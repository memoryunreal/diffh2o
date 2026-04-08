"""
CRR-Flow DiT: State-Aware Diffusion Transformer for Flow Matching.

Adapted from model/mdm_dit.py for flow matching (velocity field prediction).
Conditioned on: flow timestep t∈[0,1], CLIP text embedding, entity state labels.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


def modulate(x, shift, scale):
    """AdaLN modulation: x * (1 + scale) + shift."""
    return x * (1 + scale) + (shift if shift is not None else 0)


class SinusoidalTimestepEmbedder(nn.Module):
    """Embeds continuous flow timestep t∈[0,1] into latent space."""

    def __init__(self, latent_dim, max_period=10000):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, t):
        """
        Args:
            t: [B] continuous timestep in [0, 1]
        Returns:
            [B, latent_dim] timestep embedding
        """
        half = self.latent_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.latent_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)


class PositionalEncoding1D(nn.Module):
    """1D sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x):
        """x: [B, T, D] → [B, T, D] with positional encoding added."""
        x = x + self.pe[:x.shape[1], :].unsqueeze(0)
        return self.dropout(x)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning (pre-norm style)."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        # AdaLN-Zero: 6 modulation params (shift, scale, gate) × 2 (attn, mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        """
        Args:
            x: [B, T, D] token sequence
            c: [B, D] condition vector (timestep + text)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).unsqueeze(1).chunk(6, dim=-1)

        # Self-attention with AdaLN
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_norm, x_norm, x_norm)[0]

        # FFN with AdaLN
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_norm)
        return x


class FinalLayer(nn.Module):
    """Final projection with AdaLN-Zero modulation."""

    def __init__(self, d_model, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, output_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).unsqueeze(1).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class FlowDiT(nn.Module):
    """
    State-Aware DiT for Flow Matching motion generation.

    Predicts velocity field v_t conditioned on:
    - Flow timestep t ∈ [0, 1]
    - CLIP text embedding
    - Per-frame entity state labels

    Input:  x_t [B, T, D] noisy motion (D=398 for OakInk2)
    Output: v_t [B, T, D] predicted velocity field
    """

    def __init__(
        self,
        input_dim=360,
        latent_dim=512,
        num_layers=8,
        num_heads=8,
        ff_dim=1024,
        dropout=0.1,
        num_states=7,
        max_entities=20,
        max_seq_len=200,
        clip_version='ViT-B/32',
        cond_mask_prob=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cond_mask_prob = cond_mask_prob

        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding1D(latent_dim, max_len=max_seq_len, dropout=dropout)

        # Timestep embedding (continuous t ∈ [0,1])
        self.time_embed = SinusoidalTimestepEmbedder(latent_dim)

        # State embedding: discrete state → continuous feature
        self.state_embed = nn.Embedding(num_states + 1, latent_dim, padding_idx=num_states)
        # padding_idx maps -1 (after shifting) to zero vector
        self.max_entities = max_entities

        # CLIP text encoder (frozen)
        self.clip_model = self._load_clip(clip_version)
        self.text_proj = nn.Linear(512, latent_dim)

        # Condition MLP: combines timestep + text
        self.cond_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(latent_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.final_layer = FinalLayer(latent_dim, input_dim)

        # For compatibility with training code
        self.njoints = input_dim
        self.nfeats = 1

    def _load_clip(self, clip_version):
        """Load and freeze CLIP text encoder."""
        model, _ = clip.load(clip_version, device='cpu', jit=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def encode_text(self, raw_text):
        """Encode text strings to CLIP embeddings."""
        device = next(self.parameters()).device
        tokens = clip.tokenize(raw_text, truncate=True).to(device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens).float()
        return text_features  # [B, 512]

    def mask_cond(self, cond, force_mask=False):
        """Classifier-free guidance: randomly mask conditions during training."""
        if force_mask or (self.training and self.cond_mask_prob > 0):
            B = cond.shape[0]
            mask = torch.rand(B, 1, device=cond.device) > self.cond_mask_prob
            return cond * mask.float()
        return cond

    def parameters_wo_clip(self):
        """Return parameters excluding frozen CLIP."""
        return [p for name, p in self.named_parameters()
                if not name.startswith('clip_model') and p.requires_grad]

    def forward(self, x_t, t, y=None):
        """
        Args:
            x_t: [B, T, D] or [B, D, 1, T] noisy motion at time t
            t: [B] flow timestep in [0, 1]
            y: dict with optional keys:
                - 'text': list of strings OR 'text_emb': [B, 512] CLIP embeddings
                - 'state_labels': [B, T, E] int8 state labels (E=max_entities)
        Returns:
            v_t: [B, T, D] predicted velocity field (same shape as x_t input)
        """
        # Handle DiffH2O convention [B, D, 1, T] → [B, T, D]
        input_4d = len(x_t.shape) == 4
        if input_4d:
            B, D, _, T = x_t.shape
            x_t = x_t.squeeze(2).permute(0, 2, 1)  # [B, T, D]
        else:
            B, T, D = x_t.shape

        # Input projection
        h = self.input_proj(x_t)  # [B, T, latent_dim]

        # Positional encoding
        h = self.pos_enc(h)

        # State embedding injection
        if y is not None and 'state_labels' in y:
            states = y['state_labels']  # [B, T, E] int8
            # Shift -1 → num_states (padding_idx), clamp valid range
            states_shifted = states.clamp(min=-1) + 1  # -1→0 becomes padding
            # Actually: padding_idx=num_states, so map -1→num_states, 0→0, etc.
            states_for_embed = torch.where(
                states >= 0, states, torch.full_like(states, self.state_embed.padding_idx)
            ).long()
            # Embed and sum over entities: [B, T, E, latent] → [B, T, latent]
            state_emb = self.state_embed(states_for_embed)  # [B, T, E, latent_dim]
            # Mask padding
            mask = (states >= 0).unsqueeze(-1).float()  # [B, T, E, 1]
            state_emb = (state_emb * mask).sum(dim=2)  # [B, T, latent_dim]
            h = h + state_emb

        # Timestep embedding
        t_emb = self.time_embed(t)  # [B, latent_dim]

        # Text embedding
        if y is not None and 'text' in y:
            text_emb = self.encode_text(y['text'])  # [B, 512]
            text_emb = self.text_proj(text_emb)  # [B, latent_dim]
            text_emb = self.mask_cond(text_emb)
        elif y is not None and 'text_emb' in y:
            text_emb = self.text_proj(y['text_emb'])
            text_emb = self.mask_cond(text_emb)
        else:
            text_emb = torch.zeros(B, self.latent_dim, device=x_t.device)

        # Combined condition for AdaLN
        c = self.cond_mlp(torch.cat([t_emb, text_emb], dim=-1))  # [B, latent_dim]

        # Transformer blocks
        for block in self.blocks:
            h = block(h, c)

        # Final projection
        v_t = self.final_layer(h, c)  # [B, T, D]

        # Convert back to 4D if needed
        if input_4d:
            v_t = v_t.permute(0, 2, 1).unsqueeze(2)  # [B, D, 1, T]

        return v_t
