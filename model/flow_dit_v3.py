#!/usr/bin/env python3
"""
CRR-Flow V3: Object-Centric DiT with Entity-wise State Injection.

Key differences from V1 FlowDiT:
- Per-entity input projections (body 150D, lhand 42D, rhand 39D, obj 9D each)
- Entity-wise state embedding (not global sum)
- Spatio-temporal token sequence [B, T×7, D]
- Dual positional encoding (time + entity)
- FlashAttention-2 via F.scaled_dot_product_attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip

# =============================================================================
# Constants
# =============================================================================

NUM_ENTITIES = 7  # body + lh + rh + 4 objects
BODY_DIM = 150    # 50 markers × 3
LHAND_DIM = 42    # 14 markers × 3
RHAND_DIM = 39    # 13 markers × 3
OBJ_DIM = 9       # pos3 + rot6d
TOTAL_DIM = 267   # 231 markers + 36 objects

# Entity feature ranges within 267D input
BODY_RANGE = (0, 150)
LHAND_RANGE = (150, 192)
RHAND_RANGE = (192, 231)
OBJ_RANGES = [(231 + i * 9, 231 + (i + 1) * 9) for i in range(4)]

# State vocabulary
NUM_STATES = 6  # Static(0), Reach(1), Grasped(2), Interacting(3), Release(4), Padding(5)
STATE_PADDING = 5


# =============================================================================
# Positional Encoding
# =============================================================================

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, D]

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


# =============================================================================
# Timestep Embedding
# =============================================================================

class SinusoidalTimestepEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


# =============================================================================
# DiT Block with FlashAttention-2
# =============================================================================

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlockV3(nn.Module):
    """DiT block with AdaLN-Zero and FlashAttention-2."""

    def __init__(self, hidden_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        # Self-attention with manual Q/K/V for FlashAttention
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.attn_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout),
        )

        # AdaLN-Zero: 6 params (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # Initialize gate to zero
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, key_padding_mask=None):
        B, L, D = x.shape

        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Pre-norm self-attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, head_dim]
        q, k, v = qkv.unbind(0)

        # FlashAttention-2 via scaled_dot_product_attention
        # Convert key_padding_mask to attention bias (broadcast-friendly, no expansion)
        attn_mask = None
        if key_padding_mask is not None and key_padding_mask.any():
            # key_padding_mask: [B, L] bool (True = pad)
            # Create [B, 1, 1, L] bias: 0 for valid, -inf for padded → broadcasts to [B, H, L, L]
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).float()  # [B, 1, 1, L]
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), float('-inf'))  # pad → -inf

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        attn_out = self.attn_proj(attn_out)

        x = x + gate_msa.unsqueeze(1) * attn_out

        # Pre-norm FFN
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.ffn(x_norm)

        return x


# =============================================================================
# Final Layer
# =============================================================================

class FinalLayerV3(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        # Small random init (not zero) to allow gradient flow from step 0.
        # Zero-init would make output=0, blocking all gradients to the backbone.
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# =============================================================================
# CRR-Flow V3 Model
# =============================================================================

class CRRFlowDiT_V3(nn.Module):
    """
    Object-Centric DiT with Entity-wise State Injection for HOI generation.

    Input: [B, T, 267] (77 markers flattened + 4 objects × 9D)
    Output: [B, T, 267] (predicted velocity field)
    """

    def __init__(
        self,
        latent_dim=512,
        num_layers=8,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        max_seq_len=200,
        cond_mask_prob=0.1,
        skip_clip=False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.cond_mask_prob = cond_mask_prob
        self.num_entities = NUM_ENTITIES
        self.skip_clip = skip_clip

        # --- Per-entity input projections ---
        self.proj_body = nn.Linear(BODY_DIM, latent_dim)
        self.proj_lhand = nn.Linear(LHAND_DIM, latent_dim)
        self.proj_rhand = nn.Linear(RHAND_DIM, latent_dim)
        self.proj_obj = nn.Linear(OBJ_DIM, latent_dim)  # shared for all 4 obj slots

        # --- Per-entity output projections ---
        self.out_body = nn.Linear(latent_dim, BODY_DIM)
        self.out_lhand = nn.Linear(latent_dim, LHAND_DIM)
        self.out_rhand = nn.Linear(latent_dim, RHAND_DIM)
        self.out_obj = nn.Linear(latent_dim, OBJ_DIM)  # shared

        # --- Entity-wise state embedding ---
        self.state_embed = nn.Embedding(NUM_STATES, latent_dim, padding_idx=STATE_PADDING)

        # --- Dual positional encoding ---
        self.time_pos_enc = PositionalEncoding1D(latent_dim, max_len=max_seq_len)
        self.entity_pos_enc = nn.Embedding(NUM_ENTITIES, latent_dim)

        # --- Timestep embedding ---
        self.time_embed = SinusoidalTimestepEmbedder(latent_dim)

        # --- CLIP text encoder (frozen, optional) ---
        if not skip_clip:
            self.clip_model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
            for p in self.clip_model.parameters():
                p.requires_grad = False
        else:
            self.clip_model = None
        self.text_proj = nn.Linear(512, latent_dim)

        # --- Condition MLP ---
        self.cond_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            DiTBlockV3(latent_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # --- Final layers (per-entity) ---
        self.final_layer = FinalLayerV3(latent_dim, latent_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Input projections: Xavier for good gradient flow
        for m in [self.proj_body, self.proj_lhand, self.proj_rhand, self.proj_obj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        # Output projections: small random init (NOT zero — zero kills gradient flow)
        for m in [self.out_body, self.out_lhand, self.out_rhand, self.out_obj]:
            nn.init.xavier_uniform_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters()
                if not name.startswith('clip_model.') and p.requires_grad]

    def encode_text(self, texts):
        """Encode text using frozen CLIP. Requires skip_clip=False."""
        if self.clip_model is None:
            raise RuntimeError("CLIP model not loaded (skip_clip=True). Use precomputed text_emb.")
        device = next(self.parameters()).device
        tokens = clip.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens).float()
        return self.text_proj(text_features)  # [B, latent_dim]

    def mask_cond(self, cond, force_mask=False):
        """Classifier-free guidance: randomly mask conditioning."""
        if force_mask or (self.training and self.cond_mask_prob > 0):
            mask = torch.rand(cond.shape[0], device=cond.device) < self.cond_mask_prob
            cond = cond * (~mask).unsqueeze(-1).float()
        return cond

    def forward(self, x, t, y=None):
        """
        Args:
            x: [B, T, 267] noisy motion
            t: [B] flow timestep ∈ [0, 1]
            y: dict with 'text' (list[B]), 'state_labels' [B, T, 7]

        Returns:
            [B, T, 267] predicted velocity field
        """
        B, T, D = x.shape
        device = x.device

        # --- Split into entity features ---
        body = x[:, :, BODY_RANGE[0]:BODY_RANGE[1]]        # [B, T, 150]
        lhand = x[:, :, LHAND_RANGE[0]:LHAND_RANGE[1]]     # [B, T, 42]
        rhand = x[:, :, RHAND_RANGE[0]:RHAND_RANGE[1]]     # [B, T, 39]
        objs = [x[:, :, s:e] for s, e in OBJ_RANGES]       # 4 × [B, T, 9]

        # --- Per-entity projection ---
        h_body = self.proj_body(body)      # [B, T, D_lat]
        h_lh = self.proj_lhand(lhand)
        h_rh = self.proj_rhand(rhand)
        h_objs = [self.proj_obj(o) for o in objs]  # 4 × [B, T, D_lat]

        # --- Entity-wise state injection ---
        if y is not None and 'state_labels' in y:
            states = y['state_labels']  # [B, T, 7]
            # Clamp to valid range
            states = states.clamp(0, NUM_STATES - 1)
            # Body gets no state injection (column 0 is always Static)
            h_lh = h_lh + self.state_embed(states[:, :, 1])
            h_rh = h_rh + self.state_embed(states[:, :, 2])
            for i in range(4):
                h_objs[i] = h_objs[i] + self.state_embed(states[:, :, 3 + i])

        # --- Stack into spatio-temporal sequence [B, T, 7, D_lat] ---
        tokens = torch.stack([h_body, h_lh, h_rh] + h_objs, dim=2)  # [B, T, 7, D_lat]

        # --- Add dual positional encodings ---
        # Time positional encoding: [1, T, 1, D_lat]
        time_pe = self.time_pos_enc.pe[:, :T, :].unsqueeze(2)  # [1, T, 1, D_lat]
        tokens = tokens + time_pe

        # Entity positional encoding: [1, 1, 7, D_lat]
        entity_ids = torch.arange(NUM_ENTITIES, device=device)
        entity_pe = self.entity_pos_enc(entity_ids).unsqueeze(0).unsqueeze(0)  # [1, 1, 7, D_lat]
        tokens = tokens + entity_pe

        # --- Flatten to sequence: [B, T*7, D_lat] ---
        tokens = tokens.reshape(B, T * NUM_ENTITIES, self.latent_dim)

        # --- Build padding mask for unused object slots ---
        key_padding_mask = None
        if y is not None and 'state_labels' in y:
            states = y['state_labels']  # [B, T, 7]
            # Objects with state=PADDING are masked
            obj_padding = (states[:, :, 3:] == STATE_PADDING)  # [B, T, 4]
            # Build full mask [B, T, 7] — body/hands are never padded
            full_mask = torch.zeros(B, T, NUM_ENTITIES, device=device, dtype=torch.bool)
            full_mask[:, :, 3:] = obj_padding
            key_padding_mask = full_mask.reshape(B, T * NUM_ENTITIES)  # [B, T*7]

        # --- Condition vector ---
        t_emb = self.time_embed(t)  # [B, D_lat]

        text_emb = torch.zeros(B, self.latent_dim, device=device)
        if y is not None and 'text_emb' in y:
            text_emb = self.text_proj(y['text_emb'].to(device))
            text_emb = self.mask_cond(text_emb)
        elif y is not None and 'text' in y:
            text_emb = self.encode_text(y['text'])
            text_emb = self.mask_cond(text_emb)

        c = self.cond_mlp(torch.cat([t_emb, text_emb], dim=-1))  # [B, D_lat]

        # --- DiT blocks ---
        for block in self.blocks:
            tokens = block(tokens, c, key_padding_mask=key_padding_mask)

        # --- Final layer ---
        tokens = self.final_layer(tokens, c)  # [B, T*7, D_lat]

        # --- Unflatten and project back per entity ---
        tokens = tokens.reshape(B, T, NUM_ENTITIES, self.latent_dim)

        out_body = self.out_body(tokens[:, :, 0])    # [B, T, 150]
        out_lh = self.out_lhand(tokens[:, :, 1])     # [B, T, 42]
        out_rh = self.out_rhand(tokens[:, :, 2])     # [B, T, 39]
        out_objs = [self.out_obj(tokens[:, :, 3 + i]) for i in range(4)]  # 4 × [B, T, 9]

        # --- Concatenate back to 267D ---
        output = torch.cat([out_body, out_lh, out_rh] + out_objs, dim=-1)  # [B, T, 267]
        return output


if __name__ == "__main__":
    # Quick test
    model = CRRFlowDiT_V3(latent_dim=512, num_layers=8, num_heads=8, ff_dim=2048)
    trainable = sum(p.numel() for p in model.parameters_wo_clip()) / 1e6
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Trainable: {trainable:.1f}M, Total: {total:.1f}M (includes frozen CLIP)")

    x = torch.randn(2, 200, 267)
    t = torch.rand(2)
    y = {
        'text': ['pick up the cup', 'put down the plate'],
        'state_labels': torch.zeros(2, 200, 7, dtype=torch.long),
    }
    out = model(x, t, y=y)
    print(f"Output: {out.shape}")  # [2, 200, 267]
