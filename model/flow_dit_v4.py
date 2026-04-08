#!/usr/bin/env python3
"""
CRR-Flow V4: Entity-Centric DiT with Velocity Features + BPS Object Geometry.

Key differences from V3:
- Input includes kinematic velocity (pos + vel per entity)
- Object tokens include BPS geometric embedding (static, not noised)
- Model predicts velocity field for dynamic features only
- BPS is concatenated to object tokens before projection (not in conditioning)

Default input layout (OakInk2, dynamic 534D):
  body_pos(150) + body_vel(150) = 300D
  lh_pos(42) + lh_vel(42) = 84D
  rh_pos(39) + rh_vel(39) = 78D
  obj1_pose(9) + obj1_vel(9) = 18D × 4 objects = 72D
  Total dynamic = 534D

Static input (per object):
  BPS: 1024D × 4 objects (not noised, concatenated to object tokens)

HIMO input layout (Track B, dynamic 489D/498D):
  body: 22j×3pos + 22j×6rot + 3tsl = 201D
  lhand: 15j×3pos + 15j×6rot = 135D
  rhand: 15j×3pos + 15j×6rot = 135D
  obj: 3pos + 6rot = 9D per object (2-3 objects, padded to 4)
  BPS: 1024×3 = 3072D per object (flattened 3D coordinates)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip

# =============================================================================
# Default Constants (OakInk2 — preserved for backward compatibility)
# =============================================================================

DEFAULT_NUM_ENTITIES = 7
DEFAULT_BPS_DIM = 1024
DEFAULT_MAX_OBJECTS = 4

# Default dynamic feature dimensions (OakInk2: position + velocity)
DEFAULT_BODY_DYN_DIM = 300    # 150 pos + 150 vel
DEFAULT_LHAND_DYN_DIM = 84    # 42 pos + 42 vel
DEFAULT_RHAND_DYN_DIM = 78    # 39 pos + 39 vel
DEFAULT_OBJ_DYN_DIM = 18      # 9 pose + 9 vel (per object)

# State vocabulary
DEFAULT_NUM_STATES = 6
DEFAULT_STATE_PADDING = 5


# =============================================================================
# Reuse from V3: PositionalEncoding1D, SinusoidalTimestepEmbedder, DiTBlockV3, FinalLayerV3
# =============================================================================

class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


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


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlockV4(nn.Module):
    """DiT block with AdaLN-Zero and FlashAttention-2 (same as V3)."""

    def __init__(self, hidden_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.attn_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, key_padding_mask=None):
        B, L, D = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_mask = None
        if key_padding_mask is not None and key_padding_mask.any():
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).float()
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), float('-inf'))

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        attn_out = self.attn_proj(attn_out)

        x = x + gate_msa.unsqueeze(1) * attn_out

        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.ffn(x_norm)

        return x


class FinalLayerV4(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        # Small random init for gradient flow (lesson from V3 debugging)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# =============================================================================
# CRR-Flow V4 Model
# =============================================================================

class CRRFlowDiT_V4(nn.Module):
    """
    Entity-Centric DiT with configurable per-entity dimensions.

    Default (OakInk2):
      Dynamic input: [B, T, 534] (positions + velocities, noised)
      Static input: [B, 4, 1024] (BPS distances per object, clean)
      Output: [B, T, 534] (predicted velocity field)

    HIMO (Track B):
      Dynamic input: [B, T, 489] (2o) or [B, T, 498] (3o)
      Static input: [B, 4, 3072] (BPS 3D coords flattened, per object)
      Output: [B, T, 489/498]
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
        # Configurable entity dimensions (defaults = OakInk2)
        body_dim=DEFAULT_BODY_DYN_DIM,
        lhand_dim=DEFAULT_LHAND_DYN_DIM,
        rhand_dim=DEFAULT_RHAND_DYN_DIM,
        obj_dyn_dim=DEFAULT_OBJ_DYN_DIM,
        bps_dim=DEFAULT_BPS_DIM,
        max_objects=DEFAULT_MAX_OBJECTS,
        num_states=DEFAULT_NUM_STATES,
        state_padding=DEFAULT_STATE_PADDING,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.cond_mask_prob = cond_mask_prob
        self.skip_clip = skip_clip

        # Store configurable dimensions as instance attributes
        self.body_dim = body_dim
        self.lhand_dim = lhand_dim
        self.rhand_dim = rhand_dim
        self.obj_dyn_dim = obj_dyn_dim
        self.bps_dim = bps_dim
        self.max_objects = max_objects
        self.num_states = num_states
        self.state_padding = state_padding
        self.num_entities = 3 + max_objects  # body + lhand + rhand + N objects

        # Compute dynamic feature ranges (instance-level, not module-level)
        self.body_range = (0, body_dim)
        lh_start = body_dim
        self.lhand_range = (lh_start, lh_start + lhand_dim)
        rh_start = lh_start + lhand_dim
        self.rhand_range = (rh_start, rh_start + rhand_dim)
        obj_start = rh_start + rhand_dim
        self.obj_dyn_ranges = [
            (obj_start + i * obj_dyn_dim, obj_start + (i + 1) * obj_dyn_dim)
            for i in range(max_objects)
        ]
        self.total_dyn_dim = body_dim + lhand_dim + rhand_dim + obj_dyn_dim * max_objects

        # Object projection input: dynamic + BPS
        obj_proj_dim = obj_dyn_dim + bps_dim

        # --- Per-entity input projections ---
        self.proj_body = nn.Linear(body_dim, latent_dim)
        self.proj_lhand = nn.Linear(lhand_dim, latent_dim)
        self.proj_rhand = nn.Linear(rhand_dim, latent_dim)
        self.proj_obj = nn.Linear(obj_proj_dim, latent_dim)

        # --- Per-entity output projections (dynamic only, no BPS) ---
        self.out_body = nn.Linear(latent_dim, body_dim)
        self.out_lhand = nn.Linear(latent_dim, lhand_dim)
        self.out_rhand = nn.Linear(latent_dim, rhand_dim)
        self.out_obj = nn.Linear(latent_dim, obj_dyn_dim)

        # --- Entity-wise state embedding ---
        self.state_embed = nn.Embedding(num_states, latent_dim, padding_idx=state_padding)

        # --- Dual positional encoding ---
        self.time_pos_enc = PositionalEncoding1D(latent_dim, max_len=max_seq_len)
        self.entity_pos_enc = nn.Embedding(self.num_entities, latent_dim)

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
            DiTBlockV4(latent_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # --- Final layer ---
        self.final_layer = FinalLayerV4(latent_dim, latent_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Input projections: Xavier
        for m in [self.proj_body, self.proj_lhand, self.proj_rhand, self.proj_obj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        # Output projections: small random (not zero — prevents dead gradients)
        for m in [self.out_body, self.out_lhand, self.out_rhand, self.out_obj]:
            nn.init.xavier_uniform_(m.weight, gain=0.01)
            nn.init.zeros_(m.bias)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters()
                if not name.startswith('clip_model.') and p.requires_grad]

    def encode_text(self, texts, device=None):
        if self.clip_model is None:
            raise RuntimeError("CLIP not loaded (skip_clip=True). Use precomputed text_emb.")
        if device is None:
            device = next(self.parameters()).device
        tokens = clip.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens).float()
        return self.text_proj(text_features)

    def mask_cond(self, cond, force_mask=False):
        if force_mask or (self.training and self.cond_mask_prob > 0):
            mask = torch.rand(cond.shape[0], device=cond.device) < self.cond_mask_prob
            cond = cond * (~mask).unsqueeze(-1).float()
        return cond

    def forward(self, x_dynamic, t, y=None):
        """
        Args:
            x_dynamic: [B, T, D] noisy dynamic features
            t: [B] flow timestep in [0, 1]
            y: dict with:
                'text_emb': [B, 512] or 'text': list[B]
                'state_labels': [B, T, num_entities]
                'bps': [B, max_objects, bps_dim] static BPS per object (clean)

        Returns:
            [B, T, D] predicted velocity field for dynamic features
        """
        B, T, D = x_dynamic.shape
        device = x_dynamic.device

        # --- Split dynamic features into entities ---
        body_dyn = x_dynamic[:, :, self.body_range[0]:self.body_range[1]]
        lh_dyn = x_dynamic[:, :, self.lhand_range[0]:self.lhand_range[1]]
        rh_dyn = x_dynamic[:, :, self.rhand_range[0]:self.rhand_range[1]]
        objs_dyn = [x_dynamic[:, :, s:e] for s, e in self.obj_dyn_ranges]

        # --- Get BPS (static, clean) ---
        bps = None
        if y is not None and 'bps' in y:
            bps = y['bps'].to(device)  # [B, max_objects, bps_dim]

        # --- Per-entity projection ---
        h_body = self.proj_body(body_dyn)
        h_lh = self.proj_lhand(lh_dyn)
        h_rh = self.proj_rhand(rh_dyn)

        # Object: concat dynamic + static BPS before projection
        h_objs = []
        for i in range(self.max_objects):
            obj_dyn = objs_dyn[i]
            if bps is not None:
                obj_bps = bps[:, i, :].unsqueeze(1).expand(-1, T, -1)
                obj_full = torch.cat([obj_dyn, obj_bps], dim=-1)
            else:
                obj_full = torch.cat([obj_dyn, torch.zeros(B, T, self.bps_dim, device=device)], dim=-1)
            h_objs.append(self.proj_obj(obj_full))

        # --- Entity-wise state injection ---
        if y is not None and 'state_labels' in y:
            states = y['state_labels'].long().clamp(0, self.num_states - 1)
            h_lh = h_lh + self.state_embed(states[:, :, 1])
            h_rh = h_rh + self.state_embed(states[:, :, 2])
            for i in range(self.max_objects):
                h_objs[i] = h_objs[i] + self.state_embed(states[:, :, 3 + i])

        # --- Stack into spatio-temporal sequence ---
        tokens = torch.stack([h_body, h_lh, h_rh] + h_objs, dim=2)  # [B, T, E, D]

        # Dual positional encoding
        time_pe = self.time_pos_enc.pe[:, :T, :].unsqueeze(2)
        tokens = tokens + time_pe

        entity_ids = torch.arange(self.num_entities, device=device)
        entity_pe = self.entity_pos_enc(entity_ids).unsqueeze(0).unsqueeze(0)
        tokens = tokens + entity_pe

        # Flatten
        tokens = tokens.reshape(B, T * self.num_entities, self.latent_dim)

        # --- Padding mask ---
        key_padding_mask = None
        if y is not None and 'state_labels' in y:
            states = y['state_labels']
            obj_padding = (states[:, :, 3:] == self.state_padding)
            full_mask = torch.zeros(B, T, self.num_entities, device=device, dtype=torch.bool)
            full_mask[:, :, 3:] = obj_padding
            key_padding_mask = full_mask.reshape(B, T * self.num_entities)

        # --- Condition vector ---
        t_emb = self.time_embed(t)

        text_emb = torch.zeros(B, self.latent_dim, device=device)
        if y is not None and 'text_emb' in y:
            text_emb = self.text_proj(y['text_emb'].to(device))
            text_emb = self.mask_cond(text_emb)
        elif y is not None and 'text' in y:
            text_emb = self.encode_text(y['text'], device=device)
            text_emb = self.mask_cond(text_emb)

        c = self.cond_mlp(torch.cat([t_emb, text_emb], dim=-1))

        # --- DiT blocks ---
        for block in self.blocks:
            tokens = block(tokens, c, key_padding_mask=key_padding_mask)

        # --- Final layer ---
        tokens = self.final_layer(tokens, c)

        # --- Unflatten and per-entity output ---
        tokens = tokens.reshape(B, T, self.num_entities, self.latent_dim)

        out_body = self.out_body(tokens[:, :, 0])
        out_lh = self.out_lhand(tokens[:, :, 1])
        out_rh = self.out_rhand(tokens[:, :, 2])
        out_objs = [self.out_obj(tokens[:, :, 3 + i]) for i in range(self.max_objects)]

        output = torch.cat([out_body, out_lh, out_rh] + out_objs, dim=-1)
        return output


if __name__ == "__main__":
    # --- OakInk2 (default dimensions) ---
    print("=== OakInk2 (default) ===")
    model = CRRFlowDiT_V4(latent_dim=512, num_layers=8, num_heads=8, ff_dim=2048, skip_clip=True)
    trainable = sum(p.numel() for p in model.parameters_wo_clip()) / 1e6
    print(f"Trainable: {trainable:.1f}M params, total_dyn_dim={model.total_dyn_dim}")

    x = torch.randn(2, 200, model.total_dyn_dim)
    t = torch.rand(2)
    y = {
        'text_emb': torch.randn(2, 512),
        'state_labels': torch.zeros(2, 200, model.num_entities, dtype=torch.long),
        'bps': torch.randn(2, model.max_objects, model.bps_dim),
    }
    out = model(x, t, y=y)
    print(f"Output: {out.shape}")  # [2, 200, 534]

    # --- HIMO 2-object (Track B, 5 entities) ---
    print("\n=== HIMO 2o (Track B) ===")
    model_himo_2o = CRRFlowDiT_V4(
        latent_dim=512, num_layers=8, num_heads=8, ff_dim=2048, skip_clip=True,
        max_seq_len=300,
        body_dim=201,     # 22j×3 + 22j×6 + 3tsl
        lhand_dim=135,    # 15j×3 + 15j×6
        rhand_dim=135,    # 15j×3 + 15j×6
        obj_dyn_dim=9,    # 3pos + 6rot (no velocity)
        bps_dim=1024,     # scalar distances (same as OakInk2)
        max_objects=2,    # 2 object slots → 5 entities
    )
    trainable_h = sum(p.numel() for p in model_himo_2o.parameters_wo_clip()) / 1e6
    print(f"Trainable: {trainable_h:.1f}M params, total_dyn_dim={model_himo_2o.total_dyn_dim}, "
          f"entities={model_himo_2o.num_entities}")  # 489D, 5 entities

    x_h = torch.randn(2, 300, model_himo_2o.total_dyn_dim)
    t_h = torch.rand(2)
    y_h = {
        'text_emb': torch.randn(2, 512),
        'bps': torch.randn(2, 2, 1024),
    }
    out_h = model_himo_2o(x_h, t_h, y=y_h)
    print(f"Output: {out_h.shape}")  # [2, 300, 489] = 201+135+135+2*9

    # --- HIMO 3-object (Track B, 6 entities) ---
    print("\n=== HIMO 3o (Track B) ===")
    model_himo_3o = CRRFlowDiT_V4(
        latent_dim=512, num_layers=8, num_heads=8, ff_dim=2048, skip_clip=True,
        max_seq_len=300,
        body_dim=201, lhand_dim=135, rhand_dim=135,
        obj_dyn_dim=9, bps_dim=1024, max_objects=3,
    )
    x_3o = torch.randn(2, 300, model_himo_3o.total_dyn_dim)
    out_3o = model_himo_3o(x_3o, t_h, y={
        'text_emb': torch.randn(2, 512),
        'bps': torch.randn(2, 3, 1024),
    })
    print(f"Output: {out_3o.shape}")  # [2, 300, 498] = 201+135+135+3*9
