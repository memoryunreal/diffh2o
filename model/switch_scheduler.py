"""
Switch Scheduler module for generating transitions between motion segments.

Enhanced with:
- Dataset-agnostic njoints (398 for OakInk2, 117 for GRAB)
- Dual object BPS conditioning (source + target objects)
- Duration predictor head for inference-time duration estimation
- Classifier-free guidance support via condition dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union
from model.mdm import PositionalEncoding, TimestepEmbedder
from diffusion.gaussian_diffusion import GaussianDiffusion


class SwitchScheduler(nn.Module):
    """
    Diffusion model for generating smooth transitions between motion segments.

    Learns to generate natural transitions conditioned on:
    - Boundary frames from adjacent segments
    - Source and target text descriptions
    - Source and target object BPS encodings
    """

    def __init__(self,
                 nfeats: int = 1,
                 njoints: int = 398,
                 latent_dim: int = 512,
                 ff_dim: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 transition_length: int = 60,
                 use_text_condition: bool = True,
                 text_latent_dim: int = 512,
                 boundary_frames: int = 10,
                 arch: str = 'transformer',
                 use_obj_condition: bool = True,
                 obj_bps_dim: int = 3072,
                 cond_mask_prob: float = 0.1):
        """
        Initialize the Switch Scheduler.

        Args:
            nfeats: Number of features per joint (1 for DiffH2O convention)
            njoints: Feature dimension (398 for OakInk2, 117 for GRAB)
            latent_dim: Dimension of latent space
            ff_dim: Dimension of feedforward network
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            activation: Activation function
            transition_length: Default length of transitions
            use_text_condition: Whether to use text conditioning
            text_latent_dim: Dimension of text embeddings
            boundary_frames: Number of boundary frames to condition on
            arch: Architecture type ('transformer' or 'unet')
            use_obj_condition: Whether to use object BPS conditioning
            obj_bps_dim: Flattened dimension of object BPS encoding
            cond_mask_prob: Probability of masking conditions for CFG training
        """
        super().__init__()

        self.nfeats = nfeats
        self.njoints = njoints
        self.latent_dim = latent_dim
        self.transition_length = transition_length
        self.use_text_condition = use_text_condition
        self.use_obj_condition = use_obj_condition
        self.boundary_frames = boundary_frames
        self.arch = arch
        self.cond_mask_prob = cond_mask_prob

        # Input dimension
        self.input_dim = njoints * nfeats

        # Input projection
        self.input_projection = nn.Linear(self.input_dim, latent_dim)

        # Positional encoding for sequence positions
        self.positional_encoding = PositionalEncoding(
            latent_dim, dropout, max_len=max(transition_length * 2, 200)
        )

        # Separate positional encoding for timestep embedder (needs max_len >= diffusion steps)
        self._timestep_pos_encoder = PositionalEncoding(latent_dim, dropout=0.0, max_len=5000)

        # Timestep embedder (requires sequence_pos_encoder for sinusoidal lookup)
        self.timestep_embedder = TimestepEmbedder(latent_dim, self._timestep_pos_encoder)

        # Boundary frame encoder
        self.boundary_encoder = nn.Sequential(
            nn.Linear(self.input_dim * boundary_frames * 2, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # Text condition encoder — dual source/target text
        if use_text_condition:
            self.text_encoder = nn.Sequential(
                nn.Linear(text_latent_dim * 2, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )

        # Dual object BPS encoder — source + target objects
        if use_obj_condition:
            self.obj_encoder = nn.Sequential(
                nn.Linear(obj_bps_dim * 2, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )

        # Main architecture
        if arch == 'transformer':
            self.transformer = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                ) for _ in range(num_layers)
            ])
        else:
            self.encoder = nn.ModuleList([
                nn.Conv1d(latent_dim, latent_dim * 2, 3, padding=1),
                nn.Conv1d(latent_dim * 2, latent_dim * 2, 3, padding=1),
            ])
            self.decoder = nn.ModuleList([
                nn.Conv1d(latent_dim * 2, latent_dim * 2, 3, padding=1),
                nn.Conv1d(latent_dim * 2, latent_dim, 3, padding=1),
            ])

        # Output projection
        self.output_projection = nn.Linear(latent_dim, self.input_dim)

        # Blending weight predictor
        self.blend_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Duration predictor head (predicts original gap duration for inference)
        self.duration_predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softplus(),  # Ensures positive output
        )

    def forward(self,
                x: torch.Tensor,
                timesteps: torch.Tensor,
                y: Optional[Dict] = None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass for the Switch Scheduler.

        Accepts model_kwargs dict 'y' for compatibility with DiffH2O's diffusion
        training loop, which passes conditions as model_kwargs['y'].

        Args:
            x: Noisy transition [B, J, D, T] (DiffH2O convention) or [B, T, J, D]
            timesteps: Diffusion timesteps [B]
            y: Conditioning dict with keys:
                - boundary_before: [B, boundary_frames, input_dim]
                - boundary_after: [B, boundary_frames, input_dim]
                - text_source_emb: [B, text_dim] (optional)
                - text_target_emb: [B, text_dim] (optional)
                - source_obj_bps: [B, bps_dim] (optional)
                - target_obj_bps: [B, bps_dim] (optional)

        Returns:
            Predicted x_0 or noise, same shape as input x
        """
        # Unpack legacy kwargs for backward compatibility
        segment_a_end = kwargs.get("segment_a_end", None)
        segment_b_start = kwargs.get("segment_b_start", None)
        text_cond = kwargs.get("text_cond", None)
        source_obj_bps = kwargs.get("source_obj_bps", None)
        target_obj_bps = kwargs.get("target_obj_bps", None)

        # Unpack from y dict if provided (training loop convention)
        if y is not None:
            segment_a_end = y.get("boundary_before", segment_a_end)
            segment_b_start = y.get("boundary_after", segment_b_start)
            source_obj_bps = y.get("source_obj_bps", source_obj_bps)
            target_obj_bps = y.get("target_obj_bps", target_obj_bps)
            # Text embeddings
            if "text_source_emb" in y and "text_target_emb" in y:
                src_emb = y["text_source_emb"]
                tgt_emb = y["text_target_emb"]
                if src_emb is not None and tgt_emb is not None:
                    # Pool word embeddings to sentence level
                    if len(src_emb.shape) == 3:
                        src_emb = src_emb.mean(dim=1)
                    if len(tgt_emb.shape) == 3:
                        tgt_emb = tgt_emb.mean(dim=1)
                    text_cond = torch.cat([src_emb, tgt_emb], dim=-1)

        B = x.shape[0]

        # Handle DiffH2O convention: [B, J, D, T] -> [B, T, J*D]
        input_is_4d = len(x.shape) == 4
        if input_is_4d:
            # DiffH2O uses [B, njoints, nfeats, T]
            # nfeats=1, njoints=feature_dim
            T = x.shape[-1]
            x_flat = x.permute(0, 3, 1, 2).reshape(B, T, -1)  # [B, T, J*D]
        else:
            T = x.shape[1]
            x_flat = x.reshape(B, T, -1)

        # Project input
        h = self.input_projection(x_flat)  # [B, T, latent_dim]

        # Timestep embedding: TimestepEmbedder returns [1, B, latent_dim]
        t_emb = self.timestep_embedder(timesteps)  # [1, B, latent_dim]
        t_emb = t_emb.squeeze(0)  # [B, latent_dim]
        h = h + t_emb.unsqueeze(1)  # broadcast over T: [B, 1, latent_dim]

        # Boundary conditioning
        if segment_a_end is not None and segment_b_start is not None:
            boundary_a_flat = segment_a_end.reshape(B, -1)
            boundary_b_flat = segment_b_start.reshape(B, -1)
            boundary_features = torch.cat([boundary_a_flat, boundary_b_flat], dim=-1)

            # Classifier-free guidance: mask boundary conditioning
            if self.training and self.cond_mask_prob > 0:
                mask = (torch.rand(B, 1, device=x.device) > self.cond_mask_prob).float()
                boundary_features = boundary_features * mask

            boundary_emb = self.boundary_encoder(boundary_features)
            h = h + boundary_emb.unsqueeze(1)

        # Text conditioning
        if self.use_text_condition and text_cond is not None:
            # CFG mask
            if self.training and self.cond_mask_prob > 0:
                mask = (torch.rand(B, 1, device=x.device) > self.cond_mask_prob).float()
                text_cond = text_cond * mask

            text_emb = self.text_encoder(text_cond)
            h = h + text_emb.unsqueeze(1)

        # Object BPS conditioning
        if self.use_obj_condition and source_obj_bps is not None and target_obj_bps is not None:
            src_bps_flat = source_obj_bps.reshape(B, -1)
            tgt_bps_flat = target_obj_bps.reshape(B, -1)
            obj_features = torch.cat([src_bps_flat, tgt_bps_flat], dim=-1)

            if self.training and self.cond_mask_prob > 0:
                mask = (torch.rand(B, 1, device=x.device) > self.cond_mask_prob).float()
                obj_features = obj_features * mask

            obj_emb = self.obj_encoder(obj_features)
            h = h + obj_emb.unsqueeze(1)

        # Positional encoding: PositionalEncoding expects [T, B, D] (seq-first)
        # Our h is [B, T, D] (batch-first), so transpose around the call
        h = self.positional_encoding(h.transpose(0, 1)).transpose(0, 1)

        # Process through main architecture
        if self.arch == 'transformer':
            for layer in self.transformer:
                h = layer(h)
        else:
            h_t = h.transpose(1, 2)
            skip_connections = []
            for conv in self.encoder:
                h_t = F.gelu(conv(h_t))
                skip_connections.append(h_t)
            for i, conv in enumerate(self.decoder):
                if i < len(skip_connections):
                    h_t = h_t + skip_connections[-(i + 1)]
                h_t = F.gelu(conv(h_t))
            h = h_t.transpose(1, 2)

        # Output projection
        output = self.output_projection(h)  # [B, T, J*D]

        # Reshape back to input format
        if input_is_4d:
            output = output.reshape(B, T, self.njoints, self.nfeats)
            output = output.permute(0, 2, 3, 1)  # [B, J, D, T]
        else:
            output = output.reshape(B, T, self.njoints, self.nfeats)

        return output

    def predict_duration(self,
                         segment_a_end: torch.Tensor,
                         segment_b_start: torch.Tensor,
                         text_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict appropriate transition duration from boundary context.

        Args:
            segment_a_end: [B, boundary_frames, J*D] or [B, boundary_frames, J, D]
            segment_b_start: [B, boundary_frames, J*D] or [B, boundary_frames, J, D]
            text_cond: Optional text conditioning [B, text_dim*2]

        Returns:
            Predicted duration in frames [B, 1]
        """
        B = segment_a_end.shape[0]
        boundary_a_flat = segment_a_end.reshape(B, -1)
        boundary_b_flat = segment_b_start.reshape(B, -1)
        boundary_features = torch.cat([boundary_a_flat, boundary_b_flat], dim=-1)
        boundary_emb = self.boundary_encoder(boundary_features)

        if self.use_text_condition and text_cond is not None:
            text_emb = self.text_encoder(text_cond)
            boundary_emb = boundary_emb + text_emb

        return self.duration_predictor(boundary_emb)

    def generate_transition(self,
                            segment_a: torch.Tensor,
                            segment_b: torch.Tensor,
                            diffusion: GaussianDiffusion,
                            text_cond: Optional[torch.Tensor] = None,
                            source_obj_bps: Optional[torch.Tensor] = None,
                            target_obj_bps: Optional[torch.Tensor] = None,
                            transition_length: Optional[int] = None,
                            guidance_scale: float = 1.0,
                            inpaint_boundary: bool = True,
                            inpaint_frames: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a transition between two segments using diffusion.

        Args:
            segment_a: End portion of first segment [B, T_boundary, J, D]
            segment_b: Start portion of second segment [B, T_boundary, J, D]
            diffusion: Gaussian diffusion instance
            text_cond: Optional text conditioning
            source_obj_bps: Source object BPS encoding [B, ...]
            target_obj_bps: Target object BPS encoding [B, ...]
            transition_length: Length of transition
            guidance_scale: Classifier-free guidance scale
            inpaint_boundary: Whether to fix boundary frames during sampling
            inpaint_frames: Number of boundary frames to inpaint

        Returns:
            Tuple of (transition, blend_weights)
        """
        if transition_length is None:
            transition_length = self.transition_length

        B = segment_a.shape[0]
        device = segment_a.device

        segment_a_end = segment_a[:, -self.boundary_frames:]
        segment_b_start = segment_b[:, :self.boundary_frames]

        model_kwargs = {
            "y": {
                "boundary_before": segment_a_end,
                "boundary_after": segment_b_start,
            }
        }
        if text_cond is not None:
            model_kwargs["y"]["text_source_emb"] = text_cond
            model_kwargs["y"]["text_target_emb"] = text_cond
        if source_obj_bps is not None:
            model_kwargs["y"]["source_obj_bps"] = source_obj_bps
        if target_obj_bps is not None:
            model_kwargs["y"]["target_obj_bps"] = target_obj_bps

        # Shape: [B, njoints, nfeats, T] (DiffH2O convention)
        shape = (B, self.njoints, self.nfeats, transition_length)

        # Build inpainting mask if requested
        inpaint_dict = None
        if inpaint_boundary and inpaint_frames > 0:
            K = min(inpaint_frames, transition_length // 4)
            last_a = segment_a_end[:, -1:].expand(-1, K, -1, -1)  # [B, K, J, D]
            first_b = segment_b_start[:, :1].expand(-1, K, -1, -1)
            middle_len = transition_length - 2 * K
            middle = torch.zeros(B, middle_len, self.njoints, self.nfeats, device=device)
            inpaint_target = torch.cat([last_a, middle, first_b], dim=1)
            inpaint_target = inpaint_target.permute(0, 2, 3, 1)  # [B, J, D, T]

            inpaint_mask = torch.zeros(B, self.njoints, self.nfeats, transition_length,
                                       device=device)
            inpaint_mask[:, :, :, :K] = 1.0
            inpaint_mask[:, :, :, -K:] = 1.0

            inpaint_dict = {"inpaint_mask": inpaint_mask, "inpaint_target": inpaint_target}

        # Sample from diffusion
        with torch.no_grad():
            if inpaint_dict is not None:
                transition = diffusion.p_sample_loop(
                    self, shape,
                    model_kwargs=model_kwargs,
                    clip_denoised=True,
                    progress=False,
                    device=device,
                    init_image=inpaint_dict["inpaint_target"],
                )
                # Enforce inpainting mask
                mask = inpaint_dict["inpaint_mask"]
                target = inpaint_dict["inpaint_target"]
                transition = mask * target + (1 - mask) * transition
            else:
                transition = diffusion.p_sample_loop(
                    self, shape,
                    model_kwargs=model_kwargs,
                    clip_denoised=True,
                    progress=False,
                    device=device,
                )

        # Generate blend weights
        with torch.no_grad():
            mid_frame = transition_length // 2
            # transition is [B, J, D, T]; get mid frame
            mid_features = self.input_projection(
                transition[:, :, :, mid_frame].reshape(B, -1)
            )
            blend_weights = self.blend_predictor(mid_features)

            blend_weights = blend_weights.unsqueeze(1)
            t_range = torch.linspace(0, 1, transition_length, device=device)
            t_range = t_range.unsqueeze(0).unsqueeze(-1)
            blend_curve = torch.sigmoid((t_range - 0.5) * 10)
            blend_weights = blend_weights * blend_curve
            blend_weights = blend_weights.unsqueeze(-1)

        return transition, blend_weights

    def interpolate_boundary(self,
                             segment_a_end: torch.Tensor,
                             segment_b_start: torch.Tensor,
                             num_frames: int) -> torch.Tensor:
        """Linear interpolation between boundaries as initialization."""
        B = segment_a_end.shape[0]
        device = segment_a_end.device
        alphas = torch.linspace(0, 1, num_frames, device=device).view(1, num_frames, 1, 1)
        start = segment_a_end.unsqueeze(1)
        end = segment_b_start.unsqueeze(1)
        return (1 - alphas) * start + alphas * end


class SwitchSchedulerLightweight(SwitchScheduler):
    """Lightweight version with fewer layers for faster inference."""

    def __init__(self, **kwargs):
        kwargs.setdefault('num_layers', 4)
        kwargs.setdefault('num_heads', 4)
        kwargs.setdefault('ff_dim', 512)
        kwargs.setdefault('dropout', 0.0)
        super().__init__(**kwargs)
