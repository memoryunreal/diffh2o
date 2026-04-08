"""
Gaze/Head Guidance for Reverse Diffusion.

Applies inference-time guidance to make the head/gaze direction look toward
the target object during transitions. This follows DiffH2O's existing pattern
for grasp guidance (gradient-based guidance during reverse diffusion).

Only applicable to OakInk2 (398D) which includes full-body SMPL-X poses.

Feature indices from the 398D representation:
    Body root translation:  0:3
    Body root rotation:     3:9
    Body pose (21 joints):  9:135  (21 joints x 6D rotation)
        Neck:  joint 12 -> indices 9 + 12*6 = 81:87
        Head:  joint 15 -> indices 9 + 15*6 = 99:105
    Object 1 position:      371:374
    Object 2 position:      380:383
    Object 3 position:      389:392
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# OakInk2 feature indices
HEAD_JOINT_IDX = 15
NECK_JOINT_IDX = 12
HEAD_ROT_START = 9 + HEAD_JOINT_IDX * 6  # 99
HEAD_ROT_END = HEAD_ROT_START + 6         # 105
NECK_ROT_START = 9 + NECK_JOINT_IDX * 6  # 81
NECK_ROT_END = NECK_ROT_START + 6        # 87
ROOT_TSL_RANGE = (0, 3)
OBJ1_POS_RANGE = (371, 374)
OBJ2_POS_RANGE = (380, 383)
OBJ3_POS_RANGE = (389, 392)


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Args:
        rot_6d: [*, 6] tensor

    Returns:
        [*, 3, 3] rotation matrix
    """
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:6]

    # Gram-Schmidt orthogonalization
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


def get_head_forward_direction(x: torch.Tensor) -> torch.Tensor:
    """Extract head forward direction from OakInk2 398D features.

    The head forward direction is the Z-axis of the head rotation matrix,
    composed with root rotation.

    Args:
        x: Motion features [*, 398]

    Returns:
        Head forward direction [*, 3] (unit vector)
    """
    # Extract head rotation (6D)
    head_rot_6d = x[..., HEAD_ROT_START:HEAD_ROT_END]
    head_rot_mat = rotation_6d_to_matrix(head_rot_6d)  # [*, 3, 3]

    # Extract root rotation (6D)
    root_rot_6d = x[..., 3:9]
    root_rot_mat = rotation_6d_to_matrix(root_rot_6d)  # [*, 3, 3]

    # Compose rotations: world_head = root_rot @ head_rot
    world_head_rot = torch.matmul(root_rot_mat, head_rot_mat)

    # Forward direction is typically the Z-axis (column 2) of rotation matrix
    forward = world_head_rot[..., :, 2]  # [*, 3]

    return F.normalize(forward, dim=-1)


def get_head_position(x: torch.Tensor) -> torch.Tensor:
    """Approximate head position from root translation + offset.

    For simplicity, we use root translation with a fixed upward offset
    as head position proxy. A more accurate version would run FK.

    Args:
        x: Motion features [*, 398]

    Returns:
        Approximate head position [*, 3]
    """
    root_tsl = x[..., 0:3]
    # Approximate head as ~0.5m above root (standing human)
    head_offset = torch.zeros_like(root_tsl)
    head_offset[..., 1] = 0.5  # Y-up convention
    return root_tsl + head_offset


def get_object_position(x: torch.Tensor, obj_idx: int = 0) -> torch.Tensor:
    """Extract object position from 398D features.

    Args:
        x: Motion features [*, 398]
        obj_idx: Object index (0, 1, or 2)

    Returns:
        Object position [*, 3]
    """
    ranges = [OBJ1_POS_RANGE, OBJ2_POS_RANGE, OBJ3_POS_RANGE]
    start, end = ranges[min(obj_idx, 2)]
    return x[..., start:end]


def compute_gaze_angular_error(
    x: torch.Tensor,
    target_obj_pos: torch.Tensor,
) -> torch.Tensor:
    """Compute angular error between head gaze direction and target object.

    Args:
        x: Motion features [*, 398]
        target_obj_pos: Target object position [*, 3]

    Returns:
        Angular error (1 - cos(angle)) [*] — 0 when looking at target
    """
    head_forward = get_head_forward_direction(x)  # [*, 3]
    head_pos = get_head_position(x)  # [*, 3]

    # Direction from head to target
    to_target = target_obj_pos - head_pos
    to_target = F.normalize(to_target, dim=-1)

    # Cosine similarity (1.0 = looking at target, -1.0 = looking away)
    cos_sim = (head_forward * to_target).sum(dim=-1)

    # Angular error: 0 when perfect, 2 when looking opposite
    return 1.0 - cos_sim


def gaze_guidance_fn(
    x_t: torch.Tensor,
    t: torch.Tensor,
    model_output: torch.Tensor,
    target_obj_pos: torch.Tensor,
    scale: float = 0.3,
    max_timestep: int = 500,
    transition_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute gaze guidance gradient for reverse diffusion.

    This follows DiffH2O's inference-time guidance pattern: compute a loss
    on the current denoised estimate and return the gradient w.r.t. x_t.

    Args:
        x_t: Current noisy sample [B, J, D, T] (DiffH2O convention)
        t: Current diffusion timestep [B]
        model_output: Model's denoised prediction (x_0 estimate) [B, J, D, T]
        target_obj_pos: Target object position [B, 3] or [B, T, 3]
        scale: Guidance scale (0.1-0.5 recommended)
        max_timestep: Only apply guidance for t < max_timestep
        transition_mask: Optional [B, T] mask — only apply guidance where mask=1

    Returns:
        Gradient to subtract from x_t [B, J, D, T]
    """
    B = x_t.shape[0]
    device = x_t.device

    # Only apply guidance at lower noise levels (t < max_timestep)
    apply_mask = (t < max_timestep).float()  # [B]
    if apply_mask.sum() == 0:
        return torch.zeros_like(x_t)

    # Enable gradient computation on x_t
    x_t_grad = x_t.detach().requires_grad_(True)

    # Use model output as x_0 estimate
    # model_output shape: [B, J, D, T] where J=398, D=1
    x_0_est = model_output  # [B, 398, 1, T]

    # Reshape to [B, T, 398]
    T = x_0_est.shape[-1]
    x_0_flat = x_0_est.squeeze(2).permute(0, 2, 1)  # [B, T, 398]

    # Expand target position to match time dimension
    if target_obj_pos.dim() == 2:
        target_obj_pos = target_obj_pos.unsqueeze(1).expand(-1, T, -1)  # [B, T, 3]

    # Compute gaze angular error
    angular_error = compute_gaze_angular_error(x_0_flat, target_obj_pos)  # [B, T]

    # Apply transition mask if provided
    if transition_mask is not None:
        angular_error = angular_error * transition_mask

    # Mean angular error
    loss = angular_error.mean()

    # Compute gradient
    if x_t_grad.grad is not None:
        x_t_grad.grad.zero_()
    loss.backward()

    if x_t_grad.grad is not None:
        gradient = x_t_grad.grad.detach()
    else:
        gradient = torch.zeros_like(x_t)

    # Scale and mask
    gradient = gradient * scale * apply_mask.view(B, 1, 1, 1)

    return gradient


class GazeGuidance:
    """Wrapper class for applying gaze guidance during generation.

    Usage:
        guidance = GazeGuidance(scale=0.3)
        # During transition generation, pass as guidance_fn to diffusion
        guidance.set_target(target_obj_position)
        guidance.set_transition_frames(start=0, end=20)  # frames to guide
    """

    def __init__(self,
                 scale: float = 0.3,
                 max_timestep: int = 500,
                 ramp_frames: int = 10):
        """
        Args:
            scale: Guidance strength (0.1-0.5 recommended)
            max_timestep: Only apply for diffusion t < this value
            ramp_frames: Number of frames to ramp guidance in/out
        """
        self.scale = scale
        self.max_timestep = max_timestep
        self.ramp_frames = ramp_frames
        self.target_obj_pos = None
        self.transition_mask = None

    def set_target(self, target_obj_pos: torch.Tensor):
        """Set target object position for gaze guidance."""
        self.target_obj_pos = target_obj_pos

    def set_transition_frames(self, total_frames: int,
                               start: int = 0, end: Optional[int] = None):
        """Set which frames to apply gaze guidance to.

        Guidance is strongest in the middle and ramps at edges.
        """
        if end is None:
            end = total_frames

        mask = torch.zeros(total_frames)
        mask[start:end] = 1.0

        # Ramp in/out at edges
        ramp = min(self.ramp_frames, (end - start) // 4)
        if ramp > 0:
            ramp_in = torch.linspace(0, 1, ramp)
            ramp_out = torch.linspace(1, 0, ramp)
            mask[start:start + ramp] = ramp_in
            mask[end - ramp:end] = ramp_out

        self.transition_mask = mask

    def __call__(self, x_t, t, model_output):
        """Apply gaze guidance."""
        if self.target_obj_pos is None:
            return torch.zeros_like(x_t)

        B = x_t.shape[0]
        T = x_t.shape[-1]
        device = x_t.device

        target = self.target_obj_pos.to(device)
        mask = None
        if self.transition_mask is not None:
            mask = self.transition_mask.to(device).unsqueeze(0).expand(B, -1)
            if mask.shape[1] != T:
                # Resize mask to match
                mask = F.interpolate(
                    mask.unsqueeze(1), size=T, mode='linear', align_corners=True
                ).squeeze(1)

        return gaze_guidance_fn(
            x_t, t, model_output, target,
            scale=self.scale,
            max_timestep=self.max_timestep,
            transition_mask=mask,
        )
