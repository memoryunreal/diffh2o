"""
Transition generator for creating smooth transitions between motion segments.

Supports:
- Trained SwitchScheduler diffusion model (OakInk2 398D or GRAB 117D)
- Boundary inpainting during reverse diffusion
- Gaze guidance for OakInk2 (head orientation toward target object)
- Interpolation fallback (linear, cubic, smooth) when no model is available
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from utils.motion_data_structures import MotionSegment, TransitionSegment
from model.switch_scheduler import SwitchScheduler, SwitchSchedulerLightweight
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.respace import space_timesteps, SpacedDiffusion
from utils import dist_util
from utils.fixseed import fixseed


@dataclass
class TransitionGeneratorConfig:
    """Configuration for transition generation."""
    model_path: Optional[str] = None
    device: str = 'cuda'
    transition_length: int = 60
    boundary_frames: int = 10
    diffusion_steps: int = 1000
    noise_schedule: str = 'cosine'
    use_lightweight: bool = False
    guidance_scale: float = 1.0
    blend_method: str = 'smooth'  # 'smooth', 'linear', 'cubic'
    seed: Optional[int] = None
    # Feature dimensions
    njoints: int = 398  # 398 for OakInk2, 117 for GRAB
    nfeats: int = 1
    # Inpainting
    inpaint_boundary: bool = True
    inpaint_frames: int = 5
    # Gaze guidance (OakInk2 only)
    use_gaze_guidance: bool = False
    gaze_guidance_scale: float = 0.3
    gaze_max_timestep: int = 500
    # Normalization
    mean_path: Optional[str] = None
    std_path: Optional[str] = None


class TransitionGenerator:
    """
    Generates smooth transitions between motion segments using diffusion.
    """

    def __init__(self, config: TransitionGeneratorConfig):
        self.config = config
        self.model = None
        self.diffusion = None
        self._initialized = False
        self.mean = None
        self.std = None

        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        dist_util.setup_dist(str(self.device))

        if config.seed is not None:
            fixseed(config.seed)

        # Load normalization stats if available
        if config.mean_path and config.std_path:
            self.mean = np.load(config.mean_path)
            self.std = np.load(config.std_path)

    def initialize(self, nfeats: int = None, njoints: int = None):
        """Initialize model and diffusion."""
        if self._initialized:
            return

        if nfeats is not None:
            self.config.nfeats = nfeats
        if njoints is not None:
            self.config.njoints = njoints

        # Create model
        ModelClass = SwitchSchedulerLightweight if self.config.use_lightweight else SwitchScheduler
        self.model = ModelClass(
            nfeats=self.config.nfeats,
            njoints=self.config.njoints,
            transition_length=self.config.transition_length,
            boundary_frames=self.config.boundary_frames,
            use_text_condition=True,
            use_obj_condition=True,
        )

        # Load pretrained weights
        if self.config.model_path:
            self._load_model(self.config.model_path)

        self.model.to(self.device)
        self.model.eval()

        # Create diffusion
        self.diffusion = self._create_diffusion()

        self._initialized = True
        print(f"Transition generator initialized (njoints={self.config.njoints}).")

    def _load_model(self, model_path: str):
        """Load pretrained model weights."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_avg' in checkpoint:
                print('Loading averaged transition model')
                state_dict = checkpoint['model_avg']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Filter out unexpected keys
            model_dict = self.model.state_dict()
            filtered = {k: v for k, v in state_dict.items() if k in model_dict}
            self.model.load_state_dict(filtered, strict=False)
            print(f"Loaded transition model from {model_path} ({len(filtered)}/{len(model_dict)} keys)")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
            print("Using randomly initialized model.")

    def _create_diffusion(self) -> SpacedDiffusion:
        """Create diffusion instance."""
        from diffusion import gaussian_diffusion as gd
        from diffusion.respace import DiffusionConfig

        steps = self.config.diffusion_steps
        betas = gd.get_named_beta_schedule(self.config.noise_schedule, steps)

        return SpacedDiffusion(
            use_timesteps=space_timesteps(steps, [steps]),
            conf=DiffusionConfig(
                betas=betas,
                model_mean_type=gd.ModelMeanType.START_X,
                model_var_type=gd.ModelVarType.FIXED_SMALL,
                loss_type=gd.LossType.MSE,
                rescale_timesteps=False,
            ),
        )

    def generate_transition(self,
                            segment_a: MotionSegment,
                            segment_b: MotionSegment,
                            text_a: Optional[str] = None,
                            text_b: Optional[str] = None,
                            transition_length: Optional[int] = None,
                            target_obj_pos: Optional[torch.Tensor] = None,
                            source_obj_bps: Optional[torch.Tensor] = None,
                            target_obj_bps: Optional[torch.Tensor] = None) -> TransitionSegment:
        """
        Generate a transition between two motion segments.

        Args:
            segment_a: First motion segment
            segment_b: Second motion segment
            text_a: Text description of first segment
            text_b: Text description of second segment
            transition_length: Length of transition
            target_obj_pos: Target object position for gaze guidance [3]
            source_obj_bps: Source object BPS encoding
            target_obj_bps: Target object BPS encoding

        Returns:
            TransitionSegment connecting the two segments
        """
        if not self._initialized:
            njoints = segment_a.num_joints
            nfeats = segment_a.dim_features
            self.initialize(nfeats, njoints)

        if transition_length is None:
            transition_length = self.config.transition_length

        # Get boundary frames
        boundary_a = segment_a.get_last_frames(self.config.boundary_frames)
        boundary_b = segment_b.get_first_frames(self.config.boundary_frames)

        boundary_a = boundary_a.to(self.device).unsqueeze(0)
        boundary_b = boundary_b.to(self.device).unsqueeze(0)

        # Generate transition
        if self.config.model_path:
            transition_motion = self._generate_with_diffusion(
                boundary_a, boundary_b, transition_length,
                target_obj_pos=target_obj_pos,
                source_obj_bps=source_obj_bps,
                target_obj_bps=target_obj_bps,
            )
        else:
            transition_motion = self._generate_interpolated(
                boundary_a, boundary_b, transition_length
            )

        # Apply blending
        transition_motion = self._apply_blending(
            transition_motion, boundary_a, boundary_b
        )

        method = 'diffusion' if self.config.model_path else 'interpolation'
        transition = TransitionSegment(
            poses=transition_motion.squeeze(0),
            text=f"Transition from '{text_a or 'segment_a'}' to '{text_b or 'segment_b'}'",
            metadata={
                'method': method,
                'transition_length': transition_length,
                'blend_method': self.config.blend_method,
                'inpaint_boundary': self.config.inpaint_boundary,
                'gaze_guidance': self.config.use_gaze_guidance,
            }
        )

        return transition

    def _generate_with_diffusion(self,
                                 boundary_a: torch.Tensor,
                                 boundary_b: torch.Tensor,
                                 transition_length: int,
                                 target_obj_pos: Optional[torch.Tensor] = None,
                                 source_obj_bps: Optional[torch.Tensor] = None,
                                 target_obj_bps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate transition using trained diffusion model."""
        B = boundary_a.shape[0]

        # Prepare conditioning
        model_kwargs = {
            "y": {
                "boundary_before": boundary_a,
                "boundary_after": boundary_b,
            }
        }

        if source_obj_bps is not None:
            model_kwargs["y"]["source_obj_bps"] = source_obj_bps.to(self.device)
        if target_obj_bps is not None:
            model_kwargs["y"]["target_obj_bps"] = target_obj_bps.to(self.device)

        # Shape: [B, njoints, nfeats, T]
        shape = (B, self.config.njoints, self.config.nfeats, transition_length)

        # Build inpainting if enabled
        inpaint_target = None
        inpaint_mask = None
        if self.config.inpaint_boundary and self.config.inpaint_frames > 0:
            K = min(self.config.inpaint_frames, transition_length // 4)
            # boundary_a: [B, boundary_frames, J, D] or [B, boundary_frames, J*D]
            last_a = boundary_a[:, -1:]  # [B, 1, ...]
            first_b = boundary_b[:, :1]

            # Expand to fill K frames
            if last_a.dim() == 3:  # [B, 1, J*D]
                last_a_exp = last_a.expand(-1, K, -1)
                first_b_exp = first_b.expand(-1, K, -1)
                mid = torch.zeros(B, transition_length - 2 * K, last_a.shape[-1],
                                  device=self.device)
                target_seq = torch.cat([last_a_exp, mid, first_b_exp], dim=1)  # [B, T, D]
                # Reshape to [B, J, nfeats, T]
                target_seq = target_seq.permute(0, 2, 1).unsqueeze(2)
            else:
                last_a_exp = last_a.expand(-1, K, -1, -1)
                first_b_exp = first_b.expand(-1, K, -1, -1)
                mid = torch.zeros(B, transition_length - 2 * K,
                                  last_a.shape[-2], last_a.shape[-1],
                                  device=self.device)
                target_seq = torch.cat([last_a_exp, mid, first_b_exp], dim=1)
                target_seq = target_seq.permute(0, 2, 3, 1)

            inpaint_target = target_seq
            inpaint_mask = torch.zeros(*shape, device=self.device)
            inpaint_mask[:, :, :, :K] = 1.0
            inpaint_mask[:, :, :, -K:] = 1.0

        # Generate using diffusion
        with torch.no_grad():
            transition = self.diffusion.p_sample_loop(
                self.model,
                shape,
                model_kwargs=model_kwargs,
                clip_denoised=True,
                progress=False,
                device=self.device,
                init_image=inpaint_target,
            )

        # Enforce inpainting
        if inpaint_mask is not None and inpaint_target is not None:
            transition = inpaint_mask * inpaint_target + (1 - inpaint_mask) * transition

        # Convert from [B, J, D, T] to [B, T, J, D]
        transition = transition.permute(0, 3, 1, 2)

        return transition

    def _generate_interpolated(self,
                               boundary_a: torch.Tensor,
                               boundary_b: torch.Tensor,
                               transition_length: int) -> torch.Tensor:
        """Fallback interpolation-based transition."""
        last_a = boundary_a[:, -1]
        first_b = boundary_b[:, 0]

        if self.config.blend_method == 'cubic':
            if boundary_a.shape[1] > 1 and boundary_b.shape[1] > 1:
                vel_a = boundary_a[:, -1] - boundary_a[:, -2]
                vel_b = boundary_b[:, 1] - boundary_b[:, 0]
                transition = self._cubic_interpolation(
                    last_a, first_b, vel_a, vel_b, transition_length
                )
            else:
                transition = self._linear_interpolation(last_a, first_b, transition_length)
        else:
            transition = self._linear_interpolation(last_a, first_b, transition_length)

        return transition

    def _linear_interpolation(self, start, end, num_frames):
        device = start.device
        alphas = torch.linspace(0, 1, num_frames, device=device)
        # Handle both [B, D] and [B, J, D] shapes
        if start.dim() == 2:
            alphas = alphas.view(1, num_frames, 1)
        else:
            alphas = alphas.view(1, num_frames, 1, 1)
        return (1 - alphas) * start.unsqueeze(1) + alphas * end.unsqueeze(1)

    def _cubic_interpolation(self, start, end, vel_start, vel_end, num_frames):
        device = start.device
        t = torch.linspace(0, 1, num_frames, device=device)
        if start.dim() == 2:
            t = t.view(1, num_frames, 1)
        else:
            t = t.view(1, num_frames, 1, 1)

        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2

        s = start.unsqueeze(1)
        e = end.unsqueeze(1)
        vs = vel_start.unsqueeze(1) * 0.5
        ve = vel_end.unsqueeze(1) * 0.5

        return h00 * s + h10 * vs + h01 * e + h11 * ve

    def _apply_blending(self, transition, boundary_a, boundary_b):
        """Apply smooth blending at transition boundaries."""
        if self.config.blend_method == 'smooth':
            T = transition.shape[1]
            device = transition.device

            blend_in = torch.sigmoid(torch.linspace(-5, 5, T // 3, device=device))
            blend_out = torch.sigmoid(torch.linspace(5, -5, T // 3, device=device))
            blend_mid = torch.ones(T - 2 * (T // 3), device=device)

            blend_weights = torch.cat([blend_in, blend_mid, blend_out])

            # Reshape for broadcasting
            if transition.dim() == 4:  # [B, T, J, D]
                blend_weights = blend_weights.view(1, T, 1, 1)
            else:
                blend_weights = blend_weights.view(1, T, 1)

            last_a = boundary_a[:, -1:]
            first_b = boundary_b[:, :1]

            third = T // 3
            transition_start = transition[:, :third]
            transition_end = transition[:, -third:]
            last_a_exp = last_a.expand_as(transition_start)
            first_b_exp = first_b.expand_as(transition_end)

            transition_start = blend_weights[:, :third] * transition_start + \
                               (1 - blend_weights[:, :third]) * last_a_exp
            transition_end = blend_weights[:, -third:] * transition_end + \
                             (1 - blend_weights[:, -third:]) * first_b_exp

            transition = torch.cat([
                transition_start,
                transition[:, third:-third],
                transition_end,
            ], dim=1)

        return transition

    def generate_batch(self,
                       segment_pairs: List[Tuple[MotionSegment, MotionSegment]],
                       texts: Optional[List[Tuple[str, str]]] = None) -> List[TransitionSegment]:
        """Generate multiple transitions."""
        transitions = []
        for i, (seg_a, seg_b) in enumerate(segment_pairs):
            text_a, text_b = None, None
            if texts and i < len(texts):
                text_a, text_b = texts[i]
            transition = self.generate_transition(seg_a, seg_b, text_a, text_b)
            transitions.append(transition)
        return transitions

    def close(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
        if self.diffusion is not None:
            del self.diffusion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._initialized = False
