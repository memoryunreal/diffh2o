# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from utils.parser_util import BaseOptions, DataOptions

@dataclass
class grab_motion_rel(BaseOptions, DataOptions):
    dataset: str = 'grab'
    data_dir: str = ''
    abs_3d: bool = False

@dataclass
class grab_motion_rel_pca(BaseOptions, DataOptions):
    dataset: str = 'grab'
    data_dir: str = ''
    abs_3d: bool = False
    use_pca: bool = True

## GRASPING
@dataclass
class diffh2o_grasp(grab_motion_rel_pca):
    traj_only: bool = True
    pre_grasp: bool = True
    hands_only: bool = True
    use_random_proj: bool = True
    obj_enc: bool = True
    text_detailed: bool = False
    diffusion_steps: int = 1000
    motion_enc_frames: int = 1
    random_proj_scale: float = 10
    data_repr: str = 'diffh2o_representation_grasp'
    mean_name: str = 'Mean_diffh2o_grasp.npy'
    std_name: str = 'Std_diffh2o_grasp.npy'
    proj_matrix_name: str = 'rand_proj_diffh2o_grasp.npy'

## INTERACTION
@dataclass
class diffh2o_interaction(grab_motion_rel_pca):
    traj_only: bool = False
    hands_only: bool = False
    use_random_proj: bool = True
    random_proj_scale: float = 10
    obj_enc: bool = True
    motion_enc_frames: int = 0
    data_repr: str = 'diffh2o_representation_interaction'
    mean_name: str = 'Mean_diffh2o_interaction.npy'
    std_name: str = 'Std_diffh2o_interaction.npy'
    proj_matrix_name: str = 'rand_proj_diffh2o_interaction.npy'

### Full Simple Texts (GRASPING + INTERACTION)
@dataclass
class diffh2o_full(grab_motion_rel_pca):
    traj_only: bool = False
    hands_only: bool = False
    use_random_proj: bool = True
    random_proj_scale: float = 10
    obj_enc: bool = True
    motion_enc_frames: int = 0
    text_detailed: bool = False
    data_repr: str = 'diffh2o_representation_full'
    mean_name: str = 'Mean_diffh2o_full.npy'
    std_name: str = 'Std_diffh2o_full.npy'
    proj_matrix_name: str = 'rand_proj_diffh2o.npy'

### Full Detailed Texts (GRASPING + INTERACTION)
@dataclass
class diffh2o_full_detailed(grab_motion_rel_pca):
    traj_only: bool = False
    hands_only: bool = False
    use_random_proj: bool = True
    random_proj_scale: float = 10
    obj_enc: bool = True
    motion_enc_frames: int = 0
    text_detailed: bool = True
    data_repr: str = 'diffh2o_representation_full'
    mean_name: str = 'Mean_diffh2o_full.npy'
    std_name: str = 'Std_diffh2o_full.npy'
    proj_matrix_name: str = 'rand_proj_diffh2o.npy'

### MDM Baseline
@dataclass
class mdm_full(grab_motion_rel_pca):
    traj_only: bool = False
    hands_only: bool = False
    use_random_proj: bool = False
    random_proj_scale: float = 1
    obj_enc: bool = True
    motion_enc_frames: int = 0
    data_repr: str = 'diffh2o_representation_full'
    mean_name: str = 'Mean_diffh2o_full.npy'
    std_name: str = 'Std_diffh2o_full.npy'
    proj_matrix_name: str = ''


# ============================================
# OakInk2 Dataset Configurations
# ============================================

@dataclass
class oakink2_base(BaseOptions, DataOptions):
    """Base configuration for OakInk2 dataset."""
    dataset: str = 'oakink2'
    data_dir: str = 'dataset/OAKINK2'
    abs_3d: bool = False
    use_pca: bool = True
    max_objects: int = 3
    include_body: bool = True
    data_mode: str = 'primitive'  # 'primitive' or 'complex'


@dataclass
class oakink2_full(oakink2_base):
    """Full OakInk2 configuration with body + hands + objects (primitive mode)."""
    traj_only: bool = False
    hands_only: bool = False
    use_random_proj: bool = False
    random_proj_scale: float = 1.0
    obj_enc: bool = True
    motion_enc_frames: int = 0
    text_detailed: bool = False
    data_mode: str = 'primitive'
    data_repr: str = 'oakink2_primitive'
    mean_name: str = 'Mean_oakink2_primitive.npy'
    std_name: str = 'Std_oakink2_primitive.npy'
    proj_matrix_name: str = ''


@dataclass
class oakink2_complex(oakink2_base):
    """OakInk2 configuration for complex/long sequences."""
    traj_only: bool = False
    hands_only: bool = False
    use_random_proj: bool = False
    random_proj_scale: float = 1.0
    obj_enc: bool = True
    motion_enc_frames: int = 0
    text_detailed: bool = False
    data_mode: str = 'complex'
    data_repr: str = 'oakink2_complex'
    mean_name: str = 'Mean_oakink2_complex.npy'
    std_name: str = 'Std_oakink2_complex.npy'
    proj_matrix_name: str = ''


# ============================================
# Transition Dataset Configurations
# ============================================

@dataclass
class transition_base(BaseOptions, DataOptions):
    """Base configuration for transition datasets."""
    abs_3d: bool = False
    use_pca: bool = True
    transition_frames: int = 60
    boundary_frames: int = 10


@dataclass
class oakink2_transition(transition_base):
    """OakInk2 transition dataset (real transitions, 398D)."""
    dataset: str = 'oakink2_transition'
    data_dir: str = 'dataset/OAKINK2'
    data_mode: str = 'transition'
    traj_only: bool = False
    hands_only: bool = False
    use_random_proj: bool = False
    random_proj_scale: float = 1.0
    obj_enc: bool = True
    motion_enc_frames: int = 0
    text_detailed: bool = False
    data_repr: str = 'oakink2_transitions'
    mean_name: str = 'Mean_transitions.npy'
    std_name: str = 'Std_transitions.npy'
    proj_matrix_name: str = ''


@dataclass
class grab_transition(transition_base):
    """GRAB transition dataset (self-supervised infilling, 117D)."""
    dataset: str = 'grab_transition'
    data_dir: str = ''
    data_mode: str = 'transition'
    traj_only: bool = False
    hands_only: bool = False
    use_random_proj: bool = False
    random_proj_scale: float = 1.0
    obj_enc: bool = True
    motion_enc_frames: int = 0
    text_detailed: bool = False
    data_repr: str = 'diffh2o_representation_full'
    mean_name: str = 'Mean_diffh2o_full.npy'
    std_name: str = 'Std_diffh2o_full.npy'
    proj_matrix_name: str = ''


# ============================================
# HIMO Dataset Configurations (Track B: joint-based)
# ============================================

@dataclass
class himo_base(BaseOptions, DataOptions):
    """Base configuration for HIMO dataset (joint positions + 6D rotations).

    Feature layout per entity (position-only, no velocity):
      Body(22j): 66pos + 132rot + 3tsl = 201D
      LHand(15j): 45pos + 90rot = 135D
      RHand(15j): 45pos + 90rot = 135D
      Object: 3pos + 6rot = 9D each
      BPS: 1024 scalar distances per object (same format as OakInk2)
    """
    dataset: str = 'himo'
    data_dir: str = 'dataset/HIMO_V4'
    abs_3d: bool = False
    use_pca: bool = False
    include_body: bool = True


@dataclass
class himo_v4_2o(himo_base):
    """HIMO 2-object config for V4 entity-centric model.

    5 entities: body(201) + lhand(135) + rhand(135) + obj1(9) + obj2(9) = 489D
    """
    obj_mode: str = '2o'
    max_objects: int = 2
    num_frames: int = 300
    data_repr: str = 'processed_2o'
    mean_name: str = 'Mean_himo_2o.npy'
    std_name: str = 'Std_himo_2o.npy'
    proj_matrix_name: str = ''


@dataclass
class himo_v4_3o(himo_base):
    """HIMO 3-object config for V4 entity-centric model.

    6 entities: body(201) + lhand(135) + rhand(135) + obj1(9) + obj2(9) + obj3(9) = 498D
    """
    obj_mode: str = '3o'
    max_objects: int = 3
    num_frames: int = 300
    data_repr: str = 'processed_3o'
    mean_name: str = 'Mean_himo_3o.npy'
    std_name: str = 'Std_himo_3o.npy'
    proj_matrix_name: str = ''
