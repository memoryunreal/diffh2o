# Copyright (c) Meta Platforms, Inc. and affiliates.

from utils.parser_util import *
from dataclasses import dataclass
from configs import data, model

## Grasping
# Only hand motion prediction
@dataclass
class diffh2o_grasp(
        data.diffh2o_grasp,
        model.traj_unet_adagn_swx,
):
    save_dir: str = 'save/diffh2o_grasp'

## Interaction
# Coupled hand and object motion
@dataclass
class diffh2o_interaction(
        data.diffh2o_interaction,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/diffh2o_interaction'

### Full model simple texts (Grasp + Interaction)
@dataclass
class diffh2o_full(
        data.diffh2o_full,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/diffh2o_full'

### Full model detailed texts (Grasp + Interaction)
@dataclass
class diffh2o_full_detailed(
        data.diffh2o_full_detailed,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/diffh2o_full_detailed'

###  Baseline
@dataclass
class mdm_full(
        data.mdm_full,
        model.motion_mdm,
):
    save_dir: str = 'save/mdm_full'


# ============================================
# OakInk2 Model Cards
# ============================================

### OakInk2 Full model - Primitive mode (body + hands + objects)
@dataclass
class oakink2_full(
        data.oakink2_full,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/oakink2_full'
    # Note: OakInk2 uses 398D features (vs GRAB's 117D)


### OakInk2 Complex model - Long sequences for long motion generation
@dataclass
class oakink2_complex(
        data.oakink2_complex,
        model.motion_unet_adagn_xl,
):
    save_dir: str = 'save/oakink2_complex'
    # For long motion generation (sequences up to 20 minutes)


# ============================================
# Transition Model Cards
# ============================================

### OakInk2 Transition model (398D, real transitions)
@dataclass
class oakink2_transition(
        data.oakink2_transition,
        model.transition_transformer,
):
    save_dir: str = 'save/oakink2_transition'


### GRAB Transition model (117D, self-supervised infilling)
@dataclass
class grab_transition(
        data.grab_transition,
        model.transition_transformer,
):
    save_dir: str = 'save/grab_transition'


# ============================================
# HIMO Model Cards (Track B: joint-based, V4 entity-centric)
# ============================================

### HIMO 2-object V4 model (489D, position-only, 5 entities)
@dataclass
class himo_v4_2o(
        data.himo_v4_2o,
):
    save_dir: str = 'save/himo_v4_2o'
    # V4 entity dims: body=201, lhand=135, rhand=135, obj=9, bps=1024
    # 5 entities (body + lh + rh + 2 objects), total dynamic = 489D
    # Output matches HIMO evaluator format exactly


### HIMO 3-object V4 model (498D, position-only, 6 entities)
@dataclass
class himo_v4_3o(
        data.himo_v4_3o,
):
    save_dir: str = 'save/himo_v4_3o'
    # V4 entity dims: body=201, lhand=135, rhand=135, obj=9, bps=1024
    # 6 entities (body + lh + rh + 3 objects), total dynamic = 498D
    # Output matches HIMO evaluator format exactly
