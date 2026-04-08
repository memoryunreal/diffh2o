#!/usr/bin/env python3
"""
Preprocess OakInk2 for flow-matching model: per-entity token state trajectories.

Produces per-sequence:
  - 398D motion features (uniform subsampled)
  - Token state trajectory JSON (Reach/Grasped/Interacting/Release/Static/Idle)
  - Per-frame integer state array for efficient training
  - Entity map (entity name → column index)

Uses 3-step hierarchical contact detection:
  1. Semantic coarse filter (skip frames outside annotation)
  2. Wrist-level bounding rejection (skip if wrist > 30cm from object)
  3. Fingertip spherical proxy (15 joints × object point cloud, with radius correction)

Usage:
    python -m prepare.preprocess_oakink2_flow --num_samples 5 --num_workers 1
    python -m prepare.preprocess_oakink2_flow --num_samples 0 --num_workers 32
"""

import os
import sys
import json
import pickle
import argparse
import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare.preprocess_oakink2 import (
    parse_interval,
    load_oakink2_sequence,
    ObjectProcessor,
    OakInk2Config,
    ANNO_PROCESSED_DIR,
)
from utils.rotation_conversions import quaternion_to_matrix, matrix_to_rotation_6d

# New 360D feature dimension (body 135D + hands_6d 198D + objects 27D)
FLOW_FEATURE_DIM = 360

# =============================================================================
# State Vocabulary
# =============================================================================

STATE_IDLE = 0           # Hand: no annotation
STATE_REACH = 1          # Hand: approaching target
STATE_GRASPED = 2        # Hand/Object: rigid attachment
STATE_GRASPED_INTER = 3  # Object: grasped + tool acting on it
STATE_RELEASE = 4        # Hand: releasing target
STATE_STATIC = 5         # Object: at rest
STATE_INTERACTING = 6    # Object: being acted upon

STATE_NAMES = {
    0: "[Idle]", 1: "[Reach]", 2: "[Grasped]", 3: "[Grasped_and_Interacting]",
    4: "[Release]", 5: "[Static]", 6: "[Interacting]",
}
STATE_PAD = -1  # Padding for unused entity columns

# =============================================================================
# Topology Types
# =============================================================================

TRIADIC = "triadic"
DYADIC_RIGID = "dyadic_rigid"
FLUID_CONTAINER = "fluid_container"
STATIC_HOLD = "static_hold"
TRANSPORT = "transport"
USE_DEVICE = "use_device"

# =============================================================================
# ACTION_RULE_BASE: Primitive → Topology
# =============================================================================

ACTION_RULE_BASE = {
    # TRIADIC: hand→tool→target
    "cut": TRIADIC, "scoop": TRIADIC, "stir": TRIADIC, "wipe": TRIADIC,
    "spread": TRIADIC, "brush": TRIADIC, "brush_whiteboard": TRIADIC,
    "write_on_paper": TRIADIC, "write_on_whiteboard": TRIADIC,
    "shear_paper": TRIADIC, "staple_paper_together": TRIADIC,
    "stir_experiment_substances": TRIADIC, "sharpen_pencil": TRIADIC,
    "scrape": TRIADIC, "stab": TRIADIC, "knock": TRIADIC,
    # DYADIC_RIGID: hand→object with constrained motion
    "screw": DYADIC_RIGID, "unscrew": DYADIC_RIGID,
    "screw_into": DYADIC_RIGID, "unscrew_from": DYADIC_RIGID,
    "cap": DYADIC_RIGID, "uncap": DYADIC_RIGID,
    "cap_the_pen": DYADIC_RIGID, "uncap_alcohol_lamp": DYADIC_RIGID,
    "remove_the_pen_cap": DYADIC_RIGID,
    "flip_close_tooth_paste_cap": DYADIC_RIGID, "flip_open_tooth_paste_cap": DYADIC_RIGID,
    "open": DYADIC_RIGID, "open_book": DYADIC_RIGID, "open_gate": DYADIC_RIGID,
    "open_laptop_lid": DYADIC_RIGID,
    "shut": DYADIC_RIGID, "close": DYADIC_RIGID, "close_book": DYADIC_RIGID,
    "close_gate": DYADIC_RIGID, "close_laptop_lid": DYADIC_RIGID,
    "press_button": DYADIC_RIGID, "trigger_lever": DYADIC_RIGID,
    "pull_out_drawer": DYADIC_RIGID, "push_in_drawer": DYADIC_RIGID,
    "insert_lightbulb": DYADIC_RIGID, "insert_pencil": DYADIC_RIGID,
    "insert_usb": DYADIC_RIGID, "plug_in_power_plug": DYADIC_RIGID,
    "remove_lightbulb": DYADIC_RIGID, "remove_pencil": DYADIC_RIGID,
    "remove_usb": DYADIC_RIGID, "remove_power_plug": DYADIC_RIGID,
    "remove_lid": DYADIC_RIGID, "put_on_lid": DYADIC_RIGID,
    "remove_test_tube": DYADIC_RIGID, "remove_from_test_tube_rack": DYADIC_RIGID,
    "remove_test_tube_from_rack_with_holder": DYADIC_RIGID,
    "place_on_test_tube_rack": DYADIC_RIGID,
    "place_test_tube_on_rack_with_holder": DYADIC_RIGID,
    "assemble": DYADIC_RIGID, "connect_to": DYADIC_RIGID, "deconnect_from": DYADIC_RIGID,
    "turn": DYADIC_RIGID, "tighten": DYADIC_RIGID, "loosen": DYADIC_RIGID,
    "put_flower_into_vase": DYADIC_RIGID,
    "ignite_alcohol_lamp": DYADIC_RIGID, "put_off_alcohol_lamp": DYADIC_RIGID,
    "ignite": DYADIC_RIGID,
    # FLUID_CONTAINER
    "pour": FLUID_CONTAINER, "pour_in_lab": FLUID_CONTAINER,
    "squeeze_tooth_paste": FLUID_CONTAINER, "squeeze_out": FLUID_CONTAINER,
    "shake": FLUID_CONTAINER, "shake_lab_container": FLUID_CONTAINER,
    "flow_out": FLUID_CONTAINER,
    # STATIC_HOLD
    "hold": STATIC_HOLD, "grip": STATIC_HOLD, "secure": STATIC_HOLD,
    "support": STATIC_HOLD, "be_held": STATIC_HOLD,
    "hold_test_tube": STATIC_HOLD,
    # TRANSPORT
    "place_onto": TRANSPORT, "place_inside": TRANSPORT,
    "place_asbestos_mesh": TRANSPORT, "rearrange": TRANSPORT,
    "take_outside": TRANSPORT, "swap": TRANSPORT,
    "heat_beaker": TRANSPORT, "heat_test_tube": TRANSPORT,
    # USE_DEVICE
    "use_gamecontroller": USE_DEVICE, "use_keyboard": USE_DEVICE, "use_mouse": USE_DEVICE,
}

DEFAULT_TOPOLOGY = DYADIC_RIGID

# SMPL-X hand joint indices
LH_WRIST = 20
RH_WRIST = 21
LH_JOINTS = list(range(25, 40))   # 15 joints
RH_JOINTS = list(range(40, 55))   # 15 joints
LH_TIPS = [27, 30, 33, 36, 39]    # fingertips
RH_TIPS = [42, 45, 48, 51, 54]

# Per-joint physical radius (meters)
JOINT_RADII = np.zeros(15, dtype=np.float32)
JOINT_RADII[:] = 0.010       # base/middle joints: 1.0cm
JOINT_RADII[2] = 0.008       # index tip
JOINT_RADII[5] = 0.008       # middle tip
JOINT_RADII[8] = 0.008       # ring/pinky tip
JOINT_RADII[11] = 0.008      # ring tip
JOINT_RADII[14] = 0.012      # thumb tip


# =============================================================================
# Hierarchical Contact Detector
# =============================================================================

class HierarchicalContactDetector:
    """3-step contact detection: semantic filter → wrist rejection → fingertip proxy."""

    def __init__(self, joints, obj_transf, obj_point_clouds,
                 wrist_threshold=0.30, contact_epsilon=0.003):
        """
        Args:
            joints: [T, 127, 3] SMPL-X joint positions
            obj_transf: {obj_id: {frame_id: 4x4 matrix}}
            obj_point_clouds: {obj_id: [N, 3] canonical points}
            wrist_threshold: Step 2 rejection distance (meters)
            contact_epsilon: Step 3 contact threshold (meters)
        """
        self.joints = joints
        self.obj_transf = obj_transf
        self.obj_point_clouds = obj_point_clouds
        self.wrist_thresh = wrist_threshold
        self.epsilon = contact_epsilon
        self._cache = {}

    def _get_obj_world_points(self, obj_id, frame_idx):
        """Transform canonical object points to world frame."""
        if obj_id not in self.obj_point_clouds:
            return None
        pts_canon = self.obj_point_clouds[obj_id]  # [N, 3]
        if obj_id not in self.obj_transf:
            return pts_canon
        transf_dict = self.obj_transf[obj_id]
        if frame_idx not in transf_dict:
            return pts_canon
        T44 = transf_dict[frame_idx]
        R = T44[:3, :3]
        t = T44[:3, 3]
        return (R @ pts_canon.T).T + t

    def _get_obj_center(self, obj_id, frame_idx):
        """Get object center from transform."""
        if obj_id not in self.obj_transf:
            return None
        transf_dict = self.obj_transf[obj_id]
        if frame_idx not in transf_dict:
            return None
        return transf_dict[frame_idx][:3, 3]

    def find_contact_boundaries(self, hand, obj_id, anno_start, anno_end, mocap_frames):
        """
        Find Reach/Contact/Release boundaries within an annotated interval.

        Returns: (reach_start, contact_start, contact_end, release_end)
        """
        cache_key = (hand, obj_id, anno_start, anno_end)
        if cache_key in self._cache:
            return self._cache[cache_key]

        wrist_idx = LH_WRIST if hand == "lh" else RH_WRIST
        finger_joints = LH_JOINTS if hand == "lh" else RH_JOINTS

        # Map annotation frame IDs to array indices
        frame_to_idx = {f: i for i, f in enumerate(mocap_frames)}

        # Find frame index range
        start_idx = frame_to_idx.get(anno_start)
        end_idx = frame_to_idx.get(anno_end)
        if start_idx is None or end_idx is None:
            # Fallback: find nearest
            start_idx = min(range(len(mocap_frames)), key=lambda i: abs(mocap_frames[i] - anno_start))
            end_idx = min(range(len(mocap_frames)), key=lambda i: abs(mocap_frames[i] - anno_end))

        if end_idx <= start_idx:
            result = (anno_start, anno_start, anno_end, anno_end)
            self._cache[cache_key] = result
            return result

        # Short interval: skip splitting
        if end_idx - start_idx < 10:
            result = (anno_start, anno_start, anno_end, anno_end)
            self._cache[cache_key] = result
            return result

        contact_start_idx = None
        contact_end_idx = None

        for fi in range(start_idx, end_idx + 1):
            if fi >= len(self.joints):
                break
            frame_id = mocap_frames[fi] if fi < len(mocap_frames) else fi

            # Step 2: Wrist bounding rejection
            wrist_pos = self.joints[fi, wrist_idx]
            obj_center = self._get_obj_center(obj_id, frame_id)
            if obj_center is None:
                continue
            d_wrist = np.linalg.norm(wrist_pos - obj_center)
            if d_wrist > self.wrist_thresh:
                continue

            # Step 3: Fingertip spherical proxy
            finger_pos = self.joints[fi, finger_joints]  # [15, 3]
            obj_pts = self._get_obj_world_points(obj_id, frame_id)
            if obj_pts is None:
                # Fallback: use wrist distance with tighter threshold
                if d_wrist < 0.15:
                    if contact_start_idx is None:
                        contact_start_idx = fi
                    contact_end_idx = fi
                continue

            # Compute min distance from any joint to any surface point, minus radius
            # finger_pos: [15, 3], obj_pts: [N, 3]
            dists = np.linalg.norm(finger_pos[:, None, :] - obj_pts[None, :, :], axis=2)  # [15, N]
            min_dists_per_joint = dists.min(axis=1)  # [15]
            d_min = (min_dists_per_joint - JOINT_RADII).min()

            if d_min <= self.epsilon:
                if contact_start_idx is None:
                    contact_start_idx = fi
                contact_end_idx = fi

        # Convert back to frame IDs
        if contact_start_idx is not None and contact_end_idx is not None:
            contact_start_frame = mocap_frames[contact_start_idx] if contact_start_idx < len(mocap_frames) else anno_start
            contact_end_frame = mocap_frames[contact_end_idx] if contact_end_idx < len(mocap_frames) else anno_end
        else:
            # No contact detected: entire interval as contact (annotation fallback)
            contact_start_frame = anno_start
            contact_end_frame = anno_end

        result = (anno_start, contact_start_frame, contact_end_frame, anno_end)
        self._cache[cache_key] = result
        return result


# =============================================================================
# State Assignment
# =============================================================================

def assign_hand_states(state_array, hand_col, reach_start, contact_start, contact_end, release_end,
                       frame_to_idx):
    """Write Reach/Grasped/Release states for a hand into the state array."""
    rs = frame_to_idx.get(reach_start, 0)
    cs = frame_to_idx.get(contact_start, rs)
    ce = frame_to_idx.get(contact_end, cs)
    re = frame_to_idx.get(release_end, ce)

    # Reach phase
    for fi in range(rs, min(cs, len(state_array))):
        if state_array[fi, hand_col] == STATE_IDLE:
            state_array[fi, hand_col] = STATE_REACH
    # Grasped phase
    for fi in range(cs, min(ce + 1, len(state_array))):
        state_array[fi, hand_col] = STATE_GRASPED
    # Release phase
    for fi in range(ce + 1, min(re + 1, len(state_array))):
        if state_array[fi, hand_col] == STATE_IDLE:
            state_array[fi, hand_col] = STATE_RELEASE


def assign_object_states(state_array, obj_col, contact_start, contact_end, topology,
                         frame_to_idx, is_tool=False):
    """Write object states based on topology type."""
    cs = frame_to_idx.get(contact_start, 0)
    ce = frame_to_idx.get(contact_end, cs)

    for fi in range(cs, min(ce + 1, len(state_array))):
        current = state_array[fi, obj_col]
        if is_tool or topology in (STATIC_HOLD, TRANSPORT, FLUID_CONTAINER):
            new_state = STATE_GRASPED
        elif topology in (DYADIC_RIGID, USE_DEVICE):
            new_state = STATE_INTERACTING
        elif topology == TRIADIC:
            new_state = STATE_INTERACTING
        else:
            new_state = STATE_GRASPED

        # Merge: Grasped + Interacting = Grasped_and_Interacting
        if current == STATE_GRASPED and new_state == STATE_INTERACTING:
            state_array[fi, obj_col] = STATE_GRASPED_INTER
        elif current == STATE_INTERACTING and new_state == STATE_GRASPED:
            state_array[fi, obj_col] = STATE_GRASPED_INTER
        elif current == STATE_STATIC or current == STATE_IDLE:
            state_array[fi, obj_col] = new_state
        # else: keep existing (don't downgrade)


# =============================================================================
# Run-Length Encode for JSON Output
# =============================================================================

def rle_encode_states(state_col, mocap_frames, entity_name, entity_map_inv):
    """Convert per-frame state array to list of {frames, state, ...} spans."""
    spans = []
    if len(state_col) == 0:
        return spans

    current_state = int(state_col[0])
    start_frame = mocap_frames[0] if len(mocap_frames) > 0 else 0
    for i in range(1, len(state_col)):
        s = int(state_col[i])
        if s != current_state:
            end_frame = mocap_frames[i - 1] if i - 1 < len(mocap_frames) else i - 1
            spans.append({"frames": [int(start_frame), int(end_frame)], "state": STATE_NAMES.get(current_state, f"[{current_state}]")})
            current_state = s
            start_frame = mocap_frames[i] if i < len(mocap_frames) else i
    # Last span
    end_frame = mocap_frames[len(state_col) - 1] if len(state_col) - 1 < len(mocap_frames) else len(state_col) - 1
    spans.append({"frames": [int(start_frame), int(end_frame)], "state": STATE_NAMES.get(current_state, f"[{current_state}]")})
    return spans


# =============================================================================
# SMPL-X Forward Pass
# =============================================================================

def run_smplx_forward(anno, mocap_frames, smplx_model, device='cpu'):
    """Run SMPL-X forward pass to get joint positions for all frames."""
    T = len(mocap_frames)
    raw_smplx = anno['raw_smplx']

    body_transl = np.zeros((T, 3), dtype=np.float32)
    body_orient_aa = np.zeros((T, 3), dtype=np.float32)
    body_pose_aa = np.zeros((T, 63), dtype=np.float32)
    lhand_aa = np.zeros((T, 45), dtype=np.float32)
    rhand_aa = np.zeros((T, 45), dtype=np.float32)

    from scipy.spatial.transform import Rotation as R

    def quat_wxyz_to_aa(q):
        q_scipy = np.concatenate([q[..., 1:], q[..., :1]], axis=-1)
        orig_shape = q.shape[:-1]
        r = R.from_quat(q_scipy.reshape(-1, 4))
        return r.as_rotvec().reshape(orig_shape + (3,))

    for i, fid in enumerate(mocap_frames):
        if fid not in raw_smplx:
            continue
        s = raw_smplx[fid]
        body_transl[i] = s['world_tsl'].squeeze().numpy()
        body_orient_aa[i] = quat_wxyz_to_aa(s['world_rot'].squeeze().numpy())
        body_pose_aa[i] = quat_wxyz_to_aa(s['body_pose'].squeeze().numpy()).reshape(-1)
        lhand_aa[i] = quat_wxyz_to_aa(s['left_hand_pose'].squeeze().numpy()).reshape(-1)
        rhand_aa[i] = quat_wxyz_to_aa(s['right_hand_pose'].squeeze().numpy()).reshape(-1)

    with torch.no_grad():
        out = smplx_model(
            transl=torch.tensor(body_transl, dtype=torch.float32, device=device),
            global_orient=torch.tensor(body_orient_aa, dtype=torch.float32, device=device),
            body_pose=torch.tensor(body_pose_aa, dtype=torch.float32, device=device),
            left_hand_pose=torch.tensor(lhand_aa, dtype=torch.float32, device=device),
            right_hand_pose=torch.tensor(rhand_aa, dtype=torch.float32, device=device),
        )

    return out.joints.cpu().numpy()  # [T, 127, 3]


# =============================================================================
# Object Point Cloud Sampling
# =============================================================================

def sample_object_point_clouds(obj_list, obj_mesh_dir, n_points=1000):
    """Pre-sample surface points from object meshes."""
    clouds = {}
    for obj_id in obj_list:
        mesh_path = os.path.join(obj_mesh_dir, obj_id, "model.obj")
        if not os.path.exists(mesh_path):
            continue
        try:
            mesh = trimesh.load_mesh(mesh_path, process=False, force="mesh")
            pts = mesh.sample(n_points).astype(np.float32)
            clouds[obj_id] = pts
        except Exception:
            pass
    return clouds


# =============================================================================
# 360D Feature Extraction (all rotations as 6D)
# =============================================================================

def quat_wxyz_to_6d(quat):
    """Convert quaternion [w,x,y,z] to 6D rotation representation.

    Args:
        quat: (..., 4) quaternion in [w,x,y,z] format
    Returns:
        (..., 6) 6D rotation
    """
    q_t = torch.tensor(quat, dtype=torch.float32)
    mat = quaternion_to_matrix(q_t)  # (..., 3, 3)
    rot6d = matrix_to_rotation_6d(mat)  # (..., 6)
    return rot6d.numpy()


def mat3x3_to_6d(mat):
    """Convert 3x3 rotation matrix to 6D representation."""
    mat_t = torch.tensor(mat, dtype=torch.float32)
    if mat_t.dim() == 2:
        mat_t = mat_t.unsqueeze(0)
    rot6d = matrix_to_rotation_6d(mat_t)
    return rot6d.squeeze(0).numpy()


def extract_360d_features(anno, mocap_frames, max_objects=3):
    """Extract 360D feature vector from raw annotations.

    Layout:
      [0:3]     body_transl (3D)
      [3:9]     body_orient (6D)
      [9:135]   body_pose (21 joints × 6D = 126D)
      [135:138] lhand_wrist_pos (3D)
      [138:144] lhand_wrist_orient (6D)
      [144:234] lhand_joints_6d (15 joints × 6D = 90D)
      [234:237] rhand_wrist_pos (3D)
      [237:243] rhand_wrist_orient (6D)
      [243:333] rhand_joints_6d (15 joints × 6D = 90D)
      [333:360] objects (3 × 9D = 27D)

    Returns:
        features: [T, 360] float32
    """
    T = len(mocap_frames)
    features = np.zeros((T, FLOW_FEATURE_DIM), dtype=np.float32)

    raw_smplx = anno['raw_smplx']
    raw_mano = anno.get('raw_mano', {})
    obj_transf = anno.get('obj_transf', {})
    obj_list = anno.get('obj_list', [])[:max_objects]

    for i, fid in enumerate(mocap_frames):
        # --- Body from raw_smplx ---
        smplx = raw_smplx.get(fid)
        if smplx is None:
            continue

        # Body translation (3D)
        features[i, 0:3] = smplx['world_tsl'].squeeze().numpy()

        # Body orient: quat [w,x,y,z] → 6D
        world_rot = smplx['world_rot'].squeeze().numpy()  # (4,)
        features[i, 3:9] = quat_wxyz_to_6d(world_rot)

        # Body pose: 21 joints × quat → 21 × 6D = 126D
        body_pose = smplx['body_pose'].squeeze().numpy()  # (21, 4)
        body_6d = quat_wxyz_to_6d(body_pose)  # (21, 6)
        features[i, 9:135] = body_6d.reshape(-1)

        # --- Left hand ---
        mano = raw_mano.get(fid, {})

        # Wrist position from MANO translation
        if 'lh__tsl' in mano:
            features[i, 135:138] = mano['lh__tsl'].squeeze().numpy()

        # Wrist orient: use first joint of MANO (joint 0 = wrist)
        if 'lh__pose_coeffs' in mano:
            lh_quat = mano['lh__pose_coeffs'].squeeze().numpy()  # (16, 4)
            features[i, 138:144] = quat_wxyz_to_6d(lh_quat[0])  # wrist joint

        # Hand joints: from SMPL-X left_hand_pose (15 joints × quat → 15 × 6D)
        lh_pose = smplx['left_hand_pose'].squeeze().numpy()  # (15, 4)
        lh_6d = quat_wxyz_to_6d(lh_pose)  # (15, 6)
        features[i, 144:234] = lh_6d.reshape(-1)

        # --- Right hand ---
        if 'rh__tsl' in mano:
            features[i, 234:237] = mano['rh__tsl'].squeeze().numpy()

        if 'rh__pose_coeffs' in mano:
            rh_quat = mano['rh__pose_coeffs'].squeeze().numpy()  # (16, 4)
            features[i, 237:243] = quat_wxyz_to_6d(rh_quat[0])  # wrist joint

        rh_pose = smplx['right_hand_pose'].squeeze().numpy()  # (15, 4)
        rh_6d = quat_wxyz_to_6d(rh_pose)  # (15, 6)
        features[i, 243:333] = rh_6d.reshape(-1)

        # --- Objects ---
        for oi, oid in enumerate(obj_list):
            if oid in obj_transf and fid in obj_transf[oid]:
                tf = obj_transf[oid][fid]  # 4x4
                base = 333 + oi * 9
                features[i, base:base+3] = tf[:3, 3]  # position
                features[i, base+3:base+9] = mat3x3_to_6d(tf[:3, :3])  # rotation 6D

    return features


# =============================================================================
# Worker Function
# =============================================================================

_worker_ctx = {}


def _process_one_sequence_flow(pkl_file):
    """Process one sequence → dict with all outputs."""
    ctx = _worker_ctx
    seq_key = pkl_file.replace('.pkl', '')

    anno_dir = ctx['anno_dir']
    program_info_dir = ctx['program_info_dir']
    desc_info_dir = ctx['desc_info_dir']
    config = ctx['config']
    obj_mesh_dir = ctx['obj_mesh_dir']
    smplx_path = ctx['smplx_path']
    max_entities = ctx['max_entities']

    pkl_path = os.path.join(anno_dir, pkl_file)
    program_path = os.path.join(program_info_dir, f"{seq_key}.json")
    desc_path = os.path.join(desc_info_dir, f"{seq_key}.json")

    if not os.path.exists(program_path) or not os.path.exists(desc_path):
        return None

    try:
        with open(pkl_path, 'rb') as f:
            anno = pickle.load(f)
        with open(program_path, 'r') as f:
            program_info = json.load(f)
        with open(desc_path, 'r') as f:
            desc_info = json.load(f)
    except Exception as e:
        print(f"Error loading {seq_key}: {e}")
        return None

    mocap_frames = anno['mocap_frame_id_list']
    T = len(mocap_frames)
    obj_list = anno.get('obj_list', [])
    frame_to_idx = {f: i for i, f in enumerate(mocap_frames)}

    # Build entity list: lh, rh, then objects
    entities = ["lh", "rh"] + [f"obj_{oid}" for oid in obj_list]
    entity_to_col = {e: i for i, e in enumerate(entities)}
    E = min(len(entities), max_entities)

    # State array: [T, max_entities], init hands=IDLE, objects=STATIC
    state_array = np.full((T, max_entities), STATE_PAD, dtype=np.int8)
    state_array[:, 0] = STATE_IDLE  # lh
    state_array[:, 1] = STATE_IDLE  # rh
    for i in range(2, E):
        state_array[:, i] = STATE_STATIC

    # --- SMPL-X forward pass ---
    try:
        import smplx as smplx_lib
        smplx_model = smplx_lib.create(
            smplx_path, model_type='smplx', gender='neutral',
            use_pca=False, flat_hand_mean=True, batch_size=T, num_betas=10,
        )
        joints = run_smplx_forward(anno, mocap_frames, smplx_model)
    except Exception as e:
        print(f"  SMPL-X failed for {seq_key}: {e}")
        joints = np.zeros((T, 127, 3), dtype=np.float32)

    # --- Object point clouds ---
    obj_clouds = sample_object_point_clouds(obj_list, obj_mesh_dir)

    # --- Contact detector ---
    detector = HierarchicalContactDetector(
        joints, anno.get('obj_transf', {}), obj_clouds,
        wrist_threshold=0.30, contact_epsilon=ctx['contact_epsilon'],
    )

    # --- Process each primitive ---
    for interval_key, prim_data in program_info.items():
        try:
            lh_interval, rh_interval = parse_interval(interval_key)
        except Exception:
            continue

        primitive = prim_data.get('primitive', '')
        topology = ACTION_RULE_BASE.get(primitive, DEFAULT_TOPOLOGY)
        obj_list_prim = prim_data.get('obj_list', [])
        obj_list_rh = prim_data.get('obj_list_rh', [])
        obj_list_lh = prim_data.get('obj_list_lh', [])

        # Right hand
        if rh_interval is not None:
            rh_start, rh_end = rh_interval
            rh_objs = obj_list_rh or obj_list_prim
            for obj_id in rh_objs[:1]:  # primary object
                boundaries = detector.find_contact_boundaries("rh", obj_id, rh_start, rh_end, mocap_frames)
                reach_s, contact_s, contact_e, release_e = boundaries
                assign_hand_states(state_array, 1, reach_s, contact_s, contact_e, release_e, frame_to_idx)

                obj_key = f"obj_{obj_id}"
                if obj_key in entity_to_col and entity_to_col[obj_key] < max_entities:
                    is_tool = (topology == TRIADIC)
                    assign_object_states(state_array, entity_to_col[obj_key],
                                         contact_s, contact_e, topology if not is_tool else STATIC_HOLD,
                                         frame_to_idx, is_tool=is_tool)

            # TRIADIC: secondary object (target of tool)
            if topology == TRIADIC and len(rh_objs) > 1:
                target_id = rh_objs[1]
                target_key = f"obj_{target_id}"
                if target_key in entity_to_col and entity_to_col[target_key] < max_entities:
                    assign_object_states(state_array, entity_to_col[target_key],
                                         contact_s, contact_e, TRIADIC, frame_to_idx)

        # Left hand
        if lh_interval is not None:
            lh_start, lh_end = lh_interval
            lh_objs = obj_list_lh or obj_list_prim
            for obj_id in lh_objs[:1]:
                boundaries = detector.find_contact_boundaries("lh", obj_id, lh_start, lh_end, mocap_frames)
                reach_s, contact_s, contact_e, release_e = boundaries
                assign_hand_states(state_array, 0, reach_s, contact_s, contact_e, release_e, frame_to_idx)

                obj_key = f"obj_{obj_id}"
                if obj_key in entity_to_col and entity_to_col[obj_key] < max_entities:
                    assign_object_states(state_array, entity_to_col[obj_key],
                                         contact_s, contact_e, STATIC_HOLD, frame_to_idx, is_tool=False)

    # --- Extract features based on mode ---
    feature_mode = ctx.get('feature_mode', 'full')
    try:
        if feature_mode == 'hand_pca':
            # MANO PCA hands only: pos(3) + orient(6) + pca(21) per hand = 60D
            from prepare.preprocess_oakink2 import MANOPCAConverter
            local_mano = MANOPCAConverter(mano_path=ctx.get('mano_path'))
            T_feat = len(mocap_frames)
            features = np.zeros((T_feat, 60), dtype=np.float32)
            raw_mano = anno.get('raw_mano', {})
            for i, fid in enumerate(mocap_frames):
                m = raw_mano.get(fid, {})
                if 'lh__tsl' in m:
                    features[i, 0:3] = m['lh__tsl'].squeeze().numpy()
                if 'lh__pose_coeffs' in m:
                    lh_q = m['lh__pose_coeffs'].squeeze()  # [16, 4]
                    features[i, 3:9] = quat_wxyz_to_6d(lh_q[0].numpy())  # wrist orient
                    pca = local_mano.quaternion_to_pca(lh_q.unsqueeze(0))  # [1, 21]
                    features[i, 9:30] = pca.squeeze().numpy()
                if 'rh__tsl' in m:
                    features[i, 30:33] = m['rh__tsl'].squeeze().numpy()
                if 'rh__pose_coeffs' in m:
                    rh_q = m['rh__pose_coeffs'].squeeze()
                    features[i, 33:39] = quat_wxyz_to_6d(rh_q[0].numpy())
                    pca = local_mano.quaternion_to_pca(rh_q.unsqueeze(0))
                    features[i, 39:60] = pca.squeeze().numpy()
        elif feature_mode == 'markers':
            # 77-marker positions from SMPL-X vertices + object poses = 258D
            # Run SMPL-X forward pass to get vertices, then select 77 markers
            MARKERSET_SMPLX = [5920, 5621, 5882, 3486, 4430, 3258, 2029, 5694, 5645, 9117, 4302,
                               4319, 707, 4788, 4198, 3998, 8846, 4726, 3682, 3688, 5890, 5901,
                               3961, 4722, 5698, 5868, 3892, 3712, 4479, 4088, 3477, 4839, 5787,
                               4291, 8576, 6248, 7166, 6021, 3067, 8250, 8388, 8339, 3116, 7040,
                               7099, 2198, 7524, 7115, 6265, 6746, 8634, 7462, 6443, 6449, 8584,
                               8595, 6709, 7458, 8392, 8527, 7215, 6832, 7575, 6503, 8481, 5945,
                               5947, 8079, 7669, 7794, 7905, 8022, 5361, 4933, 5058, 5169, 5286]

            import smplx as smplx_lib
            from scipy.spatial.transform import Rotation as R_scipy

            T_feat = len(mocap_frames)
            raw_smplx_data = anno['raw_smplx']

            def quat_to_aa_batch(quats):
                """Convert [N, 4] wxyz quaternions to [N, 3] axis-angle."""
                q_scipy = np.concatenate([quats[..., 1:], quats[..., :1]], axis=-1)
                return R_scipy.from_quat(q_scipy.reshape(-1, 4)).as_rotvec().reshape(quats.shape[:-1] + (3,))

            # Collect SMPL-X params
            body_transl = np.zeros((T_feat, 3), dtype=np.float32)
            body_orient_aa = np.zeros((T_feat, 3), dtype=np.float32)
            body_pose_aa = np.zeros((T_feat, 63), dtype=np.float32)
            lhand_aa = np.zeros((T_feat, 45), dtype=np.float32)
            rhand_aa = np.zeros((T_feat, 45), dtype=np.float32)
            body_shape = np.zeros((T_feat, 300), dtype=np.float32)

            for i, fid in enumerate(mocap_frames):
                s = raw_smplx_data.get(fid)
                if s is None:
                    continue
                body_transl[i] = s['world_tsl'].squeeze().numpy()
                body_orient_aa[i] = quat_to_aa_batch(s['world_rot'].squeeze().numpy()[None])[0]
                body_pose_aa[i] = quat_to_aa_batch(s['body_pose'].squeeze().numpy()).reshape(-1)
                lhand_aa[i] = quat_to_aa_batch(s['left_hand_pose'].squeeze().numpy()).reshape(-1)
                rhand_aa[i] = quat_to_aa_batch(s['right_hand_pose'].squeeze().numpy()).reshape(-1)
                body_shape[i] = s['body_shape'].squeeze().numpy()

            # SMPL-X forward pass in chunks to get vertices
            smplx_device = torch.device('cpu')  # CPU to avoid GPU memory issues in multiprocessing
            CHUNK = 100
            all_markers = []
            for cs in range(0, T_feat, CHUNK):
                ce = min(cs + CHUNK, T_feat)
                cT = ce - cs
                smplx_model = smplx_lib.create(
                    ctx['smplx_path'], model_type='smplx', gender='neutral',
                    use_pca=False, flat_hand_mean=True, batch_size=cT, num_betas=300,
                ).to(smplx_device)
                with torch.no_grad():
                    out = smplx_model(
                        betas=torch.tensor(body_shape[cs:ce], device=smplx_device),
                        transl=torch.tensor(body_transl[cs:ce], device=smplx_device),
                        global_orient=torch.tensor(body_orient_aa[cs:ce], device=smplx_device),
                        body_pose=torch.tensor(body_pose_aa[cs:ce], device=smplx_device),
                        left_hand_pose=torch.tensor(lhand_aa[cs:ce], device=smplx_device),
                        right_hand_pose=torch.tensor(rhand_aa[cs:ce], device=smplx_device),
                    )
                    verts = out.vertices.cpu().numpy()  # [cT, 10475, 3]
                    markers = verts[:, MARKERSET_SMPLX, :]  # [cT, 77, 3]
                    all_markers.append(markers)
                del smplx_model

            all_markers = np.concatenate(all_markers, axis=0)  # [T_feat, 77, 3]
            marker_flat = all_markers.reshape(T_feat, 231)  # [T_feat, 231]

            # Extract object transforms (sorted alphabetically for consistent ordering)
            obj_transf = anno.get('obj_transf', {})
            obj_list_local = sorted(anno.get('obj_list', []))[:3]
            obj_features = np.zeros((T_feat, 27), dtype=np.float32)  # 3 objects × 9D
            for oi, oid in enumerate(obj_list_local):
                if oid in obj_transf:
                    for i, fid in enumerate(mocap_frames):
                        if fid in obj_transf[oid]:
                            tf = obj_transf[oid][fid]
                            base = oi * 9
                            obj_features[i, base:base+3] = tf[:3, 3]
                            obj_features[i, base+3:base+9] = mat3x3_to_6d(tf[:3, :3])

            features = np.concatenate([marker_flat, obj_features], axis=1)  # [T_feat, 258]

        else:
            features_full = extract_360d_features(anno, mocap_frames, max_objects=3)
            if feature_mode == 'body_only':
                features = features_full[:, :135].copy()
            elif feature_mode == 'body_hands':
                body = features_full[:, :135]
                lhand = features_full[:, 144:234]
                rhand = features_full[:, 243:333]
                features = np.concatenate([body, lhand, rhand], axis=1)  # 315D
            else:
                features = features_full  # full 360D
    except Exception as e:
        print(f"  Feature extraction failed for {seq_key}: {e}")
        dim = {'body_only': 135, 'body_hands': 315, 'hand_pca': 60, 'markers': 258}.get(feature_mode, FLOW_FEATURE_DIM)
        features = np.zeros((T, dim), dtype=np.float32)

    # --- Uniform subsample ---
    max_frames = config.max_motion_len if config.max_motion_len > 0 else 200
    T_feat = features.shape[0]
    T_state = state_array.shape[0]

    # Subsample features
    if T_feat > max_frames:
        step = T_feat / max_frames
        feat_indices = [int(i * step) for i in range(max_frames)]
        features = features[feat_indices]
    # Subsample states (use same logic but based on T_state)
    if T_state > max_frames:
        step = T_state / max_frames
        state_indices = [int(i * step) for i in range(max_frames)]
        state_array = state_array[state_indices]
        subsampled_frames = [mocap_frames[i] for i in state_indices]
    else:
        subsampled_frames = list(mocap_frames)

    # --- Build JSON ---
    entity_map = {e: i for i, e in enumerate(entities[:max_entities])}
    traj_json = {
        "metadata": {"sequence_id": seq_key, "total_frames": T, "subsampled_frames": len(subsampled_frames)},
        "entities": entities[:max_entities],
        "token_state_trajectories": {},
    }
    for ename, col in entity_map.items():
        traj_json["token_state_trajectories"][ename] = rle_encode_states(
            state_array[:, col], subsampled_frames, ename, entity_map
        )

    # --- Text descriptions ---
    text_parts = []
    for k, v in desc_info.items():
        if 'seg_desc' in v:
            text_parts.append(v['seg_desc'])
    text = " ".join(text_parts) if text_parts else seq_key

    return {
        'seq_id': seq_key,
        'features': features,
        'state_array': state_array,
        'trajectory_json': traj_json,
        'entity_map': entity_map,
        'text': text,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocess OakInk2 for flow-matching model")
    parser.add_argument("--oakink2_root", default="/hhd4/lizhe/dataset/OakInk2/data")
    parser.add_argument("--output_root", default="dataset/OAKINK2_FLOW")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument("--contact_epsilon", type=float, default=0.003)
    parser.add_argument("--max_entities", type=int, default=20)
    parser.add_argument("--mano_path", type=str, default=None)
    parser.add_argument("--smplx_path", default="/hhd4/lizhe/code/InterAct/models")
    parser.add_argument("--feature_mode", default="full",
                        choices=["full", "body_only", "body_hands", "hand_pca", "markers"],
                        help="Feature extraction mode: full(360D), body_only(135D), body_hands(315D), hand_pca(60D), markers(258D)")
    args = parser.parse_args()

    config = OakInk2Config(
        oakink2_root=args.oakink2_root,
        output_root=args.output_root,
        mode='complex',  # full sequences
        max_motion_len=args.max_frames,
    )

    # Create output directories
    for subdir in ['motion', 'state_trajectories', 'state_arrays', 'entity_maps', 'texts']:
        os.makedirs(os.path.join(args.output_root, subdir), exist_ok=True)

    print(f"\n=== OakInk2 Flow Preprocessing ===")
    print(f"Output: {args.output_root}")
    print(f"Max frames: {args.max_frames}")
    print(f"Contact epsilon: {args.contact_epsilon}m")
    print(f"Max entities: {args.max_entities}")

    # Find sequences
    anno_dir = os.path.join(args.oakink2_root, "anno_preview")
    program_info_dir = os.path.join(args.oakink2_root, "program", "program_info")
    desc_info_dir = os.path.join(args.oakink2_root, "program", "desc_info")

    pkl_files = sorted([f for f in os.listdir(anno_dir) if f.endswith('.pkl')])
    if args.num_samples > 0:
        pkl_files = pkl_files[:args.num_samples]
    print(f"Processing {len(pkl_files)} sequences...")

    # Set worker context
    global _worker_ctx
    _worker_ctx = {
        'anno_dir': anno_dir,
        'program_info_dir': program_info_dir,
        'desc_info_dir': desc_info_dir,
        'config': config,
        'obj_mesh_dir': os.path.join(args.oakink2_root, "object_repair", "align_ds"),
        'smplx_path': args.smplx_path,
        'contact_epsilon': args.contact_epsilon,
        'max_entities': args.max_entities,
        'mano_path': args.mano_path,
        'feature_mode': args.feature_mode,
    }

    # Process
    all_results = []
    if args.num_workers > 1:
        print(f"Using {args.num_workers} workers...")
        with multiprocessing.Pool(args.num_workers) as pool:
            for r in tqdm(pool.imap_unordered(_process_one_sequence_flow, pkl_files),
                          total=len(pkl_files), desc="Processing"):
                if r is not None:
                    all_results.append(r)
    else:
        for pkl_file in tqdm(pkl_files, desc="Processing"):
            r = _process_one_sequence_flow(pkl_file)
            if r is not None:
                all_results.append(r)

    print(f"\nProcessed {len(all_results)} sequences successfully")

    # Save outputs
    all_features = []
    names = []
    for idx, result in enumerate(all_results):
        np.save(os.path.join(args.output_root, 'motion', f'{idx:06d}.npy'), result['features'])
        np.save(os.path.join(args.output_root, 'state_arrays', f'{idx:06d}.npy'), result['state_array'])

        with open(os.path.join(args.output_root, 'state_trajectories', f'{idx:06d}.json'), 'w') as f:
            json.dump(result['trajectory_json'], f, indent=2)

        with open(os.path.join(args.output_root, 'entity_maps', f'{idx:06d}.json'), 'w') as f:
            json.dump(result['entity_map'], f)

        with open(os.path.join(args.output_root, 'texts', f'{idx:06d}.txt'), 'w') as f:
            f.write(f"{result['text']}##0.0#0.0\n")

        all_features.append(result['features'])
        names.append(result['seq_id'])

    # File names
    with open(os.path.join(args.output_root, 'file_names.txt'), 'w') as f:
        for idx, name in enumerate(names):
            f.write(f"{idx:06d}\t{name}\n")

    # Normalization stats
    if all_features:
        all_feats = np.concatenate(all_features, axis=0)
        mean = all_feats.mean(axis=0)
        std = all_feats.std(axis=0)
        std[std < 1e-8] = 1.0
        np.save(os.path.join(args.output_root, 'Mean_flow.npy'), mean)
        np.save(os.path.join(args.output_root, 'Std_flow.npy'), std)
        print(f"Normalization: mean shape {mean.shape}, std shape {std.shape}")

    # Train/test split (80/20)
    n = len(names)
    n_train = int(n * 0.8)
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = sorted(indices[:n_train])
    test_idx = sorted(indices[n_train:])

    with open(os.path.join(args.output_root, 'train_flow.txt'), 'w') as f:
        for i in train_idx:
            f.write(f"{i:06d}\n")
    with open(os.path.join(args.output_root, 'test_flow.txt'), 'w') as f:
        for i in test_idx:
            f.write(f"{i:06d}\n")

    # Metadata
    metadata = {
        "state_vocab": STATE_NAMES,
        "total_sequences": len(all_results),
        "train_count": len(train_idx),
        "test_count": len(test_idx),
        "max_entities": args.max_entities,
        "max_frames": args.max_frames,
        "contact_epsilon": args.contact_epsilon,
        "feature_dim": FLOW_FEATURE_DIM,
    }
    with open(os.path.join(args.output_root, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Output at {args.output_root}")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")


if __name__ == "__main__":
    main()
