#!/usr/bin/env python3
"""
CRR-Flow V3 Preprocessing: Object-Centric Marker Data Pipeline

Processes OakInk2 complex sequences into super-segments with:
- 77 SMPL-X markers (231D) + 4 object slots (36D) = 267D per frame
- Per-entity state labels [T, 7] (body + 2 hands + 4 objects)
- 10FPS downsampling with boundary expansion
- Schmitt trigger contact detection with majority vote smoothing

Usage:
    python -m prepare.preprocess_v3 --num_samples 0 --num_workers 8
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.rotation_conversions import quaternion_to_matrix, matrix_to_rotation_6d

# =============================================================================
# Constants
# =============================================================================

MARKERSET_SMPLX = [
    5920, 5621, 5882, 3486, 4430, 3258, 2029, 5694, 5645, 9117, 4302,
    4319, 707, 4788, 4198, 3998, 8846, 4726, 3682, 3688, 5890, 5901,
    3961, 4722, 5698, 5868, 3892, 3712, 4479, 4088, 3477, 4839, 5787,
    4291, 8576, 6248, 7166, 6021, 3067, 8250, 8388, 8339, 3116, 7040,
    7099, 2198, 7524, 7115, 6265, 6746, 8634, 7462, 6443, 6449, 8584,
    8595, 6709, 7458, 8392, 8527, 7215, 6832, 7575, 6503, 8481, 5945,
    5947, 8079, 7669, 7794, 7905, 8022, 5361, 4933, 5058, 5169, 5286,
]

# Body part marker indices within the 77-marker set
# SSM67: head(6) + torso/spine(19) + left_foot(7) + right_foot(7) + left_toes(5) + right_toes(6) = 50 body
# + left_hand/wrist(9) + right_hand/wrist(8) = 17 hand SSM
# + 5 left fingertips + 5 right fingertips = 10 fingertips
# Total: 50 body + 14 left_hand + 13 right_hand = 77

# Indices within the 77-marker array
# Body markers: indices 0-49 (head, torso, spine, legs, feet, toes)
# These are the SSM67 markers MINUS the hand/wrist ones
BODY_MARKER_INDICES = list(range(0, 50))  # 50 markers
LHAND_MARKER_INDICES = list(range(50, 59)) + list(range(67, 72))  # 9 SSM + 5 fingertip = 14
RHAND_MARKER_INDICES = list(range(59, 67)) + list(range(72, 77))  # 8 SSM + 5 fingertip = 13

# State vocabulary
STATE_STATIC = 0
STATE_REACH = 1
STATE_GRASPED = 2
STATE_INTERACTING = 3
STATE_RELEASE = 4
STATE_PADDING = 5

MAX_OBJECTS = 4
NUM_ENTITIES = 7  # body + 2 hands + 4 objects
FEATURE_DIM = 267  # 231 markers + 36 objects (4×9)

# Action rule base (simplified from preprocess_flow_v2.md)
TRIADIC_ACTIONS = {
    "cut", "scoop", "scrape", "stir", "spread", "wipe", "brush",
    "write/draw", "shear", "staple together", "stab", "knock",
}
DYADIC_ACTIONS = {
    "screw into", "unscrew from", "cap onto", "uncap from",
    "open", "shut", "close", "connect to", "deconnect from",
    "turn", "tighten", "loosen",
}
FLUID_ACTIONS = {"pour", "flow out", "squeeze out", "shake"}
STATIC_ACTIONS = {"hold", "grip", "secure", "support", "be held"}

# Contact detection thresholds (Schmitt trigger)
CONTACT_TRIGGER_CM = 1.5  # < 1.5cm → contact
CONTACT_RELEASE_CM = 3.0  # > 3.0cm → release
MAJORITY_VOTE_WINDOW = 5


# =============================================================================
# Phase 1: Super-Segment Extraction
# =============================================================================

def parse_interval(interval_str):
    """Parse program_info key '((lh_start, lh_end), (rh_start, rh_end))' or None."""
    import ast
    try:
        parsed = ast.literal_eval(interval_str)
        lh = parsed[0] if parsed[0] is not None else None
        rh = parsed[1] if parsed[1] is not None else None
        return lh, rh
    except Exception:
        return None, None


def extract_super_segments(program_info, total_frames, gap_threshold=300):
    """
    Merge adjacent primitives with gap <= threshold into super-segments.

    Returns:
        list of dicts: [{
            'start': int, 'end': int,
            'primitives': [prim_data_list],
            'intervals': [(start, end), ...]
        }]
    """
    # Collect all primitive intervals with their data
    all_intervals = []
    for key, prim_data in program_info.items():
        lh_interval, rh_interval = parse_interval(key)
        # Use the union of lh and rh intervals
        starts, ends = [], []
        if lh_interval:
            starts.append(lh_interval[0])
            ends.append(lh_interval[1])
        if rh_interval:
            starts.append(rh_interval[0])
            ends.append(rh_interval[1])
        if not starts:
            continue
        prim_start = min(starts)
        prim_end = max(ends)
        all_intervals.append({
            'start': prim_start, 'end': prim_end,
            'data': prim_data, 'key': key,
            'lh_interval': lh_interval, 'rh_interval': rh_interval,
        })

    if not all_intervals:
        return []

    # Sort by start frame
    all_intervals.sort(key=lambda x: x['start'])

    # Merge adjacent primitives with small gaps
    segments = []
    current_seg = {
        'start': all_intervals[0]['start'],
        'end': all_intervals[0]['end'],
        'primitives': [all_intervals[0]],
    }

    for prim in all_intervals[1:]:
        gap = prim['start'] - current_seg['end']
        if gap <= gap_threshold:
            # Merge into current segment
            current_seg['end'] = max(current_seg['end'], prim['end'])
            current_seg['primitives'].append(prim)
        else:
            # Save current and start new
            segments.append(current_seg)
            current_seg = {
                'start': prim['start'],
                'end': prim['end'],
                'primitives': [prim],
            }
    segments.append(current_seg)

    # Clamp to valid frame range
    for seg in segments:
        seg['start'] = max(0, seg['start'])
        seg['end'] = min(total_frames - 1, seg['end'])

    return segments


# =============================================================================
# Phase 2: Feature Extraction
# =============================================================================

def quat_wxyz_to_aa_batch(quats):
    """Convert [N, 4] wxyz quaternions to [N, 3] axis-angle."""
    from scipy.spatial.transform import Rotation as R
    q_scipy = np.concatenate([quats[..., 1:], quats[..., :1]], axis=-1)
    return R.from_quat(q_scipy.reshape(-1, 4)).as_rotvec().reshape(quats.shape[:-1] + (3,)).astype(np.float32)


def mat3x3_to_6d(mat):
    """Convert 3x3 rotation matrix to 6D representation."""
    mat_t = torch.tensor(mat, dtype=torch.float32)
    if mat_t.dim() == 2:
        mat_t = mat_t.unsqueeze(0)
    rot6d = matrix_to_rotation_6d(mat_t)
    return rot6d.squeeze(0).numpy()


def extract_markers(anno, frame_ids, smplx_path, device='cpu'):
    """
    Run SMPL-X forward pass and extract 77 markers.

    Returns:
        markers: [T, 77, 3] float32
    """
    import smplx as smplx_lib
    T = len(frame_ids)
    raw_smplx = anno['raw_smplx']

    # Collect SMPL-X params
    body_transl = np.zeros((T, 3), dtype=np.float32)
    body_orient_aa = np.zeros((T, 3), dtype=np.float32)
    body_pose_aa = np.zeros((T, 63), dtype=np.float32)
    lhand_aa = np.zeros((T, 45), dtype=np.float32)
    rhand_aa = np.zeros((T, 45), dtype=np.float32)
    body_shape = np.zeros((T, 300), dtype=np.float32)

    for i, fid in enumerate(frame_ids):
        s = raw_smplx.get(fid)
        if s is None:
            continue
        body_transl[i] = s['world_tsl'].squeeze().numpy()
        body_orient_aa[i] = quat_wxyz_to_aa_batch(s['world_rot'].squeeze().numpy()[None])[0]
        body_pose_aa[i] = quat_wxyz_to_aa_batch(s['body_pose'].squeeze().numpy()).reshape(-1)
        lhand_aa[i] = quat_wxyz_to_aa_batch(s['left_hand_pose'].squeeze().numpy()).reshape(-1)
        rhand_aa[i] = quat_wxyz_to_aa_batch(s['right_hand_pose'].squeeze().numpy()).reshape(-1)
        body_shape[i] = s['body_shape'].squeeze().numpy()

    # SMPL-X forward in chunks (GPU if available)
    gpu_device = torch.device('cuda:0') if torch.cuda.is_available() else device
    CHUNK = 500 if torch.cuda.is_available() else 200
    all_markers = []
    for cs in range(0, T, CHUNK):
        ce = min(cs + CHUNK, T)
        cT = ce - cs
        model = smplx_lib.create(
            smplx_path, model_type='smplx', gender='neutral',
            use_pca=False, flat_hand_mean=True, batch_size=cT, num_betas=300,
        ).to(gpu_device)
        with torch.no_grad():
            out = model(
                betas=torch.tensor(body_shape[cs:ce], device=gpu_device),
                transl=torch.tensor(body_transl[cs:ce], device=gpu_device),
                global_orient=torch.tensor(body_orient_aa[cs:ce], device=gpu_device),
                body_pose=torch.tensor(body_pose_aa[cs:ce], device=gpu_device),
                left_hand_pose=torch.tensor(lhand_aa[cs:ce], device=gpu_device),
                right_hand_pose=torch.tensor(rhand_aa[cs:ce], device=gpu_device),
            )
            verts = out.vertices.cpu().numpy()  # [cT, 10475, 3]
            markers = verts[:, MARKERSET_SMPLX, :]  # [cT, 77, 3]
            all_markers.append(markers)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.concatenate(all_markers, axis=0)  # [T, 77, 3]


def extract_object_features(anno, frame_ids, obj_slot_names):
    """
    Extract object pos+rot6d for the given object slots.

    Returns:
        obj_features: [T, max_objects*9] float32
    """
    T = len(frame_ids)
    obj_features = np.zeros((T, MAX_OBJECTS * 9), dtype=np.float32)
    obj_transf = anno.get('obj_transf', {})

    for oi, oid in enumerate(obj_slot_names[:MAX_OBJECTS]):
        if oid not in obj_transf:
            continue
        td = obj_transf[oid]
        for i, fid in enumerate(frame_ids):
            if fid in td:
                tf = td[fid]
                base = oi * 9
                obj_features[i, base:base + 3] = tf[:3, 3]
                obj_features[i, base + 3:base + 9] = mat3x3_to_6d(tf[:3, :3])

    return obj_features


# =============================================================================
# Phase 3: Per-Entity State Labels with Schmitt Trigger
# =============================================================================

def compute_fingertip_distances(markers, obj_transf, frame_ids, obj_id):
    """
    Compute min distance from 10 fingertip markers to object surface.

    Returns:
        distances: [T] float32, min distance per frame (in meters)
    """
    T = len(frame_ids)
    # Fingertip marker indices within 77-marker set: last 10 (indices 67-76)
    fingertip_indices = list(range(67, 77))
    distances = np.full(T, 999.0, dtype=np.float32)

    td = obj_transf.get(obj_id, {})
    for i, fid in enumerate(frame_ids):
        if fid not in td:
            continue
        tf = td[fid]
        obj_pos = tf[:3, 3]
        # Simplified: distance from fingertips to object center
        # (Full SDF would require mesh loading — use center distance as proxy)
        ftips = markers[i, fingertip_indices, :]  # [10, 3]
        dists = np.linalg.norm(ftips - obj_pos[None, :], axis=1)  # [10]
        distances[i] = dists.min()

    return distances


def schmitt_trigger_contact(distances, trigger_cm=1.5, release_cm=3.0):
    """
    Apply Schmitt trigger to detect contact with hysteresis.

    Returns:
        contact: [T] bool
    """
    T = len(distances)
    contact = np.zeros(T, dtype=bool)
    state = False  # not in contact

    for i in range(T):
        d_cm = distances[i] * 100  # meters to cm
        if state:  # currently in contact
            if d_cm > release_cm:
                state = False
        else:  # not in contact
            if d_cm < trigger_cm:
                state = True
        contact[i] = state

    return contact


def majority_vote_smooth(labels, window=5):
    """Apply majority vote sliding window to smooth state labels."""
    T = len(labels)
    smoothed = labels.copy()
    half = window // 2
    for i in range(half, T - half):
        window_vals = labels[i - half:i + half + 1]
        values, counts = np.unique(window_vals, return_counts=True)
        smoothed[i] = values[counts.argmax()]
    return smoothed


def get_primitive_action(prim_name):
    """Classify a primitive action into topology type."""
    clean = prim_name.strip("<>").split(",")[0].strip().lower()
    if clean in TRIADIC_ACTIONS:
        return "triadic"
    elif clean in DYADIC_ACTIONS:
        return "dyadic"
    elif clean in FLUID_ACTIONS:
        return "fluid"
    elif clean in STATIC_ACTIONS:
        return "static"
    else:
        return "dyadic"  # default


def build_entity_state_array(segment, markers, frame_ids, anno, obj_slot_names, T):
    """
    Build [T, 7] per-entity state array using primitive info and contact detection.

    Columns: [body, lh, rh, obj1, obj2, obj3, obj4]
    """
    states = np.zeros((T, NUM_ENTITIES), dtype=np.int8)
    # Body always static
    states[:, 0] = STATE_STATIC
    # Hands start idle
    states[:, 1] = STATE_STATIC  # LH idle
    states[:, 2] = STATE_STATIC  # RH idle
    # Objects start static, unused = padding
    for oi in range(MAX_OBJECTS):
        if oi < len(obj_slot_names):
            states[:, 3 + oi] = STATE_STATIC
        else:
            states[:, 3 + oi] = STATE_PADDING

    obj_transf = anno.get('obj_transf', {})
    frame_to_idx = {fid: i for i, fid in enumerate(frame_ids)}

    for prim in segment['primitives']:
        prim_data = prim['data']
        primitive_name = prim_data.get('primitive', '')
        action_type = get_primitive_action(primitive_name)

        # Left hand
        lh_interval = prim.get('lh_interval')
        prim_lh = prim_data.get('primitive_lh')
        obj_list_lh = prim_data.get('obj_list_lh') or prim_data.get('obj_list') or []

        if lh_interval and prim_lh:
            lh_start_idx = frame_to_idx.get(lh_interval[0], 0)
            lh_end_idx = frame_to_idx.get(lh_interval[1], T - 1)

            # Compute contact for LH with its target objects
            for obj_id in obj_list_lh[:1]:  # primary object
                if obj_id in obj_transf:
                    dists = compute_fingertip_distances(markers, obj_transf, frame_ids, obj_id)
                    contact = schmitt_trigger_contact(dists, CONTACT_TRIGGER_CM, CONTACT_RELEASE_CM)

                    # Assign hand states based on contact
                    for fi in range(lh_start_idx, min(lh_end_idx + 1, T)):
                        if contact[fi]:
                            states[fi, 1] = STATE_GRASPED
                        elif fi < lh_end_idx:
                            if states[fi, 1] == STATE_STATIC:
                                states[fi, 1] = STATE_REACH

                    # Assign object states
                    if obj_id in obj_slot_names:
                        obj_col = 3 + obj_slot_names.index(obj_id)
                        if obj_col < NUM_ENTITIES:
                            for fi in range(lh_start_idx, min(lh_end_idx + 1, T)):
                                if contact[fi]:
                                    if action_type == "triadic":
                                        states[fi, obj_col] = STATE_GRASPED
                                    elif action_type == "dyadic":
                                        states[fi, obj_col] = STATE_INTERACTING
                                    else:
                                        states[fi, obj_col] = STATE_GRASPED

            # Handle secondary objects for triadic
            if action_type == "triadic" and len(obj_list_lh) > 1:
                sec_obj = obj_list_lh[1]
                if sec_obj in obj_slot_names:
                    sec_col = 3 + obj_slot_names.index(sec_obj)
                    if sec_col < NUM_ENTITIES:
                        for fi in range(lh_start_idx, min(lh_end_idx + 1, T)):
                            if states[fi, 1] == STATE_GRASPED:
                                states[fi, sec_col] = STATE_INTERACTING

        # Right hand (same logic)
        rh_interval = prim.get('rh_interval')
        prim_rh = prim_data.get('primitive_rh')
        obj_list_rh = prim_data.get('obj_list_rh') or prim_data.get('obj_list') or []

        if rh_interval and prim_rh:
            rh_start_idx = frame_to_idx.get(rh_interval[0], 0)
            rh_end_idx = frame_to_idx.get(rh_interval[1], T - 1)

            for obj_id in obj_list_rh[:1]:
                if obj_id in obj_transf:
                    dists = compute_fingertip_distances(markers, obj_transf, frame_ids, obj_id)
                    contact = schmitt_trigger_contact(dists, CONTACT_TRIGGER_CM, CONTACT_RELEASE_CM)

                    for fi in range(rh_start_idx, min(rh_end_idx + 1, T)):
                        if contact[fi]:
                            states[fi, 2] = STATE_GRASPED
                        elif fi < rh_end_idx:
                            if states[fi, 2] == STATE_STATIC:
                                states[fi, 2] = STATE_REACH

                    if obj_id in obj_slot_names:
                        obj_col = 3 + obj_slot_names.index(obj_id)
                        if obj_col < NUM_ENTITIES:
                            for fi in range(rh_start_idx, min(rh_end_idx + 1, T)):
                                if contact[fi]:
                                    if action_type == "triadic":
                                        states[fi, obj_col] = STATE_GRASPED
                                    elif action_type == "dyadic":
                                        states[fi, obj_col] = STATE_INTERACTING
                                    else:
                                        states[fi, obj_col] = STATE_GRASPED

            if action_type == "triadic" and len(obj_list_rh) > 1:
                sec_obj = obj_list_rh[1]
                if sec_obj in obj_slot_names:
                    sec_col = 3 + obj_slot_names.index(sec_obj)
                    if sec_col < NUM_ENTITIES:
                        for fi in range(rh_start_idx, min(rh_end_idx + 1, T)):
                            if states[fi, 2] == STATE_GRASPED:
                                states[fi, sec_col] = STATE_INTERACTING

    # Apply majority vote smoothing to hand and object columns
    for col in range(1, NUM_ENTITIES):
        if not np.all(states[:, col] == STATE_PADDING):
            states[:, col] = majority_vote_smooth(states[:, col], MAJORITY_VOTE_WINDOW)

    return states


# =============================================================================
# Phase 4: Boundary Expansion + Downsampling
# =============================================================================

def downsample_with_boundary(markers_77, obj_features, state_array, text,
                              seg_start, seg_end, total_frames, anno,
                              boundary_pad=60, step=3, target_len=200):
    """
    1. Expand boundaries by ±60 frames
    2. Downsample by step=3 (30→10FPS)
    3. Max-pool states
    4. Uniform subsample to target_len

    Returns:
        motion: [target_len, 267], states: [target_len, 7], text: str
    """
    # Boundary expansion
    exp_start = max(0, seg_start - boundary_pad)
    exp_end = min(total_frames - 1, seg_end + boundary_pad)

    # The markers/obj/states are already for [seg_start, seg_end]
    # We need to reindex if we expanded beyond
    T_current = markers_77.shape[0]

    # Downsample: take every `step` frame
    ds_indices = list(range(0, T_current, step))
    markers_ds = markers_77[ds_indices]  # [T_ds, 77, 3]
    obj_ds = obj_features[ds_indices]
    # State max-pool: take max state in each window (preserves contacts)
    states_ds = np.zeros((len(ds_indices), NUM_ENTITIES), dtype=np.int8)
    for i, idx in enumerate(ds_indices):
        window = state_array[max(0, idx):min(T_current, idx + step)]
        for col in range(NUM_ENTITIES):
            if state_array[idx, col] == STATE_PADDING:
                states_ds[i, col] = STATE_PADDING
            else:
                states_ds[i, col] = window[:, col].max()

    T_ds = len(ds_indices)

    # Uniform subsample to target_len
    if T_ds > target_len:
        sub_step = T_ds / target_len
        sub_indices = [int(i * sub_step) for i in range(target_len)]
        markers_ds = markers_ds[sub_indices]
        obj_ds = obj_ds[sub_indices]
        states_ds = states_ds[sub_indices]
    elif T_ds < target_len:
        # Pad with last frame
        pad_len = target_len - T_ds
        markers_ds = np.concatenate([markers_ds, np.tile(markers_ds[-1:], (pad_len, 1, 1))], axis=0)
        obj_ds = np.concatenate([obj_ds, np.tile(obj_ds[-1:], (pad_len, 1))], axis=0)
        states_ds = np.concatenate([states_ds, np.tile(states_ds[-1:], (pad_len, 1))], axis=0)

    # Flatten markers and concatenate with objects
    marker_flat = markers_ds.reshape(target_len, 231)
    motion = np.concatenate([marker_flat, obj_ds], axis=1)  # [target_len, 267]

    return motion, states_ds, text


# =============================================================================
# Main Processing
# =============================================================================

def process_one_sequence(seq_id, anno_path, program_info, desc_info,
                          smplx_path, target_len=200, gap_threshold=300):
    """
    Process one complex sequence into multiple super-segment samples.

    Returns:
        list of (motion, states, text, seg_info) tuples
    """
    with open(anno_path, 'rb') as f:
        anno = pickle.load(f)

    mocap = anno['mocap_frame_id_list']
    total_frames = len(mocap)

    # Phase 1: Extract super-segments
    segments = extract_super_segments(program_info, total_frames, gap_threshold)
    if not segments:
        return []

    results = []
    for seg_idx, seg in enumerate(segments):
        seg_start = seg['start']
        seg_end = seg['end']
        seg_frames = mocap[seg_start:seg_end + 1]
        T_seg = len(seg_frames)
        if T_seg < 30:  # Skip very short segments
            continue

        # Collect objects from this segment's primitives (sorted for consistency)
        seg_objects = set()
        for prim in seg['primitives']:
            pd = prim['data']
            for key in ['obj_list', 'obj_list_lh', 'obj_list_rh']:
                val = pd.get(key)
                if val:  # handles None and empty list
                    seg_objects.update(val)
        obj_slot_names = sorted(list(seg_objects))[:MAX_OBJECTS]

        # Phase 2: Extract markers
        try:
            markers_77 = extract_markers(anno, seg_frames, smplx_path)
        except Exception as e:
            print(f"  SMPL-X failed for {seq_id} seg {seg_idx}: {e}")
            continue

        # Extract object features
        obj_features = extract_object_features(anno, seg_frames, obj_slot_names)

        # Phase 3: Build per-entity state labels
        states = build_entity_state_array(seg, markers_77, seg_frames, anno, obj_slot_names, T_seg)

        # Build text description from primitives
        texts = []
        for prim in seg['primitives']:
            pd = prim['data']
            desc = pd.get('primitive', '')
            # Try to get description from desc_info
            if desc_info:
                for dk, dv in desc_info.items():
                    if 'seg_desc' in dv:
                        texts.append(dv['seg_desc'])
                        break
            if not texts:
                texts.append(desc)
        text = ". ".join(texts[:3])  # Concatenate first 3 primitive descriptions

        # Phase 4: Downsample + boundary expand
        motion, states_ds, text = downsample_with_boundary(
            markers_77, obj_features, states, text,
            seg_start, seg_end, total_frames, anno,
            target_len=target_len,
        )

        results.append({
            'motion': motion.astype(np.float32),
            'states': states_ds.astype(np.int8),
            'text': text,
            'obj_slot_names': obj_slot_names,
            'seq_id': seq_id,
            'seg_idx': seg_idx,
            'seg_frames': (seg_start, seg_end),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="CRR-Flow V3 Preprocessing")
    parser.add_argument("--oakink2_root", default="/hhd4/lizhe/dataset/OakInk2/data")
    parser.add_argument("--output_root", default="dataset/OAKINK2_V3")
    parser.add_argument("--smplx_path", default="/hhd4/lizhe/code/InterAct/models")
    parser.add_argument("--num_samples", type=int, default=5, help="0 = all")
    parser.add_argument("--target_len", type=int, default=200)
    parser.add_argument("--gap_threshold", type=int, default=300)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    anno_dir = os.path.join(args.oakink2_root, "anno_preview")
    prog_dir = os.path.join(args.oakink2_root, "program", "program_info")
    desc_dir = os.path.join(args.oakink2_root, "program", "desc_info")

    # Get all annotation files
    pkl_files = sorted([f for f in os.listdir(anno_dir) if f.endswith('.pkl')])
    if args.num_samples > 0:
        pkl_files = pkl_files[:args.num_samples]

    print(f"=== CRR-Flow V3 Preprocessing ===")
    print(f"Output: {args.output_root}")
    print(f"Sequences: {len(pkl_files)}")
    print(f"Target length: {args.target_len} frames")

    # Output directories
    os.makedirs(os.path.join(args.output_root, "motion"), exist_ok=True)
    os.makedirs(os.path.join(args.output_root, "state_arrays"), exist_ok=True)
    os.makedirs(os.path.join(args.output_root, "texts"), exist_ok=True)

    all_samples = []
    file_names = []
    sample_idx = 0

    for pkl_file in tqdm(pkl_files, desc="Processing"):
        seq_id = pkl_file.replace('.pkl', '')
        anno_path = os.path.join(anno_dir, pkl_file)

        # Load program info — file named exactly as sequence ID
        prog_path = os.path.join(prog_dir, seq_id + '.json')
        if not os.path.exists(prog_path):
            continue
        with open(prog_path) as f:
            program_info = json.load(f)

        # Load desc info
        desc_info = None
        desc_path = os.path.join(desc_dir, seq_id + '.json')
        if os.path.exists(desc_path):
            with open(desc_path) as f:
                desc_info = json.load(f)

        try:
            results = process_one_sequence(
                seq_id, anno_path, program_info, desc_info,
                args.smplx_path, args.target_len, args.gap_threshold,
            )
        except Exception as e:
            print(f"  Failed: {seq_id}: {e}")
            continue

        for r in results:
            idx_str = f"{sample_idx:06d}"
            np.save(os.path.join(args.output_root, "motion", f"{idx_str}.npy"), r['motion'])
            np.save(os.path.join(args.output_root, "state_arrays", f"{idx_str}.npy"), r['states'])
            with open(os.path.join(args.output_root, "texts", f"{idx_str}.txt"), 'w') as f:
                f.write(f"{r['text']}##{0.0}#{0.0}\n")
            file_names.append(f"{idx_str}\t{r['seq_id']}__seg{r['seg_idx']}")
            all_samples.append(r['motion'])
            sample_idx += 1

    if not all_samples:
        print("No samples generated!")
        return

    # Compute normalization
    all_motions = np.stack(all_samples, axis=0)
    mean = all_motions.reshape(-1, FEATURE_DIM).mean(axis=0)
    std = all_motions.reshape(-1, FEATURE_DIM).std(axis=0)
    std[std < 1e-8] = 1.0

    np.save(os.path.join(args.output_root, "Mean_flow.npy"), mean)
    np.save(os.path.join(args.output_root, "Std_flow.npy"), std)

    # NOTE: Do NOT normalize and re-save here. The dataset class (flow_dataset.py)
    # applies normalization at load time using Mean_flow.npy/Std_flow.npy.
    # Saving normalized data here would cause double normalization.

    # Write file names
    with open(os.path.join(args.output_root, "file_names.txt"), 'w') as f:
        for fn in file_names:
            f.write(fn + '\n')

    # Train/test split
    n_train = int(sample_idx * args.train_ratio)
    indices = list(range(sample_idx))
    np.random.seed(42)
    np.random.shuffle(indices)

    with open(os.path.join(args.output_root, "train_flow.txt"), 'w') as f:
        for i in indices[:n_train]:
            f.write(f"{i:06d}\n")
    with open(os.path.join(args.output_root, "test_flow.txt"), 'w') as f:
        for i in indices[n_train:]:
            f.write(f"{i:06d}\n")

    # Metadata
    metadata = {
        "feature_dim": FEATURE_DIM,
        "num_entities": NUM_ENTITIES,
        "max_objects": MAX_OBJECTS,
        "target_len": args.target_len,
        "total_samples": sample_idx,
        "train_samples": n_train,
        "test_samples": sample_idx - n_train,
    }
    with open(os.path.join(args.output_root, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Total samples: {sample_idx}")
    print(f"  Train: {n_train}, Test: {sample_idx - n_train}")
    print(f"  Feature dim: {FEATURE_DIM}")
    print(f"  Output: {args.output_root}")


if __name__ == "__main__":
    main()
