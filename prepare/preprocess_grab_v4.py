#!/usr/bin/env python3
"""
Preprocess GRAB dataset → V4 format (267D markers + 267D velocity + BPS).

GRAB: 120fps → 30fps, single object, SMPL-X body params, GT per-vertex contact.
Output: 534D dynamic + 1024D BPS per object, target_len=300 frames.

Usage:
    python -m prepare.preprocess_grab_v4 --num_samples 0  # all sequences
    python -m prepare.preprocess_grab_v4 --num_samples 10  # debug
"""

import os
import json
import argparse
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from typing import Optional

# Reuse V3 constants
MARKERSET_SMPLX = [
    5920, 5621, 5882, 3486, 4430, 3258, 2029, 5694, 5645, 9117, 4302,
    4319, 707, 4788, 4198, 3998, 8846, 4726, 3682, 3688, 5890, 5901,
    3961, 4722, 5698, 5868, 3892, 3712, 4479, 4088, 3477, 4839, 5787,
    4291, 8576, 6248, 7166, 6021, 3067, 8250, 8388, 8339, 3116, 7040,
    7099, 2198, 7524, 7115, 6265, 6746, 8634, 7462, 6443, 6449, 8584,
    8595, 6709, 7458, 8392, 8527, 7215, 6832, 7575, 6503, 8481, 5945,
    5947, 8079, 7669, 7794, 7905, 8022, 5361, 4933, 5058, 5169, 5286,
]

# Fingertip indices within 77-marker set (last 10: 5 LH + 5 RH)
LH_FINGERTIP_INDICES = list(range(67, 72))  # indices 67-71
RH_FINGERTIP_INDICES = list(range(72, 77))  # indices 72-76

MAX_OBJECTS = 4
NUM_ENTITIES = 7
BPS_DIM = 1024
TARGET_FPS = 30
SOURCE_FPS = 120
DOWNSAMPLE_STEP = SOURCE_FPS // TARGET_FPS  # 4

# States
STATE_STATIC = 0
STATE_REACH = 1
STATE_GRASPED = 2
STATE_INTERACTING = 3
STATE_RELEASE = 4
STATE_PADDING = 5

# Train subjects (following InterAct convention)
TRAIN_SUBJECTS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
TEST_SUBJECTS = ['s9', 's10']


def mat3x3_to_6d(R):
    """Convert 3x3 rotation matrix to 6D representation."""
    return np.concatenate([R[:, 0], R[:, 1]])


def axis_angle_to_6d(aa):
    """Convert axis-angle [3] to 6D rotation."""
    R = Rotation.from_rotvec(aa).as_matrix()
    return mat3x3_to_6d(R)


def generate_fibonacci_basis(n):
    """Generate n points uniformly on unit sphere."""
    indices = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + np.sqrt(5)) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def compute_bps_distances(mesh, basis):
    """Compute BPS scalar distances."""
    vertices = mesh.vertices.copy() - mesh.centroid
    scale = np.max(np.abs(vertices))
    if scale > 0:
        vertices = vertices / scale
    norm_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    basis_scaled = basis * 1.5
    _, distances, _ = trimesh.proximity.closest_point(norm_mesh, basis_scaled)
    return distances.astype(np.float32)


def extract_markers_from_grab(npz_data, smplx_path, device='cuda:0'):
    """Extract 77 markers from GRAB SMPL-X parameters.

    Args:
        npz_data: loaded .npz dict
        smplx_path: path to smplx models directory

    Returns:
        markers: [T, 77, 3] at original fps
    """
    import torch
    import smplx as smplx_lib

    body = npz_data['body'].item()
    params = body['params']

    T = params['transl'].shape[0]
    gender = str(npz_data['gender'])
    n_comps = int(npz_data['n_comps'])

    chunk_size = 500
    all_markers = []

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        n = end - start

        # Create model matching exact chunk size
        model = smplx_lib.create(
            smplx_path, model_type='smplx', gender=gender,
            use_pca=True, num_pca_comps=n_comps,
            flat_hand_mean=True, batch_size=n,
        ).to(device)

        transl = torch.tensor(params['transl'][start:end], dtype=torch.float32, device=device)
        global_orient = torch.tensor(params['global_orient'][start:end], dtype=torch.float32, device=device)
        body_pose = torch.tensor(params['body_pose'][start:end], dtype=torch.float32, device=device)
        left_hand_pose = torch.tensor(params['left_hand_pose'][start:end], dtype=torch.float32, device=device)
        right_hand_pose = torch.tensor(params['right_hand_pose'][start:end], dtype=torch.float32, device=device)

        with torch.no_grad():
            output = model(
                transl=transl, global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
            )
            verts = output.vertices.cpu().numpy()

        markers = verts[:, MARKERSET_SMPLX, :]
        all_markers.append(markers)
        del model
        torch.cuda.empty_cache()

    return np.concatenate(all_markers, axis=0)


def compute_states_from_contact(contact_body, markers, obj_pos, T,
                                 intent='lift'):
    """Compute per-entity state labels from GRAB GT contact.

    Args:
        contact_body: [T, 10475] binary per-vertex contact
        markers: [T, 77, 3] marker positions
        obj_pos: [T, 3] object position
        T: number of frames
        intent: motion intent string

    Returns:
        states: [T, 7] state array
    """
    states = np.zeros((T, NUM_ENTITIES), dtype=np.int8)
    states[:, 0] = STATE_STATIC  # Body
    states[:, 3] = STATE_STATIC  # Object 1
    states[:, 4:7] = STATE_PADDING  # Objects 2-4 unused

    # Check fingertip contact using GT per-vertex labels
    lh_contact_verts = [MARKERSET_SMPLX[i] for i in LH_FINGERTIP_INDICES]
    rh_contact_verts = [MARKERSET_SMPLX[i] for i in RH_FINGERTIP_INDICES]

    lh_contact = contact_body[:, lh_contact_verts].any(axis=1)  # [T]
    rh_contact = contact_body[:, rh_contact_verts].any(axis=1)  # [T]

    # Assign hand states with Release
    for hand_col, contact in [(1, lh_contact), (2, rh_contact)]:
        seen_grasped = False
        for fi in range(T):
            if contact[fi]:
                states[fi, hand_col] = STATE_GRASPED
                seen_grasped = True
            elif seen_grasped:
                states[fi, hand_col] = STATE_RELEASE
                # Reset after sustained release (5 frames)
                if fi >= 5 and not any(contact[fi-5:fi]):
                    seen_grasped = False
                    states[fi, hand_col] = STATE_STATIC
            else:
                # Before first contact: check if approaching
                if obj_pos is not None:
                    hand_markers = markers[fi, LH_FINGERTIP_INDICES if hand_col == 1 else RH_FINGERTIP_INDICES, :]
                    dist = np.linalg.norm(hand_markers.mean(0) - obj_pos[fi])
                    if dist < 0.3:  # within 30cm → reaching
                        states[fi, hand_col] = STATE_REACH

    # Object state: grasped when any hand contacts
    for fi in range(T):
        if states[fi, 1] == STATE_GRASPED or states[fi, 2] == STATE_GRASPED:
            if intent in ['pour', 'drink', 'eat', 'cook', 'chop', 'brush', 'peel',
                           'stir', 'clean', 'screw', 'stamp', 'staple']:
                states[fi, 3] = STATE_INTERACTING
            else:
                states[fi, 3] = STATE_GRASPED

    # Majority vote smoothing (window=5)
    for col in [1, 2, 3]:
        smoothed = states[:, col].copy()
        for i in range(2, T - 2):
            window = states[i-2:i+3, col]
            vals, counts = np.unique(window, return_counts=True)
            smoothed[i] = vals[counts.argmax()]
        states[:, col] = smoothed

    return states


def process_one_sequence(npz_path, smplx_path, bps_basis, grab_mesh_dir,
                          target_len=300, device='cuda:0'):
    """Process a single GRAB sequence → V4 format.

    Returns dict with motion_dynamic [T, 534], bps [4, 1024], states [T, 7], text, etc.
    """
    data = np.load(npz_path, allow_pickle=True)

    obj_name = str(data['obj_name'])
    intent = str(data['motion_intent'])
    gender = str(data['gender'])
    subject = str(data['sbj_id'])
    n_frames = int(data['n_frames'])

    # 1. Extract 77 markers at 120fps
    markers_120 = extract_markers_from_grab(data, smplx_path, device=device)  # [T, 77, 3]

    # 2. Extract object pose at 120fps
    obj_params = data['object'].item()['params']
    obj_pos_120 = obj_params['transl'].astype(np.float32)  # [T, 3]
    obj_orient_120 = obj_params['global_orient'].astype(np.float32)  # [T, 3] axis-angle

    # 3. Downsample to 30fps
    markers = markers_120[::DOWNSAMPLE_STEP]  # [T30, 77, 3]
    obj_pos = obj_pos_120[::DOWNSAMPLE_STEP]
    obj_orient = obj_orient_120[::DOWNSAMPLE_STEP]
    T = markers.shape[0]

    # Also downsample contact labels (align to same T)
    contact_body = data['contact'].item()['body']  # [T120, 10475]
    contact_30 = np.zeros((T, contact_body.shape[1]), dtype=np.int8)
    for i in range(T):
        start = i * DOWNSAMPLE_STEP
        end = min(start + DOWNSAMPLE_STEP, contact_body.shape[0])
        contact_30[i] = contact_body[start:end].max(axis=0)

    # 4. Build 267D position features
    markers_flat = markers.reshape(T, -1)  # [T, 231]

    # Object 6D rotation
    obj_rot6d = np.zeros((T, 6), dtype=np.float32)
    for t in range(T):
        obj_rot6d[t] = axis_angle_to_6d(obj_orient[t])

    obj_feat = np.concatenate([obj_pos, obj_rot6d], axis=-1)  # [T, 9]
    # Pad 3 empty object slots
    obj_all = np.zeros((T, MAX_OBJECTS * 9), dtype=np.float32)
    obj_all[:, :9] = obj_feat

    positions = np.concatenate([markers_flat, obj_all], axis=-1)  # [T, 267]

    # 5. Subsample to target_len
    if T > target_len:
        indices = np.linspace(0, T - 1, target_len).astype(int)
        positions = positions[indices]
        contact_30 = contact_30[indices]
        markers = markers[indices]
        obj_pos = obj_pos[indices]
        T = target_len
    elif T < target_len:
        pad_len = target_len - T
        positions = np.concatenate([positions, np.tile(positions[-1:], (pad_len, 1))], axis=0)
        contact_30 = np.concatenate([contact_30, np.tile(contact_30[-1:], (pad_len, 1))], axis=0)
        markers_pad = np.tile(markers[-1:], (pad_len, 1, 1))
        markers = np.concatenate([markers, markers_pad], axis=0)
        obj_pos_pad = np.tile(obj_pos[-1:], (pad_len, 1))
        obj_pos = np.concatenate([obj_pos, obj_pos_pad], axis=0)
        T = target_len

    # 6. Compute velocity (after downsampling + subsampling)
    velocity = np.zeros_like(positions)
    velocity[1:] = positions[1:] - positions[:-1]
    velocity[0] = velocity[1]

    # 7. Dynamic features: position + velocity
    dynamic = np.concatenate([positions, velocity], axis=-1).astype(np.float32)  # [T, 534]

    # 8. Compute state labels from GT contact
    states = compute_states_from_contact(contact_30, markers, obj_pos, T, intent=intent)

    # 9. BPS for the object
    bps_array = np.zeros((MAX_OBJECTS, BPS_DIM), dtype=np.float32)
    mesh_path = os.path.join(grab_mesh_dir, f'{obj_name}.stl')
    if os.path.exists(mesh_path):
        mesh = trimesh.load(mesh_path, force='mesh')
        bps_array[0] = compute_bps_distances(mesh, bps_basis)

    # 10. Text
    text = f"the person {intent}s the {obj_name}."

    return {
        'dynamic': dynamic,      # [target_len, 534]
        'bps': bps_array,         # [4, 1024]
        'states': states,         # [target_len, 7]
        'text': text,
        'obj_name': obj_name,
        'intent': intent,
        'subject': subject,
        'gender': gender,
        'orig_frames': n_frames,
        'seq_name': os.path.splitext(os.path.basename(npz_path))[0],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grab_root", default="/hhd4/lizhe/dataset/GRAB")
    parser.add_argument("--smplx_path", default="/hhd4/lizhe/code/InterAct/models")
    parser.add_argument("--output_root", default="dataset/GRAB_V4")
    parser.add_argument("--target_len", type=int, default=300)
    parser.add_argument("--num_samples", type=int, default=0, help="0 = all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    grab_dir = os.path.join(args.grab_root, 'grab')
    grab_mesh_dir = args.grab_root  # STL files in root

    # Create output dirs
    for subdir in ['motion_dynamic', 'bps', 'state_arrays', 'texts']:
        os.makedirs(os.path.join(args.output_root, subdir), exist_ok=True)

    # BPS basis
    bps_basis = generate_fibonacci_basis(BPS_DIM)
    np.save(os.path.join(args.output_root, 'bps_basis.npy'), bps_basis)

    # Collect all sequences
    import smplx as smplx_lib
    sequences = []
    for subject in sorted(os.listdir(grab_dir)):
        subj_dir = os.path.join(grab_dir, subject)
        if not os.path.isdir(subj_dir):
            continue
        for fname in sorted(os.listdir(subj_dir)):
            if fname.endswith('.npz'):
                sequences.append((subject, os.path.join(subj_dir, fname)))

    if args.num_samples > 0:
        sequences = sequences[:args.num_samples]
    print(f"Processing {len(sequences)} GRAB sequences...")

    all_dynamics = []
    file_names = []
    train_indices = []
    test_indices = []
    sample_idx = 0

    for i, (subject, npz_path) in enumerate(sequences):
        try:
            result = process_one_sequence(
                npz_path, args.smplx_path, bps_basis, grab_mesh_dir,
                target_len=args.target_len, device=args.device,
            )
        except Exception as e:
            print(f"  SKIP {npz_path}: {e}")
            continue

        idx_str = f"{sample_idx:06d}"

        np.save(os.path.join(args.output_root, 'motion_dynamic', f'{idx_str}.npy'), result['dynamic'])
        np.save(os.path.join(args.output_root, 'bps', f'{idx_str}.npy'), result['bps'])
        np.save(os.path.join(args.output_root, 'state_arrays', f'{idx_str}.npy'), result['states'])
        with open(os.path.join(args.output_root, 'texts', f'{idx_str}.txt'), 'w') as f:
            f.write(f"{result['text']}##{0.0}#{0.0}\n")

        file_names.append(f"{idx_str}\t{subject}_{result['seq_name']}")
        all_dynamics.append(result['dynamic'])

        # Train/test by subject
        if subject in TEST_SUBJECTS:
            test_indices.append(sample_idx)
        else:
            train_indices.append(sample_idx)

        sample_idx += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sequences)} processed")

    if not all_dynamics:
        print("No samples!")
        return

    # Normalization
    all_dyn = np.stack(all_dynamics, axis=0)
    mean_dyn = all_dyn.reshape(-1, 534).mean(axis=0)
    std_dyn = all_dyn.reshape(-1, 534).std(axis=0)
    std_dyn[std_dyn < 1e-8] = 1.0

    np.save(os.path.join(args.output_root, 'Mean_flow.npy'), mean_dyn.astype(np.float32))
    np.save(os.path.join(args.output_root, 'Std_flow.npy'), std_dyn.astype(np.float32))

    # Write splits
    with open(os.path.join(args.output_root, 'file_names.txt'), 'w') as f:
        for fn in file_names:
            f.write(fn + '\n')

    with open(os.path.join(args.output_root, 'train_flow.txt'), 'w') as f:
        for i in train_indices:
            f.write(f"{i:06d}\n")
    with open(os.path.join(args.output_root, 'test_flow.txt'), 'w') as f:
        for i in test_indices:
            f.write(f"{i:06d}\n")

    # Metadata
    metadata = {
        "feature_dim_dynamic": 534,
        "bps_dim": BPS_DIM,
        "num_entities": NUM_ENTITIES,
        "max_objects": MAX_OBJECTS,
        "target_len": args.target_len,
        "source_fps": SOURCE_FPS,
        "target_fps": TARGET_FPS,
        "total_samples": sample_idx,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "train_subjects": TRAIN_SUBJECTS,
        "test_subjects": TEST_SUBJECTS,
    }
    with open(os.path.join(args.output_root, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGRAB V4 dataset saved to {args.output_root}/")
    print(f"  Samples: {sample_idx} (train={len(train_indices)}, test={len(test_indices)})")
    print(f"  Dynamic: [{args.target_len}, 534]")
    print(f"  FPS: {TARGET_FPS}")


if __name__ == "__main__":
    main()
