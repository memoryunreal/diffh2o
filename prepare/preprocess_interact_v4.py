#!/usr/bin/env python3
"""
Preprocess InterAct/OMOMO 962D → V4 534D format.

Extracts compatible features from InterAct's preprocessed data:
- Marker positions [0:231] + velocities [231:462] = 462D
- Object pose [476:485] (9D) + velocity [485:494] (9D) = 18D
- Pad 3 extra object slots with zeros → 54D
- Total: 462 + 18 + 54 = 534D
- BPS: from InterAct's object_bps/ directories (1024×3 → 1024 scalar distances)
- States: computed from contact features [494:962]

Usage:
    # InterAct GRAB subset
    python -m prepare.preprocess_interact_v4 --dataset grab --target_len 300

    # OMOMO
    python -m prepare.preprocess_interact_v4 --dataset omomo --target_len 300
"""

import os
import json
import argparse
import numpy as np
from typing import List

MAX_OBJECTS = 4
NUM_ENTITIES = 7
BPS_DIM = 1024

STATE_STATIC = 0
STATE_REACH = 1
STATE_GRASPED = 2
STATE_INTERACTING = 3
STATE_RELEASE = 4
STATE_PADDING = 5


def extract_v4_from_962(motion_962, target_len=300):
    """Extract V4 534D features from InterAct 962D.

    InterAct 962D layout:
    [0:231]   marker positions (77×3)
    [231:462] marker velocities (77×3)
    [462:476] foot contact heights (14D) — skip
    [476:485] object pose (9D: 6D rot + 3D trans)
    [485:494] object velocity (9D)
    [494:962] contact awareness (468D) — skip (used for state computation)
    """
    T = motion_962.shape[0]

    # Extract compatible features
    marker_pos = motion_962[:, 0:231]      # [T, 231]
    marker_vel = motion_962[:, 231:462]    # [T, 231]
    obj_pose = motion_962[:, 476:485]      # [T, 9]
    obj_vel = motion_962[:, 485:494]       # [T, 9]

    # Build V4 534D: pos(267) + vel(267)
    # Position: markers(231) + obj1(9) + obj2-4(27 zeros)
    pos = np.zeros((T, 267), dtype=np.float32)
    pos[:, :231] = marker_pos
    pos[:, 231:240] = obj_pose  # Object 1

    # Velocity: same layout
    vel = np.zeros((T, 267), dtype=np.float32)
    vel[:, :231] = marker_vel
    vel[:, 231:240] = obj_vel  # Object 1

    dynamic = np.concatenate([pos, vel], axis=-1)  # [T, 534]

    # Subsample/pad to target_len
    if T > target_len:
        indices = np.linspace(0, T - 1, target_len).astype(int)
        dynamic = dynamic[indices]
    elif T < target_len:
        pad_len = target_len - T
        dynamic = np.concatenate([dynamic, np.tile(dynamic[-1:], (pad_len, 1))], axis=0)

    return dynamic


def compute_states_from_contact_962(motion_962, target_len=300):
    """Compute per-entity states from InterAct's contact awareness features.

    Contact features [494:962] = 78 points × 6D.
    For each body point, first 3D = displacement to nearest object surface.
    Distance = norm(displacement). If < 5cm → contact.
    """
    T = motion_962.shape[0]
    contact_features = motion_962[:, 494:962]  # [T, 468]
    contact_features = contact_features.reshape(T, 78, 6)

    # Contact distance = norm of first 3D (displacement vector)
    contact_dist = np.linalg.norm(contact_features[:, :, :3], axis=2)  # [T, 78]

    # Fingertip marker indices within 77-marker set
    lh_tips = list(range(67, 72))  # 5 left fingertips
    rh_tips = list(range(72, 77))  # 5 right fingertips

    lh_contact = (contact_dist[:, lh_tips].min(axis=1) < 0.05)  # [T]
    rh_contact = (contact_dist[:, rh_tips].min(axis=1) < 0.05)  # [T]

    states = np.zeros((T, NUM_ENTITIES), dtype=np.int8)
    states[:, 0] = STATE_STATIC  # Body
    states[:, 3] = STATE_STATIC  # Object 1
    states[:, 4:7] = STATE_PADDING  # Objects 2-4

    # Assign hand states with Reach and Release
    for hand_col, contact in [(1, lh_contact), (2, rh_contact)]:
        seen_grasped = False
        for fi in range(T):
            if contact[fi]:
                states[fi, hand_col] = STATE_GRASPED
                seen_grasped = True
            elif seen_grasped:
                states[fi, hand_col] = STATE_RELEASE
                if fi >= 5 and not any(contact[fi-5:fi]):
                    seen_grasped = False
                    states[fi, hand_col] = STATE_STATIC
            else:
                # Check if any fingertip is approaching (within 30cm)
                tips = lh_tips if hand_col == 1 else rh_tips
                min_dist = contact_dist[fi, tips].min()
                if min_dist < 0.3:
                    states[fi, hand_col] = STATE_REACH

    # Object state
    for fi in range(T):
        if states[fi, 1] == STATE_GRASPED or states[fi, 2] == STATE_GRASPED:
            states[fi, 3] = STATE_GRASPED

    # Subsample/pad states
    if T > target_len:
        indices = np.linspace(0, T - 1, target_len).astype(int)
        states = states[indices]
    elif T < target_len:
        pad_len = target_len - T
        states = np.concatenate([states, np.tile(states[-1:], (pad_len, 1))], axis=0)

    # Majority vote smoothing
    for col in [1, 2, 3]:
        smoothed = states[:, col].copy()
        for i in range(2, target_len - 2):
            window = states[i-2:i+3, col]
            vals, counts = np.unique(window, return_counts=True)
            smoothed[i] = vals[counts.argmax()]
        states[:, col] = smoothed

    return states


def load_bps(seq_dir):
    """Load BPS encoding from InterAct's object_bps/ directory."""
    bps_array = np.zeros((MAX_OBJECTS, BPS_DIM), dtype=np.float32)
    bps_dir = os.path.join(seq_dir, 'object_bps')
    if not os.path.isdir(bps_dir):
        return bps_array

    bps_files = sorted(os.listdir(bps_dir))
    for i, fname in enumerate(bps_files[:1]):  # Only first object
        bps_path = os.path.join(bps_dir, fname)
        bps_raw = np.load(bps_path)  # [1024, 3] or [1024]
        if bps_raw.ndim == 2:
            # Convert 3D vectors to scalar distances
            bps_array[i] = np.linalg.norm(bps_raw, axis=1).astype(np.float32)
        else:
            bps_array[i] = bps_raw[:BPS_DIM].astype(np.float32)

    return bps_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['grab', 'omomo', 'behave', 'chairs'],
                        help="Which InterAct sub-dataset to process")
    parser.add_argument("--interact_root", default="/hhd4/lizhe/code/InterAct/data")
    parser.add_argument("--output_root", default="")
    parser.add_argument("--target_len", type=int, default=300)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    if not args.output_root:
        args.output_root = f"dataset/INTERACT_{args.dataset.upper()}_V4"

    seq_dir_root = os.path.join(args.interact_root, args.dataset, 'sequences_canonical')
    if not os.path.isdir(seq_dir_root):
        print(f"ERROR: {seq_dir_root} not found")
        return

    # Load test split
    test_seqs = set()
    test_seq_path = os.path.join(args.interact_root, 'test_seq.txt')
    if os.path.exists(test_seq_path):
        with open(test_seq_path) as f:
            test_seqs = set(l.strip() for l in f if l.strip())

    # Create output dirs
    for subdir in ['motion_dynamic', 'bps', 'state_arrays', 'texts', 'clip_embeddings']:
        os.makedirs(os.path.join(args.output_root, subdir), exist_ok=True)

    sequences = sorted(os.listdir(seq_dir_root))
    if args.num_samples > 0:
        sequences = sequences[:args.num_samples]
    print(f"Processing {len(sequences)} {args.dataset} sequences...")

    all_dynamics = []
    file_names = []
    train_indices = []
    test_indices = []
    sample_idx = 0

    for i, seq_name in enumerate(sequences):
        seq_dir = os.path.join(seq_dir_root, seq_name)
        motion_path = os.path.join(seq_dir, 'motion.npy')
        if not os.path.exists(motion_path):
            continue

        motion_962 = np.load(motion_path).astype(np.float32)
        if motion_962.shape[0] < 30:  # Skip very short sequences
            continue

        # Extract V4 features
        dynamic = extract_v4_from_962(motion_962, args.target_len)
        states = compute_states_from_contact_962(motion_962, args.target_len)
        bps = load_bps(seq_dir)

        # Load text
        text = ""
        text_path = os.path.join(seq_dir, 'text.txt')
        if os.path.exists(text_path):
            with open(text_path) as f:
                text = f.readline().split('#')[0].strip()

        # Save
        idx_str = f"{sample_idx:06d}"
        np.save(os.path.join(args.output_root, 'motion_dynamic', f'{idx_str}.npy'), dynamic)
        np.save(os.path.join(args.output_root, 'state_arrays', f'{idx_str}.npy'), states)
        np.save(os.path.join(args.output_root, 'bps', f'{idx_str}.npy'), bps)
        with open(os.path.join(args.output_root, 'texts', f'{idx_str}.txt'), 'w') as f:
            f.write(f"{text}##{0.0}#{0.0}\n")

        file_names.append(f"{idx_str}\t{seq_name}")

        # Split
        if seq_name in test_seqs:
            test_indices.append(sample_idx)
        else:
            train_indices.append(sample_idx)

        all_dynamics.append(dynamic)
        sample_idx += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sequences)} processed")

    if not all_dynamics:
        print("No samples!")
        return

    # If no test sequences found via test_seq.txt, use random 80/20 split
    if not test_indices:
        print("No test sequences from test_seq.txt, using random 80/20 split")
        all_indices = list(range(sample_idx))
        np.random.seed(42)
        np.random.shuffle(all_indices)
        n_train = int(sample_idx * args.train_ratio)
        train_indices = all_indices[:n_train]
        test_indices = all_indices[n_train:]

    # Normalization
    all_dyn = np.stack(all_dynamics, axis=0)
    mean = all_dyn.reshape(-1, 534).mean(axis=0)
    std = all_dyn.reshape(-1, 534).std(axis=0)
    std[std < 1e-8] = 1.0

    np.save(os.path.join(args.output_root, 'Mean_flow.npy'), mean.astype(np.float32))
    np.save(os.path.join(args.output_root, 'Std_flow.npy'), std.astype(np.float32))

    with open(os.path.join(args.output_root, 'file_names.txt'), 'w') as f:
        for fn in file_names:
            f.write(fn + '\n')
    with open(os.path.join(args.output_root, 'train_flow.txt'), 'w') as f:
        for i in train_indices:
            f.write(f"{i:06d}\n")
    with open(os.path.join(args.output_root, 'test_flow.txt'), 'w') as f:
        for i in test_indices:
            f.write(f"{i:06d}\n")

    metadata = {
        "source_dataset": args.dataset,
        "feature_dim_dynamic": 534,
        "bps_dim": BPS_DIM,
        "target_len": args.target_len,
        "total_samples": sample_idx,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
    }
    with open(os.path.join(args.output_root, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{args.dataset.upper()} V4 saved to {args.output_root}/")
    print(f"  Samples: {sample_idx} (train={len(train_indices)}, test={len(test_indices)})")


if __name__ == "__main__":
    main()
