#!/usr/bin/env python3
"""
Preprocess V3 → V4: Add kinematic velocity and BPS object embeddings.

Reads existing V3 data (267D motion + state labels) and produces V4 format:
- Dynamic motion: [200, 534] = positions(267D) + velocities(267D)
- BPS embeddings: [4, 1024] per sample (scalar distances, static)
- State labels: [200, 7] (unchanged from V3)
- Text: unchanged from V3

Usage:
    python -m prepare.preprocess_v4 --v3_root dataset/OAKINK2_V3 --output_root dataset/OAKINK2_V4
"""

import os
import json
import argparse
import numpy as np
import trimesh
from typing import Dict, Optional


# BPS config
NUM_BPS_POINTS = 1024
MAX_OBJECTS = 4


def generate_fibonacci_basis(n: int) -> np.ndarray:
    """Generate n points uniformly on unit sphere via Fibonacci spiral."""
    indices = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + np.sqrt(5)) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def compute_bps_distances(mesh: trimesh.Trimesh, basis: np.ndarray) -> np.ndarray:
    """Compute BPS scalar distances from basis points to mesh surface.

    Args:
        mesh: Object mesh (will be centered and normalized).
        basis: [N, 3] basis points on unit sphere.

    Returns:
        [N] scalar distances from each basis point to nearest mesh surface.
    """
    # Center and normalize mesh to [-1, 1]
    vertices = mesh.vertices.copy() - mesh.centroid
    scale = np.max(np.abs(vertices))
    if scale > 0:
        vertices = vertices / scale

    # Create normalized mesh for proximity query
    norm_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

    # Scale basis to encompass mesh (1.5x)
    basis_scaled = basis * 1.5

    # Find closest points on mesh surface
    closest_points, distances, _ = trimesh.proximity.closest_point(norm_mesh, basis_scaled)

    return distances.astype(np.float32)  # [1024]


def compute_velocity(positions: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame velocity: V_t = X_t - X_{t-1}.

    First frame velocity is copied from second frame.
    """
    vel = np.zeros_like(positions)
    vel[1:] = positions[1:] - positions[:-1]
    vel[0] = vel[1]  # Copy frame 1's velocity to frame 0
    return vel


def get_segment_objects(seq_id: str, seg_idx: int, prog_dir: str) -> list:
    """Reconstruct the sorted object list for a V3 segment.

    Mirrors the logic in preprocess_v3.py: collect objects from all primitives
    in the super-segment, sort alphabetically, take first MAX_OBJECTS.
    """
    prog_path = os.path.join(prog_dir, seq_id + '.json')
    if not os.path.exists(prog_path):
        return []

    with open(prog_path) as f:
        prog_info = json.load(f)

    # Collect all objects from all primitives
    all_objs = set()
    for key, prim in prog_info.items():
        for field in ['obj_list', 'obj_list_lh', 'obj_list_rh']:
            objs = prim.get(field) or []
            if isinstance(objs, list):
                all_objs.update(objs)

    return sorted(all_objs)[:MAX_OBJECTS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3_root", type=str, default="dataset/OAKINK2_V3")
    parser.add_argument("--output_root", type=str, default="dataset/OAKINK2_V4")
    parser.add_argument("--oakink2_root", type=str,
                        default="/hhd4/lizhe/dataset/OakInk2/data")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    prog_dir = os.path.join(args.oakink2_root, "program", "program_info")
    obj_mesh_dir = os.path.join(args.oakink2_root, "object_repair", "align_ds")

    # Create output dirs
    for subdir in ['motion_dynamic', 'bps', 'state_arrays', 'texts', 'clip_embeddings']:
        os.makedirs(os.path.join(args.output_root, subdir), exist_ok=True)

    # Load V3 metadata
    v3_mean = np.load(os.path.join(args.v3_root, 'Mean_flow.npy'))
    v3_std = np.load(os.path.join(args.v3_root, 'Std_flow.npy'))

    # Load file names mapping
    file_map = {}
    with open(os.path.join(args.v3_root, 'file_names.txt')) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                file_map[parts[0]] = parts[1]

    # Generate BPS basis (deterministic)
    basis = generate_fibonacci_basis(NUM_BPS_POINTS)
    np.save(os.path.join(args.output_root, 'bps_basis.npy'), basis)

    # Cache BPS per object
    bps_cache: Dict[str, np.ndarray] = {}
    mesh_cache: Dict[str, trimesh.Trimesh] = {}

    def get_bps(obj_id: str) -> Optional[np.ndarray]:
        if obj_id in bps_cache:
            return bps_cache[obj_id]
        mesh_path = os.path.join(obj_mesh_dir, obj_id, 'model.obj')
        if not os.path.exists(mesh_path):
            print(f"  WARNING: mesh not found for {obj_id}")
            return None
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            bps = compute_bps_distances(mesh, basis)
            bps_cache[obj_id] = bps
            return bps
        except Exception as e:
            print(f"  WARNING: BPS failed for {obj_id}: {e}")
            return None

    # Process each V3 sample
    sample_indices = sorted(file_map.keys())
    all_dynamics = []
    file_names_out = []
    n_success = 0
    n_bps_found = 0

    print(f"Processing {len(sample_indices)} V3 samples → V4 format...")

    for i, idx_str in enumerate(sample_indices):
        # Load V3 motion (raw, not normalized)
        v3_motion_path = os.path.join(args.v3_root, 'motion', f'{idx_str}.npy')
        v3_state_path = os.path.join(args.v3_root, 'state_arrays', f'{idx_str}.npy')
        v3_text_path = os.path.join(args.v3_root, 'texts', f'{idx_str}.txt')
        v3_clip_path = os.path.join(args.v3_root, 'clip_embeddings', f'{idx_str}.npy')

        if not os.path.exists(v3_motion_path):
            continue

        motion_raw = np.load(v3_motion_path).astype(np.float32)  # [200, 267]
        T = motion_raw.shape[0]

        # Compute velocity on raw positions (already at 10fps)
        velocity = compute_velocity(motion_raw).astype(np.float32)  # [200, 267]

        # Concatenate position + velocity → dynamic features [200, 534]
        dynamic = np.concatenate([motion_raw, velocity], axis=-1)  # [200, 534]

        # Get object BPS
        seg_info = file_map[idx_str]  # e.g. "scene_01__A001++seq__xxx__seg0"
        seg_parts = seg_info.rsplit('__seg', 1)
        seq_id = seg_parts[0]
        seg_idx = int(seg_parts[1]) if len(seg_parts) > 1 else 0

        obj_ids = get_segment_objects(seq_id, seg_idx, prog_dir)
        bps_array = np.zeros((MAX_OBJECTS, NUM_BPS_POINTS), dtype=np.float32)

        for j, obj_id in enumerate(obj_ids[:MAX_OBJECTS]):
            bps = get_bps(obj_id)
            if bps is not None:
                bps_array[j] = bps
                n_bps_found += 1

        # Save V4 outputs
        out_idx = f"{n_success:06d}"
        np.save(os.path.join(args.output_root, 'motion_dynamic', f'{out_idx}.npy'), dynamic)
        np.save(os.path.join(args.output_root, 'bps', f'{out_idx}.npy'), bps_array)

        # Copy state arrays and text
        state = np.load(v3_state_path)
        np.save(os.path.join(args.output_root, 'state_arrays', f'{out_idx}.npy'), state)

        if os.path.exists(v3_text_path):
            with open(v3_text_path) as f:
                text = f.read()
            with open(os.path.join(args.output_root, 'texts', f'{out_idx}.txt'), 'w') as f:
                f.write(text)

        if os.path.exists(v3_clip_path):
            clip_emb = np.load(v3_clip_path)
            np.save(os.path.join(args.output_root, 'clip_embeddings', f'{out_idx}.npy'), clip_emb)

        file_names_out.append(f"{out_idx}\t{seg_info}")
        all_dynamics.append(dynamic)
        n_success += 1

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(sample_indices)} processed, {len(bps_cache)} unique objects cached")

    print(f"\nProcessed {n_success} samples, {n_bps_found} BPS embeddings computed")
    print(f"Unique objects with BPS: {len(bps_cache)}")

    if not all_dynamics:
        print("No samples! Exiting.")
        return

    # Compute normalization for dynamic features (534D)
    # Only normalize position+velocity, NOT BPS (static, separate input)
    all_dyn = np.stack(all_dynamics, axis=0)  # [N, 200, 534]
    mean_dyn = all_dyn.reshape(-1, 534).mean(axis=0)
    std_dyn = all_dyn.reshape(-1, 534).std(axis=0)
    std_dyn[std_dyn < 1e-8] = 1.0

    np.save(os.path.join(args.output_root, 'Mean_flow.npy'), mean_dyn.astype(np.float32))
    np.save(os.path.join(args.output_root, 'Std_flow.npy'), std_dyn.astype(np.float32))

    # Write file names
    with open(os.path.join(args.output_root, 'file_names.txt'), 'w') as f:
        for fn in file_names_out:
            f.write(fn + '\n')

    # Train/test split (same random seed as V3)
    n_train = int(n_success * args.train_ratio)
    indices = list(range(n_success))
    np.random.seed(42)
    np.random.shuffle(indices)

    with open(os.path.join(args.output_root, 'train_flow.txt'), 'w') as f:
        for i in indices[:n_train]:
            f.write(f"{i:06d}\n")
    with open(os.path.join(args.output_root, 'test_flow.txt'), 'w') as f:
        for i in indices[n_train:]:
            f.write(f"{i:06d}\n")

    # Metadata
    metadata = {
        "feature_dim_dynamic": 534,
        "feature_dim_position": 267,
        "feature_dim_velocity": 267,
        "bps_dim": NUM_BPS_POINTS,
        "num_entities": 7,
        "max_objects": MAX_OBJECTS,
        "target_len": 200,
        "total_samples": n_success,
        "train_samples": n_train,
        "test_samples": n_success - n_train,
        "unique_objects_with_bps": len(bps_cache),
    }
    with open(os.path.join(args.output_root, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nV4 dataset saved to {args.output_root}/")
    print(f"  Dynamic: [200, 534] ({n_success} samples)")
    print(f"  BPS: [4, 1024] ({len(bps_cache)} unique objects)")
    print(f"  Train: {n_train}, Test: {n_success - n_train}")


if __name__ == "__main__":
    main()
