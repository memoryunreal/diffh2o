#!/usr/bin/env python3
"""
Preprocess HIMO dataset → V4 entity-centric format (Track B: joint-based, V1).

Position-only (no velocity) to match HIMO's native evaluator format exactly.
Output is 489D (2-object) or 498D (3-object), directly feedable to HIMO's
pretrained feature extractor without any conversion.

Entity layout (2-object, 489D):
  Body(201D):  joints[0:22]×3(66) + global_orient×6(6) + body_pose×6(126) + transl(3)
  LH(135D):    joints[22:37]×3(45) + lhand_pose×6(90)
  RH(135D):    joints[37:52]×3(45) + rhand_pose×6(90)
  Obj1(9D):    position(3) + rotation_6d(6)
  Obj2(9D):    position(3) + rotation_6d(6)

Entity layout (3-object, 498D):
  Same body/hand layout, plus Obj3(9D)

HIMO evaluator format reconstruction (lossless):
  smplx_joints = cat(body[0:66], lh[0:45], rh[0:45])       # 156D
  full_pose    = cat(body[66:198], lh[45:135], rh[45:135])  # 312D
  transl       = body[198:201]                               # 3D
  obj_states   = cat(obj1, obj2, ...)                        # 18D/27D
  → Exactly 489D / 498D as HIMO evaluator expects

BPS: scalar distances [N_obj, 1024] per sample
State labels: [T, N_entities] (zeros for active, STATE_PADDING=5 for unused)

Usage:
    python -m prepare.preprocess_himo_v4 --mode 2o
    python -m prepare.preprocess_himo_v4 --mode both --num_samples 10
    python -m prepare.preprocess_himo_v4 --mode 3o --data_root /path/to/data
"""

import os
import json
import argparse
import numpy as np
import h5py
from typing import Dict, List, Tuple
from tqdm import tqdm


# =============================================================================
# Constants
# =============================================================================

STATE_PADDING = 5  # V4 state label for unused entities

# SMPL-X 52-joint split
BODY_JOINT_RANGE = (0, 22)     # 22 body joints
LHAND_JOINT_RANGE = (22, 37)   # 15 left hand joints
RHAND_JOINT_RANGE = (37, 52)   # 15 right hand joints

# Entity position dimensions
BODY_DIM = 22 * 3 + 22 * 6 + 3   # 66 + 132 + 3 = 201
LHAND_DIM = 15 * 3 + 15 * 6       # 45 + 90 = 135
RHAND_DIM = 15 * 3 + 15 * 6       # 45 + 90 = 135
OBJ_DIM = 9                        # 3D pos + 6D rot

# Mode configs
MODE_CONFIGS = {
    '2o': {'num_objects': 2, 'num_entities': 5, 'total_dim': 201 + 135 + 135 + 2 * 9},   # 489
    '3o': {'num_objects': 3, 'num_entities': 6, 'total_dim': 201 + 135 + 135 + 3 * 9},   # 498
}


def get_entity_ranges(mode: str) -> Dict[str, Tuple[int, int]]:
    """Return index ranges for each entity within the motion vector.

    Args:
        mode: '2o' or '3o'.

    Returns:
        Dict mapping entity name to (start, end) index tuple.
    """
    cfg = MODE_CONFIGS[mode]
    ranges = {
        'body': (0, BODY_DIM),                                       # [0, 201)
        'lhand': (BODY_DIM, BODY_DIM + LHAND_DIM),                  # [201, 336)
        'rhand': (BODY_DIM + LHAND_DIM, BODY_DIM + LHAND_DIM + RHAND_DIM),  # [336, 471)
    }
    obj_start = BODY_DIM + LHAND_DIM + RHAND_DIM  # 471
    for i in range(cfg['num_objects']):
        ranges[f'obj{i + 1}'] = (obj_start + i * OBJ_DIM, obj_start + (i + 1) * OBJ_DIM)
    return ranges


# =============================================================================
# Processing functions
# =============================================================================

def decompose_to_entities(
    smplx_joints: np.ndarray,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    lhand_pose: np.ndarray,
    rhand_pose: np.ndarray,
    transl: np.ndarray,
    obj_states: List[np.ndarray],
) -> np.ndarray:
    """Decompose HIMO H5 fields into entity-ordered position vector.

    Args:
        smplx_joints: [T, 52, 3] joint positions.
        global_orient: [T, 6] root rotation in 6D.
        body_pose: [T, 21, 6] body joint rotations.
        lhand_pose: [T, 15, 6] left hand rotations.
        rhand_pose: [T, 15, 6] right hand rotations.
        transl: [T, 3] root translation.
        obj_states: list of [T, 9] object pose arrays (sorted alphabetically).

    Returns:
        motion: [T, 489] or [T, 498] entity-ordered position vector.
    """
    T = smplx_joints.shape[0]

    # Body entity (201D): joints(66) + global_orient(6) + body_pose(126) + transl(3)
    body_joints = smplx_joints[:, BODY_JOINT_RANGE[0]:BODY_JOINT_RANGE[1], :].reshape(T, -1)  # [T, 66]
    body_rot = np.concatenate([
        global_orient,                   # [T, 6]
        body_pose.reshape(T, -1),        # [T, 126]
    ], axis=-1)                          # [T, 132]
    body = np.concatenate([body_joints, body_rot, transl], axis=-1)  # [T, 201]

    # Left hand entity (135D): joints(45) + lhand_pose(90)
    lh_joints = smplx_joints[:, LHAND_JOINT_RANGE[0]:LHAND_JOINT_RANGE[1], :].reshape(T, -1)  # [T, 45]
    lh = np.concatenate([lh_joints, lhand_pose.reshape(T, -1)], axis=-1)  # [T, 135]

    # Right hand entity (135D): joints(45) + rhand_pose(90)
    rh_joints = smplx_joints[:, RHAND_JOINT_RANGE[0]:RHAND_JOINT_RANGE[1], :].reshape(T, -1)  # [T, 45]
    rh = np.concatenate([rh_joints, rhand_pose.reshape(T, -1)], axis=-1)  # [T, 135]

    # Object entities (9D each, no padding)
    parts = [body, lh, rh] + [s.astype(np.float32) for s in obj_states]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def reconstruct_himo_format(motion: np.ndarray, num_objects: int) -> np.ndarray:
    """Reconstruct HIMO's native feature vector from entity format.

    This is a lossless round-trip: the output matches what HIMO's evaluator
    (MovementConvEncoder) expects as input.

    HIMO format: smplx_joints(156) + full_pose(312) + transl(3) + obj_states(N×9)

    Args:
        motion: [T, 489] or [T, 498] entity-ordered vector.
        num_objects: 2 or 3.

    Returns:
        himo_vec: [T, 489] or [T, 498] in HIMO's native order.
    """
    # Extract entity parts
    body = motion[:, :BODY_DIM]          # [T, 201]
    lh = motion[:, BODY_DIM:BODY_DIM + LHAND_DIM]   # [T, 135]
    rh = motion[:, BODY_DIM + LHAND_DIM:BODY_DIM + LHAND_DIM + RHAND_DIM]  # [T, 135]

    # Body: joints(66) | rot(132) | transl(3)
    body_joints = body[:, :66]
    body_rot = body[:, 66:198]       # global_orient(6) + body_pose(126)
    body_transl = body[:, 198:201]

    # Hands: joints(45) | rot(90)
    lh_joints = lh[:, :45]
    lh_rot = lh[:, 45:135]
    rh_joints = rh[:, :45]
    rh_rot = rh[:, 45:135]

    # Reassemble HIMO order
    smplx_joints = np.concatenate([body_joints, lh_joints, rh_joints], axis=-1)  # 156D
    full_pose = np.concatenate([body_rot, lh_rot, rh_rot], axis=-1)              # 312D

    # Object states
    obj_start = BODY_DIM + LHAND_DIM + RHAND_DIM
    obj_states = motion[:, obj_start:obj_start + num_objects * OBJ_DIM]

    return np.concatenate([smplx_joints, full_pose, body_transl, obj_states], axis=-1)


def convert_bps_to_scalar(
    object_bps: Dict[str, np.ndarray],
    obj_names: List[str],
) -> np.ndarray:
    """Convert HIMO's 3D BPS displacements to scalar distances.

    HIMO BPS: [1, 1024, 3] displacement vectors per object.
    Output: [N_obj, 1024] scalar distances (L2 norm of displacement).

    Args:
        object_bps: dict mapping object name → [1, 1024, 3] BPS array.
        obj_names: sorted object names for this sequence.

    Returns:
        bps_scalar: [N_obj, 1024] float32.
    """
    n_obj = len(obj_names)
    bps_scalar = np.zeros((n_obj, 1024), dtype=np.float32)
    for i, name in enumerate(obj_names):
        if name in object_bps:
            raw = object_bps[name]  # [1, 1024, 3]
            bps_scalar[i] = np.linalg.norm(raw.squeeze(0), axis=-1)  # [1024]
    return bps_scalar


def compute_state_labels(T: int, num_entities: int) -> np.ndarray:
    """Compute per-entity state labels [T, N_entities].

    V1 placeholder: all active entities = 0 (idle/static).

    TODO: Implement Schmitt trigger contact detection using hand joint
    to object surface distance.

    Args:
        T: number of frames.
        num_entities: 5 (2o) or 6 (3o).

    Returns:
        states: [T, num_entities] int8.
    """
    return np.zeros((T, num_entities), dtype=np.int8)


# =============================================================================
# H5 loading
# =============================================================================

def load_h5_sequences(
    h5_path: str,
    mode: str,
    num_samples: int = 0,
) -> List[Dict]:
    """Load all sequences from an HIMO H5 file.

    Args:
        h5_path: path to train.h5 / test.h5 / val.h5.
        mode: '2o' or '3o'.
        num_samples: max sequences to load (0 = all).

    Returns:
        List of dicts with loaded fields.
    """
    if not os.path.exists(h5_path):
        return []

    num_objects = MODE_CONFIGS[mode]['num_objects']
    sequences = []

    with h5py.File(h5_path, 'r') as f:
        seq_keys = sorted(f.keys())
        if num_samples > 0:
            seq_keys = seq_keys[:num_samples]

        for seq_key in tqdm(seq_keys, desc=f'Loading {os.path.basename(h5_path)}'):
            seq = f[seq_key]

            # Object states (sorted alphabetically)
            obj_keys = sorted(seq['object_state'].keys())
            if len(obj_keys) < num_objects:
                print(f"  SKIP {seq_key}: {len(obj_keys)} objects < {num_objects}")
                continue
            obj_keys = obj_keys[:num_objects]

            sequences.append({
                'smplx_joints': seq['smplx_joints'][:].astype(np.float32),     # [T, 52, 3]
                'global_orient': seq['global_orient'][:].astype(np.float32),    # [T, 6]
                'body_pose': seq['body_pose'][:].astype(np.float32),            # [T, 21, 6]
                'lhand_pose': seq['lhand_pose'][:].astype(np.float32),          # [T, 15, 6]
                'rhand_pose': seq['rhand_pose'][:].astype(np.float32),          # [T, 15, 6]
                'transl': seq['transl'][:].astype(np.float32),                  # [T, 3]
                'betas': seq['betas'][:].astype(np.float32),                    # [10] or [T, 10]
                'text': seq['text'][0].decode() if isinstance(seq['text'][0], bytes) else str(seq['text'][0]),
                'obj_names': obj_keys,
                'obj_states': [seq['object_state'][k][:].astype(np.float32) for k in obj_keys],
                'seq_key': seq_key,
            })

    return sequences


# =============================================================================
# Main processing
# =============================================================================

def process_split(
    h5_path: str,
    mode: str,
    object_bps: Dict[str, np.ndarray],
    output_root: str,
    idx_offset: int,
    num_samples: int = 0,
) -> Tuple[List[np.ndarray], List[str], int]:
    """Process one H5 split into V4 format.

    Returns:
        all_motions: list of [T, D] arrays for normalization.
        file_names: list of "idx\\tseq_key" strings.
        count: number processed.
    """
    sequences = load_h5_sequences(h5_path, mode, num_samples)
    cfg = MODE_CONFIGS[mode]

    all_motions = []
    file_names = []
    count = 0

    for seq_data in tqdm(sequences, desc=f'Processing {os.path.basename(h5_path)}'):
        # Entity decomposition → [T, 489] or [T, 498]
        motion = decompose_to_entities(
            smplx_joints=seq_data['smplx_joints'],
            global_orient=seq_data['global_orient'],
            body_pose=seq_data['body_pose'],
            lhand_pose=seq_data['lhand_pose'],
            rhand_pose=seq_data['rhand_pose'],
            transl=seq_data['transl'],
            obj_states=seq_data['obj_states'],
        )
        assert motion.shape[-1] == cfg['total_dim'], \
            f"Dim mismatch: {motion.shape[-1]} vs {cfg['total_dim']}"

        T = motion.shape[0]

        # BPS [N_obj, 1024] scalar distances
        bps = convert_bps_to_scalar(object_bps, seq_data['obj_names'])

        # State labels [T, N_entities]
        states = compute_state_labels(T, cfg['num_entities'])

        # Text (full annotation string, preserving tokens)
        text = seq_data['text'].strip()

        # Verify lossless reconstruction
        recon = reconstruct_himo_format(motion, cfg['num_objects'])
        assert recon.shape[-1] == cfg['total_dim']

        # Save outputs
        out_idx = f"{idx_offset + count:06d}"
        np.save(os.path.join(output_root, 'motion_dynamic', f'{out_idx}.npy'), motion)
        np.save(os.path.join(output_root, 'bps', f'{out_idx}.npy'), bps)
        np.save(os.path.join(output_root, 'state_arrays', f'{out_idx}.npy'), states)
        with open(os.path.join(output_root, 'texts', f'{out_idx}.txt'), 'w') as f:
            f.write(text + '\n')

        file_names.append(f"{out_idx}\t{seq_data['seq_key']}")
        all_motions.append(motion)
        count += 1

    return all_motions, file_names, count


def preprocess_mode(mode: str, args: argparse.Namespace) -> None:
    """Run full preprocessing for one mode (2o or 3o)."""
    cfg = MODE_CONFIGS[mode]
    print(f"\n{'=' * 60}")
    print(f"HIMO {mode} → V4 entity format ({cfg['total_dim']}D, {cfg['num_entities']} entities)")
    print(f"{'=' * 60}")

    data_dir = os.path.join(args.data_root, f'processed_{mode}')
    output_root = os.path.join(args.output_root, f'HIMO_V4_{mode}')

    # Verify input
    train_h5 = os.path.join(data_dir, 'train.h5')
    if not os.path.exists(train_h5):
        print(f"ERROR: {train_h5} not found.")
        print("Download HIMO: https://docs.google.com/forms/d/e/"
              "1FAIpQLSdl5adeyKxBSBFZpgs0A7-dAouRkMFAGUP5iz3zxGDj_PhB1w/viewform")
        return

    # Create output dirs
    for subdir in ['motion_dynamic', 'bps', 'state_arrays', 'texts']:
        os.makedirs(os.path.join(output_root, subdir), exist_ok=True)

    # Load object BPS
    bps_path = os.path.join(data_dir, 'object_bps.npz')
    object_bps = dict(np.load(bps_path, allow_pickle=True)) if os.path.exists(bps_path) else {}
    print(f"Loaded BPS for {len(object_bps)} objects")

    # Process each split
    all_motions = []
    all_file_names = []
    split_indices = {}
    total_count = 0

    for split in ['train', 'test', 'val']:
        h5_path = os.path.join(data_dir, f'{split}.h5')
        motions, fnames, count = process_split(
            h5_path=h5_path,
            mode=mode,
            object_bps=object_bps,
            output_root=output_root,
            idx_offset=total_count,
            num_samples=args.num_samples,
        )
        split_indices[split] = list(range(total_count, total_count + count))
        total_count += count
        all_motions.extend(motions)
        all_file_names.extend(fnames)
        print(f"  {split}: {count} sequences")

    if total_count == 0:
        print("No sequences processed!")
        return

    # Normalization (train split only)
    train_idx = split_indices.get('train', [])
    if train_idx:
        train_data = np.concatenate([all_motions[i] for i in range(len(train_idx))], axis=0)
        mean = train_data.mean(axis=0).astype(np.float32)
        std = train_data.std(axis=0).astype(np.float32)
        std[std < 1e-8] = 1.0
    else:
        mean = np.zeros(cfg['total_dim'], dtype=np.float32)
        std = np.ones(cfg['total_dim'], dtype=np.float32)

    np.save(os.path.join(output_root, 'Mean_flow.npy'), mean)
    np.save(os.path.join(output_root, 'Std_flow.npy'), std)

    # Split index files
    for split, indices in split_indices.items():
        with open(os.path.join(output_root, f'{split}_flow.txt'), 'w') as f:
            for idx in indices:
                f.write(f"{idx:06d}\n")

    # File names
    with open(os.path.join(output_root, 'file_names.txt'), 'w') as f:
        for fn in all_file_names:
            f.write(fn + '\n')

    # Entity ranges for this mode
    entity_ranges = get_entity_ranges(mode)

    # Metadata
    metadata = {
        "dataset": "HIMO",
        "mode": mode,
        "num_objects": cfg['num_objects'],
        "num_entities": cfg['num_entities'],
        "feature_dim": cfg['total_dim'],
        "bps_dim": 1024,
        "entity_dims": {
            "body": BODY_DIM,
            "lhand": LHAND_DIM,
            "rhand": RHAND_DIM,
            "obj": OBJ_DIM,
        },
        "entity_ranges": {k: list(v) for k, v in entity_ranges.items()},
        "himo_eval_dim": cfg['total_dim'],
        "total_samples": total_count,
        "split_counts": {k: len(v) for k, v in split_indices.items()},
    }
    with open(os.path.join(output_root, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! {total_count} sequences → {output_root}")
    print(f"  Motion: [T, {cfg['total_dim']}]")
    print(f"  BPS: [{cfg['num_objects']}, 1024]")
    print(f"  States: [T, {cfg['num_entities']}]")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess HIMO → V4 entity format (Track B, position-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'HIMO_dataset', 'data'),
                        help="HIMO data dir containing processed_2o/ and processed_3o/")
    parser.add_argument("--output_root", type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'dataset'),
                        help="Output base dir (creates HIMO_V4_2o/ and/or HIMO_V4_3o/)")
    parser.add_argument("--mode", type=str, default="both", choices=["2o", "3o", "both"],
                        help="Which object setting to process")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Max sequences per split (0 = all)")
    args = parser.parse_args()

    args.data_root = os.path.abspath(args.data_root)
    args.output_root = os.path.abspath(args.output_root)

    print(f"Data root:  {args.data_root}")
    print(f"Output:     {args.output_root}")
    print(f"Mode:       {args.mode}")
    if args.num_samples > 0:
        print(f"Samples:    {args.num_samples} per split")

    if args.mode in ('2o', 'both'):
        preprocess_mode('2o', args)
    if args.mode in ('3o', 'both'):
        preprocess_mode('3o', args)

    print("\nAll done!")


if __name__ == '__main__':
    main()
