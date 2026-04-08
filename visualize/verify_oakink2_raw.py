#!/usr/bin/env python3
"""
Quick verification: Load raw_smplx from annotation directly.
Bypasses our preprocessing to verify hand poses work.

Usage:
    python visualize/verify_oakink2_raw.py

    # With specific annotation file:
    python visualize/verify_oakink2_raw.py --anno_path /path/to/anno.pkl
"""

import pickle
import numpy as np
import torch
import os
import argparse

# Paths
ANNO_DIR = "/hhd4/lizhe/dataset/OakInk2/data/anno_preview"
SMPLX_MODEL_PATH = "/hhd4/lizhe/code/InterAct/models"


def load_annotation(anno_path=None):
    """Load annotation pickle for a sequence."""
    if anno_path is None:
        # Find annotation file
        anno_files = [f for f in os.listdir(ANNO_DIR) if f.endswith('.pkl')]
        if not anno_files:
            raise FileNotFoundError(f"No annotation files in {ANNO_DIR}")
        anno_path = os.path.join(ANNO_DIR, anno_files[0])

    print(f"Loading annotation: {anno_path}")

    with open(anno_path, 'rb') as f:
        anno = pickle.load(f)

    return anno


def verify_raw_smplx(anno, frame_id=0):
    """Extract and display raw_smplx data structure."""
    raw_smplx = anno['raw_smplx'][frame_id]
    raw_mano = anno['raw_mano'][frame_id]

    print("\n" + "="*60)
    print("=== raw_smplx keys and shapes ===")
    print("="*60)
    for k, v in sorted(raw_smplx.items()):
        if hasattr(v, 'shape'):
            print(f"  {k:20s}: {str(v.shape):20s} dtype={v.dtype}")
        else:
            print(f"  {k:20s}: {type(v)}")

    print("\n" + "="*60)
    print("=== raw_mano keys and shapes ===")
    print("="*60)
    for k, v in sorted(raw_mano.items()):
        if hasattr(v, 'shape'):
            print(f"  {k:20s}: {str(v.shape):20s} dtype={v.dtype}")
        else:
            print(f"  {k:20s}: {type(v)}")

    # Compare hand poses
    print("\n" + "="*60)
    print("=== Hand pose comparison ===")
    print("="*60)

    if 'left_hand_pose' in raw_smplx:
        smplx_lh = raw_smplx['left_hand_pose']
        print(f"SMPL-X left_hand_pose: {smplx_lh.shape}")
        print(f"  First joint quat: {smplx_lh[0, 0, :].numpy()}")

        # Check if it's near identity (flat hand)
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
        dist_to_identity = np.linalg.norm(smplx_lh[0, 0, :].numpy() - identity_quat)
        print(f"  Distance from identity: {dist_to_identity:.4f}")

    if 'lh__pose_coeffs' in raw_mano:
        mano_lh = raw_mano['lh__pose_coeffs']
        print(f"\nMANO lh__pose_coeffs: {mano_lh.shape}")
        print(f"  Joint 0 (wrist) quat: {mano_lh[0, 0, :].numpy()}")
        print(f"  Joint 1 quat: {mano_lh[0, 1, :].numpy()}")

        # Check if they're related
        if 'left_hand_pose' in raw_smplx:
            smplx_lh = raw_smplx['left_hand_pose']
            if mano_lh.shape[1] == 16 and smplx_lh.shape[1] == 15:
                mano_skip_wrist = mano_lh[:, 1:, :]  # Skip joint 0 (wrist)

                # Convert to numpy for comparison
                mano_np = mano_skip_wrist.numpy() if hasattr(mano_skip_wrist, 'numpy') else mano_skip_wrist
                smplx_np = smplx_lh.numpy() if hasattr(smplx_lh, 'numpy') else smplx_lh

                match = np.allclose(mano_np, smplx_np, atol=1e-5)
                max_diff = np.abs(mano_np - smplx_np).max()
                print(f"\nMANO[1:16] == SMPLX[0:15]? {match}")
                print(f"Max difference: {max_diff:.6f}")

    return raw_smplx, raw_mano


def quat_to_aa(quat):
    """Convert quaternion [w,x,y,z] to axis-angle.

    Args:
        quat: numpy array of shape (..., 4) with [w,x,y,z] format

    Returns:
        axis-angle array of shape (..., 3)
    """
    from scipy.spatial.transform import Rotation as R

    # Convert to numpy if tensor
    if hasattr(quat, 'numpy'):
        quat = quat.numpy()

    # scipy expects [x,y,z,w] format
    quat_scipy = np.concatenate([quat[..., 1:], quat[..., :1]], axis=-1)

    # Reshape for batch processing
    original_shape = quat.shape[:-1]
    quat_flat = quat_scipy.reshape(-1, 4)

    r = R.from_quat(quat_flat)
    aa_flat = r.as_rotvec()

    aa = aa_flat.reshape(original_shape + (3,))
    return aa


def render_with_raw_smplx(raw_smplx, output_path="output/verify_raw.png"):
    """Render SMPL-X using raw_smplx data directly."""
    import smplx as smplx_lib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Create SMPL-X model
    print(f"Loading SMPL-X model from: {SMPLX_MODEL_PATH}")
    model = smplx_lib.create(
        model_path=SMPLX_MODEL_PATH,
        model_type='smplx',
        gender='neutral',
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(device)

    # Convert all parameters from quaternion to axis-angle
    world_tsl = raw_smplx['world_tsl'].squeeze().numpy()
    world_rot = quat_to_aa(raw_smplx['world_rot'].squeeze())
    body_pose = quat_to_aa(raw_smplx['body_pose'].squeeze()).reshape(-1)
    left_hand = quat_to_aa(raw_smplx['left_hand_pose'].squeeze()).reshape(-1)
    right_hand = quat_to_aa(raw_smplx['right_hand_pose'].squeeze()).reshape(-1)

    print("\n" + "="*60)
    print("=== Converted parameters (axis-angle) ===")
    print("="*60)
    print(f"world_tsl: {world_tsl}")
    print(f"world_rot: {world_rot}")
    print(f"body_pose shape: {body_pose.shape} (expected: 63 = 21*3)")
    print(f"left_hand shape: {left_hand.shape} (expected: 45 = 15*3)")
    print(f"right_hand shape: {right_hand.shape} (expected: 45 = 15*3)")

    # Check if hand poses are near zero (flat hands)
    left_hand_norm = np.linalg.norm(left_hand)
    right_hand_norm = np.linalg.norm(right_hand)
    print(f"\nLeft hand pose norm: {left_hand_norm:.4f}")
    print(f"Right hand pose norm: {right_hand_norm:.4f}")
    if left_hand_norm < 0.1:
        print("  WARNING: Left hand pose is near zero (flat hand)!")
    if right_hand_norm < 0.1:
        print("  WARNING: Right hand pose is near zero (flat hand)!")

    # Run SMPL-X
    print("\nRunning SMPL-X forward pass...")
    with torch.no_grad():
        output = model(
            transl=torch.tensor(world_tsl, dtype=torch.float32, device=device).unsqueeze(0),
            global_orient=torch.tensor(world_rot, dtype=torch.float32, device=device).unsqueeze(0),
            body_pose=torch.tensor(body_pose, dtype=torch.float32, device=device).unsqueeze(0),
            left_hand_pose=torch.tensor(left_hand, dtype=torch.float32, device=device).unsqueeze(0),
            right_hand_pose=torch.tensor(right_hand, dtype=torch.float32, device=device).unsqueeze(0),
        )

        verts = output.vertices.cpu().numpy()[0]
        joints = output.joints.cpu().numpy()[0]

    print("\n" + "="*60)
    print("=== Mesh statistics ===")
    print("="*60)
    print(f"Vertices shape: {verts.shape}")
    print(f"Joints shape: {joints.shape}")
    print(f"\nVertex bounds:")
    print(f"  X range: [{verts[:, 0].min():.4f}, {verts[:, 0].max():.4f}]")
    print(f"  Y range: [{verts[:, 1].min():.4f}, {verts[:, 1].max():.4f}]")
    print(f"  Z range: [{verts[:, 2].min():.4f}, {verts[:, 2].max():.4f}]")
    print(f"\nMin vertex Y (foot position): {verts[:, 1].min():.4f}")
    print(f"Max vertex Y (head position): {verts[:, 1].max():.4f}")
    print(f"Body height: {verts[:, 1].max() - verts[:, 1].min():.4f}")

    # Pelvis position (joint 0)
    print(f"\nPelvis position (joint 0): {joints[0]}")
    print(f"Pelvis Y: {joints[0, 1]:.4f}")

    # Floor alignment suggestion
    floor_offset = -verts[:, 1].min()
    print(f"\nSuggested floor offset: {floor_offset:.4f}")
    print(f"  (Add this to world_tsl[1] to place feet at Y=0)")

    return verts, model.faces, joints


def main():
    parser = argparse.ArgumentParser(description="Verify OakInk2 raw_smplx data")
    parser.add_argument("--anno_path", type=str, default=None,
                       help="Path to annotation pickle file")
    parser.add_argument("--frame_id", type=int, default=None,
                       help="Frame ID to verify (default: first frame)")
    args = parser.parse_args()

    # Load annotation
    anno = load_annotation(args.anno_path)

    # Get available frame IDs
    frame_ids = list(anno['raw_smplx'].keys())
    print(f"\nAvailable frames: {len(frame_ids)}")
    print(f"Frame ID type: {type(frame_ids[0])}")
    print(f"First few frame IDs: {frame_ids[:5]}")

    # Select frame
    if args.frame_id is not None:
        frame_id = args.frame_id
    else:
        frame_id = frame_ids[0]
    print(f"\nUsing frame ID: {frame_id}")

    # Verify data structure
    raw_smplx, raw_mano = verify_raw_smplx(anno, frame_id)

    # Render with raw_smplx
    verts, faces, joints = render_with_raw_smplx(raw_smplx)

    print("\n" + "="*60)
    print("=== VERIFICATION COMPLETE ===")
    print("="*60)
    print("\nNext steps based on results:")
    print("1. If hand poses show articulation (non-zero norm), raw_smplx works")
    print("2. If hands are flat, check quaternion conversion")
    print("3. For floor alignment, use the suggested floor offset")


if __name__ == "__main__":
    main()
