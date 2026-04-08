#!/usr/bin/env python3
"""
Visualize OakInk2 preprocessed motion data (398D representation).

This script renders body skeleton + hand meshes from OakInk2 .npy files
and saves the result as a video.

Usage:
    python visualize/visualize_oakink2.py \
        --motion_path dataset/OAKINK2/oakink2_primitive/000000.npy \
        --output_path output/oakink2_vis.mp4

    # With xvfb for headless servers:
    xvfb-run -s "-screen 0 1024x768x24" python visualize/visualize_oakink2.py \
        --motion_path dataset/OAKINK2/oakink2_primitive/000000.npy \
        --output_path output/oakink2_vis.mp4
"""

import argparse
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# aitviewer imports
import aitviewer
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.lines import Lines

# pytorch3d for rotation conversion
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix, quaternion_to_axis_angle as pt3d_quat_to_aa

# Local imports - vis_utils no longer needed for MANO watertight processing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# OakInk2 398D Feature Layout Constants
# =============================================================================

# Body (135D)
IDX_BODY_TRANSL = (0, 3)          # 3D translation
IDX_BODY_ORIENT = (3, 9)          # 6D rotation
IDX_BODY_POSE = (9, 135)          # 21 joints × 6D = 126D

# Left hand PCA (30D)
IDX_LHAND_POS = (135, 138)        # 3D position
IDX_LHAND_ORIENT = (138, 144)     # 6D rotation
IDX_LHAND_PCA = (144, 165)        # 21 PCA coefficients

# Right hand PCA (30D)
IDX_RHAND_POS = (165, 168)        # 3D position
IDX_RHAND_ORIENT = (168, 174)     # 6D rotation
IDX_RHAND_PCA = (174, 195)        # 21 PCA coefficients

# Left hand quaternions (67D) - auxiliary
IDX_LHAND_TRANSL = (195, 198)     # 3D translation
IDX_LHAND_QUAT = (198, 262)       # 16 joints × 4D = 64D

# Right hand quaternions (67D) - auxiliary
IDX_RHAND_TRANSL = (262, 265)     # 3D translation
IDX_RHAND_QUAT = (265, 329)       # 16 joints × 4D = 64D

# SDF (42D) - currently zeros
IDX_SDF_LEFT = (329, 350)         # 21D
IDX_SDF_RIGHT = (350, 371)        # 21D

# Objects (27D) - 3 objects × 9D
IDX_OBJ1_POS = (371, 374)
IDX_OBJ1_ROT = (374, 380)
IDX_OBJ2_POS = (380, 383)
IDX_OBJ2_ROT = (383, 389)
IDX_OBJ3_POS = (389, 392)
IDX_OBJ3_ROT = (392, 398)

# Colors
BODY_MESH_COLOR = (0.7, 0.7, 0.8, 1.0)      # Light gray-blue for body mesh
BODY_JOINT_COLOR = (0.2, 0.6, 0.8, 1.0)     # Blue for body joints
BODY_BONE_COLOR = (0.3, 0.7, 0.9, 0.8)      # Light blue for bones
HAND_L_COLOR = (158/255, 121/255, 119/255, 1.0)  # Reddish for left hand
HAND_R_COLOR = (121/255, 119/255, 158/255, 1.0)  # Purplish for right hand

# SMPL-X body joint connections (skeleton bones)
# Based on SMPL-X kinematic tree (22 body joints, excluding hands/face)
SMPLX_BODY_BONES = [
    (0, 1), (0, 2), (0, 3),      # Pelvis to legs and spine
    (1, 4), (2, 5), (3, 6),      # Upper legs and spine
    (4, 7), (5, 8), (6, 9),      # Knees and mid-spine
    (7, 10), (8, 11), (9, 12),   # Ankles and upper spine
    (12, 13), (12, 14),          # Neck to shoulders
    (13, 16), (14, 17),          # Shoulders to elbows
    (16, 18), (17, 19),          # Elbows to wrists
    (12, 15),                     # Neck to head
]

# SMPL-X hand joint connections (15 joints per hand, starting from wrist)
# Joints 25-39 = left hand, 40-54 = right hand
# Wrist (20/21) -> finger bases -> finger tips
# Each finger: base -> middle -> tip
SMPLX_LHAND_BONES = [
    (20, 25), (25, 26), (26, 27),    # Left index: wrist->base->mid->tip
    (20, 28), (28, 29), (29, 30),    # Left middle
    (20, 31), (31, 32), (32, 33),    # Left pinky
    (20, 34), (34, 35), (35, 36),    # Left ring
    (20, 37), (37, 38), (38, 39),    # Left thumb
]

SMPLX_RHAND_BONES = [
    (21, 40), (40, 41), (41, 42),    # Right index: wrist->base->mid->tip
    (21, 43), (43, 44), (44, 45),    # Right middle
    (21, 46), (46, 47), (47, 48),    # Right pinky
    (21, 49), (49, 50), (50, 51),    # Right ring
    (21, 52), (52, 53), (53, 54),    # Right thumb
]

# Colors for skeleton visualization
SKELETON_JOINT_COLOR = (0.2, 0.6, 0.9, 1.0)     # Blue for joints
SKELETON_BONE_COLOR = (0.4, 0.8, 1.0, 0.9)      # Light blue for bones
LHAND_JOINT_COLOR = (0.9, 0.3, 0.3, 1.0)        # Red for left hand
RHAND_JOINT_COLOR = (0.3, 0.9, 0.3, 1.0)        # Green for right hand


def quat_wxyz_to_aa(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w,x,y,z] to axis-angle using scipy.

    Args:
        quat: (..., 4) array in [w,x,y,z] format

    Returns:
        (..., 3) axis-angle array
    """
    # scipy expects [x,y,z,w] format
    quat_scipy = np.concatenate([quat[..., 1:], quat[..., :1]], axis=-1)

    # Reshape for batch processing
    original_shape = quat.shape[:-1]
    quat_flat = quat_scipy.reshape(-1, 4)

    r = R.from_quat(quat_flat)
    aa_flat = r.as_rotvec()

    aa = aa_flat.reshape(original_shape + (3,))
    return aa


def load_object_meshes(obj_list: list, obj_mesh_dir: str) -> dict:
    """
    Load object meshes from OBJ files.

    Args:
        obj_list: List of object IDs from annotation
        obj_mesh_dir: Path to object_repair/align_ds directory

    Returns:
        dict mapping obj_id -> (vertices, faces)
    """
    import trimesh
    obj_meshes = {}
    for obj_id in obj_list:
        obj_path = os.path.join(obj_mesh_dir, obj_id, "model.obj")
        if os.path.exists(obj_path):
            mesh = trimesh.load_mesh(obj_path, process=False, skip_materials=True, force="mesh")
            obj_meshes[obj_id] = (mesh.vertices.astype(np.float32), mesh.faces)
            print(f"  Loaded object {obj_id}: {mesh.vertices.shape[0]} vertices")
        else:
            print(f"  Warning: Object mesh not found: {obj_path}")
    return obj_meshes


def create_object_mesh_sequence(obj_meshes: dict, obj_transf: dict, T: int, floor_offset: float = 0.0):
    """
    Apply transforms to object meshes for each frame.

    Args:
        obj_meshes: dict mapping obj_id -> (vertices, faces)
        obj_transf: dict mapping obj_id -> (T, 4, 4) transforms
        T: number of frames
        floor_offset: vertical offset to apply (should match body floor alignment)

    Returns:
        list of (obj_id, vertices_sequence, faces) tuples for each object
    """
    object_sequences = []
    for obj_id, (verts, faces) in obj_meshes.items():
        if obj_id not in obj_transf:
            print(f"  Warning: No transform for object {obj_id}")
            continue
        transforms = obj_transf[obj_id]  # (T, 4, 4)

        # Apply transform to vertices for each frame
        verts_seq = np.zeros((T, verts.shape[0], 3), dtype=np.float32)
        for t in range(T):
            R = transforms[t, :3, :3]  # 3x3 rotation
            trans = transforms[t, :3, 3]  # 3 translation
            verts_seq[t] = (R @ verts.T).T + trans

        # Apply the same floor offset as the body mesh
        if floor_offset != 0.0:
            verts_seq[:, :, 1] += floor_offset

        object_sequences.append((obj_id, verts_seq, faces))
        print(f"  Created sequence for object {obj_id}: {verts_seq.shape}")
    return object_sequences


def load_raw_annotation(anno_path: str, frame_range: tuple = None):
    """
    Load raw OakInk2 annotation and extract SMPL-X parameters directly.

    This bypasses our preprocessed data to verify hand poses work correctly.

    Args:
        anno_path: Path to annotation pickle file
        frame_range: Optional (start, end) tuple to limit frames

    Returns:
        dict with body_transl, body_orient_aa, body_pose_aa, lhand_aa, rhand_aa
    """
    import pickle

    print(f"Loading raw annotation: {anno_path}")
    with open(anno_path, 'rb') as f:
        anno = pickle.load(f)

    frame_ids = sorted(list(anno['raw_smplx'].keys()))
    print(f"  Total frames in annotation: {len(frame_ids)}")

    if frame_range:
        start, end = frame_range
        frame_ids = frame_ids[start:end]
    print(f"  Processing {len(frame_ids)} frames")

    T = len(frame_ids)
    body_transl = np.zeros((T, 3), dtype=np.float32)
    body_orient_aa = np.zeros((T, 3), dtype=np.float32)
    body_pose_aa = np.zeros((T, 63), dtype=np.float32)  # 21 joints × 3
    lhand_aa = np.zeros((T, 45), dtype=np.float32)       # 15 joints × 3
    rhand_aa = np.zeros((T, 45), dtype=np.float32)       # 15 joints × 3
    body_shape = np.zeros((T, 300), dtype=np.float32)    # 300 betas

    # Store sample quaternions for debug
    sample_body_quat = None

    for i, frame_id in enumerate(frame_ids):
        raw_smplx = anno['raw_smplx'][frame_id]

        # Body shape (300D betas)
        body_shape[i] = raw_smplx['body_shape'].squeeze().numpy()

        # Translation
        body_transl[i] = raw_smplx['world_tsl'].squeeze().numpy()

        # Global orientation (quaternion -> axis-angle)
        world_rot_quat = raw_smplx['world_rot'].squeeze().numpy()  # (4,) [w,x,y,z]
        body_orient_aa[i] = quat_wxyz_to_aa(world_rot_quat)

        # Body pose (21 joints quaternion -> axis-angle)
        body_quat = raw_smplx['body_pose'].squeeze().numpy()  # (21, 4)
        body_pose_aa[i] = quat_wxyz_to_aa(body_quat).reshape(-1)  # (63,)

        # Store first frame's quaternion for debug
        if i == 0:
            sample_body_quat = body_quat

        # Hand poses from raw_smplx (15 joints each)
        lhand_quat = raw_smplx['left_hand_pose'].squeeze().numpy()  # (15, 4)
        rhand_quat = raw_smplx['right_hand_pose'].squeeze().numpy()  # (15, 4)
        lhand_aa[i] = quat_wxyz_to_aa(lhand_quat).reshape(-1)  # (45,)
        rhand_aa[i] = quat_wxyz_to_aa(rhand_quat).reshape(-1)  # (45,)

    # Debug output to verify data variation
    print(f"  === Data Verification ===")
    print(f"  body_pose quaternion frame 0, joint 0: {sample_body_quat[0] if sample_body_quat is not None else 'N/A'}")
    print(f"  body_pose_aa frame 0 norm: {np.linalg.norm(body_pose_aa[0]):.4f}")
    print(f"  body_pose_aa frame 50 norm: {np.linalg.norm(body_pose_aa[min(50, T-1)]):.4f}")
    print(f"  Variation (frame 0 vs 50): {np.linalg.norm(body_pose_aa[0] - body_pose_aa[min(50, T-1)]):.4f}")
    print(f"  lhand_aa frame 0 norm: {np.linalg.norm(lhand_aa[0]):.4f}")
    print(f"  rhand_aa frame 0 norm: {np.linalg.norm(rhand_aa[0]):.4f}")
    print(f"  body_shape frame 0 first 10: {body_shape[0, :10]}")

    # Extract object data
    obj_list = anno.get('obj_list', [])
    obj_transf = {}  # Dict[obj_id, np.ndarray(T, 4, 4)]

    print(f"  === Object Data ===")
    print(f"  Object list: {obj_list}")

    for obj_id in obj_list:
        if 'obj_transf' in anno and obj_id in anno['obj_transf']:
            obj_transf[obj_id] = np.zeros((T, 4, 4), dtype=np.float32)
            for i, frame_id in enumerate(frame_ids):
                if frame_id in anno['obj_transf'][obj_id]:
                    obj_transf[obj_id][i] = anno['obj_transf'][obj_id][frame_id]
            print(f"  Loaded transforms for {obj_id}: {obj_transf[obj_id].shape}")

    return {
        'body_transl': body_transl,
        'body_orient_aa': body_orient_aa,
        'body_pose_aa': body_pose_aa,
        'lhand_aa': lhand_aa,
        'rhand_aa': rhand_aa,
        'body_shape': body_shape,
        'frame_ids': frame_ids,
        'obj_list': obj_list,
        'obj_transf': obj_transf,
    }


def create_body_mesh_from_raw(body_transl, body_orient_aa, body_pose_aa,
                               lhand_aa, rhand_aa, body_shape, smplx_model, T,
                               align_floor=True):
    """
    Create body mesh sequence from raw annotation data (already in axis-angle format).

    Args:
        body_transl: (T, 3) body translation
        body_orient_aa: (T, 3) body root orientation (axis-angle)
        body_pose_aa: (T, 63) body pose (21 joints × 3 axis-angle)
        lhand_aa: (T, 45) left hand pose (15 joints × 3 axis-angle)
        rhand_aa: (T, 45) right hand pose (15 joints × 3 axis-angle)
        body_shape: (T, 300) body shape coefficients (betas)
        smplx_model: SMPL-X model instance
        T: number of frames
        align_floor: If True, adjust vertices so feet are at Y=0

    Returns:
        tuple of (vertices, faces, floor_offset) for mesh rendering
    """
    with torch.no_grad():
        transl = torch.tensor(body_transl, dtype=torch.float32, device=device)
        global_orient = torch.tensor(body_orient_aa, dtype=torch.float32, device=device)
        body_pose_t = torch.tensor(body_pose_aa, dtype=torch.float32, device=device)
        lhand_pose_t = torch.tensor(lhand_aa, dtype=torch.float32, device=device)
        rhand_pose_t = torch.tensor(rhand_aa, dtype=torch.float32, device=device)
        betas = torch.tensor(body_shape, dtype=torch.float32, device=device)

        output = smplx_model(
            betas=betas,
            transl=transl,
            global_orient=global_orient,
            body_pose=body_pose_t,
            left_hand_pose=lhand_pose_t,
            right_hand_pose=rhand_pose_t,
        )

        vertices = output.vertices.cpu().numpy()  # (T, V, 3)
        faces = smplx_model.faces  # (F, 3)

    # Align floor: compute minimum Y across all frames and shift up
    floor_offset = 0.0
    if align_floor:
        min_y = vertices[:, :, 1].min()
        floor_offset = -min_y
        print(f"  Floor alignment: min_y={min_y:.4f}, applying offset={floor_offset:.4f}")
        vertices[:, :, 1] += floor_offset

    return vertices, faces, floor_offset


def create_skeleton_from_raw(body_transl, body_orient_aa, body_pose_aa,
                             lhand_aa, rhand_aa, body_shape, smplx_model, T,
                             align_floor=True):
    """
    Create skeleton (joints) sequence from raw annotation data.

    Args:
        body_transl: (T, 3) body translation
        body_orient_aa: (T, 3) body root orientation (axis-angle)
        body_pose_aa: (T, 63) body pose (21 joints × 3 axis-angle)
        lhand_aa: (T, 45) left hand pose (15 joints × 3 axis-angle)
        rhand_aa: (T, 45) right hand pose (15 joints × 3 axis-angle)
        body_shape: (T, 300) body shape coefficients (betas)
        smplx_model: SMPL-X model instance
        T: number of frames
        align_floor: If True, adjust joints so feet are at Y=0

    Returns:
        joints: (T, J, 3) joint positions
        floor_offset: float, the offset applied for floor alignment
    """
    with torch.no_grad():
        transl = torch.tensor(body_transl, dtype=torch.float32, device=device)
        global_orient = torch.tensor(body_orient_aa, dtype=torch.float32, device=device)
        body_pose_t = torch.tensor(body_pose_aa, dtype=torch.float32, device=device)
        lhand_pose_t = torch.tensor(lhand_aa, dtype=torch.float32, device=device)
        rhand_pose_t = torch.tensor(rhand_aa, dtype=torch.float32, device=device)
        betas = torch.tensor(body_shape, dtype=torch.float32, device=device)

        output = smplx_model(
            betas=betas,
            transl=transl,
            global_orient=global_orient,
            body_pose=body_pose_t,
            left_hand_pose=lhand_pose_t,
            right_hand_pose=rhand_pose_t,
        )

        joints = output.joints.cpu().numpy()  # (T, J, 3), SMPL-X has 127 joints
        vertices = output.vertices.cpu().numpy()  # For floor alignment

    # Align floor using vertex minimum (same as mesh)
    floor_offset = 0.0
    if align_floor:
        min_y = vertices[:, :, 1].min()
        floor_offset = -min_y
        print(f"  Floor alignment: min_y={min_y:.4f}, applying offset={floor_offset:.4f}")
        joints[:, :, 1] += floor_offset

    return joints, floor_offset


def parse_oakink2_motion(motion_path: str, denormalize: bool = True, data_mode: str = 'primitive'):
    """
    Parse OakInk2 398D motion vector into body, hand, and object components.

    Args:
        motion_path: Path to .npy file
        denormalize: If True, apply inverse normalization
        data_mode: 'primitive' or 'complex' for correct normalization files

    Returns:
        dict with parsed components
    """
    # Load motion data (shape: T × 398)
    motion = np.load(motion_path)

    # Denormalize if needed
    if denormalize:
        data_root = os.path.dirname(os.path.dirname(motion_path))
        mean_file = os.path.join(data_root, f'Mean_oakink2_{data_mode}.npy')
        std_file = os.path.join(data_root, f'Std_oakink2_{data_mode}.npy')

        if os.path.exists(mean_file) and os.path.exists(std_file):
            mean = np.load(mean_file)
            std = np.load(std_file)
            # Avoid division by zero
            std = np.where(std < 1e-8, 1.0, std)
            motion = motion * std + mean
        else:
            print(f"Warning: Normalization files not found, using raw data")

    # Parse components
    result = {
        # Body (135D total)
        'body_transl': motion[:, IDX_BODY_TRANSL[0]:IDX_BODY_TRANSL[1]],
        'body_orient': motion[:, IDX_BODY_ORIENT[0]:IDX_BODY_ORIENT[1]],
        'body_pose': motion[:, IDX_BODY_POSE[0]:IDX_BODY_POSE[1]],

        # Left hand PCA (30D) - for generation
        'lhand_pos': motion[:, IDX_LHAND_POS[0]:IDX_LHAND_POS[1]],
        'lhand_orient': motion[:, IDX_LHAND_ORIENT[0]:IDX_LHAND_ORIENT[1]],
        'lhand_pca': motion[:, IDX_LHAND_PCA[0]:IDX_LHAND_PCA[1]],

        # Right hand PCA (30D) - for generation
        'rhand_pos': motion[:, IDX_RHAND_POS[0]:IDX_RHAND_POS[1]],
        'rhand_orient': motion[:, IDX_RHAND_ORIENT[0]:IDX_RHAND_ORIENT[1]],
        'rhand_pca': motion[:, IDX_RHAND_PCA[0]:IDX_RHAND_PCA[1]],

        # Left hand quaternions (64D) - for visualization with SMPL-X
        'lhand_quat': motion[:, IDX_LHAND_QUAT[0]:IDX_LHAND_QUAT[1]],

        # Right hand quaternions (64D) - for visualization with SMPL-X
        'rhand_quat': motion[:, IDX_RHAND_QUAT[0]:IDX_RHAND_QUAT[1]],

        # Objects (27D)
        'obj1_pos': motion[:, IDX_OBJ1_POS[0]:IDX_OBJ1_POS[1]],
        'obj1_rot': motion[:, IDX_OBJ1_ROT[0]:IDX_OBJ1_ROT[1]],
    }

    return result


def rotation_6d_to_axis_angle(rot_6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to axis-angle.

    Args:
        rot_6d: (..., 6) array of 6D rotations

    Returns:
        (..., 3) array of axis-angle rotations
    """
    # Convert to torch tensor
    rot_6d_t = torch.tensor(rot_6d, dtype=torch.float32)

    # Convert 6D to rotation matrix
    rot_mat = rotation_6d_to_matrix(rot_6d_t)  # (..., 3, 3)

    # Convert rotation matrix to axis-angle using scipy
    original_shape = rot_mat.shape[:-2]
    rot_mat_flat = rot_mat.reshape(-1, 3, 3).numpy()

    axis_angle_list = []
    for mat in rot_mat_flat:
        r = R.from_matrix(mat)
        axis_angle_list.append(r.as_rotvec())

    axis_angle = np.array(axis_angle_list).reshape(*original_shape, 3)
    return axis_angle


def quaternion_to_axis_angle(quats: np.ndarray) -> np.ndarray:
    """
    Convert quaternions to axis-angle representation.

    Args:
        quats: (..., 4) array of quaternions in [w, x, y, z] format (OakInk2 format)

    Returns:
        (..., 3) array of axis-angle rotations
    """
    # pytorch3d expects [w, x, y, z] format, which matches OakInk2
    quats_t = torch.tensor(quats, dtype=torch.float32)
    axis_angle_t = pt3d_quat_to_aa(quats_t)
    return axis_angle_t.numpy()


def get_body_joint_positions(body_transl, body_orient, body_pose, smplx_model):
    """
    Get body joint positions by running SMPL-X forward pass.

    Args:
        body_transl: (T, 3) body translation
        body_orient: (T, 6) body root orientation (6D)
        body_pose: (T, 126) body pose (21 joints × 6D)
        smplx_model: SMPL-X model instance

    Returns:
        (T, J, 3) joint positions
    """
    T = body_transl.shape[0]

    # Convert 6D rotations to axis-angle
    body_orient_aa = rotation_6d_to_axis_angle(body_orient)  # (T, 3)

    # Reshape body_pose from (T, 126) to (T, 21, 6) then to (T, 21, 3) axis-angle
    body_pose_6d = body_pose.reshape(T, 21, 6)
    body_pose_aa = rotation_6d_to_axis_angle(body_pose_6d)  # (T, 21, 3)
    body_pose_aa = body_pose_aa.reshape(T, 63)  # (T, 63) for SMPL-X body_pose

    # Run SMPL-X forward pass
    with torch.no_grad():
        transl = torch.tensor(body_transl, dtype=torch.float32, device=device)
        global_orient = torch.tensor(body_orient_aa, dtype=torch.float32, device=device)
        body_pose_t = torch.tensor(body_pose_aa, dtype=torch.float32, device=device)

        output = smplx_model(
            transl=transl,
            global_orient=global_orient,
            body_pose=body_pose_t,
        )

        joints = output.joints.cpu().numpy()  # (T, 55, 3) for SMPL-X

    return joints[:, :22, :]  # Return first 22 body joints


def create_body_mesh_sequence(body_transl, body_orient, body_pose,
                               lhand_quat, rhand_quat, smplx_model, T,
                               align_floor=True):
    """
    Create body mesh sequence for visualization with integrated hand poses.

    Args:
        body_transl: (T, 3) body translation
        body_orient: (T, 6) body root orientation (6D)
        body_pose: (T, 126) body pose (21 joints × 6D)
        lhand_quat: (T, 64) left hand quaternions (16 joints × 4D)
        rhand_quat: (T, 64) right hand quaternions (16 joints × 4D)
        smplx_model: SMPL-X model instance
        T: number of frames
        align_floor: If True, adjust vertices so feet are at Y=0

    Returns:
        tuple of (vertices, faces, floor_offset) for mesh rendering
    """
    # Convert 6D rotations to axis-angle
    body_orient_aa = rotation_6d_to_axis_angle(body_orient)  # (T, 3)

    # Reshape body_pose from (T, 126) to (T, 21, 6) then to (T, 21, 3) axis-angle
    body_pose_6d = body_pose.reshape(T, 21, 6)
    body_pose_aa = rotation_6d_to_axis_angle(body_pose_6d)  # (T, 21, 3)
    body_pose_aa = body_pose_aa.reshape(T, 63)  # (T, 63) for SMPL-X body_pose

    # Process hand quaternions
    # Hand quaternions are (T, 64) = 16 joints × 4D
    # Joint 0 = wrist (global orient, skip it - already in body)
    # Joints 1-15 = finger articulations (what SMPL-X expects)
    lhand_quat_reshaped = lhand_quat.reshape(T, 16, 4)
    rhand_quat_reshaped = rhand_quat.reshape(T, 16, 4)

    # Skip wrist (joint 0), use joints 1:16 for SMPL-X hand pose
    lhand_finger_quat = lhand_quat_reshaped[:, 1:, :]  # (T, 15, 4)
    rhand_finger_quat = rhand_quat_reshaped[:, 1:, :]  # (T, 15, 4)

    # Convert quaternions to axis-angle
    lhand_pose_aa = quaternion_to_axis_angle(lhand_finger_quat)  # (T, 15, 3)
    rhand_pose_aa = quaternion_to_axis_angle(rhand_finger_quat)  # (T, 15, 3)

    # Flatten hand poses: (T, 15, 3) -> (T, 45)
    lhand_pose_aa = lhand_pose_aa.reshape(T, 45)
    rhand_pose_aa = rhand_pose_aa.reshape(T, 45)

    # Run SMPL-X forward pass with hand poses
    with torch.no_grad():
        transl = torch.tensor(body_transl, dtype=torch.float32, device=device)
        global_orient = torch.tensor(body_orient_aa, dtype=torch.float32, device=device)
        body_pose_t = torch.tensor(body_pose_aa, dtype=torch.float32, device=device)
        lhand_pose_t = torch.tensor(lhand_pose_aa, dtype=torch.float32, device=device)
        rhand_pose_t = torch.tensor(rhand_pose_aa, dtype=torch.float32, device=device)

        output = smplx_model(
            transl=transl,
            global_orient=global_orient,
            body_pose=body_pose_t,
            left_hand_pose=lhand_pose_t,
            right_hand_pose=rhand_pose_t,
        )

        vertices = output.vertices.cpu().numpy()  # (T, V, 3)
        faces = smplx_model.faces  # (F, 3)

    # Align floor: compute minimum Y across all frames and shift up
    floor_offset = 0.0
    if align_floor:
        # Find the global minimum Y vertex (lowest point across all frames)
        min_y = vertices[:, :, 1].min()
        floor_offset = -min_y
        print(f"  Floor alignment: min_y={min_y:.4f}, applying offset={floor_offset:.4f}")
        vertices[:, :, 1] += floor_offset

    return vertices, faces, floor_offset


def render_oakink2_sequence(
    motion_path: str,
    output_path: str,
    smplx_path: str,
    resolution: str = 'medium',
    fps: int = 30,
    max_frames: int = 0,
    denormalize: bool = True,
):
    """
    Render OakInk2 motion sequence to video.

    Args:
        motion_path: Path to .npy motion file
        output_path: Path to save output video
        smplx_path: Path to SMPL-X model file
        resolution: 'high', 'medium', or 'low'
        fps: Output video FPS
        max_frames: Maximum frames to render (0 = all)
        denormalize: Whether to denormalize the data
    """
    # Set resolution
    if resolution == 'high':
        res = (2000, 1500)
    elif resolution == 'medium':
        res = (1280, 960)
    else:
        res = (640, 480)

    # Parse motion data
    print(f"Loading motion from: {motion_path}")
    data = parse_oakink2_motion(motion_path, denormalize=denormalize)

    T = data['body_transl'].shape[0]
    if max_frames > 0:
        T = min(T, max_frames)
        for key in data:
            data[key] = data[key][:T]

    print(f"Motion frames: {T}")

    # Note: Floor alignment is done AFTER SMPL-X forward pass in create_body_mesh_sequence
    # This computes actual vertex minimum Y rather than estimating from translation

    # Initialize headless renderer
    print("Initializing renderer...")
    v = HeadlessRenderer(size=res)

    # Camera settings - front view of full body
    v.scene.camera.fov = 60
    v.scene.camera.position = [0.0, 1.0, -3.5]  # Front view (negative Z), further back for full body
    v.scene.camera.target = [0.0, 0.9, 0.0]     # Focus on body center
    v.scene.camera.up = [0, 1.0, 0]
    v.scene.floor.position = [0.0, 0.0, 0.0]

    v.auto_set_camera_target = False
    v.auto_set_floor = False

    # Remove default lights and origin
    if len(v.scene.lights) > 0:
        v.scene.remove(v.scene.lights[0])
    v.scene.remove(v.scene.origin)

    # Load SMPL-X model for body mesh
    print(f"Loading SMPL-X model from: {smplx_path}")
    body_loaded = False
    try:
        import smplx
        print(f"  Creating SMPL-X model with batch_size={T}...")
        # smplx expects model_path to be the directory containing 'smplx/' folder
        smplx_model = smplx.create(
            model_path=smplx_path,
            model_type='smplx',
            gender='neutral',
            use_pca=False,
            num_pca_comps=45,
            flat_hand_mean=True,
            batch_size=T,
        ).to(device)
        print(f"  SMPL-X model created successfully")

        # Create body mesh sequence with integrated hand poses
        print("Computing body mesh vertices (with hand poses)...")
        body_verts, body_faces, floor_offset = create_body_mesh_sequence(
            data['body_transl'],
            data['body_orient'],
            data['body_pose'],
            data['lhand_quat'],
            data['rhand_quat'],
            smplx_model,
            T
        )
        print(f"  Body vertices shape: {body_verts.shape}")
        print(f"  Body faces shape: {body_faces.shape}")
        # Note: floor_offset is available for object alignment if needed in the future

        # Create body mesh renderable
        # Note: z_up=False because SMPL-X data is already in Y-up convention
        body_mesh = Meshes(
            body_verts,
            body_faces,
            z_up=False,
            color=BODY_MESH_COLOR,
            name="Body"
        )
        v.scene.add(body_mesh)
        print("Body mesh added to scene")
        body_loaded = True

    except Exception as e:
        import traceback
        print(f"Warning: Could not load SMPL-X model: {e}")
        traceback.print_exc()
        print("Skipping body mesh rendering")

    # Note: Hands are now integrated into SMPL-X body mesh (no separate MANO rendering needed)
    print("Hands integrated into SMPL-X body mesh")

    # Render and save video
    print(f"Rendering video to: {output_path}")
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else 'output'
    os.makedirs(output_dir, exist_ok=True)

    frame_dir = output_path.replace('.mp4', '_frames')
    os.makedirs(frame_dir, exist_ok=True)

    # Export frames only first, then use ffmpeg manually
    # This avoids issues with skvideo/ffmpeg pipe
    print(f"Exporting {T} frames to {frame_dir}...")

    v._init_scene()
    for frame_idx in range(T):
        v.scene.current_frame_id = frame_idx
        v.render(0, 0, export=True)
        frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
        v.export_frame(frame_path)
        if (frame_idx + 1) % 10 == 0:
            print(f"  Rendered frame {frame_idx + 1}/{T}")

    print(f"Frames saved to: {frame_dir}")

    # Create video from frames using ffmpeg
    # Try system ffmpeg first (usually has libx264), then conda ffmpeg
    import subprocess
    import shutil

    frame_pattern = os.path.join(frame_dir, "frame_%06d.png")

    # Find ffmpeg - prefer system ffmpeg which usually has libx264
    system_ffmpeg = shutil.which("ffmpeg")

    # Try with libx264 first (better quality)
    cmd = [
        system_ffmpeg or "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        output_path
    ]
    print(f"Creating video with ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Fallback: try mpeg4 codec (more compatible)
        print("libx264 not available, trying mpeg4...")
        cmd = [
            system_ffmpeg or "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "mpeg4",
            "-q:v", "5",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Video saved to: {output_path}")
    else:
        print(f"FFmpeg failed: {result.stderr}")
        print(f"Frames are available at: {frame_dir}")


def render_raw_annotation(
    anno_path: str,
    output_path: str,
    smplx_path: str,
    resolution: str = 'medium',
    fps: int = 30,
    max_frames: int = 0,
    view: str = 'front',
    render_mode: str = 'mesh',
):
    """
    Render OakInk2 motion directly from raw annotation file.

    This bypasses our preprocessed data to verify hand poses work correctly.
    Uses raw_smplx['left_hand_pose'] (15 joints) instead of raw_mano (16 joints).

    Args:
        anno_path: Path to annotation pickle file
        output_path: Path to save output video
        smplx_path: Path to SMPL-X model directory
        resolution: 'high', 'medium', or 'low'
        fps: Output video FPS
        max_frames: Maximum frames to render (0 = all)
        view: Camera view angle ('front', 'side', 'back')
        render_mode: 'mesh', 'skeleton', or 'both'
    """
    # Set resolution
    if resolution == 'high':
        res = (2000, 1500)
    elif resolution == 'medium':
        res = (1280, 960)
    else:
        res = (640, 480)

    # Load raw annotation
    frame_range = (0, max_frames) if max_frames > 0 else None
    raw_data = load_raw_annotation(anno_path, frame_range=frame_range)

    T = len(raw_data['frame_ids'])
    print(f"Motion frames: {T}")

    # Initialize headless renderer
    print("Initializing renderer...")
    v = HeadlessRenderer(size=res)

    # Camera settings based on view
    v.scene.camera.fov = 60
    if view == 'front':
        v.scene.camera.position = [0.0, 1.0, -3.5]  # Front view (negative Z)
    elif view == 'side':
        v.scene.camera.position = [3.5, 1.0, 0.0]   # Side view (positive X)
    elif view == 'back':
        v.scene.camera.position = [0.0, 1.0, 3.5]   # Back view (positive Z)
    v.scene.camera.target = [0.0, 0.9, 0.0]
    v.scene.camera.up = [0, 1.0, 0]
    v.scene.floor.position = [0.0, 0.0, 0.0]
    print(f"Camera view: {view}")

    v.auto_set_camera_target = False
    v.auto_set_floor = False

    # Remove default lights and origin
    if len(v.scene.lights) > 0:
        v.scene.remove(v.scene.lights[0])
    v.scene.remove(v.scene.origin)

    # Load SMPL-X model
    print(f"Loading SMPL-X model from: {smplx_path}")
    try:
        import smplx
        smplx_model = smplx.create(
            model_path=smplx_path,
            model_type='smplx',
            gender='neutral',
            use_pca=False,
            num_pca_comps=45,
            num_betas=300,  # OakInk2 uses 300 betas
            flat_hand_mean=True,
            batch_size=T,
        ).to(device)
        print(f"  SMPL-X model created successfully")

        print(f"Render mode: {render_mode}")

        floor_offset = 0.0  # Will be set by whichever runs first

        # Create body mesh if needed
        if render_mode in ['mesh', 'both']:
            print("Computing body mesh vertices (from raw annotation)...")
            body_verts, body_faces, floor_offset = create_body_mesh_from_raw(
                raw_data['body_transl'],
                raw_data['body_orient_aa'],
                raw_data['body_pose_aa'],
                raw_data['lhand_aa'],
                raw_data['rhand_aa'],
                raw_data['body_shape'],
                smplx_model,
                T
            )
            print(f"  Body vertices shape: {body_verts.shape}")
            print(f"  Body faces shape: {body_faces.shape}")

            # Create body mesh renderable
            body_mesh = Meshes(
                body_verts,
                body_faces,
                z_up=False,
                color=BODY_MESH_COLOR,
                name="Body"
            )
            v.scene.add(body_mesh)
            print("Body mesh added to scene")

        # Create skeleton if needed
        if render_mode in ['skeleton', 'both']:
            print("Computing skeleton joints (from raw annotation)...")
            joints, floor_offset = create_skeleton_from_raw(
                raw_data['body_transl'],
                raw_data['body_orient_aa'],
                raw_data['body_pose_aa'],
                raw_data['lhand_aa'],
                raw_data['rhand_aa'],
                raw_data['body_shape'],
                smplx_model,
                T
            )
            print(f"  Joints shape: {joints.shape}")  # (T, 127, 3) for SMPL-X

            # Create spheres for body joints (first 22 joints)
            body_joint_positions = joints[:, :22, :]  # (T, 22, 3)
            body_spheres = Spheres(
                body_joint_positions,
                radius=0.015,
                color=SKELETON_JOINT_COLOR,
                name="Body_Joints"
            )
            v.scene.add(body_spheres)
            print("  Body joints (22) added to scene")

            # Create spheres for hand joints with different colors
            # Left hand: joints 25-39 (15 joints)
            lhand_joint_positions = joints[:, 25:40, :]  # (T, 15, 3)
            lhand_spheres = Spheres(
                lhand_joint_positions,
                radius=0.008,
                color=LHAND_JOINT_COLOR,
                name="LHand_Joints"
            )
            v.scene.add(lhand_spheres)
            print("  Left hand joints (15) added to scene")

            # Right hand: joints 40-54 (15 joints)
            rhand_joint_positions = joints[:, 40:55, :]  # (T, 15, 3)
            rhand_spheres = Spheres(
                rhand_joint_positions,
                radius=0.008,
                color=RHAND_JOINT_COLOR,
                name="RHand_Joints"
            )
            v.scene.add(rhand_spheres)
            print("  Right hand joints (15) added to scene")

            # Create lines for body skeleton bones
            all_bones = SMPLX_BODY_BONES + SMPLX_LHAND_BONES + SMPLX_RHAND_BONES
            bone_lines = np.zeros((T, len(all_bones), 2, 3), dtype=np.float32)
            for b_idx, (j1, j2) in enumerate(all_bones):
                bone_lines[:, b_idx, 0, :] = joints[:, j1, :]
                bone_lines[:, b_idx, 1, :] = joints[:, j2, :]

            skeleton_lines = Lines(
                bone_lines.reshape(T, -1, 3),  # (T, num_bones*2, 3)
                r_base=0.003,
                color=SKELETON_BONE_COLOR,
                mode='lines',
                name="Skeleton_Bones"
            )
            v.scene.add(skeleton_lines)
            print(f"  Skeleton bones ({len(all_bones)}) added to scene")

        # --- Add object meshes ---
        OBJ_MESH_DIR = "/hhd4/lizhe/dataset/OakInk2/data/object_repair/align_ds"

        if raw_data['obj_list']:
            print(f"\nLoading object meshes...")
            obj_meshes = load_object_meshes(raw_data['obj_list'], OBJ_MESH_DIR)

            if obj_meshes:
                print(f"Creating object mesh sequences (floor_offset={floor_offset:.4f})...")
                obj_sequences = create_object_mesh_sequence(
                    obj_meshes, raw_data['obj_transf'], T, floor_offset=floor_offset
                )

                # Define colors for multiple objects (orange-brown tones)
                obj_colors = [
                    (0.8, 0.6, 0.3, 1.0),  # Orange-brown
                    (0.6, 0.8, 0.4, 1.0),  # Light green
                    (0.4, 0.6, 0.8, 1.0),  # Light blue
                ]

                for i, (obj_id, verts_seq, faces) in enumerate(obj_sequences):
                    color = obj_colors[i % len(obj_colors)]
                    obj_mesh = Meshes(
                        verts_seq,
                        faces,
                        z_up=False,
                        color=color,
                        name=f"Object_{obj_id}"
                    )
                    v.scene.add(obj_mesh)
                    print(f"  Added object '{obj_id}' to scene")
            else:
                print("  No object meshes loaded")
        else:
            print("No objects in this sequence")

    except Exception as e:
        import traceback
        print(f"Error loading SMPL-X model: {e}")
        traceback.print_exc()
        return

    # Render and save video
    print(f"Rendering video to: {output_path}")
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else 'output'
    os.makedirs(output_dir, exist_ok=True)

    frame_dir = output_path.replace('.mp4', '_frames')
    os.makedirs(frame_dir, exist_ok=True)

    print(f"Exporting {T} frames to {frame_dir}...")

    v._init_scene()
    for frame_idx in range(T):
        v.scene.current_frame_id = frame_idx
        v.render(0, 0, export=True)
        frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
        v.export_frame(frame_path)
        if (frame_idx + 1) % 10 == 0:
            print(f"  Rendered frame {frame_idx + 1}/{T}")

    print(f"Frames saved to: {frame_dir}")

    # Create video from frames
    import subprocess
    import shutil

    frame_pattern = os.path.join(frame_dir, "frame_%06d.png")
    system_ffmpeg = shutil.which("ffmpeg")

    cmd = [
        system_ffmpeg or "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        output_path
    ]
    print(f"Creating video with ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("libx264 not available, trying mpeg4...")
        cmd = [
            system_ffmpeg or "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "mpeg4",
            "-q:v", "5",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Video saved to: {output_path}")
    else:
        print(f"FFmpeg failed: {result.stderr}")
        print(f"Frames are available at: {frame_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize OakInk2 motion data"
    )

    parser.add_argument(
        "--motion_path",
        type=str,
        default=None,
        help="Path to OakInk2 .npy motion file (preprocessed data)"
    )
    parser.add_argument(
        "--raw_anno",
        type=str,
        default=None,
        help="Path to raw annotation .pkl file (bypasses preprocessing, uses raw_smplx)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/oakink2_vis.mp4",
        help="Path to save output video"
    )
    parser.add_argument(
        "--smplx_path",
        type=str,
        default="/hhd4/lizhe/code/InterAct/models",
        help="Path to directory containing smplx/ folder with model files"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="medium",
        choices=["high", "medium", "low"],
        help="Output video resolution"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video FPS"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum frames to render (0 = all)"
    )
    parser.add_argument(
        "--no_denormalize",
        action="store_true",
        help="Skip denormalization (use raw data)"
    )
    parser.add_argument(
        "--view",
        type=str,
        default="front",
        choices=["front", "side", "back"],
        help="Camera view angle (front, side, or back)"
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="mesh",
        choices=["mesh", "skeleton", "both"],
        help="Render mode: 'mesh' for body mesh, 'skeleton' for joints/bones, 'both' for both"
    )

    args = parser.parse_args()

    # Determine which mode to use
    if args.raw_anno:
        # Use raw annotation mode (bypasses preprocessing, for verification)
        print("=" * 60)
        print("MODE: Raw annotation visualization")
        print("  This bypasses preprocessed data to verify hand poses work")
        print("  Uses raw_smplx['left_hand_pose'] (15 joints) directly")
        print("=" * 60)
        render_raw_annotation(
            anno_path=args.raw_anno,
            output_path=args.output_path,
            smplx_path=args.smplx_path,
            resolution=args.resolution,
            fps=args.fps,
            max_frames=args.max_frames,
            view=args.view,
            render_mode=args.render_mode,
        )
    elif args.motion_path:
        # Use preprocessed .npy data
        print("=" * 60)
        print("MODE: Preprocessed motion visualization")
        print("  Uses 398D feature vector from .npy file")
        print("  NOTE: Hand poses may be incorrect due to MANO vs SMPLX mismatch")
        print("=" * 60)
        render_oakink2_sequence(
            motion_path=args.motion_path,
            output_path=args.output_path,
            smplx_path=args.smplx_path,
            resolution=args.resolution,
            fps=args.fps,
            max_frames=args.max_frames,
            denormalize=not args.no_denormalize,
        )
    else:
        print("Error: Must specify either --motion_path or --raw_anno")
        parser.print_help()


if __name__ == "__main__":
    main()
