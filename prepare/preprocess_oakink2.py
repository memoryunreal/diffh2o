#!/usr/bin/env python3
"""
Preprocess OakInk2 dataset for DiffH2O training.

Converts OakInk2 data (SMPL-X, MANO, object transforms) into DiffH2O-compatible format.

Feature vector layout (~398D):
- Body root (9D): world_tsl (3D) + world_rot (6D)
- Body pose (126D): 21 joints × 6D rotation
- Left hand PCA (30D): pos (3D) + global_orient (6D) + pca_pose (21D)
- Right hand PCA (30D): pos (3D) + global_orient (6D) + pca_pose (21D)
- Left hand quaternions (67D): tsl (3D) + pose_coeffs (16 × 4 = 64D)
- Right hand quaternions (67D): tsl (3D) + pose_coeffs (16 × 4 = 64D)
- SDF left hand (21D): signed distance from hand joints to object
- SDF right hand (21D): signed distance from hand joints to object
- Object 1 pose (9D): position (3D) + rotation (6D)
- Object 2 pose (9D): position (3D) + rotation (6D)
- Object 3 pose (9D): position (3D) + rotation (6D)
Total: 398D
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import torch
import trimesh

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.rotation_conversions import (
    quaternion_to_matrix,
    matrix_to_rotation_6d,
    quaternion_to_axis_angle,
)


@dataclass
class OakInk2Config:
    """Configuration for OakInk2 preprocessing."""
    # Input paths
    oakink2_root: str = "/hhd4/lizhe/dataset/OakInk2/data"
    anno_dir: str = "anno_preview"
    program_dir: str = "program"
    object_dir: str = "object_repair/align_ds"

    # Output paths
    output_root: str = "dataset/OAKINK2"
    motion_dir: str = "oakink2_representation"
    text_dir: str = "texts"

    # Processing parameters
    # NOTE: OakInk2 is already 30fps - no downsampling needed
    # The 4 camera views are different angles, not temporal oversampling
    max_objects: int = 3
    min_motion_len: int = 20  # Minimum frames
    max_motion_len: int = 0  # Maximum frames (0 = no limit for complex tasks)

    # Extraction mode: 'primitive', 'complex', or 'both'
    mode: str = 'primitive'

    # MANO PCA components
    num_pca_components: int = 21

    # Feature dimensions
    body_root_dim: int = 9  # 3 (tsl) + 6 (rot)
    body_pose_dim: int = 126  # 21 joints × 6D
    hand_pca_dim: int = 30  # 3 (pos) + 6 (orient) + 21 (pca)
    hand_quat_dim: int = 67  # 3 (tsl) + 64 (16 joints × 4)
    sdf_dim: int = 21  # Per hand
    object_pose_dim: int = 9  # 3 (pos) + 6 (rot)


# Feature indices for the new representation
OAKINK2_REPRESENTATION_IDCS = {
    # Body
    "body_tsl": (0, 3),
    "body_rot": (3, 9),
    "body_pose": (9, 135),  # 21 joints × 6D = 126D
    # Left hand (PCA representation for diffusion)
    "left_hand_pos": (135, 138),
    "left_hand_orient": (138, 144),
    "left_hand_pca": (144, 165),
    # Right hand (PCA representation for diffusion)
    "right_hand_pos": (165, 168),
    "right_hand_orient": (168, 174),
    "right_hand_pca": (174, 195),
    # Left hand (quaternion representation for reconstruction)
    "left_hand_tsl_quat": (195, 198),
    "left_hand_pose_quat": (198, 262),  # 16 joints × 4 = 64D
    # Right hand (quaternion representation for reconstruction)
    "right_hand_tsl_quat": (262, 265),
    "right_hand_pose_quat": (265, 329),  # 16 joints × 4 = 64D
    # SDF
    "sdf_left": (329, 350),
    "sdf_right": (350, 371),
    # Objects (up to 3)
    "object1_pos": (371, 374),
    "object1_rot": (374, 380),
    "object2_pos": (380, 383),
    "object2_rot": (383, 389),
    "object3_pos": (389, 392),
    "object3_rot": (392, 398),
}

TOTAL_FEATURE_DIM = 398


class MANOPCAConverter:
    """Convert MANO quaternion poses to PCA representation."""

    def __init__(self, mano_path: Optional[str] = None):
        """Initialize MANO PCA converter.

        Args:
            mano_path: Path to MANO model files. If None, will try common locations.
        """
        self.mano_path = mano_path
        self.pca_components = None
        self.pca_mean = None
        self._load_mano_pca()

    def _load_mano_pca(self):
        """Load MANO PCA components from model file."""
        # Try common MANO model locations
        possible_paths = [
            self.mano_path,
            "./data/smplx/mano",
            "data/smplx/mano",
            "/hhd4/lizhe/dataset/mano_v1_2/models",
            "assets/mano_v1_2/models",
            "/hhd4/lizhe/dataset/OakInk2/asset/mano_v1_2/models",
            os.path.expanduser("~/.mano/models"),
        ]

        for base_path in possible_paths:
            if base_path is None:
                continue

            right_mano = os.path.join(base_path, "MANO_RIGHT.pkl")
            if os.path.exists(right_mano):
                try:
                    with open(right_mano, 'rb') as f:
                        mano_data = pickle.load(f, encoding='latin1')

                    # Extract PCA components (hands_components in MANO)
                    if 'hands_components' in mano_data:
                        self.pca_components = torch.tensor(
                            mano_data['hands_components'][:21, :],  # Top 21 components
                            dtype=torch.float32
                        )
                        self.pca_mean = torch.tensor(
                            mano_data['hands_mean'],
                            dtype=torch.float32
                        )
                        print(f"Loaded MANO PCA from {right_mano}")
                        print(f"  PCA components shape: {self.pca_components.shape}")
                        print(f"  PCA mean shape: {self.pca_mean.shape}")
                        return
                except Exception as e:
                    print(f"Failed to load MANO from {right_mano}: {e}")

        print("WARNING: Could not load MANO PCA components. Using identity projection.")
        # Fallback: identity projection (just take first 21 dims of flattened pose)
        self.pca_components = None
        self.pca_mean = None

    def quaternion_to_pca(self, pose_quats: torch.Tensor) -> torch.Tensor:
        """Convert MANO quaternion poses to PCA representation.

        Args:
            pose_quats: Quaternion poses of shape (T, 16, 4) or (T, 15, 4)
                       where quaternions are in [w, x, y, z] format.

        Returns:
            PCA coefficients of shape (T, 21)
        """
        T = pose_quats.shape[0]
        num_joints = pose_quats.shape[1]

        # Convert quaternions to axis-angle
        # Reshape to (T*num_joints, 4) for batch processing
        quats_flat = pose_quats.reshape(-1, 4)
        axis_angles = quaternion_to_axis_angle(quats_flat)  # (T*num_joints, 3)

        # Reshape back to (T, num_joints*3)
        pose_aa = axis_angles.reshape(T, num_joints * 3)

        if self.pca_components is not None:
            # Project onto PCA space
            # MANO expects 45D pose (15 joints × 3), but we might have 16 joints
            if num_joints == 16:
                # Skip wrist joint (first joint) for MANO PCA
                pose_aa = pose_aa[:, 3:]  # Now 15 joints × 3 = 45D

            # Subtract mean and project
            pose_centered = pose_aa - self.pca_mean.unsqueeze(0)
            pca_coeffs = torch.matmul(pose_centered, self.pca_components.T)

            return pca_coeffs  # (T, 21)
        else:
            # Fallback: just take first 21 dimensions
            return pose_aa[:, :21]

    def get_hand_position_and_orient(
        self,
        mano_tsl: torch.Tensor,
        pose_quats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract hand position and global orientation from MANO params.

        Args:
            mano_tsl: Translation of shape (T, 3)
            pose_quats: Quaternion poses of shape (T, 16, 4), first is wrist

        Returns:
            position: (T, 3)
            orientation: (T, 6) in 6D rotation representation
        """
        # Position is just the translation
        position = mano_tsl

        # Global orientation is the first joint (wrist)
        wrist_quat = pose_quats[:, 0, :]  # (T, 4)
        wrist_rot_mat = quaternion_to_matrix(wrist_quat)  # (T, 3, 3)
        orientation = matrix_to_rotation_6d(wrist_rot_mat)  # (T, 6)

        return position, orientation


class ObjectProcessor:
    """Process object meshes and compute BPS/SDF."""

    def __init__(self, object_dir: str, num_bps_points: int = 1024):
        """Initialize object processor.

        Args:
            object_dir: Directory containing object mesh files.
            num_bps_points: Number of basis points for BPS encoding.
        """
        self.object_dir = object_dir
        self.num_bps_points = num_bps_points
        self.mesh_cache: Dict[str, trimesh.Trimesh] = {}
        self.bps_basis = self._generate_bps_basis()

    def _generate_bps_basis(self) -> np.ndarray:
        """Generate BPS basis points on a unit sphere."""
        # Use Fibonacci sphere for uniform distribution
        indices = np.arange(self.num_bps_points) + 0.5
        phi = np.arccos(1 - 2 * indices / self.num_bps_points)
        theta = np.pi * (1 + np.sqrt(5)) * indices

        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)

        return np.stack([x, y, z], axis=1).astype(np.float32)

    def load_mesh(self, obj_id: str) -> Optional[trimesh.Trimesh]:
        """Load object mesh from file.

        Args:
            obj_id: Object ID (e.g., 'C12001', 'O02@0038@00001')

        Returns:
            Trimesh object or None if not found.
        """
        if obj_id in self.mesh_cache:
            return self.mesh_cache[obj_id]

        mesh_path = os.path.join(self.object_dir, obj_id, "model.obj")
        if not os.path.exists(mesh_path):
            print(f"WARNING: Object mesh not found: {mesh_path}")
            return None

        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            self.mesh_cache[obj_id] = mesh
            return mesh
        except Exception as e:
            print(f"WARNING: Failed to load mesh {mesh_path}: {e}")
            return None

    def compute_bps(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Compute BPS encoding for a mesh.

        Args:
            mesh: Trimesh object.

        Returns:
            BPS encoding of shape (1024, 3) - closest points on mesh to basis points.
        """
        # Center and normalize mesh
        vertices = mesh.vertices - mesh.centroid
        scale = np.max(np.abs(vertices))
        if scale > 0:
            vertices = vertices / scale

        # Scale basis points to encompass the mesh
        basis_scaled = self.bps_basis * 1.5

        # Find closest points on mesh surface to each basis point
        closest_points, distances, triangle_ids = trimesh.proximity.closest_point(
            mesh, basis_scaled
        )

        # Return relative vectors from basis to closest points
        bps_encoding = closest_points - basis_scaled

        return bps_encoding.astype(np.float32)

    def compute_sdf(
        self,
        query_points: np.ndarray,
        mesh: trimesh.Trimesh,
        transform: np.ndarray
    ) -> np.ndarray:
        """Compute signed distance from query points to mesh surface.

        Args:
            query_points: Points of shape (N, 3) in world coordinates.
            mesh: Trimesh object in canonical pose.
            transform: 4x4 transformation matrix for the mesh.

        Returns:
            Signed distances of shape (N,)
        """
        # Transform query points to mesh canonical space
        # mesh_to_world: transform, world_to_mesh: inverse
        transform_inv = np.linalg.inv(transform)
        points_homo = np.concatenate([
            query_points,
            np.ones((query_points.shape[0], 1))
        ], axis=1)
        points_mesh = (transform_inv @ points_homo.T).T[:, :3]

        # Compute closest points on mesh
        closest_points, distances, _ = trimesh.proximity.closest_point(
            mesh, points_mesh
        )

        # Determine sign using mesh normals
        # Inside is negative, outside is positive
        try:
            signed_distances = mesh.nearest.signed_distance(points_mesh)
        except:
            # Fallback: use unsigned distance
            signed_distances = distances

        return signed_distances.astype(np.float32)


def parse_interval(interval_str: str) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Parse interval string like '((1595, 2340), None)' to tuple of frame ranges.

    Returns:
        (left_hand_interval, right_hand_interval) where each is (start, end) or None.
    """
    # Remove outer parentheses and split
    import ast
    try:
        parsed = ast.literal_eval(interval_str)
        left_interval = parsed[0]
        right_interval = parsed[1]
        return left_interval, right_interval
    except:
        return None, None


def load_oakink2_sequence(
    pkl_path: str,
    program_info_path: str,
    desc_info_path: str,
    config: OakInk2Config,
) -> List[Dict[str, Any]]:
    """Load a single OakInk2 sequence and split into primitive segments.

    Args:
        pkl_path: Path to annotation pickle file.
        program_info_path: Path to program info JSON.
        desc_info_path: Path to description info JSON.
        config: Preprocessing configuration.

    Returns:
        List of segment dictionaries, each containing:
        - 'motion': numpy array of shape (T, D)
        - 'text': text description
        - 'frame_range': (start, end) frame indices
        - 'objects': list of object IDs
    """
    # Load annotation data
    with open(pkl_path, 'rb') as f:
        anno = pickle.load(f)

    # Load program info (primitive segments)
    with open(program_info_path, 'r') as f:
        program_info = json.load(f)

    # Load descriptions
    with open(desc_info_path, 'r') as f:
        desc_info = json.load(f)

    segments = []

    # Get frame list
    mocap_frames = anno['mocap_frame_id_list']
    total_frames = len(mocap_frames)

    # Process each primitive segment
    for interval_str, prim_data in program_info.items():
        left_interval, right_interval = parse_interval(interval_str)

        # Determine frame range (use the active hand's interval)
        if right_interval is not None:
            start_frame, end_frame = right_interval
        elif left_interval is not None:
            start_frame, end_frame = left_interval
        else:
            continue

        # Get description
        desc_data = desc_info.get(interval_str, {})
        text_desc = desc_data.get('seg_desc', prim_data.get('primitive', 'unknown action'))

        # Get objects involved
        obj_list = prim_data.get('obj_list', [])

        segments.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'text': text_desc,
            'objects': obj_list,
            'primitive': prim_data.get('primitive', 'unknown'),
            'interaction_mode': prim_data.get('interaction_mode', 'rh_main'),
        })

    return segments, anno


def process_segment(
    anno: Dict,
    segment: Dict,
    config: OakInk2Config,
    mano_converter: MANOPCAConverter,
    object_processor: ObjectProcessor,
) -> Optional[np.ndarray]:
    """Process a single segment into feature representation.

    Args:
        anno: Full annotation dictionary.
        segment: Segment info dictionary.
        config: Preprocessing configuration.
        mano_converter: MANO PCA converter.
        object_processor: Object mesh processor.

    Returns:
        Feature array of shape (T, 398) or None if processing fails.
    """
    start_frame = segment['start_frame']
    end_frame = segment['end_frame']

    # Get list of mocap frames
    mocap_frames = anno['mocap_frame_id_list']

    # Find indices in the mocap frame list
    try:
        start_idx = mocap_frames.index(start_frame)
        end_idx = mocap_frames.index(end_frame)
    except ValueError:
        # Frames not found, try to find closest
        start_idx = min(range(len(mocap_frames)), key=lambda i: abs(mocap_frames[i] - start_frame))
        end_idx = min(range(len(mocap_frames)), key=lambda i: abs(mocap_frames[i] - end_frame))

    if end_idx <= start_idx:
        return None

    # OakInk2 is already 30fps - no downsampling needed
    frame_indices = list(range(start_idx, end_idx + 1))

    if len(frame_indices) < config.min_motion_len:
        return None

    # Limit to max length (0 = no limit)
    if config.max_motion_len > 0 and len(frame_indices) > config.max_motion_len:
        frame_indices = frame_indices[:config.max_motion_len]

    T = len(frame_indices)
    features = np.zeros((T, TOTAL_FEATURE_DIM), dtype=np.float32)

    # Extract data for each frame
    raw_smplx = anno['raw_smplx']
    raw_mano = anno['raw_mano']
    obj_transf = anno['obj_transf']

    for t, frame_id in enumerate(frame_indices):
        actual_frame = mocap_frames[frame_id] if frame_id < len(mocap_frames) else mocap_frames[-1]

        # Get SMPL-X data for this frame
        if actual_frame not in raw_smplx:
            # Try to find closest frame
            available_frames = list(raw_smplx.keys())
            actual_frame = min(available_frames, key=lambda x: abs(x - actual_frame))

        smplx_frame = raw_smplx[actual_frame]
        mano_frame = raw_mano.get(actual_frame, raw_mano[list(raw_mano.keys())[0]])

        # --- Body Root (9D) ---
        world_tsl = smplx_frame['world_tsl'].squeeze().numpy()  # (3,)
        world_rot_quat = smplx_frame['world_rot'].squeeze()  # (4,)
        world_rot_mat = quaternion_to_matrix(world_rot_quat.unsqueeze(0))  # (1, 3, 3)
        world_rot_6d = matrix_to_rotation_6d(world_rot_mat).squeeze().numpy()  # (6,)

        features[t, 0:3] = world_tsl
        features[t, 3:9] = world_rot_6d

        # --- Body Pose (126D) ---
        body_pose_quat = smplx_frame['body_pose'].squeeze()  # (21, 4)
        body_pose_mat = quaternion_to_matrix(body_pose_quat)  # (21, 3, 3)
        body_pose_6d = matrix_to_rotation_6d(body_pose_mat)  # (21, 6)
        features[t, 9:135] = body_pose_6d.reshape(-1).numpy()

        # --- Left Hand PCA (30D) ---
        lh_tsl = mano_frame['lh__tsl'].squeeze().numpy()  # (3,)
        lh_pose_quat = mano_frame['lh__pose_coeffs'].squeeze()  # (16, 4)

        lh_pos, lh_orient = mano_converter.get_hand_position_and_orient(
            torch.tensor(lh_tsl).unsqueeze(0),
            lh_pose_quat.unsqueeze(0)
        )
        lh_pca = mano_converter.quaternion_to_pca(lh_pose_quat.unsqueeze(0))

        features[t, 135:138] = lh_pos.squeeze().numpy()
        features[t, 138:144] = lh_orient.squeeze().numpy()
        features[t, 144:165] = lh_pca.squeeze().numpy()

        # --- Right Hand PCA (30D) ---
        rh_tsl = mano_frame['rh__tsl'].squeeze().numpy()  # (3,)
        rh_pose_quat = mano_frame['rh__pose_coeffs'].squeeze()  # (16, 4)

        rh_pos, rh_orient = mano_converter.get_hand_position_and_orient(
            torch.tensor(rh_tsl).unsqueeze(0),
            rh_pose_quat.unsqueeze(0)
        )
        rh_pca = mano_converter.quaternion_to_pca(rh_pose_quat.unsqueeze(0))

        features[t, 165:168] = rh_pos.squeeze().numpy()
        features[t, 168:174] = rh_orient.squeeze().numpy()
        features[t, 174:195] = rh_pca.squeeze().numpy()

        # --- Left Hand Quaternions (67D) ---
        features[t, 195:198] = lh_tsl
        features[t, 198:262] = lh_pose_quat.reshape(-1).numpy()

        # --- Right Hand Quaternions (67D) ---
        features[t, 262:265] = rh_tsl
        features[t, 265:329] = rh_pose_quat.reshape(-1).numpy()

        # --- SDF (42D) ---
        # TODO: Compute SDF from hand joint positions to object surfaces
        # For now, fill with zeros (will be computed in a later step)
        features[t, 329:371] = 0.0

        # --- Object Poses (27D for 3 objects) ---
        obj_list = segment['objects'][:config.max_objects]

        for obj_idx, obj_id in enumerate(obj_list):
            if obj_id in obj_transf and actual_frame in obj_transf[obj_id]:
                obj_transform = obj_transf[obj_id][actual_frame]
                obj_pos = obj_transform[:3, 3]
                obj_rot_mat = torch.tensor(obj_transform[:3, :3]).unsqueeze(0)
                obj_rot_6d = matrix_to_rotation_6d(obj_rot_mat).squeeze().numpy()

                offset = 371 + obj_idx * 9
                features[t, offset:offset+3] = obj_pos
                features[t, offset+3:offset+9] = obj_rot_6d

    return features


def create_text_annotation(text: str, tokens: Optional[List[str]] = None) -> str:
    """Create text annotation in GRAB format.

    Format: caption#tokens#start#end

    Args:
        text: Text description.
        tokens: Optional list of tokens with POS tags.

    Returns:
        Formatted annotation string.
    """
    if tokens is None:
        # Simple tokenization without POS tags
        tokens_str = ""
    else:
        tokens_str = " ".join(tokens)

    return f"{text}#{tokens_str}#0.0#0.0\n"


def load_task_targets(oakink2_root: str) -> Dict[str, str]:
    """Load high-level task descriptions from task_target.json.

    Returns:
        Dictionary mapping sequence key to high-level task description.
    """
    task_target_path = os.path.join(oakink2_root, "program", "task_target.json")
    if not os.path.exists(task_target_path):
        print(f"WARNING: task_target.json not found at {task_target_path}")
        return {}

    with open(task_target_path, 'r') as f:
        task_targets = json.load(f)
    return task_targets


def get_complex_task_description(
    seq_key: str,
    desc_info: Dict,
    task_targets: Dict[str, str]
) -> Tuple[str, str]:
    """Get both high-level and concatenated descriptions for a complex task.

    Args:
        seq_key: Sequence key (format: scene_01__A001++seq__xxx).
        desc_info: Description info for all primitives.
        task_targets: High-level task descriptions.

    Returns:
        (high_level_desc, concatenated_desc)
    """
    # High-level description from task_target.json
    # Convert key format: scene_01__A001++seq__xxx -> scene_01__A001/seq__xxx
    task_key = seq_key.replace('++seq__', '/seq__')
    high_level = task_targets.get(task_key, "")

    # Concatenate all primitive descriptions in order
    primitives = []
    for interval_str, desc_data in desc_info.items():
        seg_desc = desc_data.get('seg_desc', '')
        if seg_desc:
            primitives.append(seg_desc)

    concatenated = " ".join(primitives) if primitives else ""

    return high_level, concatenated


def process_full_sequence(
    anno: Dict,
    config: OakInk2Config,
    mano_converter: 'MANOPCAConverter',
    object_processor: 'ObjectProcessor',
    all_objects: List[str],
) -> Optional[np.ndarray]:
    """Process a full sequence (for complex task mode) into feature representation.

    Args:
        anno: Full annotation dictionary.
        config: Preprocessing configuration.
        mano_converter: MANO PCA converter.
        object_processor: Object mesh processor.
        all_objects: All objects involved in the sequence.

    Returns:
        Feature array of shape (T, 398) or None if processing fails.
    """
    # Get list of mocap frames
    mocap_frames = anno['mocap_frame_id_list']
    total_frames = len(mocap_frames)

    if total_frames < config.min_motion_len:
        return None

    # Use all frames (no downsampling, OakInk2 is already 30fps)
    frame_indices = list(range(total_frames))

    # Limit to max length if specified
    if config.max_motion_len > 0 and len(frame_indices) > config.max_motion_len:
        frame_indices = frame_indices[:config.max_motion_len]

    T = len(frame_indices)
    features = np.zeros((T, TOTAL_FEATURE_DIM), dtype=np.float32)

    # Extract data for each frame
    raw_smplx = anno['raw_smplx']
    raw_mano = anno['raw_mano']
    obj_transf = anno['obj_transf']

    for t, frame_id in enumerate(frame_indices):
        actual_frame = mocap_frames[frame_id] if frame_id < len(mocap_frames) else mocap_frames[-1]

        # Get SMPL-X data for this frame
        if actual_frame not in raw_smplx:
            # Try to find closest frame
            available_frames = list(raw_smplx.keys())
            actual_frame = min(available_frames, key=lambda x: abs(x - actual_frame))

        smplx_frame = raw_smplx[actual_frame]
        mano_frame = raw_mano.get(actual_frame, raw_mano[list(raw_mano.keys())[0]])

        # --- Body Root (9D) ---
        world_tsl = smplx_frame['world_tsl'].squeeze().numpy()  # (3,)
        world_rot_quat = smplx_frame['world_rot'].squeeze()  # (4,)
        world_rot_mat = quaternion_to_matrix(world_rot_quat.unsqueeze(0))  # (1, 3, 3)
        world_rot_6d = matrix_to_rotation_6d(world_rot_mat).squeeze().numpy()  # (6,)

        features[t, 0:3] = world_tsl
        features[t, 3:9] = world_rot_6d

        # --- Body Pose (126D) ---
        body_pose_quat = smplx_frame['body_pose'].squeeze()  # (21, 4)
        body_pose_mat = quaternion_to_matrix(body_pose_quat)  # (21, 3, 3)
        body_pose_6d = matrix_to_rotation_6d(body_pose_mat)  # (21, 6)
        features[t, 9:135] = body_pose_6d.reshape(-1).numpy()

        # --- Left Hand PCA (30D) ---
        lh_tsl = mano_frame['lh__tsl'].squeeze().numpy()  # (3,)
        lh_pose_quat = mano_frame['lh__pose_coeffs'].squeeze()  # (16, 4)

        lh_pos, lh_orient = mano_converter.get_hand_position_and_orient(
            torch.tensor(lh_tsl).unsqueeze(0),
            lh_pose_quat.unsqueeze(0)
        )
        lh_pca = mano_converter.quaternion_to_pca(lh_pose_quat.unsqueeze(0))

        features[t, 135:138] = lh_pos.squeeze().numpy()
        features[t, 138:144] = lh_orient.squeeze().numpy()
        features[t, 144:165] = lh_pca.squeeze().numpy()

        # --- Right Hand PCA (30D) ---
        rh_tsl = mano_frame['rh__tsl'].squeeze().numpy()  # (3,)
        rh_pose_quat = mano_frame['rh__pose_coeffs'].squeeze()  # (16, 4)

        rh_pos, rh_orient = mano_converter.get_hand_position_and_orient(
            torch.tensor(rh_tsl).unsqueeze(0),
            rh_pose_quat.unsqueeze(0)
        )
        rh_pca = mano_converter.quaternion_to_pca(rh_pose_quat.unsqueeze(0))

        features[t, 165:168] = rh_pos.squeeze().numpy()
        features[t, 168:174] = rh_orient.squeeze().numpy()
        features[t, 174:195] = rh_pca.squeeze().numpy()

        # --- Left Hand Quaternions (67D) ---
        features[t, 195:198] = lh_tsl
        features[t, 198:262] = lh_pose_quat.reshape(-1).numpy()

        # --- Right Hand Quaternions (67D) ---
        features[t, 262:265] = rh_tsl
        features[t, 265:329] = rh_pose_quat.reshape(-1).numpy()

        # --- SDF (42D) ---
        # TODO: Compute SDF from hand joint positions to object surfaces
        features[t, 329:371] = 0.0

        # --- Object Poses (27D for 3 objects) ---
        obj_list = all_objects[:config.max_objects]

        for obj_idx, obj_id in enumerate(obj_list):
            if obj_id in obj_transf and actual_frame in obj_transf[obj_id]:
                obj_transform = obj_transf[obj_id][actual_frame]
                obj_pos = obj_transform[:3, 3]
                obj_rot_mat = torch.tensor(obj_transform[:3, :3]).unsqueeze(0)
                obj_rot_6d = matrix_to_rotation_6d(obj_rot_mat).squeeze().numpy()

                offset = 371 + obj_idx * 9
                features[t, offset:offset+3] = obj_pos
                features[t, offset+3:offset+9] = obj_rot_6d

    return features


def main():
    parser = argparse.ArgumentParser(description="Preprocess OakInk2 dataset for DiffH2O")
    parser.add_argument("--oakink2_root", type=str,
                        default="/hhd4/lizhe/dataset/OakInk2/data",
                        help="Path to OakInk2 data directory")
    parser.add_argument("--output_root", type=str,
                        default="dataset/OAKINK2",
                        help="Output directory for processed data")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of sequences to process (0 for all)")
    parser.add_argument("--mano_path", type=str, default=None,
                        help="Path to MANO model directory")
    parser.add_argument("--mode", type=str, default="primitive",
                        choices=["primitive", "complex", "both"],
                        help="Extraction mode: primitive (short segments), complex (full sequences), or both")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Maximum frames per sample (0 for no limit)")
    args = parser.parse_args()

    config = OakInk2Config(
        oakink2_root=args.oakink2_root,
        output_root=args.output_root,
        mode=args.mode,
        max_motion_len=args.max_frames,
    )

    print(f"\n=== OakInk2 Preprocessing ===")
    print(f"Mode: {config.mode}")
    print(f"Output: {config.output_root}")
    print(f"Max frames: {'unlimited' if config.max_motion_len == 0 else config.max_motion_len}")

    # Create output directories based on mode
    if config.mode == 'primitive':
        output_dirs = [('primitive', 'oakink2_primitive', 'texts_primitive')]
    elif config.mode == 'complex':
        output_dirs = [('complex', 'oakink2_complex', 'texts_complex')]
    else:  # both
        output_dirs = [
            ('primitive', 'oakink2_primitive', 'texts_primitive'),
            ('complex', 'oakink2_complex', 'texts_complex'),
        ]

    for _, motion_subdir, text_subdir in output_dirs:
        os.makedirs(os.path.join(config.output_root, motion_subdir), exist_ok=True)
        os.makedirs(os.path.join(config.output_root, text_subdir), exist_ok=True)

    # Initialize converters
    mano_converter = MANOPCAConverter(mano_path=args.mano_path)
    object_processor = ObjectProcessor(
        os.path.join(config.oakink2_root, config.object_dir)
    )

    # Load task targets for complex mode
    task_targets = load_task_targets(config.oakink2_root)

    # Find all annotation files
    anno_dir = os.path.join(config.oakink2_root, config.anno_dir)
    program_info_dir = os.path.join(config.oakink2_root, config.program_dir, "program_info")
    desc_info_dir = os.path.join(config.oakink2_root, config.program_dir, "desc_info")

    pkl_files = sorted([f for f in os.listdir(anno_dir) if f.endswith('.pkl')])

    if args.num_samples > 0:
        pkl_files = pkl_files[:args.num_samples]

    print(f"Processing {len(pkl_files)} sequences...")

    # Track samples for each mode
    primitive_samples = {'idx': 0, 'names': [], 'features': []}
    complex_samples = {'idx': 0, 'names': [], 'features': []}

    for pkl_file in tqdm(pkl_files, desc="Processing sequences"):
        seq_key = pkl_file.replace('.pkl', '')

        pkl_path = os.path.join(anno_dir, pkl_file)
        program_path = os.path.join(program_info_dir, f"{seq_key}.json")
        desc_path = os.path.join(desc_info_dir, f"{seq_key}.json")

        if not os.path.exists(program_path) or not os.path.exists(desc_path):
            continue

        try:
            # Load annotation
            with open(pkl_path, 'rb') as f:
                anno = pickle.load(f)
            with open(desc_path, 'r') as f:
                desc_info = json.load(f)

            # Get all objects in the sequence
            all_objects = anno.get('obj_list', [])

            # === PRIMITIVE MODE ===
            if config.mode in ['primitive', 'both']:
                segments, _ = load_oakink2_sequence(
                    pkl_path, program_path, desc_path, config
                )

                for segment in segments:
                    try:
                        features = process_segment(
                            anno, segment, config, mano_converter, object_processor
                        )
                    except Exception as e:
                        continue

                    if features is None:
                        continue

                    # Save motion data
                    motion_dir = os.path.join(config.output_root, 'oakink2_primitive')
                    text_dir = os.path.join(config.output_root, 'texts_primitive')

                    motion_path = os.path.join(motion_dir, f"{primitive_samples['idx']:06d}.npy")
                    np.save(motion_path, features)

                    # Save text annotation
                    text_path = os.path.join(text_dir, f"{primitive_samples['idx']:06d}.txt")
                    with open(text_path, 'w') as f:
                        f.write(create_text_annotation(segment['text']))

                    primitive_samples['names'].append(
                        f"{seq_key}_{segment['primitive']}_{segment['start_frame']}"
                    )
                    primitive_samples['features'].append(features)
                    primitive_samples['idx'] += 1

            # === COMPLEX MODE ===
            if config.mode in ['complex', 'both']:
                try:
                    features = process_full_sequence(
                        anno, config, mano_converter, object_processor, all_objects
                    )
                except Exception as e:
                    print(f"Error processing full sequence {seq_key}: {e}")
                    features = None

                if features is not None:
                    # Save motion data
                    motion_dir = os.path.join(config.output_root, 'oakink2_complex')
                    text_dir = os.path.join(config.output_root, 'texts_complex')

                    motion_path = os.path.join(motion_dir, f"{complex_samples['idx']:06d}.npy")
                    np.save(motion_path, features)

                    # Get both text descriptions
                    high_level, concatenated = get_complex_task_description(
                        seq_key, desc_info, task_targets
                    )

                    # Save text annotation with both formats
                    # Line 1: high-level description
                    # Line 2: concatenated primitive descriptions
                    text_path = os.path.join(text_dir, f"{complex_samples['idx']:06d}.txt")
                    with open(text_path, 'w') as f:
                        # Use high-level if available, otherwise concatenated
                        main_text = high_level if high_level else concatenated
                        f.write(create_text_annotation(main_text))
                        # Also store concatenated as second line
                        if concatenated and concatenated != main_text:
                            f.write(create_text_annotation(concatenated))

                    complex_samples['names'].append(seq_key)
                    complex_samples['features'].append(features)
                    complex_samples['idx'] += 1

        except Exception as e:
            print(f"Error loading {seq_key}: {e}")
            continue

    # Print summary
    if config.mode in ['primitive', 'both']:
        print(f"\nPrimitive mode: {primitive_samples['idx']} segments")
    if config.mode in ['complex', 'both']:
        print(f"Complex mode: {complex_samples['idx']} full sequences")

    # Helper function to save mode-specific outputs
    def save_mode_outputs(samples_dict: Dict, mode_name: str, motion_subdir: str):
        """Save file names, stats, and splits for a mode."""
        if samples_dict['idx'] == 0:
            return

        # Save file names mapping
        file_names_path = os.path.join(config.output_root, f"file_names_{mode_name}.txt")
        with open(file_names_path, 'w') as f:
            for idx, name in enumerate(samples_dict['names']):
                f.write(f"{idx:06d}\t{name}\n")

        # Compute and save normalization statistics
        if samples_dict['features']:
            all_feats = np.concatenate(samples_dict['features'], axis=0)
            mean = np.mean(all_feats, axis=0)
            std = np.std(all_feats, axis=0)
            std[std < 1e-6] = 1.0  # Avoid division by zero

            np.save(os.path.join(config.output_root, f"Mean_oakink2_{mode_name}.npy"), mean)
            np.save(os.path.join(config.output_root, f"Std_oakink2_{mode_name}.npy"), std)

            print(f"\n{mode_name.capitalize()} normalization stats:")
            print(f"  Mean shape: {mean.shape}")
            print(f"  Std shape: {std.shape}")

        # Create train/test split (80/20)
        num_samples = samples_dict['idx']
        indices = list(range(num_samples))
        np.random.seed(42)
        np.random.shuffle(indices)

        split_idx = int(0.8 * num_samples)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_path = os.path.join(config.output_root, f"train_oakink2_{mode_name}.txt")
        test_path = os.path.join(config.output_root, f"test_oakink2_{mode_name}.txt")

        with open(train_path, 'w') as f:
            for idx in train_indices:
                f.write(f"{idx:06d}\n")

        with open(test_path, 'w') as f:
            for idx in test_indices:
                f.write(f"{idx:06d}\n")

        print(f"\n{mode_name.capitalize()} train/test split:")
        print(f"  Train: {len(train_indices)} samples")
        print(f"  Test: {len(test_indices)} samples")

    # Save outputs for each mode
    if config.mode in ['primitive', 'both']:
        save_mode_outputs(primitive_samples, 'primitive', 'oakink2_primitive')

    if config.mode in ['complex', 'both']:
        save_mode_outputs(complex_samples, 'complex', 'oakink2_complex')

    # Compute BPS encodings for all objects used
    print("\nComputing BPS encodings for objects...")
    bps_dict = {}
    for pkl_file in tqdm(pkl_files[:args.num_samples] if args.num_samples > 0 else pkl_files,
                         desc="Computing BPS"):
        seq_key = pkl_file.replace('.pkl', '')
        pkl_path = os.path.join(anno_dir, pkl_file)

        try:
            with open(pkl_path, 'rb') as f:
                anno = pickle.load(f)
            obj_list = anno.get('obj_list', [])

            for obj_id in obj_list:
                if obj_id in bps_dict:
                    continue

                mesh = object_processor.load_mesh(obj_id)
                if mesh is not None:
                    bps = object_processor.compute_bps(mesh)
                    bps_dict[obj_id] = bps
        except Exception as e:
            print(f"Error computing BPS for {seq_key}: {e}")

    # Save BPS encodings
    bps_path = os.path.join(config.output_root, "bps_enc_oakink2.npy")
    np.save(bps_path, bps_dict, allow_pickle=True)
    print(f"Saved BPS encodings for {len(bps_dict)} objects to {bps_path}")

    print("\nPreprocessing complete!")
    print(f"Output directory: {config.output_root}")


if __name__ == "__main__":
    main()
