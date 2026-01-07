"""
OakInk2 Dataset for DiffH2O training.

This dataset loads preprocessed OakInk2 data in DiffH2O-compatible format.
"""

import os
import codecs as cs
from os.path import join as pjoin
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from data_loaders.humanml.utils.word_vectorizer import WordVectorizer


# Feature indices for OakInk2 representation (398D)
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


class OakInk2Dataset(data.Dataset):
    """Dataset class for OakInk2 hand-object interaction data.

    This dataset loads preprocessed OakInk2 motion data and text annotations
    in a format compatible with DiffH2O's training pipeline.

    Args:
        data_root: Root directory containing preprocessed OakInk2 data.
        split: 'train' or 'test'.
        mode: 'train', 'eval', or 'gt'.
        max_motion_length: Maximum sequence length (frames).
        min_motion_len: Minimum sequence length (frames).
        max_text_len: Maximum text token length.
        max_objects: Maximum number of objects to include.
        motion_enc_frames: Number of frames for motion encoding conditioning.
    """

    def __init__(
        self,
        data_root: str = "dataset/OAKINK2",
        split: str = "train",
        mode: str = "train",
        max_motion_length: int = 200,
        min_motion_len: int = 20,
        max_text_len: int = 20,
        max_objects: int = 3,
        motion_enc_frames: int = 0,
        glove_dir: str = "glove",
        **kwargs
    ):
        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.max_motion_length = max_motion_length
        self.min_motion_len = min_motion_len
        self.max_text_len = max_text_len
        self.max_objects = max_objects
        self.motion_enc_frames = motion_enc_frames

        # Directories
        self.motion_dir = pjoin(data_root, "oakink2_representation")
        self.text_dir = pjoin(data_root, "texts")

        # Load normalization statistics
        self.mean = np.load(pjoin(data_root, "Mean_oakink2.npy"))
        self.std = np.load(pjoin(data_root, "Std_oakink2.npy"))

        # Load BPS encodings for objects
        bps_path = pjoin(data_root, "bps_enc_oakink2.npy")
        if os.path.exists(bps_path):
            self.bps = np.load(bps_path, allow_pickle=True).item()
        else:
            self.bps = {}
            print(f"WARNING: BPS file not found at {bps_path}")

        # Load file names mapping
        file_names_path = pjoin(data_root, "file_names.txt")
        self.id_dict = {}
        if os.path.exists(file_names_path):
            with open(file_names_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.id_dict[parts[0]] = parts[1]

        # Load split file
        split_file = pjoin(data_root, f"{split}_oakink2.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with cs.open(split_file, 'r') as f:
            id_list = [line.strip() for line in f.readlines()]

        # Load word vectorizer for text embeddings
        self.w_vectorizer = WordVectorizer(glove_dir, 'our_vab')

        # Load data
        self.data_dict = {}
        self.name_list = []
        self.length_list = []

        print(f"Loading OakInk2 {split} dataset...")
        for name in tqdm(id_list, desc="Loading data"):
            try:
                motion_path = pjoin(self.motion_dir, f"{name}.npy")
                if not os.path.exists(motion_path):
                    continue

                motion = np.load(motion_path)

                # Filter by length
                if len(motion) < self.min_motion_len:
                    continue
                if len(motion) > self.max_motion_length:
                    motion = motion[:self.max_motion_length]

                # Load text annotation
                text_path = pjoin(self.text_dir, f"{name}.txt")
                if not os.path.exists(text_path):
                    continue

                text_data = []
                with cs.open(text_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split('#')
                        if len(parts) >= 1:
                            caption = parts[0]
                            tokens = parts[1].split(' ') if len(parts) > 1 and parts[1] else []
                            text_data.append({
                                'caption': caption,
                                'tokens': tokens
                            })

                if not text_data:
                    continue

                # Get sequence ID from mapping
                data_id = self.id_dict.get(name, name)

                self.data_dict[name] = {
                    'motion': motion.astype(np.float32),
                    'length': len(motion),
                    'text': text_data,
                    'id': data_id,
                }
                self.name_list.append(name)
                self.length_list.append(len(motion))

            except Exception as e:
                print(f"Error loading {name}: {e}")

        self.length_arr = np.array(self.length_list)
        print(f"Loaded {len(self.name_list)} samples")

        if len(self.name_list) == 0:
            raise ValueError("No valid samples loaded. Check data paths.")

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]

        motion = data['motion'].copy()
        m_length = data['length']
        text_list = data['text']
        data_id = data['id']

        # Select text (prefer detailed if available)
        if len(text_list) > 1:
            text_data = text_list[1]
        else:
            text_data = text_list[0]

        caption = text_data['caption']
        tokens = text_data['tokens']

        # Process tokens
        if len(tokens) == 0 or (len(tokens) == 1 and tokens[0] == ''):
            # No tokens provided, use caption words
            tokens = caption.lower().replace('.', '').replace(',', '').split()
            tokens = [f"{t}/NOUN" for t in tokens[:self.max_text_len]]

        if len(tokens) < self.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        # Get word embeddings
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])

        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Keep full motion for evaluation
        motion_full = motion.copy()

        # Normalize motion
        motion = (motion - self.mean) / self.std

        # Pad to max length
        if m_length < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.tile(motion[m_length - 1], (self.max_motion_length - m_length, 1))
            ], axis=0)

        # Get object BPS (use first available object)
        obj_bps = self._get_object_bps(data_id)

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            '_'.join(tokens),
            obj_bps,
            self.motion_enc_frames,
            motion_full,
            data_id
        )

    def _get_object_bps(self, data_id: str) -> np.ndarray:
        """Get BPS encoding for objects in the sequence.

        Args:
            data_id: Sequence identifier.

        Returns:
            BPS encoding of shape (max_objects, 1024, 3) or (1024, 3).
        """
        # Try to find object BPS from the data_id
        # data_id format: scene_xx__Axxx++seq__uuid__timestamp_primitive_frame

        # For now, return the first available BPS or zeros
        if self.bps:
            # Get first available BPS as fallback
            first_obj = list(self.bps.keys())[0]
            bps = self.bps[first_obj]
            return torch.tensor(bps).unsqueeze(0)

        # Return zeros if no BPS available
        return torch.zeros(1, 1024, 3)

    def get_std_mean(self):
        """Get normalization statistics."""
        return self.std, self.mean

    def inv_transform(self, data):
        """Inverse transform normalized motion data (numpy)."""
        return data * self.std + self.mean

    def inv_transform_th(self, data, traject_only=None, use_rand_proj=None):
        """Inverse transform normalized motion data (torch).

        Args:
            data: Normalized motion tensor.
            traject_only: Unused, for compatibility with GRAB interface.
            use_rand_proj: Unused, for compatibility with GRAB interface.

        Returns:
            Denormalized motion tensor.
        """
        import torch
        std = torch.from_numpy(self.std).to(data.device).to(data.dtype)
        mean = torch.from_numpy(self.mean).to(data.device).to(data.dtype)
        return data * std + mean

    def transform_th(self, data, traject_only=None, use_rand_proj=None):
        """Forward transform motion data (torch).

        Args:
            data: Raw motion tensor.
            traject_only: Unused, for compatibility with GRAB interface.
            use_rand_proj: Unused, for compatibility with GRAB interface.

        Returns:
            Normalized motion tensor.
        """
        import torch
        std = torch.from_numpy(self.std).to(data.device).to(data.dtype)
        mean = torch.from_numpy(self.mean).to(data.device).to(data.dtype)
        return (data - mean) / std


class OakInk2(data.Dataset):
    """Wrapper class for OakInk2 dataset compatible with DiffH2O training."""

    def __init__(
        self,
        mode: str,
        split: str = "train",
        data_root: str = "dataset/OAKINK2",
        num_frames: Optional[int] = None,
        motion_enc_frames: int = 0,
        **kwargs
    ):
        self.mode = mode
        self.dataset_name = 'oakink2'
        self.dataname = 'oakink2'

        max_motion_length = num_frames if num_frames is not None else 200

        self.dataset = OakInk2Dataset(
            data_root=data_root,
            split=split,
            mode=mode,
            max_motion_length=max_motion_length,
            motion_enc_frames=motion_enc_frames,
            **kwargs
        )

        self.mean = self.dataset.mean
        self.std = self.dataset.std
        self.num_actions = 1  # Placeholder

        # Add t2m_dataset alias for compatibility with DiffH2O training code
        # Training code accesses data.dataset.t2m_dataset.inv_transform_th(), etc.
        self.t2m_dataset = self.dataset

        print(f"OakInk2 dataset loaded: {len(self.dataset)} samples")

    def __getitem__(self, item):
        return self.dataset.__getitem__(item)

    def __len__(self):
        return len(self.dataset)

    def get_std_mean(self):
        return self.std, self.mean
