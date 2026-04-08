"""
Unified Transition Dataset for training the transition diffusion model.

Supports two modes:
1. OakInk2 (398D): Real transitions extracted from natural inter-primitive gaps
2. GRAB (117D): Self-supervised infilling via random masking of existing sequences

This dual-dataset approach is a research contribution: the transition model works
with both real transition data and self-supervised infilling.
"""

import os
import codecs as cs
from os.path import join as pjoin
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

from data_loaders.humanml.utils.word_vectorizer import WordVectorizer


class TransitionDataset(data.Dataset):
    """Unified transition dataset for OakInk2 and GRAB.

    OakInk2 mode: Loads pre-extracted real transitions from oakink2_transitions/
    GRAB mode: On-the-fly masked infilling from existing GRAB sequences.

    Args:
        data_root: Root data directory
        dataset_type: 'oakink2_transition' or 'grab_transition'
        split: 'train' or 'test'
        transition_frames: Number of frames in the transition (resampled)
        boundary_frames: Number of context frames from each adjacent segment
        max_text_len: Maximum text token length
        min_mask_frames: Minimum mask window for GRAB infilling
        max_mask_frames: Maximum mask window for GRAB infilling
        augment: Whether to apply data augmentation
    """

    def __init__(
        self,
        data_root: str,
        dataset_type: str = "oakink2_transition",
        split: str = "train",
        transition_frames: int = 60,
        boundary_frames: int = 10,
        max_text_len: int = 20,
        min_mask_frames: int = 20,
        max_mask_frames: int = 60,
        augment: bool = True,
        glove_dir: str = "glove",
        **kwargs,
    ):
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.split = split
        self.transition_frames = transition_frames
        self.boundary_frames = boundary_frames
        self.max_text_len = max_text_len
        self.min_mask_frames = min_mask_frames
        self.max_mask_frames = max_mask_frames
        self.augment = augment

        # Word vectorizer for text embeddings
        self.w_vectorizer = WordVectorizer(glove_dir, "our_vab")

        if dataset_type == "oakink2_transition":
            self._init_oakink2(data_root, split)
        elif dataset_type == "grab_transition":
            self._init_grab(data_root, split)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        print(f"TransitionDataset ({dataset_type}, {split}): {len(self)} samples")

    # ----------------------------------------------------------------
    # OakInk2 initialization
    # ----------------------------------------------------------------
    def _init_oakink2(self, data_root: str, split: str):
        """Load pre-extracted OakInk2 transitions."""
        self.feature_dim = 398

        trans_dir = pjoin(data_root, "oakink2_transitions")
        before_dir = pjoin(data_root, "boundaries_before")
        after_dir = pjoin(data_root, "boundaries_after")
        text_dir = pjoin(data_root, "texts_transitions")

        # Load normalization
        mean_path = pjoin(data_root, "Mean_transitions.npy")
        std_path = pjoin(data_root, "Std_transitions.npy")

        if os.path.exists(mean_path):
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
        else:
            # Fallback to primitive normalization
            self.mean = np.load(pjoin(data_root, "Mean_oakink2_primitive.npy"))
            self.std = np.load(pjoin(data_root, "Std_oakink2_primitive.npy"))

        # Load split
        split_file = pjoin(data_root, f"{split}_transitions.txt")
        if os.path.exists(split_file):
            with open(split_file) as f:
                id_list = [line.strip() for line in f if line.strip()]
        else:
            # Use all available files
            id_list = sorted(
                f.replace(".npy", "")
                for f in os.listdir(trans_dir)
                if f.endswith(".npy")
            )

        # Load BPS
        bps_path = pjoin(data_root, "bps_enc_oakink2.npy")
        self.bps = {}
        if os.path.exists(bps_path):
            self.bps = np.load(bps_path, allow_pickle=True).item()

        # Load data
        self.data_list = []
        for name in tqdm(id_list, desc=f"Loading OakInk2 transitions ({split})"):
            tp = pjoin(trans_dir, f"{name}.npy")
            bp = pjoin(before_dir, f"{name}.npy")
            ap = pjoin(after_dir, f"{name}.npy")
            txp = pjoin(text_dir, f"{name}.txt")

            if not all(os.path.exists(p) for p in [tp, bp, ap]):
                continue

            transition = np.load(tp).astype(np.float32)
            boundary_before = np.load(bp).astype(np.float32)
            boundary_after = np.load(ap).astype(np.float32)

            # Load text
            text_source = ""
            text_target = ""
            if os.path.exists(txp):
                with open(txp) as f:
                    lines = f.readlines()
                    if len(lines) >= 3:
                        text_source = lines[1].split("#")[0].strip()
                        text_target = lines[2].split("#")[0].strip()
                    elif len(lines) >= 1:
                        text_source = lines[0].split("#")[0].strip()

            self.data_list.append({
                "transition": transition,
                "boundary_before": boundary_before,
                "boundary_after": boundary_after,
                "text_source": text_source,
                "text_target": text_target,
                "name": name,
            })

    # ----------------------------------------------------------------
    # GRAB initialization
    # ----------------------------------------------------------------
    def _init_grab(self, data_root: str, split: str):
        """Load GRAB sequences for on-the-fly infilling."""
        self.feature_dim = 117

        motion_dir = pjoin(data_root, "diffh2o_representation_full")
        text_dir = pjoin(data_root, "texts_simple")

        # Load normalization
        self.mean = np.load(pjoin(data_root, "Mean_diffh2o_full.npy"))
        self.std = np.load(pjoin(data_root, "Std_diffh2o_full.npy"))

        # Load BPS
        bps_path = pjoin(data_root, "bps_enc_obj.npy")
        self.bps = {}
        if os.path.exists(bps_path):
            self.bps = np.load(bps_path, allow_pickle=True).item()

        # Load split file
        split_file = pjoin(data_root, f"{split}.txt")
        if os.path.exists(split_file):
            with open(split_file) as f:
                id_list = [line.strip() for line in f if line.strip()]
        else:
            id_list = sorted(
                f.replace(".npy", "")
                for f in os.listdir(motion_dir)
                if f.endswith(".npy")
            )

        # Load full sequences for infilling
        self.data_list = []
        min_len = self.boundary_frames * 2 + self.min_mask_frames + 10

        for name in tqdm(id_list, desc=f"Loading GRAB sequences ({split})"):
            motion_path = pjoin(motion_dir, f"{name}.npy")
            if not os.path.exists(motion_path):
                continue

            motion = np.load(motion_path).astype(np.float32)
            if len(motion) < min_len:
                continue

            # Load text
            text = ""
            text_path = pjoin(text_dir, f"{name}.txt")
            if os.path.exists(text_path):
                with open(text_path) as f:
                    text = f.readline().split("#")[0].strip()

            self.data_list.append({
                "motion": motion,
                "text": text,
                "name": name,
            })

    # ----------------------------------------------------------------
    # Dataset interface
    # ----------------------------------------------------------------
    def __len__(self):
        if self.dataset_type == "oakink2_transition":
            return len(self.data_list)
        else:
            # GRAB: each sequence can produce many infilling samples
            # Return original count; randomness comes from mask placement
            return len(self.data_list)

    def __getitem__(self, item):
        if self.dataset_type == "oakink2_transition":
            return self._getitem_oakink2(item)
        else:
            return self._getitem_grab(item)

    def _getitem_oakink2(self, item):
        """Get an OakInk2 real transition sample."""
        entry = self.data_list[item]
        transition = entry["transition"].copy()
        boundary_before = entry["boundary_before"].copy()
        boundary_after = entry["boundary_after"].copy()

        # Data augmentation
        if self.augment and self.split == "train":
            transition, boundary_before, boundary_after = self._augment(
                transition, boundary_before, boundary_after
            )

        # Normalize
        transition = (transition - self.mean) / self.std
        boundary_before = (boundary_before - self.mean) / self.std
        boundary_after = (boundary_after - self.mean) / self.std

        # Text embeddings
        text_source_emb, text_source_len = self._encode_text(entry["text_source"])
        text_target_emb, text_target_len = self._encode_text(entry["text_target"])

        # Object BPS (placeholder — use first available)
        obj_bps = self._get_default_bps()

        return {
            "transition": torch.tensor(transition, dtype=torch.float32),
            "boundary_before": torch.tensor(boundary_before, dtype=torch.float32),
            "boundary_after": torch.tensor(boundary_after, dtype=torch.float32),
            "text_source_emb": torch.tensor(text_source_emb, dtype=torch.float32),
            "text_target_emb": torch.tensor(text_target_emb, dtype=torch.float32),
            "text_source_len": text_source_len,
            "text_target_len": text_target_len,
            "text_source": entry["text_source"],
            "text_target": entry["text_target"],
            "source_obj_bps": torch.tensor(obj_bps, dtype=torch.float32),
            "target_obj_bps": torch.tensor(obj_bps, dtype=torch.float32),
            "m_length": self.transition_frames,
            "name": entry["name"],
        }

    def _getitem_grab(self, item):
        """Get a GRAB self-supervised infilling sample."""
        entry = self.data_list[item]
        motion = entry["motion"].copy()
        T = len(motion)

        # Random mask window
        mask_len = np.random.randint(self.min_mask_frames, self.max_mask_frames + 1)
        mask_len = min(mask_len, T - 2 * self.boundary_frames - 2)

        # Random mask position (must leave room for boundaries)
        earliest_start = self.boundary_frames
        latest_start = T - self.boundary_frames - mask_len
        if latest_start <= earliest_start:
            # Sequence too short; use fixed position
            mask_start = self.boundary_frames
            mask_len = min(mask_len, T - 2 * self.boundary_frames)
        else:
            mask_start = np.random.randint(earliest_start, latest_start + 1)

        mask_end = mask_start + mask_len

        # Extract segments
        boundary_before = motion[mask_start - self.boundary_frames : mask_start]
        boundary_after = motion[mask_end : mask_end + self.boundary_frames]
        transition_raw = motion[mask_start:mask_end]

        # Pad boundaries if needed
        if len(boundary_before) < self.boundary_frames:
            pad = np.tile(boundary_before[0:1], (self.boundary_frames - len(boundary_before), 1))
            boundary_before = np.concatenate([pad, boundary_before], axis=0)
        if len(boundary_after) < self.boundary_frames:
            pad = np.tile(boundary_after[-1:], (self.boundary_frames - len(boundary_after), 1))
            boundary_after = np.concatenate([boundary_after, pad], axis=0)

        # Resample transition to fixed length
        transition = self._resample(transition_raw, self.transition_frames)

        # Data augmentation
        if self.augment and self.split == "train":
            transition, boundary_before, boundary_after = self._augment(
                transition, boundary_before, boundary_after
            )

        # Normalize
        transition = (transition - self.mean) / self.std
        boundary_before = (boundary_before - self.mean) / self.std
        boundary_after = (boundary_after - self.mean) / self.std

        # Text embeddings (same text for source and target in infilling)
        text_emb, text_len = self._encode_text(entry["text"])
        obj_bps = self._get_default_bps()

        return {
            "transition": torch.tensor(transition, dtype=torch.float32),
            "boundary_before": torch.tensor(boundary_before, dtype=torch.float32),
            "boundary_after": torch.tensor(boundary_after, dtype=torch.float32),
            "text_source_emb": torch.tensor(text_emb, dtype=torch.float32),
            "text_target_emb": torch.tensor(text_emb, dtype=torch.float32),
            "text_source_len": text_len,
            "text_target_len": text_len,
            "text_source": entry["text"],
            "text_target": entry["text"],
            "source_obj_bps": torch.tensor(obj_bps, dtype=torch.float32),
            "target_obj_bps": torch.tensor(obj_bps, dtype=torch.float32),
            "m_length": self.transition_frames,
            "name": entry["name"],
        }

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _resample(self, motion: np.ndarray, target_frames: int) -> np.ndarray:
        """Resample motion to target frame count via linear interpolation."""
        T, D = motion.shape
        if T == target_frames:
            return motion
        old_idx = np.linspace(0, T - 1, T)
        new_idx = np.linspace(0, T - 1, target_frames)
        result = np.zeros((target_frames, D))
        for d in range(D):
            result[:, d] = np.interp(new_idx, old_idx, motion[:, d])
        return result

    def _augment(
        self,
        transition: np.ndarray,
        before: np.ndarray,
        after: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply data augmentation to transition and boundaries."""
        # Time reversal (50% chance)
        if np.random.random() < 0.5:
            transition = transition[::-1].copy()
            before, after = after[::-1].copy(), before[::-1].copy()

        # Speed perturbation (0.8-1.2x)
        if np.random.random() < 0.3:
            speed = np.random.uniform(0.8, 1.2)
            new_len = max(4, int(len(transition) * speed))
            transition = self._resample(transition, new_len)
            transition = self._resample(transition, self.transition_frames)

        # Small noise injection
        if np.random.random() < 0.2:
            noise_scale = 0.001
            transition = transition + np.random.randn(*transition.shape) * noise_scale

        return transition, before, after

    def _encode_text(self, text: str) -> Tuple[np.ndarray, int]:
        """Encode text to word embeddings."""
        if not text:
            text = "transition motion"

        tokens = text.lower().replace(".", "").replace(",", "").split()
        tokens = [f"{t}/NOUN" for t in tokens[: self.max_text_len]]

        if len(tokens) < self.max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[: self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)

        word_embeddings = []
        for token in tokens:
            word_emb, _ = self.w_vectorizer[token]
            word_embeddings.append(word_emb)

        return np.array(word_embeddings, dtype=np.float32), sent_len

    def _get_default_bps(self) -> np.ndarray:
        """Get a default BPS encoding."""
        if self.bps:
            first_key = next(iter(self.bps))
            return self.bps[first_key]
        return np.zeros((1024, 3), dtype=np.float32)

    def get_std_mean(self):
        return self.std, self.mean

    def inv_transform(self, data_arr):
        return data_arr * self.std + self.mean

    def inv_transform_th(self, data_tensor, **kwargs):
        std = torch.from_numpy(self.std).to(data_tensor.device).to(data_tensor.dtype)
        mean = torch.from_numpy(self.mean).to(data_tensor.device).to(data_tensor.dtype)
        return data_tensor * std + mean

    def transform_th(self, data_tensor, **kwargs):
        std = torch.from_numpy(self.std).to(data_tensor.device).to(data_tensor.dtype)
        mean = torch.from_numpy(self.mean).to(data_tensor.device).to(data_tensor.dtype)
        return (data_tensor - mean) / std

    @property
    def max_motion_length(self):
        return self.transition_frames


def transition_collate_fn(batch):
    """Custom collate function for transition batches."""
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], (int, float)):
            collated[key] = torch.tensor(values)
        else:
            collated[key] = values

    # Restructure for compatibility with training loop
    # Training loop expects (motion, cond) tuple
    motion = collated["transition"]  # [B, T, D]
    # Reshape to [B, D, 1, T] to match DiffH2O convention [B, J, nfeats, T]
    # Since nfeats=1, J=feature_dim: motion is [B, T, D] -> [B, D, 1, T]
    motion = motion.permute(0, 2, 1).unsqueeze(2)  # [B, D, 1, T]

    # Create mask (all True since transitions are fixed-length after resampling)
    batch_size = motion.shape[0]
    seq_len = motion.shape[-1]  # T dimension after permute
    lengths = collated["m_length"]
    mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)

    cond = {
        "y": {
            "mask": mask,
            "boundary_before": collated["boundary_before"],
            "boundary_after": collated["boundary_after"],
            "text_source_emb": collated["text_source_emb"],
            "text_target_emb": collated["text_target_emb"],
            "text_source_len": collated.get("text_source_len", None),
            "text_target_len": collated.get("text_target_len", None),
            "text_source": collated.get("text_source", None),
            "text_target": collated.get("text_target", None),
            "source_obj_bps": collated["source_obj_bps"],
            "target_obj_bps": collated["target_obj_bps"],
            "lengths": lengths,
        }
    }

    return motion, cond


class TransitionDatasetWrapper(data.Dataset):
    """Wrapper for DiffH2O training compatibility."""

    def __init__(
        self,
        mode: str,
        dataset_type: str = "oakink2_transition",
        split: str = "train",
        data_root: str = "dataset/OAKINK2",
        transition_frames: int = 60,
        boundary_frames: int = 10,
        **kwargs,
    ):
        self.mode = mode
        self.dataset_name = dataset_type
        self.dataname = dataset_type

        self.dataset = TransitionDataset(
            data_root=data_root,
            dataset_type=dataset_type,
            split=split,
            transition_frames=transition_frames,
            boundary_frames=boundary_frames,
            **kwargs,
        )

        self.mean = self.dataset.mean
        self.std = self.dataset.std
        self.num_actions = 1

        # Compatibility alias
        self.t2m_dataset = self.dataset

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def get_std_mean(self):
        return self.std, self.mean
