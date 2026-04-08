"""
OakInk2 Flow Dataset: loads motion (398D) + state arrays + text for flow matching.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OakInk2FlowDataset(Dataset):
    """Dataset for OakInk2 flow matching with state labels."""

    def __init__(self, data_root='dataset/OAKINK2_FLOW', split='train',
                 max_motion_length=200, num_frames=200):
        self.data_root = data_root
        self.split = split
        self.max_motion_length = max_motion_length

        # Load normalization (auto-detect dimension from file)
        self.mean = np.load(os.path.join(data_root, 'Mean_flow.npy'))
        self.std = np.load(os.path.join(data_root, 'Std_flow.npy'))
        self.std[self.std < 1e-8] = 1.0
        self.feature_dim = len(self.mean)

        # Load split
        split_file = os.path.join(data_root, f'{split}_flow.txt')
        with open(split_file) as f:
            self.indices = [line.strip() for line in f if line.strip()]

        # Load file names
        self.name_list = []
        name_map = {}
        with open(os.path.join(data_root, 'file_names.txt')) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    name_map[parts[0]] = parts[1]

        # Check for precomputed CLIP embeddings
        clip_emb_dir = os.path.join(data_root, 'clip_embeddings')
        self.use_clip_emb = os.path.isdir(clip_emb_dir)
        if self.use_clip_emb:
            print(f"Found precomputed CLIP embeddings at {clip_emb_dir}")

        # Pre-load all data
        self.motions = []
        self.states = []
        self.texts = []       # raw text strings (fallback)
        self.clip_embs = []   # precomputed CLIP [512] embeddings
        self.lengths = []

        for idx_str in self.indices:
            motion_path = os.path.join(data_root, 'motion', f'{idx_str}.npy')
            state_path = os.path.join(data_root, 'state_arrays', f'{idx_str}.npy')
            text_path = os.path.join(data_root, 'texts', f'{idx_str}.txt')

            if not os.path.exists(motion_path):
                continue

            motion = np.load(motion_path).astype(np.float32)  # [T, D]
            state = np.load(state_path).astype(np.int8)  # [T, E]

            # Normalize motion
            motion = (motion - self.mean) / self.std

            # Load text or precomputed embedding
            text = ""
            clip_emb = None
            if self.use_clip_emb:
                emb_path = os.path.join(clip_emb_dir, f'{idx_str}.npy')
                if os.path.exists(emb_path):
                    clip_emb = np.load(emb_path).astype(np.float32)  # [512]
            if clip_emb is None and os.path.exists(text_path):
                with open(text_path) as f:
                    text = f.readline().split('#')[0].strip()

            T = motion.shape[0]
            self.motions.append(motion)
            self.states.append(state)
            self.texts.append(text)
            self.clip_embs.append(clip_emb)
            self.lengths.append(T)
            self.name_list.append(name_map.get(idx_str, idx_str))

        print(f"OakInk2FlowDataset ({split}): {len(self.motions)} samples loaded"
              f" (clip_emb={'yes' if self.use_clip_emb else 'no'})")

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        motion = self.motions[idx].copy()  # [T, D]
        state = self.states[idx].copy()  # [T, E]
        text = self.texts[idx]
        clip_emb = self.clip_embs[idx]
        m_length = self.lengths[idx]

        # Pad to max_motion_length
        T = motion.shape[0]
        if T < self.max_motion_length:
            pad_len = self.max_motion_length - T
            motion = np.concatenate([motion, np.tile(motion[-1:], (pad_len, 1))], axis=0)
            state = np.concatenate([state, np.tile(state[-1:], (pad_len, 1))], axis=0)

        item = {
            'motion': torch.tensor(motion, dtype=torch.float32),  # [T, D]
            'state_labels': torch.tensor(state, dtype=torch.long),  # [T, E]
            'text': text,
            'm_length': m_length,
        }
        if clip_emb is not None:
            item['clip_emb'] = torch.tensor(clip_emb, dtype=torch.float32)  # [512]
        return item

    def inv_transform(self, data):
        """Denormalize motion data."""
        return data * self.std + self.mean

    def inv_transform_th(self, data, **kwargs):
        """Denormalize torch tensor."""
        std = torch.from_numpy(self.std).to(data.device).to(data.dtype)
        mean = torch.from_numpy(self.mean).to(data.device).to(data.dtype)
        return data * std + mean


def flow_collate_fn(batch):
    """Collate function for flow dataset."""
    motions = torch.stack([b['motion'] for b in batch])  # [B, T, D]
    states = torch.stack([b['state_labels'] for b in batch])  # [B, T, E]
    lengths = torch.tensor([b['m_length'] for b in batch])

    y = {
        'state_labels': states,
        'lengths': lengths,
    }

    # Use precomputed CLIP embeddings if available, else raw text
    if 'clip_emb' in batch[0]:
        y['text_emb'] = torch.stack([b['clip_emb'] for b in batch])  # [B, 512]
    else:
        y['text'] = [b['text'] for b in batch]

    return motions, {'y': y}


class OakInk2FlowV4Dataset(OakInk2FlowDataset):
    """V4 dataset: dynamic motion (534D) + BPS embeddings (4x1024)."""

    def __init__(self, data_root='dataset/OAKINK2_V4', split='train',
                 max_motion_length=200, num_frames=200,
                 state_dir=None):
        # V4 uses motion_dynamic/ instead of motion/
        self.data_root = data_root
        self.split = split
        self.max_motion_length = max_motion_length

        # Load normalization (534D for dynamic features)
        self.mean = np.load(os.path.join(data_root, 'Mean_flow.npy'))
        self.std = np.load(os.path.join(data_root, 'Std_flow.npy'))
        self.std[self.std < 1e-8] = 1.0
        self.feature_dim = len(self.mean)

        split_file = os.path.join(data_root, f'{split}_flow.txt')
        with open(split_file) as f:
            self.indices = [line.strip() for line in f if line.strip()]

        self.name_list = []
        name_map = {}
        with open(os.path.join(data_root, 'file_names.txt')) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    name_map[parts[0]] = parts[1]

        clip_emb_dir = os.path.join(data_root, 'clip_embeddings')
        self.use_clip_emb = os.path.isdir(clip_emb_dir)
        if self.use_clip_emb:
            print(f"Found precomputed CLIP embeddings at {clip_emb_dir}")

        self.motions = []
        self.bps_data = []
        self.states = []
        self.texts = []
        self.clip_embs = []
        self.lengths = []

        # State arrays directory (default or custom fixed states)
        self.state_dir = state_dir or os.path.join(data_root, 'state_arrays')
        if state_dir:
            print(f"Using custom state arrays from {state_dir}")

        for idx_str in self.indices:
            motion_path = os.path.join(data_root, 'motion_dynamic', f'{idx_str}.npy')
            bps_path = os.path.join(data_root, 'bps', f'{idx_str}.npy')
            state_path = os.path.join(self.state_dir, f'{idx_str}.npy')
            text_path = os.path.join(data_root, 'texts', f'{idx_str}.txt')

            if not os.path.exists(motion_path):
                continue

            motion = np.load(motion_path).astype(np.float32)  # [T, 534]
            motion = (motion - self.mean) / self.std

            bps = np.zeros((4, 1024), dtype=np.float32)
            if os.path.exists(bps_path):
                bps = np.load(bps_path).astype(np.float32)  # [4, 1024]

            state = np.load(state_path).astype(np.int8)

            text = ""
            clip_emb = None
            if self.use_clip_emb:
                emb_path = os.path.join(clip_emb_dir, f'{idx_str}.npy')
                if os.path.exists(emb_path):
                    clip_emb = np.load(emb_path).astype(np.float32)
            if clip_emb is None and os.path.exists(text_path):
                with open(text_path) as f:
                    text = f.readline().split('#')[0].strip()

            T = motion.shape[0]
            self.motions.append(motion)
            self.bps_data.append(bps)
            self.states.append(state)
            self.texts.append(text)
            self.clip_embs.append(clip_emb)
            self.lengths.append(T)
            self.name_list.append(name_map.get(idx_str, idx_str))

        print(f"OakInk2FlowV4Dataset ({split}): {len(self.motions)} samples, "
              f"dim={self.feature_dim}, clip_emb={'yes' if self.use_clip_emb else 'no'}")

    def __getitem__(self, idx):
        motion = self.motions[idx].copy()
        bps = self.bps_data[idx].copy()
        state = self.states[idx].copy()
        text = self.texts[idx]
        clip_emb = self.clip_embs[idx]
        m_length = self.lengths[idx]

        T = motion.shape[0]
        if T < self.max_motion_length:
            pad_len = self.max_motion_length - T
            motion = np.concatenate([motion, np.tile(motion[-1:], (pad_len, 1))], axis=0)
            state = np.concatenate([state, np.tile(state[-1:], (pad_len, 1))], axis=0)

        item = {
            'motion': torch.tensor(motion, dtype=torch.float32),
            'bps': torch.tensor(bps, dtype=torch.float32),
            'state_labels': torch.tensor(state, dtype=torch.long),
            'text': text,
            'm_length': m_length,
        }
        if clip_emb is not None:
            item['clip_emb'] = torch.tensor(clip_emb, dtype=torch.float32)
        return item


def flow_v4_collate_fn(batch):
    """Collate function for V4 dataset (includes BPS)."""
    motions = torch.stack([b['motion'] for b in batch])
    bps = torch.stack([b['bps'] for b in batch])
    states = torch.stack([b['state_labels'] for b in batch])
    lengths = torch.tensor([b['m_length'] for b in batch])

    y = {
        'state_labels': states,
        'bps': bps,
        'lengths': lengths,
    }

    if 'clip_emb' in batch[0]:
        y['text_emb'] = torch.stack([b['clip_emb'] for b in batch])
    else:
        y['text'] = [b['text'] for b in batch]

    return motions, {'y': y}


class OakInk2FlowWrapper:
    """Wrapper for compatibility with DiffH2O training interfaces."""

    def __init__(self, split='train', data_root='dataset/OAKINK2_FLOW',
                 num_frames=200, batch_size=32, model_version='v3',
                 state_dir=None, **kwargs):
        if model_version == 'v4':
            self.dataset = OakInk2FlowV4Dataset(
                data_root=data_root, split=split,
                max_motion_length=num_frames, num_frames=num_frames,
                state_dir=state_dir,
            )
            self.collate_fn = flow_v4_collate_fn
        else:
            self.dataset = OakInk2FlowDataset(
                data_root=data_root, split=split,
                max_motion_length=num_frames, num_frames=num_frames,
            )
            self.collate_fn = flow_collate_fn
        self.t2m_dataset = self.dataset
        self.batch_size = batch_size

    def get_loader(self, shuffle=True):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=4,
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )
