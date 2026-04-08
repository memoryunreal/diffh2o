"""
HIMO V4 Flow Dataset: loads HIMO motion (489D/498D) + state arrays + BPS + text
for CRR-Flow V4 entity-centric training.

Supports both 2-object (489D, 5 entities) and 3-object (498D, 6 entities) modes.
Compatible with train_flow.py via HIMOFlowWrapper.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HIMOFlowV4Dataset(Dataset):
    """HIMO V4 dataset: position-only motion + scalar BPS + state labels.

    Data layout (2o, 489D):
        body: [0, 201)    = 22j×3pos + 22j×6rot + 3tsl = 201D
        lhand: [201, 336) = 15j×3pos + 15j×6rot = 135D
        rhand: [336, 471) = 15j×3pos + 15j×6rot = 135D
        obj1: [471, 480)  = 3pos + 6rot = 9D
        obj2: [480, 489)  = 3pos + 6rot = 9D

    Data layout (3o, 498D):
        Same as 2o + obj3: [489, 498) = 9D
    """

    def __init__(self, data_root, split='train',
                 max_motion_length=300, num_frames=300,
                 max_objects=4, state_dir=None):
        self.data_root = data_root
        self.split = split
        self.max_motion_length = max_motion_length
        self.max_objects = max_objects

        # Load metadata
        meta_path = os.path.join(data_root, 'metadata.json')
        with open(meta_path) as f:
            self.metadata = json.load(f)

        self.num_objects = self.metadata['num_objects']
        self.num_entities = self.metadata['num_entities']

        # Load normalization
        self.mean = np.load(os.path.join(data_root, 'Mean_flow.npy'))
        self.std = np.load(os.path.join(data_root, 'Std_flow.npy'))
        self.std[self.std < 1e-2] = 1e-2
        self.feature_dim = len(self.mean)

        # Load split file
        split_file = os.path.join(data_root, f'{split}_flow.txt')
        with open(split_file) as f:
            self.indices = [line.strip() for line in f if line.strip()]

        # Load file names mapping
        name_map = {}
        names_path = os.path.join(data_root, 'file_names.txt')
        if os.path.exists(names_path):
            with open(names_path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        name_map[parts[0]] = parts[1]

        # Check for precomputed CLIP embeddings
        clip_emb_dir = os.path.join(data_root, 'clip_embeddings')
        self.use_clip_emb = os.path.isdir(clip_emb_dir)
        if self.use_clip_emb:
            print(f"Found precomputed CLIP embeddings at {clip_emb_dir}")

        # State arrays directory
        self.state_dir = state_dir or os.path.join(data_root, 'state_arrays')
        if state_dir:
            print(f"Using custom state arrays from {state_dir}")

        # Pre-load all data
        self.motions = []
        self.bps_data = []
        self.states = []
        self.texts = []
        self.clip_embs = []
        self.lengths = []
        self.name_list = []

        skipped = 0
        for idx_str in self.indices:
            motion_path = os.path.join(data_root, 'motion_dynamic', f'{idx_str}.npy')
            bps_path = os.path.join(data_root, 'bps', f'{idx_str}.npy')
            state_path = os.path.join(self.state_dir, f'{idx_str}.npy')
            text_path = os.path.join(data_root, 'texts', f'{idx_str}.txt')

            if not os.path.exists(motion_path):
                skipped += 1
                continue

            motion = np.load(motion_path).astype(np.float32)  # [T, 489/498]

            # Normalize motion
            motion = (motion - self.mean) / self.std

            # Load BPS [N_obj, 1024] → pad to [max_objects, 1024]
            bps = np.zeros((max_objects, 1024), dtype=np.float32)
            if os.path.exists(bps_path):
                raw_bps = np.load(bps_path).astype(np.float32)  # [N_obj, 1024]
                n_obj = min(raw_bps.shape[0], max_objects)
                bps[:n_obj] = raw_bps[:n_obj]

            # Load state [T, N_entities]
            state = np.load(state_path).astype(np.int8)  # [T, 5 or 6]

            # Load text or CLIP embedding
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

        if skipped > 0:
            print(f"Warning: skipped {skipped} missing motion files")
        print(f"HIMOFlowV4Dataset ({split}): {len(self.motions)} samples, "
              f"dim={self.feature_dim}, mode={self.metadata['mode']}, "
              f"clip_emb={'yes' if self.use_clip_emb else 'no'}")

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, idx):
        motion = self.motions[idx].copy()  # [T, D]
        bps = self.bps_data[idx].copy()    # [max_objects, 1024]
        state = self.states[idx].copy()    # [T, N_entities]
        text = self.texts[idx]
        clip_emb = self.clip_embs[idx]
        m_length = self.lengths[idx]

        # Pad temporal dimension to max_motion_length
        T = motion.shape[0]
        if T < self.max_motion_length:
            pad_len = self.max_motion_length - T
            motion = np.concatenate(
                [motion, np.tile(motion[-1:], (pad_len, 1))], axis=0)
            state = np.concatenate(
                [state, np.tile(state[-1:], (pad_len, 1))], axis=0)

        item = {
            'motion': torch.tensor(motion, dtype=torch.float32),       # [T, D]
            'bps': torch.tensor(bps, dtype=torch.float32),             # [max_obj, 1024]
            'state_labels': torch.tensor(state, dtype=torch.long),     # [T, E]
            'text': text,
            'm_length': m_length,
        }
        if clip_emb is not None:
            item['clip_emb'] = torch.tensor(clip_emb, dtype=torch.float32)
        return item

    def inv_transform(self, data):
        """Denormalize motion data (numpy)."""
        return data * self.std + self.mean

    def inv_transform_th(self, data, **kwargs):
        """Denormalize motion data (torch)."""
        std = torch.from_numpy(self.std).to(data.device).to(data.dtype)
        mean = torch.from_numpy(self.mean).to(data.device).to(data.dtype)
        return data * std + mean


def himo_flow_v4_collate_fn(batch):
    """Collate function for HIMO V4 dataset (same protocol as OakInk2 V4)."""
    motions = torch.stack([b['motion'] for b in batch])          # [B, T, D]
    bps = torch.stack([b['bps'] for b in batch])                 # [B, max_obj, 1024]
    states = torch.stack([b['state_labels'] for b in batch])     # [B, T, E]
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


class HIMOFlowWrapper:
    """Wrapper for HIMO V4 dataset, compatible with train_flow.py.

    Usage:
        wrapper = HIMOFlowWrapper(
            split='train',
            data_root='dataset/HIMO_V4_2o',
            num_frames=300,
            batch_size=32,
        )
        loader = wrapper.get_loader()
    """

    def __init__(self, split='train', data_root='dataset/HIMO_V4_2o',
                 num_frames=300, batch_size=32, max_objects=4,
                 state_dir=None, **kwargs):
        self.dataset = HIMOFlowV4Dataset(
            data_root=data_root,
            split=split,
            max_motion_length=num_frames,
            num_frames=num_frames,
            max_objects=max_objects,
            state_dir=state_dir,
        )
        self.collate_fn = himo_flow_v4_collate_fn
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
