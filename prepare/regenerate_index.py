#!/usr/bin/env python3
"""Regenerate index files for OakInk2 primitives without reprocessing features."""

import os
import json
import ast
import numpy as np
from tqdm import tqdm

config_root = '/hhd4/lizhe/dataset/OakInk2/data'
anno_dir = os.path.join(config_root, 'anno_preview')
program_info_dir = os.path.join(config_root, 'program/program_info')
desc_info_dir = os.path.join(config_root, 'program/desc_info')
output_root = 'dataset/OAKINK2'
primitive_dir = os.path.join(output_root, 'oakink2_primitive')

pkl_files = sorted([f for f in os.listdir(anno_dir) if f.endswith('.pkl')])

# 1. Enumerate all segment names in order
segment_names = []
for pkl_file in pkl_files:
    seq_key = pkl_file.replace('.pkl', '')
    program_path = os.path.join(program_info_dir, f'{seq_key}.json')
    desc_path = os.path.join(desc_info_dir, f'{seq_key}.json')
    if not os.path.exists(program_path) or not os.path.exists(desc_path):
        continue
    with open(program_path, 'r') as f:
        program_info = json.load(f)
    for interval_str, prim_data in program_info.items():
        try:
            parsed = ast.literal_eval(interval_str)
            left_interval = parsed[0]
            right_interval = parsed[1]
        except:
            continue
        if right_interval is not None:
            start_frame, end_frame = right_interval
        elif left_interval is not None:
            start_frame, end_frame = left_interval
        else:
            continue
        primitive_type = prim_data.get('primitive', 'unknown')
        segment_names.append(f'{seq_key}_{primitive_type}_{start_frame}')

assert len(segment_names) == 2840, f"Expected 2840, got {len(segment_names)}"
print(f"Enumerated {len(segment_names)} segments")

# 2. Write file_names_primitive.txt
file_names_path = os.path.join(output_root, 'file_names_primitive.txt')
with open(file_names_path, 'w') as f:
    for idx, name in enumerate(segment_names):
        f.write(f'{idx:06d}\t{name}\n')
print(f'Wrote {len(segment_names)} entries to {file_names_path}')

# 3. Compute normalization stats from all .npy files
print('Computing normalization statistics from 2840 files...')
all_feats = []
for i in tqdm(range(2840), desc="Loading features"):
    feat = np.load(os.path.join(primitive_dir, f'{i:06d}.npy'))
    all_feats.append(feat)

all_feats_concat = np.concatenate(all_feats, axis=0)
print(f'Total frames: {all_feats_concat.shape[0]}')

mean = np.mean(all_feats_concat, axis=0)
std = np.std(all_feats_concat, axis=0)
std[std < 1e-6] = 1.0  # Avoid division by zero

np.save(os.path.join(output_root, 'Mean_oakink2_primitive.npy'), mean)
np.save(os.path.join(output_root, 'Std_oakink2_primitive.npy'), std)
print(f'Normalization stats: Mean shape {mean.shape}, Std shape {std.shape}')

# 4. Create train/test split (80/20, same seed as preprocessing)
indices = list(range(2840))
np.random.seed(42)
np.random.shuffle(indices)
split_idx = int(0.8 * 2840)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

with open(os.path.join(output_root, 'train_oakink2_primitive.txt'), 'w') as f:
    for idx in train_indices:
        f.write(f'{idx:06d}\n')

with open(os.path.join(output_root, 'test_oakink2_primitive.txt'), 'w') as f:
    for idx in test_indices:
        f.write(f'{idx:06d}\n')

print(f'Train/test split: {len(train_indices)} / {len(test_indices)}')
print('Done! All index files regenerated.')
