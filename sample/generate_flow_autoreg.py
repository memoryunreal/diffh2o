#!/usr/bin/env python3
"""
Autoregressive block-by-block generation for multi-segment complex tasks.

Chains multiple test segments sequentially, using the last K frames of each
generated segment as history for the next.

Usage:
    python -m sample.generate_flow_autoreg \
        --model_path save/v4_autoreg_release_200k/model000060000.pt \
        --model_version v4 --data_root dataset/OAKINK2_V4 \
        --state_dir dataset/OAKINK2_V4/state_arrays_fixed \
        --segment_indices 000200,000201,000205,000206 \
        --output_dir output/ar_chain_rearrange_plate \
        --device 5
"""

import os
import json
import argparse
import numpy as np
import torch
from torchdiffeq import odeint

from utils.fixseed import fixseed
from data_loaders.humanml.data.flow_dataset import OakInk2FlowWrapper
from sample.generate_flow import create_model


def generate_segment_with_history(model, y, x_hist, seq_len, input_dim,
                                   history_len=20, num_steps=10, device='cuda'):
    """Generate one segment conditioned on history from previous segment.

    Args:
        model: trained V4 model
        y: conditioning dict (text_emb, state_labels, bps)
        x_hist: [1, K, D] clean history from previous segment (normalized)
        seq_len: total frames (K + L)
        input_dim: feature dimension (534)
        history_len: K frames of history
        num_steps: ODE integration steps

    Returns:
        full_motion: [1, seq_len, D] = [history | generated_target]
    """
    model.eval()
    B = 1
    K = history_len
    L = seq_len - K

    # ODE integration on target frames only
    x0_target = torch.randn(B, L, input_dim, device=device)

    def ode_fn(t_scalar, x_target):
        t_batch = t_scalar.expand(B).to(device)
        x_full = torch.cat([x_hist, x_target], dim=1)  # [1, T, D]
        with torch.no_grad():
            v_full = model(x_full, t_batch, y=y)
        return v_full[:, K:, :]  # velocity for target only

    t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    trajectory = odeint(ode_fn, x0_target, t_span, method='euler')
    x1_target = trajectory[-1]  # [1, L, D]

    return torch.cat([x_hist, x1_target], dim=1)  # [1, T, D]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_version", default="v4")
    parser.add_argument("--data_root", default="dataset/OAKINK2_V4")
    parser.add_argument("--state_dir", default="")
    parser.add_argument("--segment_indices", required=True,
                        help="Comma-separated test indices forming a complex task, e.g. 000200,000201,000205")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history_len", type=int, default=20)
    parser.add_argument("--num_ode_steps", type=int, default=10)
    args = parser.parse_args()

    fixseed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    seg_indices = [s.strip() for s in args.segment_indices.split(',')]
    print(f"Autoregressive chain: {len(seg_indices)} segments: {seg_indices}")

    # Load training config
    model_dir = os.path.dirname(args.model_path)
    with open(os.path.join(model_dir, 'args.json')) as f:
        train_args = json.load(f)

    # Load dataset
    state_dir = args.state_dir if args.state_dir else None
    data_wrapper = OakInk2FlowWrapper(
        split='test', data_root=args.data_root,
        num_frames=train_args.get('num_frames', 200),
        batch_size=1, model_version=args.model_version,
        state_dir=state_dir,
    )
    mean = data_wrapper.dataset.mean
    std = data_wrapper.dataset.std
    input_dim = data_wrapper.dataset.feature_dim

    # Build index → dataset position mapping
    test_indices = data_wrapper.dataset.indices if hasattr(data_wrapper.dataset, 'indices') else []
    idx_to_pos = {idx: pos for pos, idx in enumerate(test_indices)}

    # Load model
    use_clip = data_wrapper.dataset.use_clip_emb
    model = create_model(args.model_version, train_args, input_dim, use_clip, device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"Model loaded: {args.model_path}")

    K = args.history_len
    all_gen_segments = []
    all_gt_segments = []
    all_texts = []

    # Initialize history as zero (rest pose)
    x_hist = torch.zeros(1, K, input_dim, device=device)

    for seg_i, seg_idx in enumerate(seg_indices):
        # Find this segment in test dataset
        if seg_idx not in idx_to_pos:
            print(f"  WARNING: {seg_idx} not in test set, skipping")
            continue
        pos = idx_to_pos[seg_idx]

        # Get sample from dataset
        sample = data_wrapper.dataset[pos]
        gt_motion = sample['motion'].unsqueeze(0).to(device)  # [1, T, D] normalized
        seq_len = sample['m_length']

        # Build conditioning
        y = {'state_labels': sample['state_labels'].unsqueeze(0).to(device)}
        if 'clip_emb' in sample:
            y['text_emb'] = sample['clip_emb'].unsqueeze(0).to(device)
        if 'bps' in sample:
            y['bps'] = sample['bps'].unsqueeze(0).to(device)

        text = sample.get('text', '')
        print(f"\n  Segment {seg_i}: [{seg_idx}] '{text[:60]}' ({seq_len} frames)")

        # Generate with history conditioning
        gen_motion = generate_segment_with_history(
            model, y, x_hist, seq_len=seq_len, input_dim=input_dim,
            history_len=K, num_steps=args.num_ode_steps, device=device,
        )

        # Denormalize
        gen_np = gen_motion[0].cpu().numpy() * std + mean
        gt_np = gt_motion[0].cpu().numpy() * std + mean

        # Save per-segment
        np.save(os.path.join(args.output_dir, f'seg_{seg_i:02d}_gen.npy'), gen_np[:seq_len])
        np.save(os.path.join(args.output_dir, f'seg_{seg_i:02d}_gt.npy'), gt_np[:seq_len])

        # Compute segment MSE
        pos_gen = gen_np[:seq_len, :267] if gen_np.shape[1] > 267 else gen_np[:seq_len]
        pos_gt = gt_np[:seq_len, :267] if gt_np.shape[1] > 267 else gt_np[:seq_len]
        seg_mse = ((pos_gen[:, :231] - pos_gt[:, :231]) ** 2).mean()
        print(f"    Marker MSE: {seg_mse:.6f}")

        # For next segment: use last K frames of generated as history (normalized)
        gen_norm = gen_motion[0, -K:, :].unsqueeze(0)  # [1, K, D] still normalized
        x_hist = gen_norm.clone()

        all_gen_segments.append(gen_np[:seq_len])
        all_gt_segments.append(gt_np[:seq_len])
        all_texts.append(text)

    # Concatenate full long-horizon motion (removing overlapping K frames)
    if len(all_gen_segments) > 0:
        full_gen = [all_gen_segments[0]]
        full_gt = [all_gt_segments[0]]
        for i in range(1, len(all_gen_segments)):
            full_gen.append(all_gen_segments[i][K:])  # skip history overlap
            full_gt.append(all_gt_segments[i][K:])

        full_gen = np.concatenate(full_gen, axis=0)
        full_gt = np.concatenate(full_gt, axis=0)

        np.save(os.path.join(args.output_dir, 'full_gen.npy'), full_gen)
        np.save(os.path.join(args.output_dir, 'full_gt.npy'), full_gt)

        # Extract positions for visualization
        if full_gen.shape[1] > 267:
            full_gen_pos = full_gen[:, :267]
            full_gt_pos = full_gt[:, :267]
        else:
            full_gen_pos = full_gen
            full_gt_pos = full_gt

        np.save(os.path.join(args.output_dir, 'full_gen_pos.npy'), full_gen_pos)
        np.save(os.path.join(args.output_dir, 'full_gt_pos.npy'), full_gt_pos)

        total_mse = ((full_gen_pos[:, :231] - full_gt_pos[:, :231]) ** 2).mean()
        total_frames = full_gen.shape[0]

        print(f"\n{'='*60}")
        print(f"Long-horizon result:")
        print(f"  Segments: {len(all_gen_segments)}")
        print(f"  Total frames: {total_frames} ({total_frames/10:.1f}s at 10fps)")
        print(f"  Full marker MSE: {total_mse:.6f}")
        print(f"  Texts: {' → '.join(t[:40] for t in all_texts)}")

        # Save metadata
        info = {
            'model_path': args.model_path,
            'segment_indices': seg_indices,
            'texts': all_texts,
            'total_frames': int(total_frames),
            'history_len': K,
            'full_marker_mse': float(total_mse),
            'per_segment_frames': [s.shape[0] for s in all_gen_segments],
        }
        with open(os.path.join(args.output_dir, 'chain_info.json'), 'w') as f:
            json.dump(info, f, indent=2)

        print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
