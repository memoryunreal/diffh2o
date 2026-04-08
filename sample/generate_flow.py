#!/usr/bin/env python3
"""
Generate motions using trained CRR-Flow model (V1/V3/V4).

Usage:
    # V3
    python -m sample.generate_flow \
        --model_path save/v3_entity_200k_fixed/model000010000.pt \
        --model_version v3 --data_root dataset/OAKINK2_V3 \
        --num_samples 4 --output_dir output/v3_10k_vis

    # V4
    python -m sample.generate_flow \
        --model_path save/v4_entity_200k/model000010000.pt \
        --model_version v4 --data_root dataset/OAKINK2_V4 \
        --num_samples 4 --output_dir output/v4_10k_vis
"""

import os
import json
import argparse

import torch
import numpy as np
from torchdiffeq import odeint

from utils.fixseed import fixseed
from data_loaders.humanml.data.flow_dataset import OakInk2FlowWrapper


def generate_motion(model, y, seq_len, input_dim, num_steps=10, device='cuda'):
    """Generate motion via ODE integration of learned velocity field."""
    model.eval()
    B = 1

    x0 = torch.randn(B, seq_len, input_dim, device=device)

    def ode_fn(t_scalar, x):
        t_batch = t_scalar.expand(x.shape[0]).to(device)
        with torch.no_grad():
            return model(x, t_batch, y=y)

    t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    trajectory = odeint(ode_fn, x0, t_span, method='euler')

    return trajectory[-1]  # [1, T, D]


def generate_motion_autoregressive(model, y, seq_len, input_dim, history_len=20,
                                    num_steps=10, device='cuda'):
    """Generate motion with block-wise autoregressive conditioning.

    First K frames are clean history, model generates remaining L frames.
    For the first segment, history is rest pose with zero velocity.
    """
    model.eval()
    B = 1
    K = history_len
    T_full = K + (seq_len - K)  # = seq_len

    # History: from y if provided, else zeros (rest pose)
    if 'x_hist' in y:
        x_hist = y['x_hist']  # [1, K, D] pre-provided history
    else:
        x_hist = torch.zeros(B, K, input_dim, device=device)

    # Generate target frames via ODE
    x0_target = torch.randn(B, seq_len - K, input_dim, device=device)

    def ode_fn(t_scalar, x_target):
        t_batch = t_scalar.expand(B).to(device)
        # Concat clean history + noisy target
        x_full = torch.cat([x_hist, x_target], dim=1)  # [1, T, D]
        with torch.no_grad():
            v_full = model(x_full, t_batch, y=y)
        # Return only target velocity
        return v_full[:, K:, :]

    t_span = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    trajectory = odeint(ode_fn, x0_target, t_span, method='euler')

    x1_target = trajectory[-1]  # [1, L, D]

    # Full sequence: history + generated target
    return torch.cat([x_hist, x1_target], dim=1)  # [1, T, D]


def create_model(model_version, train_args, input_dim, use_precomputed_clip, device):
    """Create the appropriate model based on version."""
    latent_dim = train_args.get('latent_dim', 512)
    num_layers = train_args.get('num_layers', 8)
    num_heads = train_args.get('num_heads', 8)
    ff_dim = train_args.get('ff_dim', 2048)
    max_seq_len = train_args.get('num_frames', 200)

    if model_version == 'v4':
        from model.flow_dit_v4 import CRRFlowDiT_V4
        model = CRRFlowDiT_V4(
            latent_dim=latent_dim, num_layers=num_layers,
            num_heads=num_heads, ff_dim=ff_dim,
            dropout=0.0, max_seq_len=max_seq_len,
            cond_mask_prob=0.0,
            skip_clip=use_precomputed_clip,
        )
    elif model_version == 'v3':
        from model.flow_dit_v3 import CRRFlowDiT_V3
        model = CRRFlowDiT_V3(
            latent_dim=latent_dim, num_layers=num_layers,
            num_heads=num_heads, ff_dim=ff_dim,
            dropout=0.0, max_seq_len=max_seq_len,
            cond_mask_prob=0.0,
            skip_clip=use_precomputed_clip,
        )
    else:
        from model.flow_dit import FlowDiT
        model = FlowDiT(
            input_dim=input_dim,
            latent_dim=latent_dim, num_layers=num_layers,
            num_heads=num_heads, ff_dim=ff_dim,
            dropout=0.0, cond_mask_prob=0.0,
        )

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Generate motions with CRR-Flow")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_version", type=str, default="v1",
                        choices=["v1", "v3", "v4"])
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="output/flow_gen")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=10,
                        help="ODE integration steps")
    parser.add_argument("--data_root", type=str, default="dataset/OAKINK2_FLOW")
    args = parser.parse_args()

    fixseed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load training args
    model_dir = os.path.dirname(args.model_path)
    args_path = os.path.join(model_dir, 'args.json')
    with open(args_path) as f:
        train_args = json.load(f)

    # Load dataset (test split, GT state labels for conditioning)
    data_wrapper = OakInk2FlowWrapper(
        split='test', data_root=args.data_root,
        num_frames=train_args.get('num_frames', 200),
        batch_size=1, model_version=args.model_version,
    )
    dataloader = data_wrapper.get_loader(shuffle=False)

    # Create model
    input_dim = data_wrapper.dataset.feature_dim
    use_clip = data_wrapper.dataset.use_clip_emb
    model = create_model(args.model_version, train_args, input_dim, use_clip, device)

    # Load checkpoint
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"Model loaded: {args.model_path} (version={args.model_version}, dim={input_dim})")

    # Normalization stats
    mean = data_wrapper.dataset.mean
    std = data_wrapper.dataset.std

    print(f"Generating {args.num_samples} samples ({args.num_steps} ODE steps)...")

    for i, (gt_motion, cond) in enumerate(dataloader):
        if i >= args.num_samples:
            break

        y = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in cond['y'].items()}

        # Get text for logging
        text = ""
        if 'text' in y:
            text = y['text'][0] if isinstance(y['text'], list) else str(y['text'])
        seq_len = y['lengths'][0].item()

        print(f"\nSample {i}: '{text[:60]}' ({seq_len} frames)")

        # Generate
        gen_motion = generate_motion(
            model, y, seq_len=seq_len, input_dim=input_dim,
            num_steps=args.num_steps, device=device,
        )

        # Denormalize
        gen_np = gen_motion[0].cpu().numpy()  # [T, D]
        gen_denorm = gen_np * std + mean

        gt_np = gt_motion[0].numpy()
        gt_denorm = gt_np * std + mean

        # For V4: extract position part (first 267D) for visualization
        if args.model_version == 'v4':
            gen_pos = gen_denorm[:seq_len, :267]  # positions only
            gt_pos = gt_denorm[:seq_len, :267]
            # Also save full 534D
            np.save(os.path.join(args.output_dir, f'gen_full_{i:04d}.npy'), gen_denorm[:seq_len])
        else:
            gen_pos = gen_denorm[:seq_len]
            gt_pos = gt_denorm[:seq_len]

        # Save position-space motions (for marker visualization)
        np.save(os.path.join(args.output_dir, f'gen_{i:04d}.npy'), gen_pos)
        np.save(os.path.join(args.output_dir, f'gt_{i:04d}.npy'), gt_pos)

        # Extract markers for quick diagnostic
        gen_markers = gen_pos[:, :231].reshape(-1, 77, 3)
        gt_markers = gt_pos[:, :231].reshape(-1, 77, 3)
        marker_mse = ((gen_markers - gt_markers) ** 2).mean()
        marker_range_gen = (gen_markers.max(0) - gen_markers.min(0)).mean()
        marker_range_gt = (gt_markers.max(0) - gt_markers.min(0)).mean()
        print(f"  Marker MSE={marker_mse:.4f}, range: GT={marker_range_gt:.4f} Gen={marker_range_gen:.4f}")

    # Save generation info
    info = {
        "model_path": args.model_path,
        "model_version": args.model_version,
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "input_dim": input_dim,
    }
    with open(os.path.join(args.output_dir, 'gen_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
