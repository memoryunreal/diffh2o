#!/usr/bin/env python3
"""
Generate transitions using the trained SwitchScheduler diffusion model.

Loads checkpoint, gets conditioning from dataset, runs diffusion sampling,
saves .npy files and prints diagnostic stats (GT vs generated).

Usage:
    python -m sample.generate_transition \
        --model_path save/oakink2_transition/model000200000.pt \
        --num_samples 4 --output_dir output/gen_transitions
"""

import argparse
import json
import os

import numpy as np
import torch

from utils.fixseed import fixseed
from utils import dist_util
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from utils.model_util import create_gaussian_diffusion
from train.train_transition import create_transition_model


CHANNEL_GROUPS = {
    "Body transl (0:3)": (0, 3),
    "Body orient (3:9)": (3, 9),
    "Body pose (9:135)": (9, 135),
    "LHand PCA (135:165)": (135, 165),
    "RHand PCA (165:195)": (165, 195),
    "LHand quat (198:262)": (198, 262),
    "RHand quat (265:329)": (265, 329),
    "Obj1 pos (371:374)": (371, 374),
    "Obj1 rot (374:380)": (374, 380),
}


def print_diagnostics(gt_dn, gen_dn, sample_idx=0):
    """Print per-channel comparison between GT and generated."""
    gt = gt_dn[sample_idx].cpu().numpy()
    gen = gen_dn[sample_idx].cpu().numpy()

    print(f"\n{'Channel':<30s} {'GT_range':>10s} {'Gen_range':>10s} {'MSE':>10s}")
    print("-" * 65)
    for name, (s, e) in CHANNEL_GROUPS.items():
        gt_range = (gt[:, s:e].max(axis=0) - gt[:, s:e].min(axis=0)).mean()
        gen_range = (gen[:, s:e].max(axis=0) - gen[:, s:e].min(axis=0)).mean()
        mse = ((gt[:, s:e] - gen[:, s:e]) ** 2).mean()
        print(f"{name:<30s} {gt_range:10.6f} {gen_range:10.6f} {mse:10.6f}")


def main():
    parser = argparse.ArgumentParser(description="Generate transitions with SwitchScheduler")
    parser.add_argument("--model_path", type=str,
                        default="save/oakink2_transition/model000200000.pt")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="output/gen_transitions")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    cli_args = parser.parse_args()

    fixseed(cli_args.seed)

    # Load saved training args
    args_path = os.path.join(os.path.dirname(cli_args.model_path), "args.json")
    with open(args_path) as f:
        saved_args = json.load(f)

    class Args:
        pass

    args = Args()
    for k, v in saved_args.items():
        setattr(args, k, v)

    dist_util.setup_dist(cli_args.device)
    device = dist_util.dev()

    # Load dataset
    print("Loading dataset...")
    data_conf = DatasetConfig(
        name=args.dataset,
        batch_size=cli_args.num_samples,
        num_frames=args.num_frames,
    )
    data = get_dataset_loader(data_conf, shuffle=False)
    print(f"Dataset: {len(data.dataset)} samples")

    # Determine feature dim
    njoints = 398 if "oakink2" in args.dataset else 117

    # Create model and load checkpoint
    print("Creating model...")
    model = create_transition_model(args, njoints=njoints, nfeats=1)
    ckpt = torch.load(cli_args.model_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    print(f"Model loaded from {cli_args.model_path} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    # Create diffusion
    print("Creating diffusion...")
    diffusion = create_gaussian_diffusion(args, data)

    # Get one batch
    motion_gt, cond = next(iter(data))
    motion_gt = motion_gt.to(device)
    y = {}
    for k, v in cond["y"].items():
        y[k] = v.to(device) if hasattr(v, "to") else v

    B = motion_gt.shape[0]
    print(f"\nBatch size: {B}")
    print(f"GT shape: {motion_gt.shape}  range: [{motion_gt.min():.3f}, {motion_gt.max():.3f}]  std: {motion_gt.std():.3f}")

    # Print text descriptions
    for i in range(min(B, 4)):
        src = y.get("text_source", [""])[i] if "text_source" in y else ""
        tgt = y.get("text_target", [""])[i] if "text_target" in y else ""
        print(f"  Sample {i}: '{src}' -> '{tgt}'")

    # Run diffusion sampling
    print(f"\nGenerating {B} transitions ({args.diffusion_steps} diffusion steps)...")
    shape = motion_gt.shape

    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            model, shape,
            model_kwargs={"y": y},
            clip_denoised=True,
            progress=True,
        )

    print(f"Generated shape: {sample.shape}  range: [{sample.min():.3f}, {sample.max():.3f}]  std: {sample.std():.3f}")

    # Denormalize: [B, 398, 1, 60] -> [B, 60, 398]
    mean_np = data.dataset.t2m_dataset.mean
    std_np = data.dataset.t2m_dataset.std
    mean_t = torch.tensor(mean_np, device=device, dtype=sample.dtype)
    std_t = torch.tensor(std_np, device=device, dtype=sample.dtype)

    gt_flat = motion_gt.squeeze(2).permute(0, 2, 1)   # [B, 60, 398]
    gen_flat = sample.squeeze(2).permute(0, 2, 1)       # [B, 60, 398]
    gt_dn = gt_flat * std_t + mean_t
    gen_dn = gen_flat * std_t + mean_t

    # Print diagnostics for each sample
    for i in range(min(B, 4)):
        print(f"\n=== Sample {i} ===")
        print_diagnostics(gt_dn, gen_dn, sample_idx=i)

    # Save outputs
    os.makedirs(cli_args.output_dir, exist_ok=True)

    # Also denormalize boundary frames for full sequence assembly
    bnd_before = y["boundary_before"]  # [B, 10, 398]
    bnd_after = y["boundary_after"]    # [B, 10, 398]
    bnd_before_dn = bnd_before * std_t + mean_t
    bnd_after_dn = bnd_after * std_t + mean_t

    stats_lines = []
    for i in range(B):
        gen_np = gen_dn[i].cpu().numpy()   # [60, 398]
        gt_np = gt_dn[i].cpu().numpy()

        # Full 80-frame sequence: boundary_before + generated + boundary_after
        full_np = np.concatenate([
            bnd_before_dn[i].cpu().numpy(),
            gen_np,
            bnd_after_dn[i].cpu().numpy(),
        ], axis=0)  # [80, 398]

        np.save(os.path.join(cli_args.output_dir, f"gen_{i:06d}.npy"), gen_np)
        np.save(os.path.join(cli_args.output_dir, f"gt_{i:06d}.npy"), gt_np)
        np.save(os.path.join(cli_args.output_dir, f"full_{i:06d}.npy"), full_np)

        src = y.get("text_source", [""])[i] if "text_source" in y else ""
        tgt = y.get("text_target", [""])[i] if "text_target" in y else ""
        body_range_gt = (gt_np[:, 9:135].max(axis=0) - gt_np[:, 9:135].min(axis=0)).mean()
        body_range_gen = (gen_np[:, 9:135].max(axis=0) - gen_np[:, 9:135].min(axis=0)).mean()
        stats_lines.append(
            f"Sample {i}: src='{src}' tgt='{tgt}'\n"
            f"  GT body_pose_range={body_range_gt:.6f}  Gen body_pose_range={body_range_gen:.6f}\n"
        )

    stats_path = os.path.join(cli_args.output_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.writelines(stats_lines)

    print(f"\nSaved {B} samples to {cli_args.output_dir}/")
    print(f"  gen_*.npy  — generated transitions [60, 398]")
    print(f"  gt_*.npy   — ground truth transitions [60, 398]")
    print(f"  full_*.npy — full sequences [80, 398] (boundary + gen + boundary)")
    print(f"  stats.txt  — diagnostic summary")


if __name__ == "__main__":
    main()
