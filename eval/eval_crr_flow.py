#!/usr/bin/env python3
"""
Evaluate CRR-Flow models: Marker MSE, Object Pose Error, Contact F1, Motion Range.

Usage:
    # Single model evaluation
    python -m eval.eval_crr_flow \
        --model_path save/v4_entity_200k/model000130000.pt \
        --model_version v4 --data_root dataset/OAKINK2_V4 \
        --num_samples 0 --device 5

    # 0 = all test samples
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from utils.fixseed import fixseed
from data_loaders.humanml.data.flow_dataset import OakInk2FlowWrapper
from sample.generate_flow import create_model, generate_motion

# State constants
STATE_GRASPED = 2
STATE_PADDING = 5

# Fingertip indices within 77-marker set
LH_TIPS = list(range(67, 72))  # 5 left fingertips (within 77 markers → flat indices)
RH_TIPS = list(range(72, 77))  # 5 right fingertips
# In 231D marker space: tip_i at indices [tip_idx*3 : tip_idx*3+3]


def geodesic_rotation_error(rot6d_a, rot6d_b):
    """Compute geodesic rotation error in degrees between two 6D rotations."""
    def rot6d_to_mat(r6):
        a1, a2 = r6[:3], r6[3:6]
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / (np.linalg.norm(b2) + 1e-8)
        b3 = np.cross(b1, b2)
        return np.stack([b1, b2, b3], axis=-1)

    R_a = rot6d_to_mat(rot6d_a)
    R_b = rot6d_to_mat(rot6d_b)
    R_diff = R_a @ R_b.T
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(trace))


def compute_contact_f1(gen_markers_231, gen_obj_pos, states, contact_thresh=0.05):
    """Compute per-hand contact F1 score.

    Args:
        gen_markers_231: [T, 231] generated marker positions
        gen_obj_pos: [T, 3] primary object position
        states: [T, 7] state labels
        contact_thresh: distance threshold in meters

    Returns:
        dict with lh_f1, rh_f1, avg_f1
    """
    T = gen_markers_231.shape[0]
    markers = gen_markers_231.reshape(T, 77, 3)
    results = {}

    for hand_name, tip_indices, col in [('lh', LH_TIPS, 1), ('rh', RH_TIPS, 2)]:
        tips = markers[:, tip_indices, :]  # [T, 5, 3]
        dists = np.linalg.norm(tips - gen_obj_pos[:, None, :], axis=2)  # [T, 5]
        min_dist = dists.min(axis=1)  # [T]

        gen_contact = min_dist < contact_thresh
        gt_contact = states[:, col] == STATE_GRASPED

        tp = (gen_contact & gt_contact).sum()
        fp = (gen_contact & ~gt_contact).sum()
        fn = (~gen_contact & gt_contact).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results[f'{hand_name}_precision'] = float(precision)
        results[f'{hand_name}_recall'] = float(recall)
        results[f'{hand_name}_f1'] = float(f1)

    results['avg_f1'] = (results['lh_f1'] + results['rh_f1']) / 2
    return results


def evaluate_single_sample(gen_motion, gt_motion, states=None, model_version='v4'):
    """Compute all metrics for a single generated sample.

    Args:
        gen_motion: [T, D] generated (denormalized)
        gt_motion: [T, D] ground truth (denormalized)
        states: [T, 7] state labels (optional, for contact F1)
        model_version: 'v3' or 'v4'
    """
    # For V4: extract position part
    if model_version == 'v4' and gen_motion.shape[1] > 267:
        gen_pos = gen_motion[:, :267]
        gt_pos = gt_motion[:, :267]
        gen_vel = gen_motion[:, 267:534] if gen_motion.shape[1] >= 534 else None
        gt_vel = gt_motion[:, 267:534] if gt_motion.shape[1] >= 534 else None
    else:
        gen_pos = gen_motion
        gt_pos = gt_motion
        gen_vel = None
        gt_vel = None

    T = min(gen_pos.shape[0], gt_pos.shape[0])
    gen_pos = gen_pos[:T]
    gt_pos = gt_pos[:T]

    metrics = {}

    # 1. Marker MSE (77 markers × 3D = 231D)
    gen_markers = gen_pos[:, :231]
    gt_markers = gt_pos[:, :231]
    metrics['marker_mse'] = float(((gen_markers - gt_markers) ** 2).mean())

    # Per-part MSE
    metrics['body_mse'] = float(((gen_pos[:, :150] - gt_pos[:, :150]) ** 2).mean())
    metrics['lh_mse'] = float(((gen_pos[:, 150:192] - gt_pos[:, 150:192]) ** 2).mean())
    metrics['rh_mse'] = float(((gen_pos[:, 192:231] - gt_pos[:, 192:231]) ** 2).mean())

    # 2. Object Pose Error (primary object, slot 0)
    gen_obj = gen_pos[:, 231:240]  # [T, 9]
    gt_obj = gt_pos[:, 231:240]

    # Position error (mm)
    obj_pos_err = np.linalg.norm(gen_obj[:, :3] - gt_obj[:, :3], axis=1).mean() * 1000
    metrics['obj_pos_err_mm'] = float(obj_pos_err)

    # Rotation error (degrees) — average over frames
    rot_errs = []
    for t in range(T):
        try:
            err = geodesic_rotation_error(gen_obj[t, 3:9], gt_obj[t, 3:9])
            if not np.isnan(err):
                rot_errs.append(err)
        except Exception:
            pass
    metrics['obj_rot_err_deg'] = float(np.mean(rot_errs)) if rot_errs else 0.0

    # 3. Motion range (marker spatial extent)
    gen_markers_3d = gen_markers.reshape(T, 77, 3)
    gt_markers_3d = gt_markers.reshape(T, 77, 3)
    metrics['gen_range'] = float((gen_markers_3d.max(0) - gen_markers_3d.min(0)).mean())
    metrics['gt_range'] = float((gt_markers_3d.max(0) - gt_markers_3d.min(0)).mean())

    # 4. Velocity MSE (V4 only)
    if gen_vel is not None and gt_vel is not None:
        gt_vel = gt_vel[:T]
        gen_vel = gen_vel[:T]
        metrics['vel_mse'] = float(((gen_vel[:, :231] - gt_vel[:, :231]) ** 2).mean())

    # 5. Contact F1 (if states available)
    if states is not None:
        states = states[:T]
        obj_pos = gen_pos[:, 231:234]  # [T, 3]
        contact = compute_contact_f1(gen_markers, obj_pos, states)
        metrics.update(contact)

    # 6. Jitter (3rd order temporal derivative)
    if T > 3:
        jitter = np.diff(gen_markers_3d, n=3, axis=0)
        metrics['jitter'] = float(np.linalg.norm(jitter, axis=2).mean())

    return metrics


def run_evaluation(args):
    """Run full evaluation on a model checkpoint."""
    fixseed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load training args
    model_dir = os.path.dirname(args.model_path)
    args_path = os.path.join(model_dir, 'args.json')
    with open(args_path) as f:
        train_args = json.load(f)

    # Determine model version
    model_version = args.model_version or train_args.get('model_version', 'v3')

    # Load dataset (test split) — use num_frames from training config
    num_frames = train_args.get('num_frames', 200)
    print(f"Using num_frames={num_frames} from training config")

    data_wrapper = OakInk2FlowWrapper(
        split='test', data_root=args.data_root,
        num_frames=num_frames,
        batch_size=1, model_version=model_version,
        state_dir=args.state_dir if args.state_dir else None,
    )
    dataloader = data_wrapper.get_loader(shuffle=False)
    mean = data_wrapper.dataset.mean
    std = data_wrapper.dataset.std

    # Create and load model
    input_dim = data_wrapper.dataset.feature_dim
    use_clip = data_wrapper.dataset.use_clip_emb
    model = create_model(model_version, train_args, input_dim, use_clip, device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    step = ''.join(c for c in os.path.basename(args.model_path) if c.isdigit())
    print(f"Evaluating: {args.model_path} (version={model_version}, dim={input_dim}, step={step})")
    print(f"Test set: {len(data_wrapper.dataset)} samples")

    # Evaluate
    num_samples = args.num_samples if args.num_samples > 0 else len(data_wrapper.dataset)
    all_metrics = []
    start_time = time.time()

    for i, (gt_motion_batch, cond) in enumerate(dataloader):
        if i >= num_samples:
            break

        y = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in cond['y'].items()}
        seq_len = y['lengths'][0].item()

        # Generate
        gen_motion = generate_motion(
            model, y, seq_len=seq_len, input_dim=input_dim,
            num_steps=args.num_ode_steps, device=device,
        )

        # Denormalize
        gen_np = gen_motion[0].cpu().numpy() * std + mean
        gt_np = gt_motion_batch[0].numpy() * std + mean

        # Get state labels
        states = None
        if 'state_labels' in cond['y']:
            states = cond['y']['state_labels'][0].numpy()

        # Compute metrics
        metrics = evaluate_single_sample(gen_np[:seq_len], gt_np[:seq_len],
                                          states=states[:seq_len] if states is not None else None,
                                          model_version=model_version)
        all_metrics.append(metrics)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  {i+1}/{num_samples} ({elapsed:.0f}s) "
                  f"marker_mse={metrics['marker_mse']:.4f}")

    # Aggregate
    agg = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics if key in m]
        agg[f'{key}_mean'] = float(np.mean(vals))
        agg[f'{key}_std'] = float(np.std(vals))

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Results ({len(all_metrics)} samples, {elapsed:.0f}s):")
    print(f"  Marker MSE:     {agg['marker_mse_mean']:.6f} ± {agg['marker_mse_std']:.6f}")
    print(f"  Body MSE:       {agg['body_mse_mean']:.6f}")
    print(f"  LH MSE:         {agg['lh_mse_mean']:.6f}")
    print(f"  RH MSE:         {agg['rh_mse_mean']:.6f}")
    print(f"  Obj Pos (mm):   {agg['obj_pos_err_mm_mean']:.2f} ± {agg['obj_pos_err_mm_std']:.2f}")
    print(f"  Obj Rot (deg):  {agg['obj_rot_err_deg_mean']:.2f} ± {agg['obj_rot_err_deg_std']:.2f}")
    print(f"  Jitter:         {agg.get('jitter_mean', 0):.6f}")
    print(f"  Gen Range:      {agg['gen_range_mean']:.4f}")
    print(f"  GT Range:       {agg['gt_range_mean']:.4f}")
    if 'vel_mse_mean' in agg:
        print(f"  Vel MSE:        {agg['vel_mse_mean']:.6f}")
    if 'avg_f1_mean' in agg:
        print(f"  Contact F1:     {agg['avg_f1_mean']:.4f}")
        print(f"  LH F1:          {agg['lh_f1_mean']:.4f}")
        print(f"  RH F1:          {agg['rh_f1_mean']:.4f}")

    # Save results
    result = {
        'model_path': args.model_path,
        'model_version': model_version,
        'data_root': args.data_root,
        'step': step,
        'num_samples': len(all_metrics),
        'num_ode_steps': args.num_ode_steps,
        'metrics': agg,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_tag = os.path.basename(args.data_root).lower()
    result_path = os.path.join(args.output_dir, f'eval_{dataset_tag}_{model_version}_{step}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_version", type=str, default="")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--state_dir", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=0, help="0 = all test")
    parser.add_argument("--num_ode_steps", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    run_evaluation(args)


if __name__ == "__main__":
    main()
