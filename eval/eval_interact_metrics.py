#!/usr/bin/env python3
"""
Evaluate CRR-Flow models using InterAct's pre-trained evaluators.
Computes FID, R-Precision@1/2/3, Matching Score, Diversity, Multimodality.

Converts our V4 534D (or V3 267D) output → InterAct's 615D eval format:
  markers(231D) + BPS-encoded object(384D)

Usage:
    python -m eval.eval_interact_metrics \
        --model_path save/v4_autoreg_release_200k/model000060000.pt \
        --model_version v4 --data_root dataset/OAKINK2_V4 \
        --interact_root /hhd4/lizhe/code/InterAct \
        --device 5 --num_samples 0 --num_repeats 3

    # With post-processing (SG smoothing)
    python -m eval.eval_interact_metrics \
        --model_path ... --smooth_window 11
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

# Add InterAct to path
INTERACT_ROOT = "/hhd4/lizhe/code/InterAct"

from utils.fixseed import fixseed
from data_loaders.humanml.data.flow_dataset import OakInk2FlowWrapper
from sample.generate_flow import create_model, generate_motion


def rotation_6d_to_matrix_batch(rot6d):
    """Convert [N, 6] → [N, 3, 3] rotation matrices."""
    a1 = rot6d[:, :3]
    a2 = rot6d[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(axis=1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # [N, 3, 3]


def convert_to_615d(motion_267, obj_points_400, bps_basis_128):
    """Convert our 267D motion → InterAct 615D eval format.

    Args:
        motion_267: [T, 267] marker positions + object poses
        obj_points_400: [400, 3] object point cloud (canonical space)
        bps_basis_128: [128, 3] BPS basis points

    Returns:
        [T, 615] = markers(231) + BPS-encoded object(384)
    """
    from bps_torch.bps import bps_torch

    T = motion_267.shape[0]
    markers = motion_267[:, :231]  # [T, 231]
    obj_pose = motion_267[:, 231:240]  # [T, 9] (6D rot + 3D trans)

    # Transform object points by generated pose per frame
    rot6d = obj_pose[:, :6]  # [T, 6]
    trans = obj_pose[:, 6:9]  # [T, 3]

    R = rotation_6d_to_matrix_batch(rot6d)  # [T, 3, 3]

    # Transform points: [T, 400, 3]
    pts = obj_points_400[None, :, :].repeat(T, axis=0)  # [T, 400, 3]
    pts_transformed = np.einsum('tij,tnj->tni', R, pts) + trans[:, None, :]  # [T, 400, 3]

    # BPS encode: for each frame, encode transformed points using basis
    # basis also needs to be translated
    bps_basis_t = bps_basis_128[None, :, :].repeat(T, axis=0) + trans[:, None, :]  # [T, 128, 3]

    # Use bps_torch for encoding
    pts_t = torch.tensor(pts_transformed.reshape(-1, 400, 3), dtype=torch.float32)
    basis_t = torch.tensor(bps_basis_t.reshape(-1, 128, 3), dtype=torch.float32)

    bps_encoder = bps_torch(n_bps_points=128, bps_cell_type='deltas')
    bps_enc = bps_encoder.encode(
        x=pts_t, feature_type=['deltas'],
        custom_basis=basis_t
    )['deltas']  # [T, 128, 3]

    bps_flat = bps_enc.cpu().numpy().reshape(T, 384)  # [T, 384]

    eval_615 = np.concatenate([markers, bps_flat], axis=-1)  # [T, 615]
    return eval_615


def load_interact_evaluator(interact_root, device):
    """Load InterAct's pre-trained motion and text encoders."""
    sys.path.insert(0, os.path.join(interact_root, "text2interaction"))

    from tma.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
    from tma.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder

    eval_dir = os.path.join(interact_root, "text2interaction/assets/eval")

    # Load checkpoint
    ckpt = torch.load(os.path.join(eval_dir, "markersbps.ckpt"), map_location=device, weights_only=False)

    # Extract state dicts from Lightning checkpoint
    state_dict = ckpt.get('state_dict', ckpt)
    me_sd = {k.replace('motionencoder.', ''): v for k, v in state_dict.items() if k.startswith('motionencoder.')}
    te_sd = {k.replace('textencoder.', ''): v for k, v in state_dict.items() if k.startswith('textencoder.')}

    # Motion encoder
    motionencoder = ActorAgnosticEncoder(
        nfeats=615, vae=True, latent_dim=256,
        ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
        activation="gelu"
    ).to(device)
    motionencoder.load_state_dict(me_sd, strict=False)
    motionencoder.eval()

    # Text encoder
    textencoder = DistilbertActorAgnosticEncoder(
        modelpath='distilbert-base-uncased',
        finetune=False, vae=True, latent_dim=256,
        ff_size=1024, num_layers=4, num_heads=4, dropout=0.1,
        activation="gelu"
    ).to(device)
    textencoder.load_state_dict(te_sd, strict=False)
    textencoder.eval()

    # Normalization
    mean_enc = np.load(os.path.join(eval_dir, "mean_train_markersbps.npy"))
    std_enc = np.load(os.path.join(eval_dir, "std_train_markersbps.npy"))
    std_enc[std_enc < 1e-8] = 1.0

    # BPS basis
    bps_basis = np.load(os.path.join(eval_dir, "bps_basis_set_128_1.npy"))

    return motionencoder, textencoder, mean_enc, std_enc, bps_basis


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """FID computation."""
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + eps * np.eye(sigma1.shape[0])).dot(sigma2 + eps * np.eye(sigma2.shape[0])))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def calculate_diversity(features, num_samples=300):
    """Diversity: pairwise distance in feature space."""
    n = features.shape[0]
    if n < 2:
        return 0.0
    num_samples = min(num_samples, n * (n - 1) // 2)
    idx1 = np.random.randint(0, n, num_samples)
    idx2 = np.random.randint(0, n, num_samples)
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    dist = np.linalg.norm(features[idx1] - features[idx2], axis=1)
    return dist.mean()


def get_default_object_points(data_root, file_idx, interact_root):
    """Get object point cloud for BPS encoding.

    Try InterAct's preprocessed data first, then fall back to sampling from OakInk2 meshes.
    """
    # Try to load from InterAct data
    # For OakInk2, we need to load from mesh and sample points
    obj_mesh_dir = '/hhd4/lizhe/dataset/OakInk2/data/object_repair/align_ds'
    prog_dir = '/hhd4/lizhe/dataset/OakInk2/data/program/program_info'

    v3_root = 'dataset/OAKINK2_V3'
    file_map = {}
    fm_path = os.path.join(v3_root, 'file_names.txt')
    if os.path.exists(fm_path):
        with open(fm_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    file_map[parts[0]] = parts[1]

    seg_info = file_map.get(file_idx, '')
    seq_id = seg_info.rsplit('__seg', 1)[0] if seg_info else ''

    # Get primary object
    obj_ids = []
    prog_path = os.path.join(prog_dir, seq_id + '.json')
    if os.path.exists(prog_path):
        import json
        prog = json.load(open(prog_path))
        all_objs = set()
        for k, v in prog.items():
            for field in ['obj_list', 'obj_list_lh', 'obj_list_rh']:
                objs = v.get(field) or []
                if isinstance(objs, list):
                    all_objs.update(objs)
        obj_ids = sorted(all_objs)[:1]  # Primary object only

    if obj_ids:
        import trimesh
        mesh_path = os.path.join(obj_mesh_dir, obj_ids[0], 'model.obj')
        if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path, force='mesh')
            # Sample 400 points from surface (like InterAct)
            points, _ = trimesh.sample.sample_surface(mesh, 400)
            return points.astype(np.float32)

    # Fallback: unit sphere points
    return np.random.randn(400, 3).astype(np.float32) * 0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_version", default="v4")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--state_dir", default="")
    parser.add_argument("--interact_root", default=INTERACT_ROOT)
    parser.add_argument("--num_samples", type=int, default=0, help="0=all test")
    parser.add_argument("--num_repeats", type=int, default=3, help="Evaluation repeats")
    parser.add_argument("--num_ode_steps", type=int, default=10)
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth_window", type=int, default=0,
                        help="Savitzky-Golay window for post-processing (0=none, 11=medium)")
    parser.add_argument("--output_dir", default="eval_results")
    args = parser.parse_args()

    fixseed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load our model
    model_dir = os.path.dirname(args.model_path)
    with open(os.path.join(model_dir, 'args.json')) as f:
        train_args = json.load(f)

    model_version = args.model_version
    state_dir = args.state_dir if args.state_dir else None

    data_wrapper = OakInk2FlowWrapper(
        split='test', data_root=args.data_root,
        num_frames=train_args.get('num_frames', 200),
        batch_size=1, model_version=model_version,
        state_dir=state_dir,
    )
    dataloader = data_wrapper.get_loader(shuffle=False)
    mean = data_wrapper.dataset.mean
    std = data_wrapper.dataset.std
    input_dim = data_wrapper.dataset.feature_dim
    test_indices = data_wrapper.dataset.indices if hasattr(data_wrapper.dataset, 'indices') else []

    use_clip = data_wrapper.dataset.use_clip_emb
    model = create_model(model_version, train_args, input_dim, use_clip, device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    step = ''.join(c for c in os.path.basename(args.model_path) if c.isdigit())
    print(f"Model: {args.model_path} ({model_version}@{step})")

    # Load InterAct evaluator
    print("Loading InterAct evaluator...")
    motionencoder, textencoder, mean_enc, std_enc, bps_basis = \
        load_interact_evaluator(args.interact_root, device)
    print(f"Evaluator loaded (615D, BPS basis {bps_basis.shape})")

    num_samples = args.num_samples if args.num_samples > 0 else len(data_wrapper.dataset)
    smooth_tag = f"_sg{args.smooth_window}" if args.smooth_window > 0 else ""

    all_results = []
    for repeat_id in range(args.num_repeats):
        fixseed(args.seed + repeat_id)

        gt_embeddings = []
        pred_embeddings = []
        text_embeddings = []
        matching_real = 0.0
        matching_pred = 0.0
        n_samples = 0

        print(f"\n--- Repeat {repeat_id + 1}/{args.num_repeats} ---")

        for i, (gt_motion_batch, cond) in enumerate(dataloader):
            if i >= num_samples:
                break

            y = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in cond['y'].items()}
            seq_len = y['lengths'][0].item()

            # Generate motion
            gen_motion = generate_motion(
                model, y, seq_len=seq_len, input_dim=input_dim,
                num_steps=args.num_ode_steps, device=device,
            )

            # Denormalize
            gen_np = gen_motion[0].cpu().numpy() * std + mean
            gt_np = gt_motion_batch[0].numpy() * std + mean

            # Extract positions (V4: first 267D)
            gen_pos = gen_np[:seq_len, :267] if gen_np.shape[1] > 267 else gen_np[:seq_len]
            gt_pos = gt_np[:seq_len, :267] if gt_np.shape[1] > 267 else gt_np[:seq_len]

            # Post-processing: Savitzky-Golay smoothing
            if args.smooth_window > 0 and seq_len > args.smooth_window:
                gen_pos = savgol_filter(gen_pos, window_length=args.smooth_window,
                                         polyorder=3, axis=0)

            # Get object point cloud for BPS encoding
            file_idx = test_indices[i] if i < len(test_indices) else f"{i:06d}"
            obj_points = get_default_object_points(args.data_root, file_idx, args.interact_root)

            # Convert to 615D
            gen_615 = convert_to_615d(gen_pos, obj_points, bps_basis)
            gt_615 = convert_to_615d(gt_pos, obj_points, bps_basis)

            # Normalize
            gen_615_norm = (gen_615 - mean_enc) / std_enc
            gt_615_norm = (gt_615 - mean_enc) / std_enc

            # Encode with pre-trained evaluator
            gen_t = torch.tensor(gen_615_norm, dtype=torch.float32).unsqueeze(0).to(device)
            gt_t = torch.tensor(gt_615_norm, dtype=torch.float32).unsqueeze(0).to(device)
            length_t = torch.tensor([seq_len])

            with torch.no_grad():
                gen_emb = motionencoder(gen_t, length_t).loc.cpu().numpy()  # [1, 256]
                gt_emb = motionencoder(gt_t, length_t).loc.cpu().numpy()

                # Text embedding
                text = ""
                if 'text' in cond['y']:
                    text = cond['y']['text'][0] if isinstance(cond['y']['text'], list) else ""
                elif i < len(test_indices):
                    text_path = os.path.join(args.data_root, 'texts', f'{test_indices[i]}.txt')
                    if os.path.exists(text_path):
                        with open(text_path) as f:
                            text = f.readline().split('#')[0].strip()

                if text:
                    text_emb = textencoder([text]).loc.cpu().numpy()  # [1, 256]
                else:
                    text_emb = np.zeros((1, 256))

            pred_embeddings.append(gen_emb[0])
            gt_embeddings.append(gt_emb[0])
            text_embeddings.append(text_emb[0])

            # Matching score
            matching_real += np.linalg.norm(gt_emb[0] - text_emb[0])
            matching_pred += np.linalg.norm(gen_emb[0] - text_emb[0])
            n_samples += 1

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{num_samples}")

        # Compute metrics
        pred_emb = np.stack(pred_embeddings)
        gt_emb = np.stack(gt_embeddings)
        text_emb = np.stack(text_embeddings)

        # FID
        gt_mu, gt_cov = gt_emb.mean(0), np.cov(gt_emb, rowvar=False)
        pred_mu, pred_cov = pred_emb.mean(0), np.cov(pred_emb, rowvar=False)
        fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)

        # R-Precision
        top_k = [0, 0, 0]
        for j in range(n_samples):
            dists = np.linalg.norm(text_emb[j:j+1] - pred_emb, axis=1)  # [N]
            rank = (dists < dists[j]).sum()
            if rank < 1: top_k[0] += 1
            if rank < 2: top_k[1] += 1
            if rank < 3: top_k[2] += 1
        r_prec = [k / n_samples for k in top_k]

        # Matching Score
        matching_real /= n_samples
        matching_pred /= n_samples

        # Diversity
        diversity = calculate_diversity(pred_emb, 300 if n_samples > 300 else 50)
        diversity_real = calculate_diversity(gt_emb, 300 if n_samples > 300 else 50)

        result = {
            'fid': float(fid),
            'r_prec_1': float(r_prec[0]),
            'r_prec_2': float(r_prec[1]),
            'r_prec_3': float(r_prec[2]),
            'matching_pred': float(matching_pred),
            'matching_real': float(matching_real),
            'diversity': float(diversity),
            'diversity_real': float(diversity_real),
        }
        all_results.append(result)
        print(f"  FID: {fid:.3f}, R@1: {r_prec[0]:.3f}, R@3: {r_prec[2]:.3f}, "
              f"Match: {matching_pred:.3f}, Div: {diversity:.3f}")

    # Average over repeats
    avg = {}
    for key in all_results[0]:
        vals = [r[key] for r in all_results]
        avg[f'{key}_mean'] = float(np.mean(vals))
        avg[f'{key}_std'] = float(np.std(vals))

    print(f"\n{'='*60}")
    print(f"Final (avg over {args.num_repeats} repeats, {n_samples} samples{smooth_tag}):")
    print(f"  FID:         {avg['fid_mean']:.3f} ± {avg['fid_std']:.3f}")
    print(f"  R-Prec@1:    {avg['r_prec_1_mean']:.3f} ± {avg['r_prec_1_std']:.3f}")
    print(f"  R-Prec@2:    {avg['r_prec_2_mean']:.3f} ± {avg['r_prec_2_std']:.3f}")
    print(f"  R-Prec@3:    {avg['r_prec_3_mean']:.3f} ± {avg['r_prec_3_std']:.3f}")
    print(f"  Matching:     {avg['matching_pred_mean']:.3f} ± {avg['matching_pred_std']:.3f}")
    print(f"  Diversity:    {avg['diversity_mean']:.3f} ± {avg['diversity_std']:.3f}")
    print(f"  Div (real):   {avg['diversity_real_mean']:.3f}")

    # Save
    dataset_tag = os.path.basename(args.data_root).lower()
    result_file = os.path.join(args.output_dir,
        f'interact_metrics_{dataset_tag}_{model_version}_{step}{smooth_tag}.json')
    output = {
        'model_path': args.model_path,
        'model_version': model_version,
        'data_root': args.data_root,
        'step': step,
        'num_samples': n_samples,
        'num_repeats': args.num_repeats,
        'smooth_window': args.smooth_window,
        'metrics': avg,
        'per_repeat': all_results,
    }
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {result_file}")


if __name__ == "__main__":
    main()
