#!/usr/bin/env python3
"""
Compute all possible metrics for PriorMDM DoubleTake on OakInk2.

PriorMDM outputs 22 body joints (no hands, no objects).
We compute:
- FID, Diversity, Matching: by embedding 22 joints into 615D eval space (zero-pad markers + no BPS)
- Body Jitter: 3rd-order temporal derivative on 22 joints
- Body-only MSE: map 22 joints → nearest body markers in GT, compare
- Long-horizon metrics: same on chained sequences

Usage:
    python -m eval.eval_priormdm_metrics \
        --priormdm_results priorMDM/save/my_humanml_trans_enc_512/samples_*/results.npy \
        --data_root dataset/OAKINK2_V4 \
        --device 5
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

INTERACT_ROOT = "/hhd4/lizhe/code/InterAct"

# HumanML3D 22-joint order → approximate SMPL-X body marker mapping
# These are the closest body markers (from our 50 body markers) to each SMPL joint
# Joint: 0=pelvis, 1=l_hip, 2=r_hip, 3=spine1, 4=l_knee, 5=r_knee,
#        6=spine2, 7=l_ankle, 8=r_ankle, 9=spine3, 10=l_foot, 11=r_foot,
#        12=neck, 13=l_collar, 14=r_collar, 15=head, 16=l_shoulder, 17=r_shoulder,
#        18=l_elbow, 19=r_elbow, 20=l_wrist, 21=r_wrist

# Approximate mapping: joint_i → body_marker_index (within 50 body markers)
# These are rough correspondences — joints are rotation centers, markers are surface points
JOINT_TO_MARKER = {
    0: 23,   # pelvis → hip marker
    1: 38,   # l_hip → left hip marker
    2: 34,   # r_hip → right hip marker
    3: 19,   # spine1 → lower spine marker
    4: 40,   # l_knee → left knee marker
    5: 36,   # r_knee → right knee marker
    6: 17,   # spine2 → mid spine marker
    7: 42,   # l_ankle → left ankle marker
    8: 30,   # r_ankle → right ankle marker
    9: 3,    # spine3 → upper spine marker
    10: 44,  # l_foot → left foot marker
    11: 32,  # r_foot → right foot marker
    12: 0,   # neck → neck marker
    13: 7,   # l_collar → left shoulder area
    14: 4,   # r_collar → right shoulder area
    15: 1,   # head → head marker
    16: 7,   # l_shoulder → left shoulder marker
    17: 4,   # r_shoulder → right shoulder marker
    18: 9,   # l_elbow → left elbow marker
    19: 6,   # r_elbow → right elbow marker
    20: 11,  # l_wrist → left wrist marker
    21: 8,   # r_wrist → right wrist marker
}


def joints_to_615d(joints_22, bps_basis):
    """Convert 22 joints → 615D InterAct eval format (body only, no objects).

    Embeds 22 joints into 231D marker space (zero-pad unused), adds zero BPS.
    """
    T = joints_22.shape[0]

    # Create 231D marker-like representation (77 markers × 3)
    markers_231 = np.zeros((T, 231), dtype=np.float32)

    # Map 22 joints to their approximate marker positions
    for joint_idx, marker_idx in JOINT_TO_MARKER.items():
        # Each marker occupies 3D: [marker_idx*3 : marker_idx*3+3]
        markers_231[:, marker_idx * 3: marker_idx * 3 + 3] = joints_22[:, joint_idx, :]

    # BPS: zeros (no object)
    bps_384 = np.zeros((T, 384), dtype=np.float32)

    return np.concatenate([markers_231, bps_384], axis=-1)  # [T, 615]


def compute_body_mse(joints_22, gt_markers_50):
    """Compute MSE between 22 joints and GT 50 body markers (mapped subset)."""
    T = min(joints_22.shape[0], gt_markers_50.shape[0])
    mse_vals = []

    for joint_idx, marker_idx in JOINT_TO_MARKER.items():
        joint_pos = joints_22[:T, joint_idx, :]  # [T, 3]
        marker_pos = gt_markers_50[:T, marker_idx, :]  # [T, 3]
        mse = ((joint_pos - marker_pos) ** 2).mean()
        mse_vals.append(mse)

    return np.mean(mse_vals)


def compute_jitter(positions):
    """3rd-order temporal derivative."""
    if positions.shape[0] <= 3:
        return 0.0
    jitter = np.diff(positions, n=3, axis=0)
    return float(np.linalg.norm(jitter, axis=-1).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--priormdm_dir", default="priorMDM/save/my_humanml_trans_enc_512")
    parser.add_argument("--data_root", default="dataset/OAKINK2_V4")
    parser.add_argument("--interact_root", default=INTERACT_ROOT)
    parser.add_argument("--device", type=int, default=5)
    parser.add_argument("--output_dir", default="eval_results")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load InterAct evaluator
    sys.path.insert(0, os.path.join(args.interact_root, "text2interaction"))
    from tma.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
    from tma.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder

    eval_dir = os.path.join(args.interact_root, "text2interaction/assets/eval")
    ckpt = torch.load(os.path.join(eval_dir, "markersbps.ckpt"), map_location=device, weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)

    me_sd = {k.replace('motionencoder.', ''): v for k, v in state_dict.items() if k.startswith('motionencoder.')}
    te_sd = {k.replace('textencoder.', ''): v for k, v in state_dict.items() if k.startswith('textencoder.')}

    motionencoder = ActorAgnosticEncoder(nfeats=615, vae=True, latent_dim=256,
        ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu").to(device)
    motionencoder.load_state_dict(me_sd, strict=False)
    motionencoder.eval()

    textencoder = DistilbertActorAgnosticEncoder(modelpath='distilbert-base-uncased',
        finetune=False, vae=True, latent_dim=256,
        ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu").to(device)
    textencoder.load_state_dict(te_sd, strict=False)
    textencoder.eval()

    mean_enc = np.load(os.path.join(eval_dir, "mean_train_markersbps.npy"))
    std_enc = np.load(os.path.join(eval_dir, "std_train_markersbps.npy"))
    std_enc[std_enc < 1e-8] = 1.0
    bps_basis = np.load(os.path.join(eval_dir, "bps_basis_set_128_1.npy"))

    # Load GT data
    data_root = args.data_root
    gt_mean = np.load(os.path.join(data_root, 'Mean_flow.npy'))
    gt_std = np.load(os.path.join(data_root, 'Std_flow.npy'))
    gt_std[gt_std < 1e-8] = 1.0

    with open(os.path.join(data_root, 'test_flow.txt')) as f:
        test_indices = [l.strip() for l in f if l.strip()]

    # Load PriorMDM per-sample results
    # Find results from single-sample generation
    results_dir = None
    for d in sorted(os.listdir(args.priormdm_dir)):
        if d.startswith('samples_') and 'test_single' in d:
            results_dir = os.path.join(args.priormdm_dir, d)
            break

    # If no per-sample results, use the chain outputs split into segments
    if results_dir and os.path.exists(os.path.join(results_dir, 'results.npy')):
        print(f"Loading per-sample PriorMDM results from {results_dir}")
        data = np.load(os.path.join(results_dir, 'results.npy'), allow_pickle=True).item()
        prior_motion = data['motion']  # [N, 22, 3, T]
        N = prior_motion.shape[0]
        print(f"  {N} samples, shape per sample: {prior_motion.shape}")
    else:
        # Use chain output, split into ~196-frame segments
        print("No per-sample results. Using 50-text chain split into segments.")
        chain_path = 'output/priormdm_oakink2_eval/full_chain_joints.npy'
        if os.path.exists(chain_path):
            chain = np.load(chain_path)  # [T_total, 22, 3]
            seg_len = 196  # ~10s at 20fps
            N = min(50, chain.shape[0] // seg_len)
            prior_motion = np.zeros((N, 22, 3, seg_len))
            for i in range(N):
                seg = chain[i * seg_len:(i + 1) * seg_len]
                prior_motion[i] = seg.transpose(1, 2, 0)  # [22, 3, T]
            print(f"  Split chain into {N} segments of {seg_len} frames")
        else:
            print("ERROR: No PriorMDM data found")
            return

    # Load texts
    texts = []
    text_path = 'output/priormdm_oakink2_eval/test_texts.txt'
    if os.path.exists(text_path):
        with open(text_path) as f:
            texts = [l.strip() for l in f][:N]

    # Compute metrics
    print(f"\nComputing metrics for {N} PriorMDM samples...")

    pred_embeddings = []
    gt_embeddings = []
    text_embeddings = []
    body_mses = []
    jitters = []
    matching_pred = 0.0
    matching_real = 0.0

    for i in range(min(N, len(test_indices))):
        # PriorMDM joints
        if prior_motion.ndim == 4:
            joints = prior_motion[i].transpose(2, 0, 1)  # [T, 22, 3]
        else:
            joints = prior_motion[i]
        T_prior = joints.shape[0]

        # GT markers
        gt_path = os.path.join(data_root, 'motion_dynamic', f'{test_indices[i]}.npy')
        if not os.path.exists(gt_path):
            continue
        gt_motion = np.load(gt_path).astype(np.float32) * gt_std + gt_mean
        gt_pos = gt_motion[:, :267] if gt_motion.shape[1] > 267 else gt_motion
        gt_body_markers = gt_pos[:, :150].reshape(-1, 50, 3)  # [T_gt, 50, 3]
        T_gt = gt_body_markers.shape[0]

        # Align lengths (PriorMDM at 20fps, GT at 10fps — subsample PriorMDM by 2x)
        if T_prior > T_gt * 2:
            joints = joints[:T_gt * 2:2]  # subsample
        elif T_prior > T_gt:
            indices = np.linspace(0, T_prior - 1, T_gt).astype(int)
            joints = joints[indices]
        T = min(joints.shape[0], T_gt)
        joints = joints[:T]
        gt_body_markers = gt_body_markers[:T]

        # Body MSE (22 joints vs nearest 22 markers)
        body_mse = compute_body_mse(joints, gt_body_markers)
        body_mses.append(body_mse)

        # Jitter
        jitter = compute_jitter(joints)
        jitters.append(jitter)

        # Convert to 615D for FID/R-Precision
        prior_615 = joints_to_615d(joints, bps_basis)
        gt_615 = np.concatenate([gt_pos[:T, :231], np.zeros((T, 384))], axis=-1)

        # Normalize
        prior_615_norm = (prior_615 - mean_enc) / std_enc
        gt_615_norm = (gt_615 - mean_enc) / std_enc

        # Encode
        prior_t = torch.tensor(prior_615_norm, dtype=torch.float32).unsqueeze(0).to(device)
        gt_t = torch.tensor(gt_615_norm, dtype=torch.float32).unsqueeze(0).to(device)
        length_t = torch.tensor([T])

        with torch.no_grad():
            pred_emb = motionencoder(prior_t, length_t).loc.cpu().numpy()[0]
            gt_emb = motionencoder(gt_t, length_t).loc.cpu().numpy()[0]

            text = texts[i] if i < len(texts) else "A person manipulates an object."
            text_emb = textencoder([text]).loc.cpu().numpy()[0]

        pred_embeddings.append(pred_emb)
        gt_embeddings.append(gt_emb)
        text_embeddings.append(text_emb)

        matching_pred += np.linalg.norm(pred_emb - text_emb)
        matching_real += np.linalg.norm(gt_emb - text_emb)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N}")

    n = len(pred_embeddings)
    pred_emb = np.stack(pred_embeddings)
    gt_emb = np.stack(gt_embeddings)
    text_emb_arr = np.stack(text_embeddings)

    # FID
    from scipy import linalg
    gt_mu, gt_cov = gt_emb.mean(0), np.cov(gt_emb, rowvar=False)
    pred_mu, pred_cov = pred_emb.mean(0), np.cov(pred_emb, rowvar=False)
    diff = gt_mu - pred_mu
    covmean, _ = linalg.sqrtm(gt_cov.dot(pred_cov), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(gt_cov) + np.trace(pred_cov) - 2 * np.trace(covmean)

    # R-Precision
    top_k = [0, 0, 0]
    for j in range(n):
        dists = np.linalg.norm(text_emb_arr[j:j + 1] - pred_emb, axis=1)
        rank = (dists < dists[j]).sum()
        if rank < 1: top_k[0] += 1
        if rank < 2: top_k[1] += 1
        if rank < 3: top_k[2] += 1
    r_prec = [k / n for k in top_k]

    # Matching
    matching_pred /= n
    matching_real /= n

    # Diversity
    num_d = min(300, n * (n - 1) // 2)
    idx1 = np.random.randint(0, n, num_d)
    idx2 = np.random.randint(0, n, num_d)
    mask = idx1 != idx2
    diversity = np.linalg.norm(pred_emb[idx1[mask]] - pred_emb[idx2[mask]], axis=1).mean() if mask.sum() > 0 else 0

    # Long-horizon metrics (from 3 chains)
    lh_jitters = []
    lh_mses = []
    for name in ['scoop_cheese', 'lightbulb', 'donut']:
        jp = f'output/priormdm_doubletake/{name}_joints.npy'
        if os.path.exists(jp):
            j = np.load(jp)
            lh_jitters.append(compute_jitter(j))

    print(f"\n{'=' * 60}")
    print(f"PriorMDM DoubleTake Results on OakInk2 ({n} samples)")
    print(f"{'=' * 60}")
    print(f"  FID:           {fid:.3f}")
    print(f"  R-Prec@1:      {r_prec[0]:.3f}")
    print(f"  R-Prec@3:      {r_prec[2]:.3f}")
    print(f"  Matching:       {matching_pred:.3f}")
    print(f"  Diversity:      {diversity:.3f}")
    print(f"  Body MSE*:      {np.mean(body_mses):.6f} ± {np.std(body_mses):.6f}")
    print(f"  Body Jitter:    {np.mean(jitters):.6f} ± {np.std(jitters):.6f}")
    if lh_jitters:
        print(f"  LH Jitter (3ch):{np.mean(lh_jitters):.6f}")
    print(f"  Obj Pos:        N/A (no objects)")
    print(f"  Contact F1:     N/A (no hands)")
    print(f"\n  *Body MSE: 22 joints vs nearest 22 of 50 body markers (approximate)")

    result = {
        'method': 'PriorMDM DoubleTake',
        'num_samples': n,
        'fid': float(fid),
        'r_prec_1': float(r_prec[0]),
        'r_prec_2': float(r_prec[1]),
        'r_prec_3': float(r_prec[2]),
        'matching': float(matching_pred),
        'diversity': float(diversity),
        'body_mse_mean': float(np.mean(body_mses)),
        'body_mse_std': float(np.std(body_mses)),
        'jitter_mean': float(np.mean(jitters)),
        'jitter_std': float(np.std(jitters)),
        'lh_jitter_mean': float(np.mean(lh_jitters)) if lh_jitters else None,
        'obj_pos': None,
        'contact_f1': None,
    }

    out_path = os.path.join(args.output_dir, 'priormdm_oakink2_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
