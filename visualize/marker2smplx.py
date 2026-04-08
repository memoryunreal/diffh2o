#!/usr/bin/env python3
"""
SMPL-X inverse fitting from 77 markers → full body mesh.

Adapted from InterAct's SmplhOptmize10 for SMPL-X with 77-marker set.
Uses LBFGS optimization with marker reconstruction + smoothness losses.

Usage (as module):
    from visualize.marker2smplx import fit_smplx_to_markers
    vertices, faces = fit_smplx_to_markers(markers, device='cuda:0')
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import smplx as smplx_lib

# 77-marker vertex indices on SMPL-X mesh (from preprocess_v3.py)
MARKERSET_SMPLX = [
    5920, 5621, 5882, 3486, 4430, 3258, 2029, 5694, 5645, 9117, 4302,
    4319, 707, 4788, 4198, 3998, 8846, 4726, 3682, 3688, 5890, 5901,
    3961, 4722, 5698, 5868, 3892, 3712, 4479, 4088, 3477, 4839, 5787,
    4291, 8576, 6248, 7166, 6021, 3067, 8250, 8388, 8339, 3116, 7040,
    7099, 2198, 7524, 7115, 6265, 6746, 8634, 7462, 6443, 6449, 8584,
    8595, 6709, 7458, 8392, 8527, 7215, 6832, 7575, 6503, 8481, 5945,
    5947, 8079, 7669, 7794, 7905, 8022, 5361, 4933, 5058, 5169, 5286,
]

# Torso marker indices (for initial translation guess)
TORSO_MARKERS = [3, 4, 5, 6, 17, 18, 19, 20]  # Indices within 77-marker set


class SmplxOptimize(nn.Module):
    """Optimize SMPL-X parameters from 77 marker positions via LBFGS."""

    def __init__(self, smplx_path, num_frames, device='cuda:0'):
        super().__init__()
        self.device = device
        self.num_frames = num_frames

        # Load SMPL-X model
        self.smplx_model = smplx_lib.create(
            smplx_path, model_type='smplx', gender='neutral',
            use_pca=False, flat_hand_mean=True,
            batch_size=num_frames, num_betas=10,
        ).to(device)

        # Optimizable parameters
        self.transl = Variable(torch.zeros(num_frames, 3, device=device), requires_grad=True)
        self.global_orient = Variable(torch.zeros(num_frames, 3, device=device), requires_grad=True)
        self.body_pose = Variable(torch.zeros(num_frames, 63, device=device), requires_grad=True)
        self.left_hand_pose = Variable(torch.zeros(num_frames, 45, device=device), requires_grad=True)
        self.right_hand_pose = Variable(torch.zeros(num_frames, 45, device=device), requires_grad=True)
        self.betas = Variable(torch.zeros(1, 10, device=device), requires_grad=True)

    def forward_smplx(self):
        output = self.smplx_model(
            transl=self.transl,
            global_orient=self.global_orient,
            body_pose=self.body_pose,
            left_hand_pose=self.left_hand_pose,
            right_hand_pose=self.right_hand_pose,
            betas=self.betas.expand(self.num_frames, -1),
        )
        return output.vertices  # [T, 10475, 3]

    def get_pred_markers(self):
        verts = self.forward_smplx()
        return verts[:, MARKERSET_SMPLX, :]  # [T, 77, 3]

    def init_translation(self, markers_gt):
        """Initialize translation from torso marker centroid."""
        with torch.no_grad():
            verts = self.forward_smplx()
            pred_torso = verts[:, [MARKERSET_SMPLX[i] for i in TORSO_MARKERS], :].mean(dim=1)
            gt_torso = markers_gt[:, TORSO_MARKERS, :].mean(dim=1)
            offset = (gt_torso - pred_torso).float()
        self.transl = Variable(copy.deepcopy(offset), requires_grad=True)

    def smooth_loss(self):
        loss = torch.sum((self.body_pose[1:] - self.body_pose[:-1]) ** 2)
        loss += torch.sum((self.transl[1:] - self.transl[:-1]) ** 2)
        loss += torch.sum((self.global_orient[1:] - self.global_orient[:-1]) ** 2)
        loss += torch.sum((self.left_hand_pose[1:] - self.left_hand_pose[:-1]) ** 2)
        loss += torch.sum((self.right_hand_pose[1:] - self.right_hand_pose[:-1]) ** 2)
        return loss

    def optimize(self, markers_gt, stage1_iters=10, stage2_iters=50, verbose=True):
        """Two-stage optimization: translation first, then full body.

        Args:
            markers_gt: [T, 77, 3] target marker positions (torch tensor on device)
            stage1_iters: LBFGS iterations for translation+orientation
            stage2_iters: LBFGS iterations for full body

        Returns:
            vertices: [T, 10475, 3] optimized SMPL-X mesh vertices
            faces: [F, 3] mesh faces
        """
        self.init_translation(markers_gt)

        # Stage 1: Optimize translation + global orientation only
        if verbose:
            print("  Stage 1: optimizing translation + orientation...")
        opt1 = torch.optim.LBFGS(
            [self.transl, self.global_orient],
            max_iter=100, lr=1e-2, line_search_fn='strong_wolfe',
        )
        for i in range(stage1_iters):
            def closure1():
                opt1.zero_grad()
                pred = self.get_pred_markers()
                loss = 100 * torch.sum((pred - markers_gt) ** 2)
                loss += 5 * self.smooth_loss()
                loss.backward()
                return loss
            opt1.step(closure1)

        # Stage 2: Optimize all parameters
        if verbose:
            print("  Stage 2: optimizing full body...")
        opt2 = torch.optim.LBFGS(
            [self.transl, self.global_orient, self.body_pose, self.betas,
             self.left_hand_pose, self.right_hand_pose],
            max_iter=100, lr=1e-2, line_search_fn='strong_wolfe',
        )
        for i in range(stage2_iters):
            def closure2():
                opt2.zero_grad()
                pred = self.get_pred_markers()
                loss_markers = 100 * torch.sum((pred - markers_gt) ** 2)
                loss_smooth = 5 * self.smooth_loss()
                loss_betas = torch.sum(self.betas ** 2)
                loss = loss_markers + loss_smooth + loss_betas
                loss.backward()
                return loss
            opt2.step(closure2)

        # Final result
        with torch.no_grad():
            verts = self.forward_smplx()
            pred = verts[:, MARKERSET_SMPLX, :]
            final_mse = ((pred - markers_gt) ** 2).mean().item()
            if verbose:
                print(f"  Final marker MSE: {final_mse:.6f}")

        return verts.detach().cpu().numpy(), self.smplx_model.faces


def fit_smplx_to_markers(markers_np, smplx_path='/hhd4/lizhe/code/InterAct/models',
                          device='cuda:0', chunk_size=200):
    """Fit SMPL-X mesh to 77-marker positions.

    Args:
        markers_np: [T, 77, 3] numpy array of marker positions
        smplx_path: Path to SMPL-X model directory
        device: CUDA device
        chunk_size: Process in chunks to avoid OOM

    Returns:
        vertices: [T, 10475, 3] mesh vertices
        faces: [F, 3] mesh face indices
    """
    T = markers_np.shape[0]
    all_verts = []
    faces = None

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = markers_np[start:end]
        n_frames = chunk.shape[0]

        print(f"Fitting frames {start}-{end} ({n_frames} frames)...")
        markers_t = torch.tensor(chunk, dtype=torch.float32, device=device)

        optimizer = SmplxOptimize(smplx_path, n_frames, device=device)
        verts, faces = optimizer.optimize(markers_t)
        all_verts.append(verts)

        # Free GPU memory
        del optimizer, markers_t
        torch.cuda.empty_cache()

    return np.concatenate(all_verts, axis=0), faces


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--markers_path", required=True, help="[T, 267] motion .npy (first 231D = markers)")
    parser.add_argument("--output_path", required=True, help="Output [T, 10475, 3] vertices .npy")
    parser.add_argument("--smplx_path", default="/hhd4/lizhe/code/InterAct/models")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    motion = np.load(args.markers_path)
    if motion.shape[1] > 231:
        motion = motion[:, :231]  # Extract marker dims only
    markers = motion.reshape(-1, 77, 3)

    vertices, faces = fit_smplx_to_markers(markers, args.smplx_path, args.device)
    np.save(args.output_path, vertices)
    print(f"Saved vertices {vertices.shape} to {args.output_path}")
