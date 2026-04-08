#!/usr/bin/env python3
"""
Train CRR-Flow: Flow Matching or Diffusion DiT for motion generation.

Usage:
    # Flow matching (default)
    python -m train.train_flow --device 0 --num_steps 30000 --save_interval 5000

    # Diffusion loss (Way 1 verification)
    python -m train.train_flow --device 0 --num_steps 30000 --use_diffusion --save_interval 5000

    # Overfit on 1 sequence
    python -m train.train_flow --device 0 --num_steps 30000 --batch_size 1 \
        --overfit_single --save_dir save/flow_debug --use_wandb
"""

import os
import json
import argparse
import math
import time

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from utils.fixseed import fixseed
from model.flow_dit import FlowDiT
from data_loaders.humanml.data.flow_dataset import OakInk2FlowWrapper
from data_loaders.humanml.data.himo_flow_dataset import HIMOFlowWrapper


# =============================================================================
# Cosine noise schedule for diffusion mode
# =============================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in improved DDPM."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod[:timesteps]


def main():
    parser = argparse.ArgumentParser(description="Train CRR-Flow model")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="save/oakink2_flow")
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--data_root", type=str, default="dataset/OAKINK2_FLOW")
    parser.add_argument("--split", type=str, default="train", help="Training split name (e.g. train, train_20)")
    # Model
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cond_mask_prob", type=float, default=0.1)
    parser.add_argument("--num_frames", type=int, default=200)
    # Loss mode
    parser.add_argument("--use_diffusion", action="store_true",
                        help="Use standard diffusion loss (predict x0) instead of flow matching")
    parser.add_argument("--diffusion_steps", type=int, default=1000,
                        help="Number of diffusion timesteps (only with --use_diffusion)")
    # Model version
    parser.add_argument("--model_version", type=str, default="v1",
                        choices=["v1", "v3", "v4"],
                        help="Model version: v1 (FlowDiT), v3 (entity-centric), v4 (velocity+BPS)")
    # V4 entity dimensions (configurable for HIMO Track B)
    parser.add_argument("--body_dim", type=int, default=0,
                        help="V4 body entity dim (0=use default 300 for OakInk2)")
    parser.add_argument("--lhand_dim", type=int, default=0,
                        help="V4 left hand entity dim (0=use default 84)")
    parser.add_argument("--rhand_dim", type=int, default=0,
                        help="V4 right hand entity dim (0=use default 78)")
    parser.add_argument("--obj_dyn_dim", type=int, default=0,
                        help="V4 per-object dynamic dim (0=use default 18)")
    parser.add_argument("--bps_dim", type=int, default=0,
                        help="V4 BPS dim per object (0=use default 1024)")
    parser.add_argument("--max_objects", type=int, default=4,
                        help="Max object slots in V4 model")
    # Multi-GPU
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use DDP multi-GPU training (launch with torchrun)")
    parser.add_argument("--data_parallel", type=str, default="",
                        help="Use nn.DataParallel on specified GPUs, e.g. '0,1,2,3'")
    # Debug
    parser.add_argument("--overfit_single", action="store_true",
                        help="Overfit on a single sample for debugging")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint")
    # Autoregressive
    parser.add_argument("--autoregressive", action="store_true",
                        help="Block-wise autoregressive: K history frames clean, L target frames noised")
    parser.add_argument("--history_len", type=int, default=20,
                        help="Number of clean history frames (K) for autoregressive mode")
    # State directory
    parser.add_argument("--state_dir", type=str, default="",
                        help="Custom state arrays directory (e.g. state_arrays_fixed)")
    # Dataset type
    parser.add_argument("--dataset_type", type=str, default="oakink2",
                        choices=["oakink2", "himo"],
                        help="Dataset type: oakink2 or himo")
    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="crr-flow")
    args = parser.parse_args()

    fixseed(args.seed)

    # --- Multi-GPU DDP setup ---
    is_main = True
    local_rank = 0
    if args.multi_gpu:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        print(f"[rank {local_rank}] before init_process_group", flush=True)
        dist.init_process_group(backend='nccl')
        print(f"[rank {local_rank}] after init_process_group", flush=True)
        torch.cuda.set_device(local_rank)
        is_main = (local_rank == 0)

    if is_main:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    if args.multi_gpu:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    loss_mode = "diffusion" if args.use_diffusion else "flow_matching"

    # --- Data ---
    print("Loading dataset...")
    state_dir = args.state_dir if args.state_dir else None
    if args.dataset_type == 'himo':
        data_wrapper = HIMOFlowWrapper(
            split=args.split, data_root=args.data_root,
            num_frames=args.num_frames, batch_size=args.batch_size,
            max_objects=args.max_objects,
            state_dir=state_dir,
        )
    else:
        data_wrapper = OakInk2FlowWrapper(
            split=args.split, data_root=args.data_root,
            num_frames=args.num_frames, batch_size=args.batch_size,
            model_version=args.model_version,
            state_dir=state_dir,
        )
    if args.multi_gpu:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(data_wrapper.dataset, shuffle=not args.overfit_single)
        dataloader = torch.utils.data.DataLoader(
            data_wrapper.dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=4, collate_fn=data_wrapper.collate_fn, drop_last=True,
        )
    else:
        dataloader = data_wrapper.get_loader(shuffle=not args.overfit_single)
    if is_main:
        print(f"Dataset: {len(data_wrapper.dataset)} samples, batch_size={args.batch_size}")

    # --- Model ---
    input_dim = data_wrapper.dataset.feature_dim
    # Skip CLIP loading when precomputed embeddings are available (saves 600MB, enables DDP)
    use_precomputed_clip = data_wrapper.dataset.use_clip_emb
    if args.model_version == 'v4':
        from model.flow_dit_v4 import CRRFlowDiT_V4
        # Build kwargs for configurable entity dimensions (0 = use defaults)
        v4_kwargs = {}
        if args.body_dim > 0:
            v4_kwargs['body_dim'] = args.body_dim
        if args.lhand_dim > 0:
            v4_kwargs['lhand_dim'] = args.lhand_dim
        if args.rhand_dim > 0:
            v4_kwargs['rhand_dim'] = args.rhand_dim
        if args.obj_dyn_dim > 0:
            v4_kwargs['obj_dyn_dim'] = args.obj_dyn_dim
        if args.bps_dim > 0:
            v4_kwargs['bps_dim'] = args.bps_dim
        if args.max_objects != 4:
            v4_kwargs['max_objects'] = args.max_objects
        dims_str = f"input_dim={input_dim}" if not v4_kwargs else f"input_dim={input_dim}, custom_dims={v4_kwargs}"
        print(f"Creating CRRFlowDiT_V4 ({dims_str}, loss_mode={loss_mode}, skip_clip={use_precomputed_clip})...")
        model = CRRFlowDiT_V4(
            latent_dim=args.latent_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            max_seq_len=args.num_frames,
            cond_mask_prob=args.cond_mask_prob,
            skip_clip=use_precomputed_clip,
            **v4_kwargs,
        ).to(device)
    elif args.model_version == 'v3':
        from model.flow_dit_v3 import CRRFlowDiT_V3
        print(f"Creating CRRFlowDiT_V3 (entity-centric, input_dim={input_dim}, loss_mode={loss_mode}, skip_clip={use_precomputed_clip})...")
        model = CRRFlowDiT_V3(
            latent_dim=args.latent_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            max_seq_len=args.num_frames,
            cond_mask_prob=args.cond_mask_prob,
            skip_clip=use_precomputed_clip,
        ).to(device)
    else:
        print(f"Creating FlowDiT model (input_dim={input_dim}, loss_mode={loss_mode})...")
        model = FlowDiT(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            num_states=7,
            max_entities=20,
            max_seq_len=args.num_frames,
            cond_mask_prob=args.cond_mask_prob,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters_wo_clip()) / 1e6
    clip_params = 0.0
    if hasattr(model, 'clip_model') and model.clip_model is not None:
        clip_params = sum(p.numel() for p in model.clip_model.parameters()) / 1e6
    if is_main:
        clip_str = f" + {clip_params:.1f}M frozen CLIP" if clip_params > 0 else " (CLIP skipped)"
        print(f"Model params: {total_params:.1f}M trainable{clip_str}")

    # --- Multi-GPU wrap ---
    if args.multi_gpu:
        print(f"[rank {local_rank}] before DDP()", flush=True)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        print(f"[rank {local_rank}] after DDP()", flush=True)
        raw_model = model.module
    elif args.data_parallel:
        gpu_ids = [int(g) for g in args.data_parallel.split(',')]
        print(f"Using nn.DataParallel on GPUs {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        raw_model = model.module
    else:
        raw_model = model

    # --- Optimizer ---
    optimizer = AdamW(raw_model.parameters_wo_clip(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Loss setup ---
    if args.use_diffusion:
        # Precompute cosine schedule
        alphas_cumprod = cosine_beta_schedule(args.diffusion_steps).to(device)
        print(f"Using DIFFUSION loss (predict x0, {args.diffusion_steps} steps, cosine schedule)")
    else:
        from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
        FM = ConditionalFlowMatcher(sigma=0.0)
        print("Using FLOW MATCHING loss (predict velocity, independent coupling)")

    # --- Resume ---
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        if is_main:
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt, strict=False)
        # Try to load optimizer state and infer start_step from checkpoint name
        opt_path = args.resume.replace('model', 'opt')
        if os.path.exists(opt_path):
            if is_main:
                print(f"Loading optimizer state from {opt_path}")
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        # Extract step number from checkpoint name (e.g. model000050000.pt -> 50000)
        ckpt_name = os.path.basename(args.resume)
        try:
            start_step = int(''.join(c for c in ckpt_name if c.isdigit()))
            if is_main:
                print(f"Resuming from step {start_step}")
        except ValueError:
            pass

    # --- WandB Logging ---
    if args.use_wandb and is_main:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=os.path.basename(args.save_dir),
            config=vars(args),
        )

    # Synchronize all ranks before training
    if args.multi_gpu:
        print(f"[rank {local_rank}] before barrier", flush=True)
        dist.barrier()
        print(f"[rank {local_rank}] after barrier", flush=True)

    # --- Training Loop ---
    if is_main:
        print(f"\nTraining for {args.num_steps} steps (loss={loss_mode})...")
    model.train()
    data_iter = iter(dataloader)
    running_loss = 0.0
    start_time = time.time()

    for step in range(start_step, args.num_steps):
        # Get batch
        try:
            motions, cond = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            motions, cond = next(data_iter)

        if args.multi_gpu and step == start_step:
            print(f"[rank {local_rank}] first batch loaded, shape={motions.shape}", flush=True)

        x1 = motions.to(device)  # [B, T, D] clean motion
        y = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in cond['y'].items()}

        # Overfit single
        if args.overfit_single:
            x1 = x1[:1].expand(max(1, args.batch_size), -1, -1)
            y = {k: (v[:1].expand(max(1, args.batch_size), *v.shape[1:])
                      if isinstance(v, torch.Tensor) else v[:1] * max(1, args.batch_size))
                 for k, v in y.items()}

        B, T, D = x1.shape

        if args.use_diffusion:
            # === DIFFUSION LOSS: predict x0 ===
            # Sample random timesteps
            t_int = torch.randint(0, args.diffusion_steps, (B,), device=device)
            t_float = t_int.float() / args.diffusion_steps  # normalize to [0, 1] for model

            # Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar)*x_0, (1-alpha_bar)*I)
            alpha_bar = alphas_cumprod[t_int]  # [B]
            alpha_bar = alpha_bar.view(B, 1, 1)  # [B, 1, 1] for broadcasting

            noise = torch.randn_like(x1)
            x_t = torch.sqrt(alpha_bar) * x1 + torch.sqrt(1.0 - alpha_bar) * noise

            # Model predicts x0
            x0_pred = model(x_t, t_float, y=y)

            # MSE loss: predict x0
            loss = ((x0_pred - x1) ** 2).mean()

        else:
            # === FLOW MATCHING LOSS: predict velocity ===
            if args.autoregressive:
                # Block-wise autoregressive: K clean history + L noised target
                K = args.history_len
                x1_hist = x1[:, :K, :]      # [B, K, D] clean history
                x1_target = x1[:, K:, :]     # [B, L, D] target

                # Flow matching only on target
                x0_target = torch.randn_like(x1_target)
                t, xt_target, ut_target = FM.sample_location_and_conditional_flow(
                    x0_target, x1_target)
                t = t.to(device)
                xt_target = xt_target.to(device)
                ut_target = ut_target.to(device)

                # Concat: [B, K+L, D] = [B, T, D], history is clean
                xt_full = torch.cat([x1_hist, xt_target], dim=1)

                # Forward (model sees full T frames)
                vt_pred_full = model(xt_full, t, y=y)

                # Loss only on target frames (masked)
                vt_pred_target = vt_pred_full[:, K:, :]
                loss = ((vt_pred_target - ut_target) ** 2).mean()
            else:
                x0 = torch.randn_like(x1)
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                t = t.to(device)
                xt = xt.to(device)
                ut = ut.to(device)

                vt_pred = model(xt, t, y=y)
                loss = ((vt_pred - ut) ** 2).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(raw_model.parameters_wo_clip(), args.grad_clip)
        optimizer.step()

        if args.multi_gpu and step == start_step:
            print(f"[rank {local_rank}] first step done, loss={loss.item():.4f}, "
                  f"GPU mem={torch.cuda.memory_allocated(device)/1e9:.1f}GB", flush=True)

        running_loss += loss.item()

        # Logging
        if is_main and (step + 1) % args.log_interval == 0:
            avg_loss = running_loss / args.log_interval
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1 - start_step) / elapsed
            print(f"Step {step+1}/{args.num_steps} | loss: {avg_loss:.6f} | "
                  f"{steps_per_sec:.1f} steps/s | mode={loss_mode}")
            if args.use_wandb:
                import wandb
                wandb.log({"loss": avg_loss, "step": step + 1,
                           "steps_per_sec": steps_per_sec})
            running_loss = 0.0

        # Save checkpoint (main rank only)
        if is_main and ((step + 1) % args.save_interval == 0 or step == args.num_steps - 1):
            ckpt_path = os.path.join(args.save_dir, f"model{step+1:09d}.pt")
            torch.save(raw_model.state_dict(), ckpt_path)
            opt_path = os.path.join(args.save_dir, f"opt{step+1:09d}.pt")
            torch.save(optimizer.state_dict(), opt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    if is_main:
        print(f"\nTraining complete. Checkpoints at {args.save_dir}/")
    if args.use_wandb and is_main:
        import wandb
        wandb.finish()

    if args.multi_gpu:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
