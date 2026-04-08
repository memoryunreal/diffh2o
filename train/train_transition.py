#!/usr/bin/env python3
"""
Train a transition diffusion model.

Supports two dataset modes:
- OakInk2 (398D): Real transitions extracted from natural inter-primitive gaps
- GRAB (117D): Self-supervised infilling via random masking

Usage:
    # Train on OakInk2 transitions
    python -m train.train_transition --dataset oakink2_transition

    # Train on GRAB infilling
    python -m train.train_transition --dataset grab_transition

    # With WandB logging
    python -m train.train_transition --dataset oakink2_transition --train_platform_type WandbPlatform
"""

import os
import sys
import json
import argparse
from pprint import pprint

import torch
import numpy as np

from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from utils.model_util import create_gaussian_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandbPlatform
from configs import card
from model.switch_scheduler import SwitchScheduler, SwitchSchedulerLightweight


def create_transition_model(args, njoints: int, nfeats: int = 1):
    """Create a SwitchScheduler model for transition generation.

    Args:
        args: Training arguments
        njoints: Feature dimension (398 for OakInk2, 117 for GRAB)
        nfeats: Features per joint (always 1 for DiffH2O)

    Returns:
        SwitchScheduler model
    """
    model = SwitchScheduler(
        nfeats=nfeats,
        njoints=njoints,
        latent_dim=args.latent_dim,
        ff_dim=args.ff_size,
        num_layers=args.layers,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        transition_length=args.num_frames,
        use_text_condition=True,
        text_latent_dim=300,  # GloVe embedding dim
        boundary_frames=getattr(args, 'boundary_frames', 10),
        arch='transformer',
        use_obj_condition=True,
        obj_bps_dim=3072,
        cond_mask_prob=args.cond_mask_prob,
    )

    # The SwitchScheduler needs to masquerade as an MDM-like model
    # for compatibility with the training loop and diffusion code
    model.cond_mode = 'text'
    model.njoints = njoints
    model.nfeats = nfeats

    # Dummy methods for compatibility with training loop
    def parameters_wo_clip():
        return model.parameters()
    model.parameters_wo_clip = parameters_wo_clip

    return model


def main():
    # Determine which card to use based on --dataset argument
    # Parse dataset from sys.argv before full arg parsing
    dataset = 'oakink2_transition'
    for i, arg in enumerate(sys.argv):
        if arg == '--dataset' and i + 1 < len(sys.argv):
            dataset = sys.argv[i + 1]
            break

    if dataset == 'grab_transition':
        base_cls = card.grab_transition
    else:
        base_cls = card.oakink2_transition

    args = train_args(base_cls=base_cls)
    pprint(args.__dict__)
    fixseed(args.seed)

    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    # Create data loader
    print(f"Creating {dataset} data loader...")
    data_root = 'dataset/OAKINK2' if dataset == 'oakink2_transition' else 'dataset/GRAB_HANDS'

    data_conf = DatasetConfig(
        name=dataset,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
    )
    data = get_dataset_loader(data_conf)
    print(f"Dataset loaded: {len(data.dataset)} samples")

    # Determine feature dimensions
    if dataset == 'oakink2_transition':
        njoints = 398
    else:
        njoints = 117
    nfeats = 1

    # Create transition model
    print(f"Creating transition model (njoints={njoints})...")
    model = create_transition_model(args, njoints=njoints, nfeats=nfeats)
    model.to(dist_util.dev())

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Total params: {total_params:.2f}M')

    # Create diffusion
    print("Creating diffusion...")
    diffusion = create_gaussian_diffusion(args, data)

    # Set inverse transform function
    def inv_transform_th(motion):
        mean = torch.tensor(data.dataset.mean, device=motion.device, dtype=motion.dtype)
        std = torch.tensor(data.dataset.std, device=motion.device, dtype=motion.dtype)
        return motion * std + mean

    diffusion.data_inv_transform_fn = inv_transform_th

    # Train
    print(f"Training transition model on {dataset}...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()
