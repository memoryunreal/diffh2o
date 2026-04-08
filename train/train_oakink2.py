#!/usr/bin/env python3
"""
Train a diffusion model on OakInk2 dataset.

This script trains a diffusion model for full-body hand-object interaction
generation using the OakInk2 dataset.
"""

import os
import json
from pprint import pprint
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandbPlatform
from configs import card


def main():
    # Use OakInk2 full model configuration
    args = train_args(base_cls=card.oakink2_full)

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

    print("Creating OakInk2 data loader...")
    data_conf = DatasetConfig(
        name='oakink2',  # Use OakInk2 dataset
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        motion_enc_frames=getattr(args, 'motion_enc_frames', 0),
    )

    data = get_dataset_loader(data_conf)
    print(f"Dataset loaded: {len(data.dataset)} samples")

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())

    # Set inverse transform function for evaluation
    # OakInk2 uses a simpler inverse transform
    def inv_transform_th(motion):
        """Inverse transform for OakInk2 motion data."""
        import torch
        mean = torch.tensor(data.dataset.mean, device=motion.device, dtype=motion.dtype)
        std = torch.tensor(data.dataset.std, device=motion.device, dtype=motion.dtype)
        return motion * std + mean

    diffusion.data_inv_transform_fn = inv_transform_th

    print('Total params: %.2fM' %
          (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training OakInk2 model...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()
