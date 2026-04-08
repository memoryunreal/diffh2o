#!/usr/bin/env python3
"""
Generate motion samples from OakInk2-trained diffusion model.

This script generates full-body hand-object interaction motions from text
descriptions using the OakInk2 dataset representation (398D features).

Usage:
    # Generate from dataset (default)
    python -m sample.generate_oakink2 \
        --model_path ./save/oakink2_full/model000200000.pt \
        --num_samples 16 \
        --batch_size 16

    # Generate from single text prompt
    python -m sample.generate_oakink2 \
        --model_path ./save/oakink2_full/model000200000.pt \
        --text_prompt "Press the button on the scale."

    # Generate from text file
    python -m sample.generate_oakink2 \
        --model_path ./save/oakink2_full/model000200000.pt \
        --input_text ./prompts.txt

    # Evaluate entire test set
    python -m sample.generate_oakink2 \
        --model_path ./save/oakink2_full/model000200000.pt \
        --eval_entire_set \
        --batch_size 32
"""

import os
import json
import shutil
import numpy as np
import torch
from pprint import pprint

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import generate_args
from utils.output_util import sample_to_motion_oakink2, recover_from_ric_oakink2
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from data_loaders.humanml.data.oakink2_dataset import OAKINK2_REPRESENTATION_IDCS


def load_oakink2_dataset(args, max_frames, n_frames, data_mode='primitive'):
    """Load OakInk2 dataset with simplified configuration.

    Args:
        args: Generation arguments
        max_frames: Maximum number of frames
        n_frames: Fixed sequence length
        data_mode: 'primitive' or 'complex'

    Returns:
        Data loader for OakInk2 dataset
    """
    conf = DatasetConfig(
        name='oakink2',
        batch_size=args.batch_size,
        num_frames=max_frames,
        split=args.gen_split,
        hml_mode='text_only',
        data_mode=data_mode,
        motion_enc_frames=0,
    )
    data = get_dataset_loader(conf, shuffle=args.random_order)
    data.fixed_length = n_frames
    return data


def main():
    # Parse arguments (load from model checkpoint)
    args = generate_args()

    # Override dataset for OakInk2
    args.dataset = 'oakink2'

    pprint(args.__dict__)
    print("##### OakInk2 Generation Mode #####")

    # Get model iteration from path
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')

    fixseed(args.seed)

    # Output path setup
    out_path = args.output_dir
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path), f'samples_{niter}')

    # Frame settings (OakInk2 uses 200 frames max, 30 FPS)
    max_frames = 200
    n_frames = max_frames
    print(f'n_frames: {n_frames}')

    dist_util.setup_dist(args.device)

    # Text input handling (optional - if not specified, uses dataset texts)
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 3
        args.num_repetitions = 1
        print(f"Using text prompt: {args.text_prompt}")
    elif args.input_text != '':
        assert os.path.exists(args.input_text), f"Input text file not found: {args.input_text}"
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.strip() for s in texts]
        args.num_samples = len(texts)
        print(f"Loaded {len(texts)} text prompts from file")
    elif args.action_file != '':
        assert os.path.exists(args.action_file), f"Action file not found: {args.action_file}"
        with open(args.action_file, 'r') as fr:
            texts = fr.readlines()
        texts = [s.strip() for s in texts]
        args.num_samples = len(texts)
        print(f"Loaded {len(texts)} action prompts from file")
    else:
        # Will use texts from dataset
        texts = None
        print("Using texts from dataset")

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'

    args.batch_size = args.num_samples

    # Load OakInk2 dataset
    print('Loading OakInk2 dataset...')
    data_mode = getattr(args, 'data_mode', 'primitive')
    data = load_oakink2_dataset(args, max_frames, n_frames, data_mode=data_mode)

    print(f"Dataset loaded: {len(data.dataset)} samples")

    # Calculate number of repetitions
    if args.eval_entire_set:
        num_reps = len(data.dataset.t2m_dataset.name_list) // args.batch_size * args.num_repetitions
    else:
        num_reps = args.num_repetitions
    print(f"NUM REPS: {num_reps}")

    total_num_samples = args.num_samples * num_reps

    # Storage for results
    all_motions = []
    all_lengths = []
    all_text = []
    all_data_id = []
    all_gt_kf = []

    # Create output directory
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    # Save args
    args_path = os.path.join(out_path, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    # Model setup
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoint from [{args.model_path}]...")
    load_saved_model(model, args.model_path)

    # Wrap with CFG sampler if needed
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)

    model.to(dist_util.dev())
    model.eval()

    print(f"Model njoints: {model.njoints}, nfeats: {model.nfeats}")

    # Set up inverse transform function
    def inv_transform_th(motion):
        """Inverse transform for OakInk2 motion data."""
        mean = torch.tensor(data.dataset.mean, device=motion.device, dtype=motion.dtype)
        std = torch.tensor(data.dataset.std, device=motion.device, dtype=motion.dtype)
        return motion * std + mean

    diffusion.data_inv_transform_fn = inv_transform_th
    diffusion.data_transform_fn = data.dataset.t2m_dataset.transform_th
    diffusion.data_get_mean_fn = data.dataset.t2m_dataset.get_std_mean

    # Sample function
    sample_fn = diffusion.p_sample_loop
    dump_steps = [999]
    num_dump_step = len(dump_steps)
    args.num_dump_step = num_dump_step

    # Get data iterator
    iterator = iter(data)

    # Generation loop
    for rep_i in range(num_reps):
        print(f"REP {rep_i}")

        # Get batch from dataset
        motion_gt, model_kwargs = next(iterator)

        # Store ground truth keyframes for reference
        motion_gt_unnorm = data.dataset.t2m_dataset.inv_transform_th(
            motion_gt.cpu().permute(0, 2, 3, 1)
        ).float().permute(0, 1, 3, 2)
        motion_gt_kf = motion_gt_unnorm[..., [0, n_frames-1]]  # First and last frames
        all_gt_kf.extend(motion_gt_kf.swapaxes(2, 3))

        # Add CFG scale
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(
                args.batch_size, device=dist_util.dev()
            ) * args.guidance_param

        model_kwargs['y']['log_id'] = rep_i

        # Generate sample
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, n_frames),
            clip_denoised=not args.predict_xstart,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=dump_steps,
            noise=None,
            const_noise=False,
            cond_fn=None,
        )

        # Cut to effective length
        gen_eff_len = min(sample[0].shape[-1], n_frames)
        print(f'Generated motion length: {gen_eff_len} frames')

        for j in range(len(sample)):
            sample[j] = sample[j][:, :, :, :gen_eff_len]

        # Convert to motion
        cur_motions_pose_space, cur_motions, cur_lengths, cur_texts = sample_to_motion_oakink2(
            sample, args, model_kwargs, model, gen_eff_len,
            data.dataset.t2m_dataset.inv_transform
        )

        cur_motions_pose_space = np.array(cur_motions_pose_space)[0]

        all_motions.extend(cur_motions_pose_space[:, np.newaxis])
        all_lengths.extend(cur_lengths)
        all_text.extend(cur_texts)
        all_data_id.extend(model_kwargs['y']['data_id'])

    # Concatenate results
    total_num_samples = args.num_samples * num_reps * num_dump_step
    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[-total_num_samples:]
    all_text = all_text[:total_num_samples]
    all_data_id = all_data_id[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_gt_kf = np.concatenate(all_gt_kf, axis=0)[-total_num_samples:]

    # Extract motion components for logging
    motion_components = recover_from_ric_oakink2(all_motions[:, :, :n_frames], OAKINK2_REPRESENTATION_IDCS)
    print(f"Generated motion shapes:")
    print(f"  - Full motion: {all_motions.shape}")
    print(f"  - Body pose: {motion_components['body_pose'].shape}")
    print(f"  - Left hand PCA: {motion_components['left_hand_pca'].shape}")
    print(f"  - Right hand PCA: {motion_components['right_hand_pca'].shape}")
    print(f"  - Object 1 pose: {motion_components['object1_pose'].shape}")

    # Save results
    result_file = f'results_oakink2_{args.gen_split}.npy'
    npy_path = os.path.join(out_path, result_file)

    print(f"Saving results to [{npy_path}]")
    np.save(npy_path, {
        'motion': all_motions[:, :, :n_frames],
        'text': all_text,
        'lengths': all_lengths,
        'num_samples': args.num_samples,
        'num_repetitions': args.num_repetitions,
        'gt_kf': all_gt_kf[:, :, :n_frames],
        'data_id': all_data_id,
        'feature_dim': 398,
    })

    # Save text file
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))

    # Save lengths file
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    # Save motion components for easy access
    components_path = os.path.join(out_path, 'motion_components.npy')
    np.save(components_path, {k: v.numpy() for k, v in motion_components.items()})

    print(f"[Done] Results saved to [{os.path.abspath(out_path)}]")
    print(f"  - Main results: {npy_path}")
    print(f"  - Text prompts: {npy_path.replace('.npy', '.txt')}")
    print(f"  - Motion components: {components_path}")


if __name__ == "__main__":
    main()
