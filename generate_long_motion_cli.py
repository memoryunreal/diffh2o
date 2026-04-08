#!/usr/bin/env python
"""
CLI interface for generating long-term hand-object motion from multi-sentence text.

Usage:
    # Basic generation (heuristic planner, interpolation transitions)
    python generate_long_motion_cli.py \
        --prompt "The person picks up the apple, examines it, then places it down." \
        --model_path save/diffh2o_full/model000200000.pt

    # Full pipeline (template planner, trained transitions, gaze guidance)
    python generate_long_motion_cli.py \
        --prompt "Pick up the mug, drink from it, then place it down and grab the phone." \
        --model_path save/oakink2_full/model000200000.pt \
        --planner_type template \
        --dataset_type oakink2 \
        --transition_model_path save/oakink2_transition/model000200000.pt \
        --use_trained_transitions \
        --use_gaze_guidance \
        --output output/long_motion
"""

import argparse
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sample.generate_long_motion import LongMotionGenerator, LongMotionConfig
from utils.fixseed import fixseed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate long-term hand-object motion from multi-sentence text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--prompt", type=str, required=True,
                        help="Multi-sentence text prompt")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to DiffH2O model checkpoint")

    # Optional model paths
    parser.add_argument("--grasp_model_path", type=str, default=None)
    parser.add_argument("--transition_model_path", type=str, default=None,
                        help="Path to trained transition model")

    # Output
    parser.add_argument("--output", type=str, default="output/long_motion")
    parser.add_argument("--save_intermediate", action="store_true", default=True)

    # Generation
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--guidance_param", type=float, default=2.5)

    # Segment
    parser.add_argument("--segment_frames", type=int, default=100)
    parser.add_argument("--min_segment_frames", type=int, default=50)
    parser.add_argument("--max_segment_frames", type=int, default=200)

    # Transition
    parser.add_argument("--transition_frames", type=int, default=60)
    parser.add_argument("--boundary_frames", type=int, default=10)
    parser.add_argument("--blend_method", type=str, default="smooth",
                        choices=["smooth", "linear", "cubic"])
    parser.add_argument("--use_trained_transitions", action="store_true",
                        help="Use trained transition model")
    parser.add_argument("--inpaint_boundary", action="store_true", default=True,
                        help="Fix boundary frames during reverse diffusion")
    parser.add_argument("--inpaint_frames", type=int, default=5)

    # Planner
    parser.add_argument("--planner_type", type=str, default="heuristic",
                        choices=["heuristic", "template"],
                        help="Planning strategy: heuristic (simple) or template (verb->primitive)")
    parser.add_argument("--dataset_type", type=str, default="oakink2",
                        choices=["oakink2", "grab"],
                        help="Dataset type (determines feature dim and object vocabulary)")

    # Gaze guidance (OakInk2 only)
    parser.add_argument("--use_gaze_guidance", action="store_true",
                        help="Apply gaze guidance during transitions (OakInk2 only)")
    parser.add_argument("--gaze_guidance_scale", type=float, default=0.3,
                        help="Gaze guidance strength (0.1-0.5)")

    # Text
    parser.add_argument("--text_detailed", action="store_true")
    parser.add_argument("--no_nltk", action="store_true")

    # Other
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file (overrides CLI args)")
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--batch_file", type=str, default=None)

    return parser.parse_args()


def load_config(args):
    config_dict = {
        'model_path': args.model_path,
        'grasp_model_path': args.grasp_model_path,
        'transition_model_path': args.transition_model_path,
        'device': args.device,
        'seed': args.seed,
        'guidance_param': args.guidance_param,
        'default_segment_frames': args.segment_frames,
        'min_segment_frames': args.min_segment_frames,
        'max_segment_frames': args.max_segment_frames,
        'transition_frames': args.transition_frames,
        'boundary_frames': args.boundary_frames,
        'blend_method': args.blend_method,
        'use_trained_transitions': args.use_trained_transitions,
        'inpaint_boundary': args.inpaint_boundary,
        'inpaint_frames': args.inpaint_frames,
        'planner_type': args.planner_type,
        'dataset_type': args.dataset_type,
        'use_gaze_guidance': args.use_gaze_guidance,
        'gaze_guidance_scale': args.gaze_guidance_scale,
        'text_detailed': args.text_detailed,
        'use_nltk': not args.no_nltk,
        'output_dir': args.output,
        'save_intermediate': args.save_intermediate,
        'verbose': args.verbose,
    }

    if args.config:
        with open(args.config, 'r') as f:
            config_dict.update(json.load(f))

    return LongMotionConfig(**config_dict)


def load_prompt(args):
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            return f.read().strip()
    elif args.batch_file:
        with open(args.batch_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return args.prompt


def main():
    args = parse_args()
    config = load_config(args)
    prompts = load_prompt(args)

    if isinstance(prompts, list):
        print(f"Processing {len(prompts)} prompts...")
        for i, prompt in enumerate(prompts):
            print(f"\n[{i + 1}/{len(prompts)}] Processing: {prompt[:100]}...")
            config.output_dir = os.path.join(args.output, f"prompt_{i:03d}")
            try:
                generator = LongMotionGenerator(config)
                sequence = generator.generate_long_motion(prompt)
                generator.close()
                print(f"  Generated {sequence.total_frames} frames -> {config.output_dir}")
            except Exception as e:
                print(f"  Error: {e}")
    else:
        prompt = prompts
        print(f"Prompt: {prompt[:200]}...")
        print(f"Planner: {config.planner_type} | Dataset: {config.dataset_type}")
        print(f"Transitions: {'trained' if config.use_trained_transitions else 'interpolation'}")

        generator = LongMotionGenerator(config)
        try:
            sequence = generator.generate_long_motion(prompt)

            print("\n" + "=" * 50)
            print("GENERATION COMPLETE")
            print("=" * 50)
            print(f"Total frames: {sequence.total_frames}")
            print(f"Segments: {sequence.num_segments}")
            print(f"Transitions: {sequence.num_transitions}")
            print(f"Output: {config.output_dir}")

            if config.verbose:
                print("\nSegment Details:")
                for start, end, seg_type in sequence.get_segment_boundaries():
                    print(f"  [{start:4d}-{end:4d}] {seg_type}")

            print("\nFiles created:")
            for fn in ["full_sequence.npy", "full_sequence.pkl",
                        "generation_plan.json", "config.json", "boundaries.json"]:
                fp = os.path.join(config.output_dir, fn)
                if os.path.exists(fp):
                    size = os.path.getsize(fp) / 1024
                    print(f"  - {fn} ({size:.1f} KB)")

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            generator.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
