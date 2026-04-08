"""
Main module for generating long-term hand-object motion from multi-sentence text.

Supports:
- Template-based planner (verb→primitive mapping) or heuristic planner
- Trained transition diffusion model or interpolation fallback
- Gaze guidance during transitions (OakInk2 only)
- Both OakInk2 (398D) and GRAB (117D) datasets
"""

import os
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import json
from tqdm import tqdm

from utils.motion_data_structures import MotionSegment, MotionSequence, TransitionSegment
from sample.text_orchestration import TextOrchestrator, TemplateOrchestrator
from sample.segment_generator import SegmentGenerator, SegmentGeneratorConfig
from sample.transition_generator import TransitionGenerator, TransitionGeneratorConfig
from utils.fixseed import fixseed
from utils import dist_util


@dataclass
class LongMotionConfig:
    """Configuration for long-term motion generation."""
    # Model paths
    model_path: str
    grasp_model_path: Optional[str] = None
    transition_model_path: Optional[str] = None

    # Generation parameters
    device: str = 'cuda'
    seed: Optional[int] = None
    guidance_param: float = 2.5

    # Segment parameters
    default_segment_frames: int = 100
    min_segment_frames: int = 50
    max_segment_frames: int = 200

    # Transition parameters
    transition_frames: int = 60
    boundary_frames: int = 10
    blend_method: str = 'smooth'
    use_trained_transitions: bool = False
    inpaint_boundary: bool = True
    inpaint_frames: int = 5

    # Planner
    planner_type: str = 'heuristic'  # 'heuristic' or 'template'
    dataset_type: str = 'oakink2'    # 'oakink2' or 'grab'

    # Gaze guidance (OakInk2 only)
    use_gaze_guidance: bool = False
    gaze_guidance_scale: float = 0.3

    # Text processing
    text_detailed: bool = False
    use_nltk: bool = True

    # Output
    output_dir: str = 'output/long_motion'
    save_intermediate: bool = True
    verbose: bool = True


class LongMotionGenerator:
    """
    Orchestrates long-term motion generation from multi-sentence text.
    """

    def __init__(self, config: LongMotionConfig):
        self.config = config

        # Select orchestrator based on planner type
        if config.planner_type == 'template':
            self.text_orchestrator = TemplateOrchestrator(
                default_segment_frames=config.default_segment_frames,
                transition_frames=config.transition_frames,
                dataset_type=config.dataset_type,
            )
        else:
            self.text_orchestrator = TextOrchestrator(
                default_segment_frames=config.default_segment_frames,
                transition_frames=config.transition_frames,
            )

        # Segment generator
        seg_config = SegmentGeneratorConfig(
            model_path=config.model_path,
            grasp_model_path=config.grasp_model_path,
            device=config.device,
            guidance_param=config.guidance_param,
            seed=config.seed,
            text_detailed=config.text_detailed,
        )
        self.segment_generator = SegmentGenerator(seg_config)

        # Transition generator
        njoints = 398 if config.dataset_type == 'oakink2' else 117
        trans_config = TransitionGeneratorConfig(
            model_path=config.transition_model_path if config.use_trained_transitions else None,
            device=config.device,
            transition_length=config.transition_frames,
            boundary_frames=config.boundary_frames,
            blend_method=config.blend_method,
            seed=config.seed,
            njoints=njoints,
            inpaint_boundary=config.inpaint_boundary,
            inpaint_frames=config.inpaint_frames,
            use_gaze_guidance=config.use_gaze_guidance,
            gaze_guidance_scale=config.gaze_guidance_scale,
        )
        self.transition_generator = TransitionGenerator(trans_config)

        os.makedirs(config.output_dir, exist_ok=True)

        if config.seed is not None:
            fixseed(config.seed)

        self._log("Long-term motion generator initialized.")
        self._log(f"  Planner: {config.planner_type}")
        self._log(f"  Dataset: {config.dataset_type}")
        self._log(f"  Transitions: {'trained' if config.use_trained_transitions else 'interpolation'}")
        self._log(f"  Gaze guidance: {config.use_gaze_guidance}")

    def _log(self, message: str):
        if self.config.verbose:
            print(f"[LongMotion] {message}")

    def generate_long_motion(self,
                              prompt: str,
                              object_repr: Optional[torch.Tensor] = None) -> MotionSequence:
        """Main entry point for generating long-term motion."""
        self._log(f"Processing prompt: {prompt[:100]}...")

        # Step 1: Process text
        plan = self.text_orchestrator.process_text(prompt)
        self._log(f"Found {len(plan['sentences'])} sentences.")

        is_valid, issues = self.text_orchestrator.validate_plan(plan)
        if not is_valid:
            self._log(f"Warning: Plan validation issues: {issues}")

        # Step 2: Initialize generators
        self._log("Initializing generators...")
        self.segment_generator.initialize()

        # Step 3: Generate segments
        self._log("Generating motion segments...")
        segments = self._generate_segments(plan, object_repr)

        # Step 4: Generate transitions
        self._log("Generating transitions...")
        transitions = self._generate_transitions(segments, plan)

        # Step 5: Create motion sequence
        motion_sequence = MotionSequence(
            segments=segments,
            transitions=transitions,
            full_text=prompt,
            object_repr=object_repr,
        )

        # Step 6: Post-processing
        motion_sequence = self._post_process(motion_sequence)

        # Step 7: Save outputs
        if self.config.save_intermediate:
            self._save_outputs(motion_sequence, plan)

        self._log(f"Generation complete! Total frames: {motion_sequence.total_frames}")
        return motion_sequence

    def _generate_segments(self, plan: Dict,
                            object_repr: Optional[torch.Tensor]) -> List[MotionSegment]:
        """Generate motion segments for each sentence."""
        segments = []
        generation_steps = [s for s in plan['generation_plan'] if s['type'] == 'segment']

        for step in tqdm(generation_steps, desc="Generating segments"):
            sentence = step['sentence']
            duration = step['duration']

            self._log(f"  Segment {step['index']}: '{sentence[:50]}...' ({duration} frames)")

            segment = self.segment_generator.generate_segment_from_sentence(
                sentence=sentence,
                object_repr=object_repr,
                num_frames=duration,
            )

            segment.metadata.update({
                'segment_index': step['index'],
                'temporal_type': step.get('temporal_type', 'sequential'),
                'object': step.get('object', None),
                'primitive': step.get('primitive', None),
                'hand_dominance': step.get('hand_dominance', 'right'),
            })

            segments.append(segment)

            if self.config.save_intermediate:
                path = os.path.join(self.config.output_dir, f"segment_{step['index']:02d}.npy")
                np.save(path, segment.poses.cpu().numpy())

        return segments

    def _generate_transitions(self, segments: List[MotionSegment],
                                plan: Dict) -> List[TransitionSegment]:
        """Generate transitions between segments."""
        transitions = []

        if not self.config.use_trained_transitions and len(segments) > 0:
            njoints = segments[0].num_joints
            nfeats = segments[0].dim_features
            self.transition_generator.initialize(nfeats, njoints)

        # Get transition steps from plan for metadata
        trans_steps = [s for s in plan['generation_plan'] if s['type'] == 'transition']

        for i in range(len(segments) - 1):
            self._log(f"  Transition {i}: segment {i} -> segment {i + 1}")

            # Get target object position for gaze guidance
            target_obj_pos = None
            if (self.config.use_gaze_guidance and
                    self.config.dataset_type == 'oakink2' and
                    i < len(segments) - 1):
                # Extract target object position from first frame of next segment
                next_poses = segments[i + 1].poses
                if next_poses.shape[-1] >= 374 or next_poses.shape[-2] >= 374:
                    # Get object 1 position from the target segment
                    if next_poses.dim() == 3:  # [T, J, D]
                        target_obj_pos = next_poses[0, 371:374, 0]
                    elif next_poses.dim() == 2:  # [T, D]
                        target_obj_pos = next_poses[0, 371:374]

            # Get transition type from plan
            trans_type = "object_switch"
            if i < len(trans_steps):
                trans_type = trans_steps[i].get('transition_type', 'object_switch')

            transition = self.transition_generator.generate_transition(
                segment_a=segments[i],
                segment_b=segments[i + 1],
                text_a=segments[i].text,
                text_b=segments[i + 1].text,
                transition_length=self.config.transition_frames,
                target_obj_pos=target_obj_pos,
            )

            transition.source_segment_idx = i
            transition.target_segment_idx = i + 1
            transition.metadata['transition_type'] = trans_type

            transitions.append(transition)

            if self.config.save_intermediate:
                path = os.path.join(self.config.output_dir, f"transition_{i:02d}_{i + 1:02d}.npy")
                np.save(path, transition.poses.cpu().numpy())

        return transitions

    def _post_process(self, sequence: MotionSequence) -> MotionSequence:
        """Apply post-processing."""
        sequence = self._smooth_velocities(sequence)
        sequence = self._enforce_hand_constraints(sequence)
        sequence = self._normalize_timing(sequence)
        return sequence

    def _smooth_velocities(self, sequence: MotionSequence) -> MotionSequence:
        """Smooth velocity discontinuities at boundaries."""
        full_motion = sequence.to_tensor(include_transitions=True)
        if full_motion.shape[0] < 3:
            return sequence

        boundaries = sequence.get_segment_boundaries()
        for start, end, seg_type in boundaries:
            if seg_type == 'transition':
                if start > 0 and end < full_motion.shape[0]:
                    window = 3
                    for j in range(max(0, start - window), min(full_motion.shape[0], end + window)):
                        if 0 < j < full_motion.shape[0] - 1:
                            full_motion[j] = (0.25 * full_motion[j - 1] +
                                              0.5 * full_motion[j] +
                                              0.25 * full_motion[j + 1])

        # Update segments
        current_idx = 0
        for i, segment in enumerate(sequence.segments):
            seg_len = segment.num_frames
            segment.poses = full_motion[current_idx:current_idx + seg_len]
            current_idx += seg_len
            if i < len(sequence.transitions):
                trans = sequence.transitions[i]
                trans_len = trans.num_frames
                trans.poses = full_motion[current_idx:current_idx + trans_len]
                current_idx += trans_len

        return sequence

    def _enforce_hand_constraints(self, sequence: MotionSequence) -> MotionSequence:
        """Placeholder for hand physical constraints."""
        return sequence

    def _normalize_timing(self, sequence: MotionSequence) -> MotionSequence:
        total_frames = sequence.total_frames
        if total_frames > 5000:
            self._log(f"Warning: Sequence very long ({total_frames} frames).")
        elif total_frames < 50:
            self._log(f"Warning: Sequence very short ({total_frames} frames).")
        return sequence

    def _save_outputs(self, sequence: MotionSequence, plan: Dict):
        """Save generation outputs."""
        output_path = os.path.join(self.config.output_dir, "full_sequence.npy")
        sequence.save(output_path)
        self._log(f"Saved full sequence to {output_path}")

        # Save plan
        plan_path = os.path.join(self.config.output_dir, "generation_plan.json")
        with open(plan_path, 'w') as f:
            plan_serializable = {
                k: v for k, v in plan.items()
                if k != 'temporal_info'
            }
            plan_serializable['sentences'] = plan['sentences']
            plan_serializable['objects'] = plan.get('objects', [])
            plan_serializable['durations'] = plan.get('durations', [])
            plan_serializable['total_frames'] = plan.get('total_frames', 0)
            json.dump(plan_serializable, f, indent=2)

        # Save config
        config_path = os.path.join(self.config.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)

        # Save boundaries
        boundaries_path = os.path.join(self.config.output_dir, "boundaries.json")
        boundaries = sequence.get_segment_boundaries()
        with open(boundaries_path, 'w') as f:
            json.dump(boundaries, f, indent=2)

    def close(self):
        self.segment_generator.close()
        self.transition_generator.close()


def generate_long_motion(prompt: str,
                          model_path: str,
                          output_dir: str = 'output/long_motion',
                          config: Optional[LongMotionConfig] = None,
                          **kwargs) -> MotionSequence:
    """Convenience function for generating long-term motion."""
    if config is None:
        config = LongMotionConfig(
            model_path=model_path,
            output_dir=output_dir,
            **kwargs,
        )

    generator = LongMotionGenerator(config)
    sequence = generator.generate_long_motion(prompt)
    generator.close()
    return sequence
