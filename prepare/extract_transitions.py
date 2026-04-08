#!/usr/bin/env python3
"""
Extract transition segments from OakInk2 sequences.

Transitions are the gaps between consecutive annotated primitives where the person
releases object A, repositions, and approaches object B. These serve as real
transition training data for the transition diffusion model.

Usage:
    python -m prepare.extract_transitions --num_samples 0  # all sequences
    python -m prepare.extract_transitions --num_samples 10 --visualize
"""

import os
import sys
import json
import argparse
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TransitionInfo:
    """Metadata for an extracted transition segment."""
    sequence_id: str
    transition_idx: int
    # Frame ranges in the original sequence
    source_prim_end: int
    target_prim_start: int
    # Including boundary context
    extract_start: int  # source_prim_end - boundary_frames
    extract_end: int    # target_prim_start + boundary_frames
    # Duration
    gap_frames: int
    total_frames: int  # including boundary context
    resampled_frames: int
    # Text descriptions
    text_source: str
    text_target: str
    # Primitive types
    prim_source: str
    prim_target: str
    # Object IDs
    obj_source: List[str]
    obj_target: List[str]
    # Whether objects differ (object switch vs same object)
    is_object_switch: bool


def parse_frame_range(key_str: str) -> Optional[Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]]:
    """Parse program_info key string into frame ranges.

    Keys look like:
        "(None, (966, 5954))"           -> single hand
        "((1431, 4025), (1431, 4025))"  -> bimanual
    Returns (lh_range, rh_range) where each is (start, end) or None.
    """
    try:
        parsed = ast.literal_eval(key_str)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(parsed, tuple) or len(parsed) != 2:
        return None

    lh, rh = parsed

    def to_range(v):
        if v is None:
            return None
        if isinstance(v, tuple) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        return None

    return (to_range(lh), to_range(rh))


def get_effective_range(lh_range, rh_range) -> Optional[Tuple[int, int]]:
    """Get the effective frame range from left/right hand ranges."""
    ranges = [r for r in [lh_range, rh_range] if r is not None]
    if not ranges:
        return None
    start = min(r[0] for r in ranges)
    end = max(r[1] for r in ranges)
    return (start, end)


def resample_motion(motion: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample motion to target number of frames using linear interpolation.

    Args:
        motion: Input motion array [T, D]
        target_frames: Target number of frames

    Returns:
        Resampled motion [target_frames, D]
    """
    T, D = motion.shape
    if T == target_frames:
        return motion

    old_indices = np.linspace(0, T - 1, T)
    new_indices = np.linspace(0, T - 1, target_frames)
    resampled = np.zeros((target_frames, D))
    for d in range(D):
        resampled[:, d] = np.interp(new_indices, old_indices, motion[:, d])
    return resampled


def extract_transitions_from_sequence(
    seq_id: str,
    program_info: Dict,
    desc_info: Dict,
    anno_dir: str,
    boundary_frames: int = 10,
    min_gap: int = 30,
    max_gap: int = 600,
    target_frames: int = 60,
) -> List[Tuple[TransitionInfo, np.ndarray, np.ndarray, np.ndarray]]:
    """Extract transitions from a single OakInk2 sequence.

    Args:
        seq_id: Sequence identifier
        program_info: Primitive annotations for the sequence
        desc_info: Text descriptions for the sequence
        anno_dir: Directory containing preprocessed annotation pickles
        boundary_frames: Number of context frames from each adjacent primitive
        min_gap: Minimum gap frames to consider as transition
        max_gap: Maximum gap frames
        target_frames: Target frames for resampled transition

    Returns:
        List of (TransitionInfo, transition_motion, boundary_before, boundary_after)
        where transition_motion is [target_frames, 398] and boundaries are [boundary_frames, 398]
    """
    # Parse all primitives and sort by start frame
    primitives = []
    for key_str, prim_data in program_info.items():
        parsed = parse_frame_range(key_str)
        if parsed is None:
            continue
        lh_range, rh_range = parsed
        effective = get_effective_range(lh_range, rh_range)
        if effective is None:
            continue

        text = desc_info.get(key_str, {}).get("seg_desc", "")
        primitives.append({
            "key": key_str,
            "start": effective[0],
            "end": effective[1],
            "primitive": prim_data.get("primitive", ""),
            "obj_list": prim_data.get("obj_list", []),
            "text": text,
        })

    # Sort by start frame
    primitives.sort(key=lambda p: p["start"])

    if len(primitives) < 2:
        return []

    # Load the full sequence motion data
    # Try to load from preprocessed complex mode data first
    motion_path = None
    complex_dir = "dataset/OAKINK2/oakink2_complex"
    # We need the raw annotation to get full-sequence motion
    # The preprocessed data is already segmented, so we need to load from anno_processed
    anno_path = os.path.join(anno_dir, f"{seq_id}.pkl")
    if not os.path.exists(anno_path):
        # Try loading from preprocessed complex sequences
        # Map seq_id to complex file
        return []

    import pickle
    try:
        with open(anno_path, "rb") as f:
            anno_data = pickle.load(f)
    except Exception:
        return []

    # Get preprocessed 398D representation from anno_data
    # The anno_processed files should contain the full-sequence feature vectors
    if "features_398d" in anno_data:
        full_motion = anno_data["features_398d"]
    elif "motion" in anno_data:
        full_motion = anno_data["motion"]
    else:
        return []

    total_seq_frames = len(full_motion)

    # Extract transitions between consecutive primitives
    results = []
    for i in range(len(primitives) - 1):
        src = primitives[i]
        tgt = primitives[i + 1]

        gap_start = src["end"]
        gap_end = tgt["start"]
        gap_frames = gap_end - gap_start

        # Filter by gap duration
        if gap_frames < min_gap or gap_frames > max_gap:
            continue

        # Compute extraction boundaries with context
        extract_start = max(0, gap_start - boundary_frames)
        extract_end = min(total_seq_frames, gap_end + boundary_frames)

        if extract_end <= extract_start:
            continue

        # Extract boundary context from adjacent primitives
        boundary_before_start = max(0, gap_start - boundary_frames)
        boundary_before_end = gap_start
        boundary_after_start = gap_end
        boundary_after_end = min(total_seq_frames, gap_end + boundary_frames)

        # Ensure we have enough boundary frames
        actual_before = boundary_before_end - boundary_before_start
        actual_after = boundary_after_end - boundary_after_start
        if actual_before < 2 or actual_after < 2:
            continue

        # Extract motion segments
        boundary_before = full_motion[boundary_before_start:boundary_before_end]
        boundary_after = full_motion[boundary_after_start:boundary_after_end]
        transition_raw = full_motion[gap_start:gap_end]

        # Pad boundaries to exact boundary_frames if needed
        if len(boundary_before) < boundary_frames:
            pad = np.tile(boundary_before[0:1], (boundary_frames - len(boundary_before), 1))
            boundary_before = np.concatenate([pad, boundary_before], axis=0)
        if len(boundary_after) < boundary_frames:
            pad = np.tile(boundary_after[-1:], (boundary_frames - len(boundary_after), 1))
            boundary_after = np.concatenate([boundary_after, pad], axis=0)

        # Resample transition to target frames
        transition_resampled = resample_motion(transition_raw, target_frames)

        # Determine if objects differ
        obj_src = set(src["obj_list"])
        obj_tgt = set(tgt["obj_list"])
        is_switch = len(obj_src & obj_tgt) == 0 and len(obj_src) > 0 and len(obj_tgt) > 0

        info = TransitionInfo(
            sequence_id=seq_id,
            transition_idx=i,
            source_prim_end=gap_start,
            target_prim_start=gap_end,
            extract_start=extract_start,
            extract_end=extract_end,
            gap_frames=gap_frames,
            total_frames=extract_end - extract_start,
            resampled_frames=target_frames,
            text_source=src["text"],
            text_target=tgt["text"],
            prim_source=src["primitive"],
            prim_target=tgt["primitive"],
            obj_source=src["obj_list"],
            obj_target=tgt["obj_list"],
            is_object_switch=is_switch,
        )

        results.append((info, transition_resampled, boundary_before, boundary_after))

    return results


def extract_transitions_from_preprocessed(
    preprocessed_dir: str = "dataset/OAKINK2",
    program_dir: str = "/hhd4/lizhe/dataset/OakInk2/data/program",
    boundary_frames: int = 10,
    min_gap: int = 30,
    max_gap: int = 600,
    target_frames: int = 60,
    num_samples: int = 0,
) -> Tuple[List[TransitionInfo], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Extract transitions using preprocessed complex-mode data.

    Instead of loading raw annotations, this uses the already preprocessed
    complex-mode sequences which contain full 398D features.

    Returns:
        (infos, transitions, boundaries_before, boundaries_after)
    """
    prog_info_dir = os.path.join(program_dir, "program_info")
    desc_info_dir = os.path.join(program_dir, "desc_info")

    complex_motion_dir = os.path.join(preprocessed_dir, "oakink2_complex")
    file_names_path = os.path.join(preprocessed_dir, "file_names_complex.txt")

    # Load file name mappings: idx -> seq_id
    idx_to_seq = {}
    if os.path.exists(file_names_path):
        with open(file_names_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    idx_to_seq[parts[0]] = parts[1]

    # Build reverse mapping: seq_id -> list of complex motion file indices
    seq_to_files = {}
    for idx, seq_id in idx_to_seq.items():
        # seq_id contains the full sequence identifier
        # Extract the base sequence name for matching with program_info
        seq_to_files.setdefault(seq_id, []).append(idx)

    # Load program info files
    prog_files = sorted(os.listdir(prog_info_dir))
    if num_samples > 0:
        prog_files = prog_files[:num_samples]

    all_infos = []
    all_transitions = []
    all_before = []
    all_after = []

    for prog_file in tqdm(prog_files, desc="Extracting transitions"):
        seq_id = prog_file.replace(".json", "")

        # Load program info
        with open(os.path.join(prog_info_dir, prog_file)) as f:
            program_info = json.load(f)

        # Load desc info
        desc_path = os.path.join(desc_info_dir, prog_file)
        desc_info = {}
        if os.path.exists(desc_path):
            with open(desc_path) as f:
                desc_info = json.load(f)

        # Parse primitives and sort by start frame
        primitives = []
        for key_str, prim_data in program_info.items():
            parsed = parse_frame_range(key_str)
            if parsed is None:
                continue
            lh_range, rh_range = parsed
            effective = get_effective_range(lh_range, rh_range)
            if effective is None:
                continue

            text = desc_info.get(key_str, {}).get("seg_desc", "")
            primitives.append({
                "key": key_str,
                "start": effective[0],
                "end": effective[1],
                "primitive": prim_data.get("primitive", ""),
                "obj_list": prim_data.get("obj_list", []),
                "text": text,
            })

        primitives.sort(key=lambda p: p["start"])

        if len(primitives) < 2:
            continue

        # Try to load the complex-mode full sequence
        # The complex file covers the entire sequence
        complex_motion = None
        for idx, mapped_seq in idx_to_seq.items():
            if seq_id in mapped_seq or mapped_seq in seq_id:
                motion_path = os.path.join(complex_motion_dir, f"{idx}.npy")
                if os.path.exists(motion_path):
                    complex_motion = np.load(motion_path)
                    break

        if complex_motion is None:
            # Fallback: try loading by matching sequence name patterns
            # Complex files might be named differently
            continue

        total_seq_frames = len(complex_motion)

        # Extract transitions between consecutive primitives
        for i in range(len(primitives) - 1):
            src = primitives[i]
            tgt = primitives[i + 1]

            gap_start = src["end"]
            gap_end = tgt["start"]
            gap_frames = gap_end - gap_start

            if gap_frames < min_gap or gap_frames > max_gap:
                continue

            # Frame indices might need scaling if complex mode used different fps
            # OakInk2 is 30fps natively, complex mode preserves this
            if gap_start >= total_seq_frames or gap_end > total_seq_frames:
                continue

            # Extract boundaries
            bb_start = max(0, gap_start - boundary_frames)
            ba_end = min(total_seq_frames, gap_end + boundary_frames)

            boundary_before = complex_motion[bb_start:gap_start]
            boundary_after = complex_motion[gap_end:ba_end]
            transition_raw = complex_motion[gap_start:gap_end]

            if len(boundary_before) < 2 or len(boundary_after) < 2 or len(transition_raw) < 2:
                continue

            # Pad boundaries
            if len(boundary_before) < boundary_frames:
                pad = np.tile(boundary_before[0:1], (boundary_frames - len(boundary_before), 1))
                boundary_before = np.concatenate([pad, boundary_before], axis=0)
            else:
                boundary_before = boundary_before[-boundary_frames:]

            if len(boundary_after) < boundary_frames:
                pad = np.tile(boundary_after[-1:], (boundary_frames - len(boundary_after), 1))
                boundary_after = np.concatenate([boundary_after, pad], axis=0)
            else:
                boundary_after = boundary_after[:boundary_frames]

            # Resample transition
            transition_resampled = resample_motion(transition_raw, target_frames)

            obj_src = set(src["obj_list"])
            obj_tgt = set(tgt["obj_list"])
            is_switch = len(obj_src & obj_tgt) == 0 and len(obj_src) > 0 and len(obj_tgt) > 0

            info = TransitionInfo(
                sequence_id=seq_id,
                transition_idx=i,
                source_prim_end=gap_start,
                target_prim_start=gap_end,
                extract_start=bb_start,
                extract_end=ba_end,
                gap_frames=gap_frames,
                total_frames=ba_end - bb_start,
                resampled_frames=target_frames,
                text_source=src["text"],
                text_target=tgt["text"],
                prim_source=src["primitive"],
                prim_target=tgt["primitive"],
                obj_source=src["obj_list"],
                obj_target=tgt["obj_list"],
                is_object_switch=is_switch,
            )

            all_infos.append(info)
            all_transitions.append(transition_resampled.astype(np.float32))
            all_before.append(boundary_before.astype(np.float32))
            all_after.append(boundary_after.astype(np.float32))

    return all_infos, all_transitions, all_before, all_after


def save_transitions(
    output_dir: str,
    infos: List[TransitionInfo],
    transitions: List[np.ndarray],
    boundaries_before: List[np.ndarray],
    boundaries_after: List[np.ndarray],
    train_ratio: float = 0.8,
):
    """Save extracted transitions to disk with train/test split.

    Output structure:
        output_dir/
            transitions/          # [target_frames, 398] motion files
            boundaries_before/    # [boundary_frames, 398] context
            boundaries_after/     # [boundary_frames, 398] context
            texts_transitions/    # Text annotations
            transition_metadata.json
            Mean_transitions.npy
            Std_transitions.npy
            train_transitions.txt
            test_transitions.txt
    """
    trans_dir = os.path.join(output_dir, "oakink2_transitions")
    before_dir = os.path.join(output_dir, "boundaries_before")
    after_dir = os.path.join(output_dir, "boundaries_after")
    text_dir = os.path.join(output_dir, "texts_transitions")

    for d in [trans_dir, before_dir, after_dir, text_dir]:
        os.makedirs(d, exist_ok=True)

    # Save individual files
    for i, (info, trans, before, after) in enumerate(
        zip(infos, transitions, boundaries_before, boundaries_after)
    ):
        name = f"{i:06d}"
        np.save(os.path.join(trans_dir, f"{name}.npy"), trans)
        np.save(os.path.join(before_dir, f"{name}.npy"), before)
        np.save(os.path.join(after_dir, f"{name}.npy"), after)

        # Write text annotation in DiffH2O format
        text_line = (
            f"Transition from: {info.text_source} to: {info.text_target}"
            f"#{info.prim_source}_to_{info.prim_target}#0.0#0.0\n"
            f"{info.text_source}##0.0#0.0\n"
            f"{info.text_target}##0.0#0.0"
        )
        with open(os.path.join(text_dir, f"{name}.txt"), "w") as f:
            f.write(text_line)

    # Compute normalization statistics from all transitions
    if transitions:
        all_trans = np.concatenate(transitions, axis=0)  # [N*T, 398]
        mean = all_trans.mean(axis=0)
        std = all_trans.std(axis=0)
        std[std < 1e-6] = 1.0  # Avoid division by zero
    else:
        mean = np.zeros(398)
        std = np.ones(398)

    np.save(os.path.join(output_dir, "Mean_transitions.npy"), mean)
    np.save(os.path.join(output_dir, "Std_transitions.npy"), std)

    # Train/test split
    n = len(infos)
    indices = np.random.RandomState(42).permutation(n)
    n_train = int(n * train_ratio)
    train_indices = sorted(indices[:n_train])
    test_indices = sorted(indices[n_train:])

    with open(os.path.join(output_dir, "train_transitions.txt"), "w") as f:
        for idx in train_indices:
            f.write(f"{idx:06d}\n")

    with open(os.path.join(output_dir, "test_transitions.txt"), "w") as f:
        for idx in test_indices:
            f.write(f"{idx:06d}\n")

    # Save metadata
    metadata = {
        "total_transitions": n,
        "train_count": len(train_indices),
        "test_count": len(test_indices),
        "target_frames": infos[0].resampled_frames if infos else 60,
        "feature_dim": transitions[0].shape[-1] if transitions else 398,
        "transitions": [asdict(info) for info in infos],
    }

    with open(os.path.join(output_dir, "transition_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Print statistics
    gap_durations = [info.gap_frames for info in infos]
    obj_switches = sum(1 for info in infos if info.is_object_switch)
    print(f"\nTransition Extraction Statistics:")
    print(f"  Total transitions: {n}")
    print(f"  Train/Test: {len(train_indices)}/{len(test_indices)}")
    print(f"  Object switches: {obj_switches} ({100*obj_switches/max(n,1):.1f}%)")
    if gap_durations:
        print(f"  Gap duration (frames): min={min(gap_durations)}, "
              f"max={max(gap_durations)}, median={np.median(gap_durations):.0f}, "
              f"mean={np.mean(gap_durations):.0f}")
    print(f"  Feature dim: {transitions[0].shape[-1] if transitions else 'N/A'}")
    print(f"  Output dir: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract OakInk2 transitions")
    parser.add_argument("--preprocessed_dir", type=str, default="dataset/OAKINK2",
                        help="Directory with preprocessed OakInk2 data")
    parser.add_argument("--program_dir", type=str,
                        default="/hhd4/lizhe/dataset/OakInk2/data/program",
                        help="Directory with program_info and desc_info")
    parser.add_argument("--output_dir", type=str, default="dataset/OAKINK2",
                        help="Output directory for transitions")
    parser.add_argument("--boundary_frames", type=int, default=10,
                        help="Number of boundary context frames")
    parser.add_argument("--min_gap", type=int, default=30,
                        help="Minimum gap frames (1s at 30fps)")
    parser.add_argument("--max_gap", type=int, default=600,
                        help="Maximum gap frames (20s at 30fps)")
    parser.add_argument("--target_frames", type=int, default=60,
                        help="Resample transitions to this many frames")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of sequences to process (0=all)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train/test split ratio")

    args = parser.parse_args()

    print("Extracting transitions from OakInk2...")
    print(f"  Preprocessed dir: {args.preprocessed_dir}")
    print(f"  Program dir: {args.program_dir}")
    print(f"  Boundary frames: {args.boundary_frames}")
    print(f"  Gap range: [{args.min_gap}, {args.max_gap}] frames")
    print(f"  Target frames: {args.target_frames}")

    # Method 1: Real transitions from complex-mode sequences
    infos, transitions, before, after = extract_transitions_from_preprocessed(
        preprocessed_dir=args.preprocessed_dir,
        program_dir=args.program_dir,
        boundary_frames=args.boundary_frames,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        target_frames=args.target_frames,
        num_samples=args.num_samples,
    )
    print(f"\n  Real transitions from complex data: {len(infos)}")

    # Method 2: Synthetic transitions from primitive pairs (always run)
    prim_infos, prim_trans, prim_before, prim_after = extract_transitions_from_primitives(
        preprocessed_dir=args.preprocessed_dir,
        program_dir=args.program_dir,
        boundary_frames=args.boundary_frames,
        target_frames=args.target_frames,
        num_samples=args.num_samples,
    )
    print(f"  Synthetic transitions from primitive pairs: {len(prim_infos)}")

    # Combine both sources
    infos.extend(prim_infos)
    transitions.extend(prim_trans)
    before.extend(prim_before)
    after.extend(prim_after)

    if infos:
        save_transitions(
            args.output_dir, infos, transitions, before, after,
            train_ratio=args.train_ratio,
        )
    else:
        print("No transitions found. Check data paths and preprocessing.")


def extract_transitions_from_primitives(
    preprocessed_dir: str = "dataset/OAKINK2",
    program_dir: str = "/hhd4/lizhe/dataset/OakInk2/data/program",
    boundary_frames: int = 10,
    target_frames: int = 60,
    num_samples: int = 0,
) -> Tuple[List[TransitionInfo], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Fallback: Create cross-primitive transitions from primitive-mode files.

    When full complex sequences aren't available, we create transitions by
    pairing the end of one primitive with the start of the next primitive
    from the same sequence. The transition itself is synthesized via
    interpolation as a training target (the model learns to improve on this).
    """
    prim_dir = os.path.join(preprocessed_dir, "oakink2_primitive")
    text_dir = os.path.join(preprocessed_dir, "texts_primitive")
    file_names_path = os.path.join(preprocessed_dir, "file_names_primitive.txt")

    if not os.path.exists(prim_dir):
        return [], [], [], []

    # Load file names mapping
    idx_to_info = {}
    if os.path.exists(file_names_path):
        with open(file_names_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    idx_to_info[parts[0]] = parts[1]

    # Group primitives by sequence
    # Primitive names look like: {seq_id}_{primitive}_{start_frame}
    # where seq_id is a pkl filename without .pkl extension
    # e.g. "scene_01__A001++seq__<hash>__<date>_grip_966"
    # Multi-word primitives like "take_outside" make simple rsplit unreliable,
    # so we match against actual sequence IDs from program_info files.
    from collections import defaultdict

    # Build set of known sequence IDs from program_info directory
    known_seq_ids = set()
    prog_info_path = os.path.join(program_dir, "program_info")
    if os.path.exists(prog_info_path):
        for fname in os.listdir(prog_info_path):
            if fname.endswith(".json"):
                known_seq_ids.add(fname.replace(".json", ""))
    # Sort by length descending so longer IDs match first (prevents partial matches)
    sorted_seq_ids = sorted(known_seq_ids, key=len, reverse=True)

    def extract_seq_name(info_str):
        """Extract sequence ID from primitive file name."""
        for seq_id in sorted_seq_ids:
            if info_str.startswith(seq_id + "_"):
                return seq_id
        # Fallback: return the full string (each primitive becomes its own group)
        return info_str

    seq_groups = defaultdict(list)
    for idx, info_str in idx_to_info.items():
        seq_name = extract_seq_name(info_str)
        seq_groups[seq_name].append(idx)

    # Sort each group and create pairs
    all_infos = []
    all_transitions = []
    all_before = []
    all_after = []

    prog_info_dir = os.path.join(program_dir, "program_info")
    desc_info_dir = os.path.join(program_dir, "desc_info")

    processed = 0
    for seq_name, indices in tqdm(sorted(seq_groups.items()), desc="Processing primitive pairs"):
        if num_samples > 0 and processed >= num_samples:
            break

        # Sort indices numerically
        indices = sorted(indices, key=lambda x: int(x))

        if len(indices) < 2:
            continue

        # Create transitions between consecutive primitives
        for j in range(len(indices) - 1):
            idx_a = indices[j]
            idx_b = indices[j + 1]

            motion_a_path = os.path.join(prim_dir, f"{idx_a}.npy")
            motion_b_path = os.path.join(prim_dir, f"{idx_b}.npy")

            if not os.path.exists(motion_a_path) or not os.path.exists(motion_b_path):
                continue

            motion_a = np.load(motion_a_path)
            motion_b = np.load(motion_b_path)

            if len(motion_a) < boundary_frames + 2 or len(motion_b) < boundary_frames + 2:
                continue

            # Boundary frames
            boundary_before = motion_a[-boundary_frames:]
            boundary_after = motion_b[:boundary_frames]

            # Create interpolated transition as training target
            last_a = motion_a[-1:]
            first_b = motion_b[0:1]
            alphas = np.linspace(0, 1, target_frames).reshape(-1, 1)
            transition = (1 - alphas) * last_a + alphas * first_b

            # Load texts
            text_a = ""
            text_b = ""
            text_path_a = os.path.join(text_dir, f"{idx_a}.txt")
            text_path_b = os.path.join(text_dir, f"{idx_b}.txt")
            if os.path.exists(text_path_a):
                with open(text_path_a) as f:
                    text_a = f.readline().split("#")[0].strip()
            if os.path.exists(text_path_b):
                with open(text_path_b) as f:
                    text_b = f.readline().split("#")[0].strip()

            info = TransitionInfo(
                sequence_id=seq_name,
                transition_idx=j,
                source_prim_end=len(motion_a),
                target_prim_start=0,
                extract_start=len(motion_a) - boundary_frames,
                extract_end=boundary_frames,
                gap_frames=target_frames,
                total_frames=target_frames + 2 * boundary_frames,
                resampled_frames=target_frames,
                text_source=text_a,
                text_target=text_b,
                prim_source="unknown",
                prim_target="unknown",
                obj_source=[],
                obj_target=[],
                is_object_switch=text_a != text_b,
            )

            all_infos.append(info)
            all_transitions.append(transition.astype(np.float32))
            all_before.append(boundary_before.astype(np.float32))
            all_after.append(boundary_after.astype(np.float32))

        processed += 1

    return all_infos, all_transitions, all_before, all_after


if __name__ == "__main__":
    main()
