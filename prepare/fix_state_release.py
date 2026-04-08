#!/usr/bin/env python3
"""
Fix state labels: add STATE_RELEASE (4) for frames after contact ends.

The original preprocess_v3.py never assigns Release. This script post-processes
existing state arrays to insert Release states between the last Grasped frame
and the next Static/Reach block.

Logic per hand column:
  Forward scan: track whether we've seen Grasped.
  After last Grasped frame in a block → assign Release until next Reach or Static.

Usage:
    python -m prepare.fix_state_release \
        --data_root dataset/OAKINK2_V4 \
        --output_dir dataset/OAKINK2_V4/state_arrays_fixed
"""

import os
import argparse
import numpy as np

STATE_STATIC = 0
STATE_REACH = 1
STATE_GRASPED = 2
STATE_INTERACTING = 3
STATE_RELEASE = 4
STATE_PADDING = 5

STATE_NAMES = {0: 'Static', 1: 'Reach', 2: 'Grasped', 3: 'Interacting', 4: 'Release', 5: 'Padding'}


def fix_hand_column(col):
    """Fix a single hand column [T] to insert Release after Grasped blocks.

    Pattern detection:
      ... Reach Reach Grasped Grasped Grasped Reach Reach Static ...
                                              ^^^^^ ^^^^^ these should be Release
    """
    T = len(col)
    fixed = col.copy()
    seen_grasped = False

    for i in range(T):
        if fixed[i] == STATE_GRASPED:
            seen_grasped = True
        elif seen_grasped:
            # We were in Grasped, now we're not → this is Release
            if fixed[i] in (STATE_REACH, STATE_STATIC):
                fixed[i] = STATE_RELEASE
            if fixed[i] == STATE_STATIC:
                # Fully released, reset
                seen_grasped = False
            # If it's Reach after Grasped, mark as Release and continue
            # until we hit Static (fully disengaged)

    return fixed


def fix_object_column(col, hand_col_fixed):
    """Fix object column: add Release for objects when corresponding hand releases.

    If hand transitions from Grasped→Release, the object should also transition
    from Grasped/Interacting→Static (object is released).
    Object Release isn't used (objects go directly to Static), but we keep
    the object's Grasped/Interacting status consistent with the hand.
    """
    # Objects don't need Release state — they go back to Static when released.
    # But we should ensure object is Static when hand is in Release.
    T = len(col)
    fixed = col.copy()
    for i in range(T):
        if fixed[i] == STATE_PADDING:
            continue
        if hand_col_fixed[i] == STATE_RELEASE and fixed[i] in (STATE_GRASPED, STATE_INTERACTING):
            fixed[i] = STATE_STATIC
    return fixed


def fix_state_array(states):
    """Fix a [T, 7] state array to include Release states.

    Args:
        states: [T, 7] original state array

    Returns:
        [T, 7] fixed state array with Release states
    """
    fixed = states.copy()

    # Fix left hand (col 1)
    fixed[:, 1] = fix_hand_column(fixed[:, 1])

    # Fix right hand (col 2)
    fixed[:, 2] = fix_hand_column(fixed[:, 2])

    # Fix objects based on their corresponding hand's state
    # Objects 1-4 (cols 3-6): check which hand was grasping them
    # Simple heuristic: if any hand released, object goes Static
    for obj_col in range(3, 7):
        if np.all(fixed[:, obj_col] == STATE_PADDING):
            continue
        # Check against both hands
        fixed[:, obj_col] = fix_object_column(fixed[:, obj_col], fixed[:, 1])
        fixed[:, obj_col] = fix_object_column(fixed[:, obj_col], fixed[:, 2])

    # Apply majority vote smoothing to Release regions (window=3, smaller to preserve boundaries)
    for col in range(1, 3):  # hands only
        T = len(fixed[:, col])
        smoothed = fixed[:, col].copy()
        for i in range(1, T - 1):
            window = fixed[i-1:i+2, col]
            vals, counts = np.unique(window, return_counts=True)
            smoothed[i] = vals[counts.argmax()]
        fixed[:, col] = smoothed

    return fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset/OAKINK2_V4")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = os.path.join(args.data_root, 'state_arrays_fixed')
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.data_root, 'state_arrays')
    files = sorted(f for f in os.listdir(state_dir) if f.endswith('.npy'))
    print(f"Processing {len(files)} state arrays...")

    total_release_frames = 0
    total_frames = 0

    for fname in files:
        states = np.load(os.path.join(state_dir, fname)).astype(np.int8)
        fixed = fix_state_array(states)
        np.save(os.path.join(args.output_dir, fname), fixed.astype(np.int8))

        total_release_frames += (fixed[:, 1:3] == STATE_RELEASE).sum()
        total_frames += fixed.shape[0] * 2  # 2 hand columns

    # Print stats
    all_fixed = []
    for fname in files:
        all_fixed.append(np.load(os.path.join(args.output_dir, fname)))
    all_fixed = np.concatenate(all_fixed, axis=0)

    print(f"\nState distribution after fix:")
    entities = ['Body', 'LH', 'RH', 'Obj1', 'Obj2', 'Obj3', 'Obj4']
    for e in range(7):
        col = all_fixed[:, e]
        unique, counts = np.unique(col, return_counts=True)
        total = counts.sum()
        dist = ', '.join(f'{STATE_NAMES[u]}={c/total*100:.1f}%' for u, c in zip(unique, counts))
        print(f'  {entities[e]:5s}: {dist}')

    print(f"\nRelease frames: {total_release_frames} / {total_frames} hand frames "
          f"({100*total_release_frames/total_frames:.1f}%)")
    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
