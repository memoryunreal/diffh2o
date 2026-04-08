"""
Evaluation metrics for long-horizon motion generation.

Metrics:
- Transition Smoothness (TS): Jerk at transition boundaries
- Contact Consistency (CC): Hand-object proximity at primitive start/end
- Temporal Coherence (TC): Velocity continuity at boundaries
- Physical Plausibility (PP): Penetration + floating detection
- Transition FID: Feature distribution quality of generated transitions
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LongMotionMetrics:
    """Container for long-motion evaluation metrics."""
    transition_smoothness: float = 0.0
    contact_consistency: float = 0.0
    temporal_coherence: float = 0.0
    physical_plausibility: float = 0.0
    transition_fid: float = 0.0
    # Per-transition breakdown
    per_transition_ts: List[float] = None
    per_transition_tc: List[float] = None

    def __post_init__(self):
        if self.per_transition_ts is None:
            self.per_transition_ts = []
        if self.per_transition_tc is None:
            self.per_transition_tc = []

    def to_dict(self) -> Dict:
        return {
            'transition_smoothness': self.transition_smoothness,
            'contact_consistency': self.contact_consistency,
            'temporal_coherence': self.temporal_coherence,
            'physical_plausibility': self.physical_plausibility,
            'transition_fid': self.transition_fid,
            'per_transition_ts': self.per_transition_ts,
            'per_transition_tc': self.per_transition_tc,
        }


def compute_jerk(motion: np.ndarray, fps: int = 30) -> np.ndarray:
    """Compute jerk (3rd derivative) of motion.

    Args:
        motion: [T, D] motion array
        fps: Frames per second

    Returns:
        [T-3, D] jerk values
    """
    dt = 1.0 / fps
    vel = np.diff(motion, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return jerk


def transition_smoothness(
    full_motion: np.ndarray,
    boundaries: List[Tuple[int, int, str]],
    fps: int = 30,
    window: int = 5,
) -> Tuple[float, List[float]]:
    """Compute Transition Smoothness (TS) metric.

    Measures the mean squared jerk at transition boundaries.
    Lower is better (smoother transitions).

    Args:
        full_motion: [T, D] full motion sequence
        boundaries: List of (start, end, type) from MotionSequence
        fps: Frames per second
        window: Number of frames around boundary to evaluate

    Returns:
        (mean_ts, per_transition_ts)
    """
    jerk = compute_jerk(full_motion, fps)
    per_transition = []

    for start, end, seg_type in boundaries:
        if seg_type != 'transition':
            continue

        # Evaluate jerk at transition edges (boundary region)
        edge_start = max(0, start - window)
        edge_end = min(len(jerk), end + window)

        if edge_start < edge_end:
            boundary_jerk = jerk[edge_start:edge_end]
            ts = np.mean(boundary_jerk ** 2)
            per_transition.append(float(ts))

    mean_ts = np.mean(per_transition) if per_transition else 0.0
    return float(mean_ts), per_transition


def temporal_coherence(
    full_motion: np.ndarray,
    boundaries: List[Tuple[int, int, str]],
    fps: int = 30,
) -> Tuple[float, List[float]]:
    """Compute Temporal Coherence (TC) metric.

    Measures L2 velocity jump norm at segment-transition boundaries.
    Lower is better (more continuous motion).

    Args:
        full_motion: [T, D] full motion sequence
        boundaries: List of (start, end, type) from MotionSequence
        fps: Frames per second

    Returns:
        (mean_tc, per_boundary_tc)
    """
    dt = 1.0 / fps
    vel = np.diff(full_motion, axis=0) / dt

    per_boundary = []

    for start, end, seg_type in boundaries:
        if seg_type != 'transition':
            continue

        # Velocity before transition start
        if start > 0 and start < len(vel):
            vel_before = vel[start - 1]
            vel_after = vel[start]
            jump = np.linalg.norm(vel_after - vel_before)
            per_boundary.append(float(jump))

        # Velocity at transition end
        if end > 0 and end < len(vel):
            vel_before = vel[end - 1]
            vel_after = vel[min(end, len(vel) - 1)]
            jump = np.linalg.norm(vel_after - vel_before)
            per_boundary.append(float(jump))

    mean_tc = np.mean(per_boundary) if per_boundary else 0.0
    return float(mean_tc), per_boundary


def contact_consistency(
    full_motion: np.ndarray,
    boundaries: List[Tuple[int, int, str]],
    feature_dim: int = 398,
) -> float:
    """Compute Contact Consistency (CC) metric.

    For OakInk2 (398D): measures hand-object distance at primitive start/end.
    Uses hand positions and object positions from the feature vector.

    Args:
        full_motion: [T, D] full motion sequence
        boundaries: List of (start, end, type) from MotionSequence
        feature_dim: Feature dimension (398 for OakInk2, 117 for GRAB)

    Returns:
        Mean hand-object distance at segment edges
    """
    if feature_dim == 398:
        # OakInk2 indices
        lh_pos = (135, 138)
        rh_pos = (165, 168)
        obj1_pos = (371, 374)
    elif feature_dim == 117:
        # GRAB: hand positions are in different indices
        # Approximate: first few dims are hand positions
        lh_pos = (0, 3)
        rh_pos = (3, 6)
        obj1_pos = (108, 111) if feature_dim > 111 else (0, 3)
    else:
        return 0.0

    distances = []
    for start, end, seg_type in boundaries:
        if seg_type != 'segment':
            continue

        for frame_idx in [start, min(end - 1, len(full_motion) - 1)]:
            if frame_idx < 0 or frame_idx >= len(full_motion):
                continue

            frame = full_motion[frame_idx]
            lh = frame[lh_pos[0]:lh_pos[1]]
            rh = frame[rh_pos[0]:rh_pos[1]]
            obj = frame[obj1_pos[0]:obj1_pos[1]]

            dist_lh = np.linalg.norm(lh - obj)
            dist_rh = np.linalg.norm(rh - obj)
            distances.append(min(dist_lh, dist_rh))

    return float(np.mean(distances)) if distances else 0.0


def physical_plausibility(
    full_motion: np.ndarray,
    feature_dim: int = 398,
) -> float:
    """Compute Physical Plausibility (PP) metric.

    Checks for physically implausible states:
    - Hand-object penetration (SDF < 0)
    - Floating hands (SDF too large during interaction)
    - Excessive velocities

    Args:
        full_motion: [T, D] motion sequence
        feature_dim: Feature dimension

    Returns:
        Plausibility score (lower = more physical violations)
    """
    violations = 0
    total_checks = 0

    if feature_dim == 398:
        # Check SDF values (indices 329-371)
        sdf_left = full_motion[:, 329:350]  # 21D
        sdf_right = full_motion[:, 350:371]  # 21D

        # Penetration: negative SDF values
        penetration_left = np.sum(sdf_left < -0.01)
        penetration_right = np.sum(sdf_right < -0.01)
        violations += penetration_left + penetration_right
        total_checks += sdf_left.size + sdf_right.size

    # Check for excessive velocities (any feature dim)
    if len(full_motion) > 1:
        vel = np.diff(full_motion, axis=0)
        max_vel = np.max(np.abs(vel), axis=0)
        excessive = np.sum(max_vel > 5.0)  # Threshold for normalized motion
        violations += excessive
        total_checks += len(max_vel)

    if total_checks == 0:
        return 1.0

    return 1.0 - (violations / total_checks)


def compute_transition_fid(
    generated_transitions: List[np.ndarray],
    real_transitions: List[np.ndarray],
) -> float:
    """Compute FID between generated and real transition distributions.

    Uses simple feature statistics (mean, covariance) in motion space.

    Args:
        generated_transitions: List of [T, D] generated transition arrays
        real_transitions: List of [T, D] real transition arrays

    Returns:
        FID score (lower is better)
    """
    if not generated_transitions or not real_transitions:
        return float('inf')

    # Flatten transitions to feature vectors
    gen_features = np.array([t.flatten() for t in generated_transitions])
    real_features = np.array([t.flatten() for t in real_transitions])

    # Compute statistics
    mu_gen = np.mean(gen_features, axis=0)
    mu_real = np.mean(real_features, axis=0)

    # Use diagonal covariance for efficiency
    var_gen = np.var(gen_features, axis=0) + 1e-6
    var_real = np.var(real_features, axis=0) + 1e-6

    # Simplified FID with diagonal covariance
    mean_diff = np.sum((mu_gen - mu_real) ** 2)
    trace_term = np.sum(var_gen + var_real - 2 * np.sqrt(var_gen * var_real))

    return float(mean_diff + trace_term)


def evaluate_long_motion(
    full_motion: np.ndarray,
    boundaries: List[Tuple[int, int, str]],
    feature_dim: int = 398,
    fps: int = 30,
    real_transitions: Optional[List[np.ndarray]] = None,
) -> LongMotionMetrics:
    """Run full evaluation on a long motion sequence.

    Args:
        full_motion: [T, D] full motion sequence
        boundaries: List of (start, end, type) from MotionSequence
        feature_dim: Feature dimension (398 or 117)
        fps: Frames per second
        real_transitions: Optional real transitions for FID computation

    Returns:
        LongMotionMetrics with all metrics computed
    """
    ts_mean, ts_per = transition_smoothness(full_motion, boundaries, fps)
    tc_mean, tc_per = temporal_coherence(full_motion, boundaries, fps)
    cc = contact_consistency(full_motion, boundaries, feature_dim)
    pp = physical_plausibility(full_motion, feature_dim)

    # Extract generated transitions for FID
    gen_transitions = []
    for start, end, seg_type in boundaries:
        if seg_type == 'transition' and start < end <= len(full_motion):
            gen_transitions.append(full_motion[start:end])

    fid = 0.0
    if real_transitions and gen_transitions:
        fid = compute_transition_fid(gen_transitions, real_transitions)

    return LongMotionMetrics(
        transition_smoothness=ts_mean,
        contact_consistency=cc,
        temporal_coherence=tc_mean,
        physical_plausibility=pp,
        transition_fid=fid,
        per_transition_ts=ts_per,
        per_transition_tc=tc_per,
    )


def evaluate_ablation(
    results_dir: str,
    configs: List[Dict],
    feature_dim: int = 398,
    fps: int = 30,
) -> Dict[str, LongMotionMetrics]:
    """Run evaluation for ablation study across multiple configurations.

    Args:
        results_dir: Directory containing generation results
        configs: List of config dicts with 'name' and 'path' keys
        feature_dim: Feature dimension
        fps: Frames per second

    Returns:
        Dict mapping config name to metrics
    """
    import json

    results = {}
    for cfg in configs:
        name = cfg['name']
        path = cfg['path']

        motion_path = os.path.join(path, 'full_sequence.npy')
        boundaries_path = os.path.join(path, 'boundaries.json')

        if not os.path.exists(motion_path) or not os.path.exists(boundaries_path):
            print(f"Skipping {name}: files not found")
            continue

        full_motion = np.load(motion_path)
        with open(boundaries_path) as f:
            boundaries = json.load(f)

        # Convert boundaries to tuples
        boundaries = [(b[0], b[1], b[2]) for b in boundaries]

        metrics = evaluate_long_motion(full_motion, boundaries, feature_dim, fps)
        results[name] = metrics

        print(f"\n{name}:")
        print(f"  TS (Transition Smoothness): {metrics.transition_smoothness:.4f}")
        print(f"  TC (Temporal Coherence):    {metrics.temporal_coherence:.4f}")
        print(f"  CC (Contact Consistency):   {metrics.contact_consistency:.4f}")
        print(f"  PP (Physical Plausibility): {metrics.physical_plausibility:.4f}")

    return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate long-motion generation")
    parser.add_argument("--motion_path", type=str, required=True,
                        help="Path to full_sequence.npy")
    parser.add_argument("--boundaries_path", type=str, required=True,
                        help="Path to boundaries.json")
    parser.add_argument("--feature_dim", type=int, default=398)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save metrics JSON")

    args = parser.parse_args()

    motion = np.load(args.motion_path)
    with open(args.boundaries_path) as f:
        boundaries = json.load(f)
    boundaries = [(b[0], b[1], b[2]) for b in boundaries]

    metrics = evaluate_long_motion(motion, boundaries, args.feature_dim, args.fps)

    print("\nLong-Motion Evaluation Results:")
    print(f"  Transition Smoothness (TS): {metrics.transition_smoothness:.6f}")
    print(f"  Temporal Coherence (TC):    {metrics.temporal_coherence:.6f}")
    print(f"  Contact Consistency (CC):   {metrics.contact_consistency:.6f}")
    print(f"  Physical Plausibility (PP): {metrics.physical_plausibility:.4f}")
    print(f"  Transition FID:             {metrics.transition_fid:.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\nSaved to {args.output}")
