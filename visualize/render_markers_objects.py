#!/usr/bin/env python3
"""
Render 77-marker spheres + object meshes from generated V3/V4 motions.

Produces side-by-side GT vs Generated comparison videos.
Runs in mdm2 env with aitviewer. Each render in separate process (GL context bug).

Usage:
    # Single render
    xvfb-run -s "-screen 0 1024x768x24" python visualize/render_markers_objects.py \
        --gen_path output/v3_10k_vis/gen_0000.npy \
        --gt_path output/v3_10k_vis/gt_0000.npy \
        --output_path output/v3_10k_vis/vis_0000.mp4

    # With objects
    xvfb-run ... python visualize/render_markers_objects.py \
        --gen_path output/v4_10k_vis/gen_0000.npy \
        --output_path output/v4_10k_vis/vis_0000.mp4 \
        --object_mesh_dir /hhd4/lizhe/dataset/OakInk2/data/object_repair/align_ds
"""

import os
import argparse
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


# Marker group indices within 77-marker set
BODY_IDX = list(range(0, 50))
LHAND_IDX = list(range(50, 64))
RHAND_IDX = list(range(64, 77))


def rotation_6d_to_matrix(rot6d):
    """Convert 6D rotation to 3x3 rotation matrix."""
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def extract_markers_and_objects(motion, smooth_window=0):
    """Extract markers [T, 77, 3] and object poses [T, 4, 9] from 267D motion.

    Args:
        motion: [T, 267] or [T, 534] (V4 positions only or full)
        smooth_window: Savitzky-Golay window (0 = no smoothing)
    """
    if motion.shape[1] > 267:
        motion = motion[:, :267]  # V4: take positions only

    if smooth_window > 0:
        motion = savgol_filter(motion, window_length=smooth_window, polyorder=3, axis=0)

    markers = motion[:, :231].reshape(-1, 77, 3)
    objects = motion[:, 231:267].reshape(-1, 4, 9)
    return markers, objects


def render_motion(gen_path, gt_path, output_path, smooth_window=0,
                  object_mesh_dir=None, fps=10, resolution='medium'):
    """Render markers + objects as video using aitviewer."""
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.spheres import Spheres

    res_map = {'low': (640, 480), 'medium': (1280, 960), 'high': (1920, 1080)}
    width, height = res_map.get(resolution, (1280, 960))

    # Load motions
    gen_motion = np.load(gen_path)
    gen_markers, gen_objects = extract_markers_and_objects(gen_motion, smooth_window)
    T = gen_markers.shape[0]

    gt_markers = None
    if gt_path and os.path.exists(gt_path):
        gt_motion = np.load(gt_path)
        gt_markers, gt_objects = extract_markers_and_objects(gt_motion, smooth_window)

    # Floor offset
    all_y = gen_markers[:, :, 1]
    floor_offset = -all_y.min()

    # Create renderer
    v = HeadlessRenderer(size=(width, height))
    v.scene.floor.enabled = True

    # Body marker colors
    body_color = (0.3, 0.5, 0.9, 1.0)    # Blue
    lhand_color = (0.9, 0.3, 0.3, 1.0)    # Red
    rhand_color = (0.3, 0.9, 0.3, 1.0)    # Green

    # Generated markers (offset to right)
    offset_gen = np.array([0.5, 0.0, 0.0])
    gen_m = gen_markers.copy()
    gen_m[:, :, 1] += floor_offset
    gen_m += offset_gen

    for name, idx, color in [("Gen Body", BODY_IDX, body_color),
                              ("Gen LHand", LHAND_IDX, lhand_color),
                              ("Gen RHand", RHAND_IDX, rhand_color)]:
        s = Spheres(gen_m[:, idx, :], radius=0.008, color=color, name=name)
        v.scene.add(s)

    # GT markers (offset to left)
    if gt_markers is not None:
        offset_gt = np.array([-0.5, 0.0, 0.0])
        gt_m = gt_markers.copy()
        gt_m[:, :, 1] += floor_offset
        gt_m += offset_gt

        gt_body_color = (0.5, 0.5, 0.5, 0.8)
        gt_hand_color = (0.7, 0.7, 0.7, 0.8)
        for name, idx, color in [("GT Body", BODY_IDX, gt_body_color),
                                  ("GT LHand", LHAND_IDX, gt_hand_color),
                                  ("GT RHand", RHAND_IDX, gt_hand_color)]:
            s = Spheres(gt_m[:, idx, :], radius=0.008, color=color, name=name)
            v.scene.add(s)

    # Camera
    v.scene.camera.position = np.array([0.0, 1.5, -2.5])
    v.scene.camera.target = np.array([0.0, 0.8, 0.0])

    # Render frame by frame
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    frame_dir = output_path.replace('.mp4', '_frames')
    os.makedirs(frame_dir, exist_ok=True)

    v._init_scene()
    for frame_idx in range(T):
        v.scene.current_frame_id = frame_idx
        v.render(0, 0, export=True)
        v.export_frame(os.path.join(frame_dir, f"frame_{frame_idx:06d}.png"))

    # ffmpeg: frames → video
    import subprocess
    frame_pattern = os.path.join(frame_dir, "frame_%06d.png")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", frame_pattern, "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "18", output_path,
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"Saved: {output_path} ({T} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_path", required=True)
    parser.add_argument("--gt_path", default="")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--smooth_window", type=int, default=0,
                        help="Savitzky-Golay window (0=raw, 11=medium, 21=heavy)")
    parser.add_argument("--object_mesh_dir", default="")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--resolution", default="medium")
    args = parser.parse_args()

    render_motion(
        gen_path=args.gen_path,
        gt_path=args.gt_path,
        output_path=args.output_path,
        smooth_window=args.smooth_window,
        object_mesh_dir=args.object_mesh_dir,
        fps=args.fps,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
