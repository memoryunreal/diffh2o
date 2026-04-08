#!/usr/bin/env python3
"""
Render PriorMDM DoubleTake output (22 joints) as skeleton video using aitviewer.

Usage (mdm2 env):
    xvfb-run -a -s "-screen 0 1024x768x24" python -m visualize.render_priormdm \
        --joints_path output/priormdm_doubletake/rearrange_plate_joints.npy \
        --output_path output/priormdm_doubletake/rearrange_plate_aitviewer.mp4
"""

import os
import argparse
import numpy as np


# HumanML3D 22-joint kinematic chain (parent-child bones)
T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],       # right leg
    [0, 1, 4, 7, 10],       # left leg
    [0, 3, 6, 9, 12, 15],   # spine + head
    [9, 14, 17, 19, 21],    # right arm
    [9, 13, 16, 18, 20],    # left arm
]

# Bone connections as pairs
BONES = []
for chain in T2M_KINEMATIC_CHAIN:
    for i in range(len(chain) - 1):
        BONES.append((chain[i], chain[i + 1]))


def render_skeleton(joints_path, output_path, fps=20, resolution='medium'):
    """Render 22-joint skeleton as video."""
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.spheres import Spheres
    from aitviewer.renderables.lines import Lines

    res_map = {'low': (640, 480), 'medium': (1280, 960), 'high': (1920, 1080)}
    width, height = res_map.get(resolution, (1280, 960))

    joints = np.load(joints_path)  # [T, 22, 3]
    T_frames = joints.shape[0]

    # Floor offset
    floor_offset = -joints[:, :, 1].min()
    joints[:, :, 1] += floor_offset

    # Init renderer
    v = HeadlessRenderer(size=(width, height))
    v.scene.camera.fov = 60
    v.scene.camera.position = np.array([0.0, 1.5, -4.0])
    v.scene.camera.target = np.array([0.0, 1.0, 0.0])
    v.scene.camera.up = np.array([0, 1.0, 0])
    v.scene.floor.position = np.array([0.0, 0.0, 0.0])
    v.auto_set_camera_target = False
    v.auto_set_floor = False

    if len(v.scene.lights) > 0:
        v.scene.remove(v.scene.lights[0])
    v.scene.remove(v.scene.origin)

    # Joint spheres (color by body part)
    body_color = (0.3, 0.5, 0.9, 1.0)
    s = Spheres(joints, radius=0.03, color=body_color, name="Joints")
    v.scene.add(s)

    # Bone lines
    for parent, child in BONES:
        bone_positions = np.stack([joints[:, parent], joints[:, child]], axis=1)  # [T, 2, 3]
        line = Lines(bone_positions, r_base=0.015, color=(0.8, 0.8, 0.8, 1.0), name=f"Bone_{parent}_{child}")
        v.scene.add(line)

    # Render frames
    frame_dir = output_path.replace('.mp4', '_frames')
    os.makedirs(frame_dir, exist_ok=True)

    v._init_scene()
    for frame_idx in range(T_frames):
        v.scene.current_frame_id = frame_idx
        v.render(0, 0, export=True)
        v.export_frame(os.path.join(frame_dir, f"frame_{frame_idx:06d}.png"))

    # Add text overlay
    try:
        from PIL import Image, ImageDraw
        for frame_idx in range(T_frames):
            fp = os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
            img = Image.open(fp)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "PriorMDM DoubleTake (22 joints, body only)", fill='white')
            draw.text((10, height - 25), f"Frame {frame_idx}/{T_frames}", fill='white')
            img.save(fp)
    except ImportError:
        pass

    # ffmpeg
    import subprocess
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    frame_pattern = os.path.join(frame_dir, "frame_%06d.png")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", frame_pattern, "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "18", output_path,
    ]
    subprocess.run(cmd, capture_output=True)
    print(f"Saved: {output_path} ({T_frames} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joints_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--resolution", default="medium")
    args = parser.parse_args()
    render_skeleton(args.joints_path, args.output_path, args.fps, args.resolution)


if __name__ == "__main__":
    main()
