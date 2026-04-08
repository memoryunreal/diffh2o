#!/usr/bin/env python3
"""
Render PriorMDM 22-joint skeleton + GT object meshes using aitviewer.

Usage (mdm2 env):
    xvfb-run -a -s "-screen 0 1024x768x24" python -m visualize.render_priormdm_with_objects \
        --joints_path output/priormdm_doubletake/rearrange_plate_joints.npy \
        --obj_dir output/priormdm_doubletake/rearrange_plate_gt_objects \
        --output_path output/priormdm_doubletake/rearrange_plate_with_gt_objects.mp4
"""

import os
import argparse
import numpy as np

T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

BONES = []
for chain in T2M_KINEMATIC_CHAIN:
    for i in range(len(chain) - 1):
        BONES.append((chain[i], chain[i + 1]))


def render(joints_path, obj_dir, output_path, fps=20, resolution='low'):
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.spheres import Spheres
    from aitviewer.renderables.meshes import Meshes

    res_map = {'low': (640, 480), 'medium': (1280, 960)}
    width, height = res_map.get(resolution, (640, 480))

    joints = np.load(joints_path)  # [T_prior, 22, 3]
    T_prior = joints.shape[0]

    # Load GT objects (may have different frame count — subsample/pad to match)
    obj_data = []
    if os.path.isdir(obj_dir):
        for fn in sorted(os.listdir(obj_dir)):
            if fn.startswith('obj_') and fn.endswith('_verts.npy'):
                verts = np.load(os.path.join(obj_dir, fn))  # [T_gt, V, 3]
                faces_fn = fn.replace('_verts.npy', '_faces.npy')
                faces = np.load(os.path.join(obj_dir, faces_fn))
                obj_data.append((verts, faces))

    # Align object frames to PriorMDM frame count
    # PriorMDM uses 20fps, OakInk2 GT uses 10fps — subsample GT by 2x
    for i, (verts, faces) in enumerate(obj_data):
        T_gt = verts.shape[0]
        if T_gt != T_prior:
            indices = np.linspace(0, T_gt - 1, T_prior).astype(int)
            obj_data[i] = (verts[indices], faces)

    # Floor offset from joints
    floor_offset = -joints[:, :, 1].min()
    joints[:, :, 1] += floor_offset

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

    # Joint spheres
    s = Spheres(joints, radius=0.03, color=(0.3, 0.5, 0.9, 1.0), name="Joints")
    v.scene.add(s)

    # Object meshes
    obj_colors = [(0.9, 0.4, 0.3, 0.9), (0.3, 0.9, 0.4, 0.9),
                  (0.9, 0.8, 0.3, 0.9), (0.9, 0.4, 0.9, 0.9)]
    for i, (obj_v, obj_f) in enumerate(obj_data):
        obj_v_copy = obj_v.copy()
        obj_v_copy[:, :, 1] += floor_offset
        m = Meshes(obj_v_copy, obj_f, color=obj_colors[i % 4], name=f"GT_Obj_{i}")
        v.scene.add(m)

    # Render
    frame_dir = output_path.replace('.mp4', '_frames')
    os.makedirs(frame_dir, exist_ok=True)

    v._init_scene()
    for fi in range(T_prior):
        v.scene.current_frame_id = fi
        v.render(0, 0, export=True)
        v.export_frame(os.path.join(frame_dir, f"frame_{fi:06d}.png"))

    # Text overlay
    try:
        from PIL import Image, ImageDraw
        for fi in range(T_prior):
            fp = os.path.join(frame_dir, f"frame_{fi:06d}.png")
            img = Image.open(fp)
            draw = ImageDraw.Draw(img)
            for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                draw.text((10+dx, 10+dy), "PriorMDM DoubleTake + GT Objects", fill='black')
            draw.text((10, 10), "PriorMDM DoubleTake + GT Objects", fill='white')
            draw.text((10, height - 25), f"Frame {fi}/{T_prior}", fill='white')
            img.save(fp)
    except ImportError:
        pass

    import subprocess
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i",
           os.path.join(frame_dir, "frame_%06d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output_path]
    subprocess.run(cmd, capture_output=True)
    print(f"Saved: {output_path} ({T_prior} frames, {len(obj_data)} GT objects)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joints_path", required=True)
    parser.add_argument("--obj_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--resolution", default="low")
    args = parser.parse_args()
    render(args.joints_path, args.obj_dir, args.output_path, args.fps, args.resolution)


if __name__ == "__main__":
    main()
