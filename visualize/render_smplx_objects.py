#!/usr/bin/env python3
"""
Render generated motions as SMPL-X mesh + object meshes.

Pipeline: 267D motion → extract 77 markers → SMPL-X inverse fitting → mesh + objects → video

Usage:
    # Step 1: Inverse fitting (rog_env, needs GPU)
    python visualize/render_smplx_objects.py --mode fit \
        --motion_path output/v4_10k_vis/gen_0000.npy \
        --output_dir output/v4_10k_vis/smplx_0000

    # Step 2: Render (mdm2 env, headless)
    xvfb-run -a -s "-screen 0 1024x768x24" python visualize/render_smplx_objects.py --mode render \
        --verts_path output/v4_10k_vis/smplx_0000/vertices.npy \
        --motion_path output/v4_10k_vis/gen_0000.npy \
        --output_path output/v4_10k_vis/smplx_0000.mp4
"""

import os
import json
import argparse
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


MARKERSET_SMPLX = [
    5920, 5621, 5882, 3486, 4430, 3258, 2029, 5694, 5645, 9117, 4302,
    4319, 707, 4788, 4198, 3998, 8846, 4726, 3682, 3688, 5890, 5901,
    3961, 4722, 5698, 5868, 3892, 3712, 4479, 4088, 3477, 4839, 5787,
    4291, 8576, 6248, 7166, 6021, 3067, 8250, 8388, 8339, 3116, 7040,
    7099, 2198, 7524, 7115, 6265, 6746, 8634, 7462, 6443, 6449, 8584,
    8595, 6709, 7458, 8392, 8527, 7215, 6832, 7575, 6503, 8481, 5945,
    5947, 8079, 7669, 7794, 7905, 8022, 5361, 4933, 5058, 5169, 5286,
]


def rotation_6d_to_matrix(rot6d):
    """Convert 6D rotation to 3x3 matrix."""
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def extract_motion_components(motion, smooth_window=0):
    """Extract markers [T,77,3] and object poses from 267D/534D motion."""
    if motion.shape[1] > 267:
        motion = motion[:, :267]
    if smooth_window > 0:
        motion = savgol_filter(motion, window_length=smooth_window, polyorder=3, axis=0)
    markers = motion[:, :231].reshape(-1, 77, 3)
    objects = motion[:, 231:267].reshape(-1, 4, 9)
    return markers, objects


def fit_markers_to_smplx(motion_path, output_dir, device='cuda:0', smooth_window=11):
    """Run SMPL-X inverse fitting from motion file."""
    from visualize.marker2smplx import fit_smplx_to_markers

    os.makedirs(output_dir, exist_ok=True)
    motion = np.load(motion_path)
    markers, objects = extract_motion_components(motion, smooth_window=smooth_window)

    print(f"Fitting SMPL-X to {markers.shape[0]} frames...")
    vertices, faces = fit_smplx_to_markers(markers, device=device)

    np.save(os.path.join(output_dir, 'vertices.npy'), vertices)
    np.save(os.path.join(output_dir, 'faces.npy'), faces)
    np.save(os.path.join(output_dir, 'objects.npy'), objects)
    print(f"Saved to {output_dir}/")
    return vertices, faces, objects


def render_smplx_video(verts_path, motion_path, output_path, gt_verts_path=None,
                        fps=10, resolution='low', smooth_window=11):
    """Render SMPL-X mesh + objects as video using aitviewer."""
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.meshes import Meshes
    import trimesh

    res_map = {'low': (640, 480), 'medium': (1280, 960), 'high': (1920, 1080)}
    width, height = res_map.get(resolution, (1280, 960))

    # Load fitted vertices
    vertices = np.load(verts_path)  # [T, 10475, 3]
    faces_path = verts_path.replace('vertices.npy', 'faces.npy')
    faces = np.load(faces_path) if os.path.exists(faces_path) else None

    # Load objects
    motion = np.load(motion_path)
    _, objects = extract_motion_components(motion, smooth_window=smooth_window)
    T = vertices.shape[0]

    # Floor offset
    floor_offset = -vertices[:, :, 1].min()
    vertices[:, :, 1] += floor_offset

    # Init renderer
    v = HeadlessRenderer(size=(width, height))
    v.scene.camera.fov = 60
    v.scene.camera.position = np.array([0.0, 1.2, -3.0])
    v.scene.camera.target = np.array([0.0, 0.9, 0.0])
    v.scene.camera.up = np.array([0, 1.0, 0])
    v.scene.floor.position = np.array([0.0, 0.0, 0.0])
    v.auto_set_camera_target = False
    v.auto_set_floor = False

    # Remove defaults
    if len(v.scene.lights) > 0:
        v.scene.remove(v.scene.lights[0])
    v.scene.remove(v.scene.origin)

    # Add body mesh
    if faces is not None:
        body_mesh = Meshes(
            vertices, faces,
            color=(0.5, 0.6, 0.9, 0.9),
            name="Body",
        )
        v.scene.add(body_mesh)

    # Add object meshes (from generated poses)
    obj_mesh_dir = '/hhd4/lizhe/dataset/OakInk2/data/object_repair/align_ds'
    # Try to load object info from gen_info.json or file_names
    gen_dir = os.path.dirname(motion_path)
    info_path = os.path.join(gen_dir, 'gen_info.json')

    # Render frames
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    frame_dir = output_path.replace('.mp4', '_frames')
    os.makedirs(frame_dir, exist_ok=True)

    v._init_scene()
    for frame_idx in range(T):
        v.scene.current_frame_id = frame_idx
        v.render(0, 0, export=True)
        v.export_frame(os.path.join(frame_dir, f"frame_{frame_idx:06d}.png"))
        if (frame_idx + 1) % 50 == 0:
            print(f"  Rendered {frame_idx + 1}/{T} frames")

    # ffmpeg
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
    parser.add_argument("--mode", choices=["fit", "render", "both"], default="both")
    parser.add_argument("--motion_path", required=True, help="Generated motion .npy [T, 267/534]")
    parser.add_argument("--gt_motion_path", default="", help="GT motion for comparison")
    parser.add_argument("--output_dir", default="", help="Dir for fit output (vertices, faces)")
    parser.add_argument("--output_path", default="", help="Output video path")
    parser.add_argument("--verts_path", default="", help="Pre-fitted vertices .npy (for render mode)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smooth_window", type=int, default=11)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--resolution", default="low")
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = args.motion_path.replace('.npy', '_smplx')
    if not args.output_path:
        args.output_path = args.motion_path.replace('.npy', '_smplx.mp4')

    if args.mode in ("fit", "both"):
        fit_markers_to_smplx(args.motion_path, args.output_dir,
                              device=args.device, smooth_window=args.smooth_window)

    if args.mode in ("render", "both"):
        verts_path = args.verts_path or os.path.join(args.output_dir, 'vertices.npy')
        render_smplx_video(verts_path, args.motion_path, args.output_path,
                           fps=args.fps, resolution=args.resolution,
                           smooth_window=args.smooth_window)


if __name__ == "__main__":
    main()
