#!/usr/bin/env python3
"""
Full visualization: SMPL-X mesh + object meshes + text overlay.

Usage:
    # Step 1: Fit (rog_env)
    python -m visualize.render_full --mode fit \
        --motion_path output/v4_10k_vis/gen_0000.npy \
        --output_dir output/v4_10k_vis/full_gen_0 \
        --sample_idx 0 --data_root dataset/OAKINK2_V3

    # Step 2: Render (mdm2 env)
    xvfb-run -a -s "-screen 0 1024x768x24" python -m visualize.render_full --mode render \
        --output_dir output/v4_10k_vis/full_gen_0 \
        --output_path output/v4_10k_vis/full_gen_0.mp4
"""

import os
import json
import argparse
import numpy as np
from scipy.signal import savgol_filter
import trimesh


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


def get_sample_metadata(sample_idx, data_root='dataset/OAKINK2_V3'):
    """Get text, object IDs, and object mesh paths for a test sample."""
    # Read test indices
    with open(os.path.join(data_root, 'test_flow.txt')) as f:
        test_indices = [l.strip() for l in f]

    if sample_idx >= len(test_indices):
        return {"text": "", "obj_ids": [], "obj_meshes": {}}

    idx_str = test_indices[sample_idx]

    # Text
    text = ""
    text_path = os.path.join(data_root, 'texts', f'{idx_str}.txt')
    if os.path.exists(text_path):
        with open(text_path) as f:
            text = f.readline().split('#')[0].strip()

    # Object IDs from file_names → seq_id → program_info
    file_map = {}
    with open(os.path.join(data_root, 'file_names.txt')) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                file_map[parts[0]] = parts[1]

    seg_info = file_map.get(idx_str, '')
    seq_id = seg_info.rsplit('__seg', 1)[0]

    prog_dir = '/hhd4/lizhe/dataset/OakInk2/data/program/program_info'
    obj_mesh_dir = '/hhd4/lizhe/dataset/OakInk2/data/object_repair/align_ds'

    obj_ids = []
    prog_path = os.path.join(prog_dir, seq_id + '.json')
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            prog = json.load(f)
        all_objs = set()
        for k, v in prog.items():
            for field in ['obj_list', 'obj_list_lh', 'obj_list_rh']:
                objs = v.get(field) or []
                if isinstance(objs, list):
                    all_objs.update(objs)
        obj_ids = sorted(all_objs)[:4]

    # Load meshes
    obj_meshes = {}
    for oid in obj_ids:
        mesh_path = os.path.join(obj_mesh_dir, oid, 'model.obj')
        if os.path.exists(mesh_path):
            mesh = trimesh.load(mesh_path, force='mesh')
            obj_meshes[oid] = mesh

    return {"text": text, "obj_ids": obj_ids, "obj_meshes": obj_meshes}


def fit_and_prepare(motion_path, output_dir, sample_idx, data_root, device, smooth_window):
    """Fit SMPL-X + prepare object transforms + save metadata."""
    from visualize.marker2smplx import fit_smplx_to_markers

    os.makedirs(output_dir, exist_ok=True)

    motion = np.load(motion_path)
    if motion.shape[1] > 267:
        motion = motion[:, :267]
    if smooth_window > 0:
        motion = savgol_filter(motion, window_length=smooth_window, polyorder=3, axis=0)

    markers = motion[:, :231].reshape(-1, 77, 3)
    obj_poses = motion[:, 231:267].reshape(-1, 4, 9)  # [T, 4, 9]
    T = markers.shape[0]

    # Fit SMPL-X
    print(f"Fitting SMPL-X ({T} frames)...")
    vertices, faces = fit_smplx_to_markers(markers, device=device)

    # Get metadata
    meta = get_sample_metadata(sample_idx, data_root)

    # Compute object vertex sequences
    obj_vert_sequences = {}
    for slot_idx, oid in enumerate(meta['obj_ids']):
        if oid not in meta['obj_meshes']:
            continue
        mesh = meta['obj_meshes'][oid]
        obj_verts_base = mesh.vertices.astype(np.float32)
        obj_faces = mesh.faces

        # Apply per-frame transforms
        obj_verts_seq = np.zeros((T, len(obj_verts_base), 3), dtype=np.float32)
        for t in range(T):
            pos = obj_poses[t, slot_idx, :3]
            rot6d = obj_poses[t, slot_idx, 3:9]
            R = rotation_6d_to_matrix(rot6d)
            # Apply Z(-90°) rotation fix for OakInk2 coordinate convention
            Rz = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
            R_fixed = R @ Rz
            obj_verts_seq[t] = (obj_verts_base @ R_fixed.T) + pos

        obj_vert_sequences[oid] = {
            'vertices': obj_verts_seq,
            'faces': obj_faces,
        }

    # Save everything
    np.save(os.path.join(output_dir, 'body_vertices.npy'), vertices)
    np.save(os.path.join(output_dir, 'body_faces.npy'), faces)

    for oid, data in obj_vert_sequences.items():
        safe_name = oid.replace('@', '_').replace('/', '_')
        np.save(os.path.join(output_dir, f'obj_{safe_name}_verts.npy'), data['vertices'])
        np.save(os.path.join(output_dir, f'obj_{safe_name}_faces.npy'), data['faces'])

    info = {
        'text': meta['text'],
        'obj_ids': meta['obj_ids'],
        'obj_files': [oid.replace('@', '_').replace('/', '_') for oid in meta['obj_ids']
                       if oid in meta['obj_meshes']],
        'num_frames': T,
        'sample_idx': sample_idx,
    }
    with open(os.path.join(output_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"Saved to {output_dir}/ (body + {len(obj_vert_sequences)} objects)")


def render_video(output_dir, output_path, fps=10, resolution='low'):
    """Render SMPL-X body + objects + text overlay."""
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.meshes import Meshes

    res_map = {'low': (640, 480), 'medium': (1280, 960), 'high': (1920, 1080)}
    width, height = res_map.get(resolution, (1280, 960))

    # Load info
    with open(os.path.join(output_dir, 'info.json')) as f:
        info = json.load(f)

    body_verts = np.load(os.path.join(output_dir, 'body_vertices.npy'))
    body_faces = np.load(os.path.join(output_dir, 'body_faces.npy'))
    T = body_verts.shape[0]

    # Floor offset from body
    floor_offset = -body_verts[:, :, 1].min()
    body_verts[:, :, 1] += floor_offset

    # Init renderer
    v = HeadlessRenderer(size=(width, height))
    v.scene.camera.fov = 60
    v.scene.camera.position = np.array([0.0, 1.2, -3.0])
    v.scene.camera.target = np.array([0.0, 0.9, 0.0])
    v.scene.camera.up = np.array([0, 1.0, 0])
    v.scene.floor.position = np.array([0.0, 0.0, 0.0])
    v.auto_set_camera_target = False
    v.auto_set_floor = False

    if len(v.scene.lights) > 0:
        v.scene.remove(v.scene.lights[0])
    v.scene.remove(v.scene.origin)

    # Add body mesh
    body_mesh = Meshes(body_verts, body_faces,
                       color=(0.7, 0.7, 0.8, 1.0), z_up=False, name="Body")
    v.scene.add(body_mesh)

    # Add object meshes
    obj_colors = [
        (0.9, 0.4, 0.3, 1.0),   # Red
        (0.3, 0.9, 0.4, 1.0),   # Green
        (0.9, 0.8, 0.3, 1.0),   # Yellow
        (0.9, 0.4, 0.9, 1.0),   # Purple
    ]
    for i, safe_name in enumerate(info.get('obj_files', [])):
        verts_path = os.path.join(output_dir, f'obj_{safe_name}_verts.npy')
        faces_path = os.path.join(output_dir, f'obj_{safe_name}_faces.npy')
        if os.path.exists(verts_path):
            obj_v = np.load(verts_path)
            obj_f = np.load(faces_path)
            obj_v[:, :, 1] += floor_offset
            color = obj_colors[i % len(obj_colors)]
            obj_mesh = Meshes(obj_v, obj_f, color=color, z_up=False, name=f"Obj_{i}")
            v.scene.add(obj_mesh)

    # Render frames
    frame_dir = output_path.replace('.mp4', '_frames')
    os.makedirs(frame_dir, exist_ok=True)

    v._init_scene()
    for frame_idx in range(T):
        v.scene.current_frame_id = frame_idx
        v.render(0, 0, export=True)
        v.export_frame(os.path.join(frame_dir, f"frame_{frame_idx:06d}.png"))

    # Add text overlay to frames
    text = info.get('text', '')
    if text:
        try:
            from PIL import Image, ImageDraw, ImageFont
            for frame_idx in range(T):
                fp = os.path.join(frame_dir, f"frame_{frame_idx:06d}.png")
                img = Image.open(fp)
                draw = ImageDraw.Draw(img)
                # White text with black outline
                x, y = 10, 10
                for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    draw.text((x+dx, y+dy), text[:80], fill='black')
                draw.text((x, y), text[:80], fill='white')
                # Frame counter
                draw.text((x, height - 25), f"Frame {frame_idx}/{T} ({frame_idx/10:.1f}s)", fill='white')
                img.save(fp)
        except ImportError:
            print("PIL not available, skipping text overlay")

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
    print(f"Saved: {output_path} ({T} frames, text: '{text[:50]}...')")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fit", "render", "both"], default="both")
    parser.add_argument("--motion_path", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_path", default="")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--data_root", default="dataset/OAKINK2_V3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smooth_window", type=int, default=11)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--resolution", default="low")
    args = parser.parse_args()

    if not args.output_path:
        args.output_path = args.output_dir + '.mp4'

    if args.mode in ("fit", "both"):
        fit_and_prepare(args.motion_path, args.output_dir, args.sample_idx,
                        args.data_root, args.device, args.smooth_window)

    if args.mode in ("render", "both"):
        render_video(args.output_dir, args.output_path,
                     fps=args.fps, resolution=args.resolution)


if __name__ == "__main__":
    main()
