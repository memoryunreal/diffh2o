#!/usr/bin/env python3
"""
Render raw SMPL-X mesh (from OakInk2 annotations) + GT object meshes.

This renders the original SMPL-X mesh directly from annotation parameters,
NOT from marker fitting. The mesh has no missing faces since it uses the
full SMPL-X forward pass.

Usage:
    # Step 1: Extract mesh data (rog_env, needs GPU + smplx)
    python -m visualize.render_raw_smplx_gt_objects --mode fit \
        --seq_id scene_02__A001++seq__ac736a20fe59cf0a1df3__2023-04-15-19-49-11 \
        --seg_indices 0 1 2 \
        --output_dir output/ar_chain_lightbulb/raw_smplx_gt_objects \
        --fps_out 10

    # Step 2: Render video (mdm2 env, headless)
    xvfb-run -a -s "-screen 0 1280x960x24" python -m visualize.render_raw_smplx_gt_objects \
        --mode render \
        --output_dir output/ar_chain_lightbulb/raw_smplx_gt_objects \
        --output_path output/ar_chain_lightbulb/raw_smplx_gt_objects.mp4 \
        --resolution medium
"""

import os
import json
import pickle
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

ANNO_DIR = "/hhd4/lizhe/dataset/OakInk2/data/anno_preview"
PROG_DIR = "/hhd4/lizhe/dataset/OakInk2/data/program/program_info"
OBJ_MESH_DIR = "/hhd4/lizhe/dataset/OakInk2/data/object_repair/align_ds"
SMPLX_MODEL_PATH = "/hhd4/lizhe/code/InterAct/models"
MAX_OBJECTS = 4


def parse_interval(key_str: str):
    """Parse program_info key string to (lh_interval, rh_interval)."""
    import ast
    try:
        parsed = ast.literal_eval(key_str)
        if isinstance(parsed, tuple) and len(parsed) == 2:
            lh, rh = parsed
            lh_interval = tuple(lh) if lh is not None else None
            rh_interval = tuple(rh) if rh is not None else None
            return lh_interval, rh_interval
    except Exception:
        pass
    return None, None


def extract_super_segments(program_info, total_frames, gap_threshold=300):
    """Merge adjacent primitives into super-segments."""
    all_intervals = []
    for key, prim_data in program_info.items():
        lh_interval, rh_interval = parse_interval(key)
        starts, ends = [], []
        if lh_interval:
            starts.append(lh_interval[0])
            ends.append(lh_interval[1])
        if rh_interval:
            starts.append(rh_interval[0])
            ends.append(rh_interval[1])
        if not starts:
            continue
        all_intervals.append({
            'start': min(starts), 'end': max(ends),
            'data': prim_data, 'key': key,
        })

    if not all_intervals:
        return []

    all_intervals.sort(key=lambda x: x['start'])

    segments = []
    cur = {'start': all_intervals[0]['start'], 'end': all_intervals[0]['end'],
           'primitives': [all_intervals[0]]}

    for prim in all_intervals[1:]:
        if prim['start'] - cur['end'] <= gap_threshold:
            cur['end'] = max(cur['end'], prim['end'])
            cur['primitives'].append(prim)
        else:
            segments.append(cur)
            cur = {'start': prim['start'], 'end': prim['end'], 'primitives': [prim]}
    segments.append(cur)

    for seg in segments:
        seg['start'] = max(0, seg['start'])
        seg['end'] = min(total_frames - 1, seg['end'])

    return segments


def quat_wxyz_to_aa(quats):
    """Convert [N, 4] wxyz quaternions to [N, 3] axis-angle."""
    # scipy expects xyzw
    xyzw = np.concatenate([quats[..., 1:], quats[..., :1]], axis=-1)
    return R.from_quat(xyzw.reshape(-1, 4)).as_rotvec().reshape(quats.shape[:-1] + (3,))


def rotation_6d_to_matrix(rot6d):
    """Convert 6D rotation to 3x3 matrix (Gram-Schmidt)."""
    a1 = rot6d[:3]
    a2 = rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def fit_raw_smplx(seq_id, seg_indices, output_dir,
                  fps_out=10, device='cuda:0'):
    """Extract raw SMPL-X vertices + GT object meshes from OakInk2 annotations."""
    import torch
    import smplx as smplx_lib
    import trimesh

    os.makedirs(output_dir, exist_ok=True)

    # Load annotation
    anno_path = os.path.join(ANNO_DIR, f"{seq_id}.pkl")
    print(f"Loading annotation: {anno_path}")
    with open(anno_path, 'rb') as f:
        anno = pickle.load(f)

    mocap = anno['mocap_frame_id_list']
    total_frames = len(mocap)

    # Load program info to get segment boundaries
    prog_path = os.path.join(PROG_DIR, f"{seq_id}.json")
    with open(prog_path) as f:
        program_info = json.load(f)

    segments = extract_super_segments(program_info, total_frames)
    print(f"Found {len(segments)} super-segments, using indices: {seg_indices}")

    # Collect all frame IDs from requested segments
    all_frame_ids = []
    seg_boundaries = []
    for si in seg_indices:
        if si >= len(segments):
            print(f"  Warning: segment {si} out of range ({len(segments)} segments)")
            continue
        seg = segments[si]
        seg_frames = mocap[seg['start']:seg['end'] + 1]
        seg_boundaries.append((len(all_frame_ids), len(all_frame_ids) + len(seg_frames)))
        all_frame_ids.extend(seg_frames)

    T_full = len(all_frame_ids)
    print(f"Total frames at 30fps: {T_full} ({T_full / 30:.1f}s)")

    # Downsample 30fps -> fps_out
    step = 30 // fps_out
    ds_indices = list(range(0, T_full, step))
    ds_frame_ids = [all_frame_ids[i] for i in ds_indices]
    T_ds = len(ds_frame_ids)
    print(f"After downsampling to {fps_out}fps: {T_ds} frames ({T_ds / fps_out:.1f}s)")

    # Extract SMPL-X parameters for downsampled frames
    print("Extracting SMPL-X parameters...")
    transl_list, orient_list, body_list, lhand_list, rhand_list, shape_list = [], [], [], [], [], []
    for fid in ds_frame_ids:
        rs = anno['raw_smplx'][fid]
        transl_list.append(rs['world_tsl'].squeeze().numpy())
        orient_list.append(quat_wxyz_to_aa(rs['world_rot'].squeeze().numpy()[None])[0])
        body_list.append(quat_wxyz_to_aa(rs['body_pose'].squeeze().numpy()).reshape(-1))
        lhand_list.append(quat_wxyz_to_aa(rs['left_hand_pose'].squeeze().numpy()).reshape(-1))
        rhand_list.append(quat_wxyz_to_aa(rs['right_hand_pose'].squeeze().numpy()).reshape(-1))
        shape_list.append(rs['body_shape'].squeeze().numpy())

    transl = np.array(transl_list, dtype=np.float32)
    orient = np.array(orient_list, dtype=np.float32)
    body_pose = np.array(body_list, dtype=np.float32)
    lhand = np.array(lhand_list, dtype=np.float32)
    rhand = np.array(rhand_list, dtype=np.float32)
    shape = np.array(shape_list, dtype=np.float32)

    # SMPL-X forward pass in chunks
    if 'cuda' in device and torch.cuda.is_available():
        dev = torch.device(device)
    else:
        dev = torch.device('cpu')
    print(f"Running SMPL-X forward pass on {dev}...")
    CHUNK = 200

    all_verts = []
    faces = None
    for cs in range(0, T_ds, CHUNK):
        ce = min(cs + CHUNK, T_ds)
        bs = ce - cs
        model = smplx_lib.create(
            model_path=SMPLX_MODEL_PATH, model_type='smplx', gender='neutral',
            use_pca=False, flat_hand_mean=True, batch_size=bs, num_betas=300,
        ).to(dev)

        with torch.no_grad():
            output = model(
                betas=torch.from_numpy(shape[cs:ce]).to(dev),
                transl=torch.from_numpy(transl[cs:ce]).to(dev),
                global_orient=torch.from_numpy(orient[cs:ce]).to(dev),
                body_pose=torch.from_numpy(body_pose[cs:ce]).to(dev),
                left_hand_pose=torch.from_numpy(lhand[cs:ce]).to(dev),
                right_hand_pose=torch.from_numpy(rhand[cs:ce]).to(dev),
            )
            all_verts.append(output.vertices.cpu().numpy())
            if faces is None:
                faces = model.faces.copy()
        print(f"  Chunk {cs}-{ce} done")

    body_vertices = np.concatenate(all_verts, axis=0)  # [T_ds, 10475, 3]
    body_faces = faces.astype(np.int32)
    print(f"Body vertices: {body_vertices.shape}, faces: {body_faces.shape}")

    # Save body mesh
    np.save(os.path.join(output_dir, 'body_vertices.npy'), body_vertices)
    np.save(os.path.join(output_dir, 'body_faces.npy'), body_faces)

    # Extract GT object meshes + transforms
    obj_transf = anno.get('obj_transf', {})
    obj_list = sorted(list(obj_transf.keys()))[:MAX_OBJECTS]
    print(f"Objects: {obj_list}")

    # Z(-90) rotation fix for OakInk2 coordinate convention
    Rz_neg90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)

    obj_files = []
    for obj_id in obj_list:
        safe_name = obj_id.replace('@', '_')
        obj_files.append(safe_name)

        # Load base mesh
        mesh_path = os.path.join(OBJ_MESH_DIR, obj_id, 'model.obj')
        if not os.path.exists(mesh_path):
            print(f"  Warning: mesh not found: {mesh_path}")
            continue

        mesh = trimesh.load(mesh_path, force='mesh')
        base_verts = mesh.vertices.astype(np.float32)
        obj_faces = mesh.faces.astype(np.int32)

        # Apply per-frame transforms
        td = obj_transf[obj_id]
        obj_verts = np.zeros((T_ds, len(base_verts), 3), dtype=np.float32)

        for t_idx, fid in enumerate(ds_frame_ids):
            if fid in td:
                tf = td[fid]  # [4, 4] SE(3)
                rot = tf[:3, :3].astype(np.float32)
                pos = tf[:3, 3].astype(np.float32)
                # Apply Rz(-90) fix then transform
                obj_verts[t_idx] = (base_verts @ Rz_neg90.T @ rot.T) + pos
            elif t_idx > 0:
                obj_verts[t_idx] = obj_verts[t_idx - 1]  # Hold last frame

        np.save(os.path.join(output_dir, f'obj_{safe_name}_verts.npy'), obj_verts)
        np.save(os.path.join(output_dir, f'obj_{safe_name}_faces.npy'), obj_faces)
        print(f"  Saved {safe_name}: verts {obj_verts.shape}, faces {obj_faces.shape}")

    # Get text description
    desc_path = os.path.join(os.path.dirname(PROG_DIR), "desc_info", f"{seq_id}.json")
    text = ""
    if os.path.exists(desc_path):
        with open(desc_path) as f:
            desc_info = json.load(f)
        texts = [v.get('seg_desc', '') for v in desc_info.values() if 'seg_desc' in v]
        text = ". ".join(texts[:5])

    # Save info
    info = {
        'text': text or f"Raw SMPL-X: {seq_id}",
        'obj_ids': obj_list,
        'obj_files': obj_files,
        'num_frames': T_ds,
        'fps': fps_out,
        'original_fps': 30,
        'seq_id': seq_id,
        'seg_indices': seg_indices,
        'seg_boundaries_30fps': seg_boundaries,
    }
    with open(os.path.join(output_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"Done! Saved to {output_dir}")


def render_video(output_dir, output_path, fps=10, resolution='low'):
    """Render SMPL-X body + objects as MP4 (reuses render_full.py logic)."""
    from visualize.render_full import render_video as _render
    _render(output_dir, output_path, fps=fps, resolution=resolution)


def main():
    parser = argparse.ArgumentParser(description="Render raw SMPL-X + GT objects")
    parser.add_argument("--mode", choices=["fit", "render", "both"], default="both")
    parser.add_argument("--seq_id", default="", help="OakInk2 sequence ID")
    parser.add_argument("--seg_indices", type=int, nargs='+', default=[0, 1, 2],
                        help="Segment indices to concatenate")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_path", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps_out", type=int, default=10, help="Output FPS (downsample from 30fps)")
    parser.add_argument("--fps_render", type=int, default=10, help="Render playback FPS")
    parser.add_argument("--resolution", default="medium")
    args = parser.parse_args()

    if not args.output_path:
        args.output_path = args.output_dir.rstrip('/') + '.mp4'

    if args.mode in ("fit", "both"):
        fit_raw_smplx(args.seq_id, args.seg_indices, args.output_dir,
                      fps_out=args.fps_out, device=args.device)

    if args.mode in ("render", "both"):
        render_video(args.output_dir, args.output_path,
                     fps=args.fps_render, resolution=args.resolution)


if __name__ == "__main__":
    main()
