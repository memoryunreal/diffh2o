#!/usr/bin/env python3
"""
Rebuild GT body + objects with proper frame alignment from raw OakInk2 annotations.

The AR chain concatenates segments that have DIFFERENT object slot mappings,
causing misalignment. This script bypasses the preprocessed 267D data and
directly loads GT body + ALL objects from annotations at the correct frame IDs.

Usage:
    # Step 1: Build aligned GT data (needs rog_env for smplx)
    python -m visualize.rebuild_gt_aligned --mode fit \
        --scene scoop_cheese --output_dir debug_vis/scoop_cheese/gt_aligned

    # Step 2: Render (needs mdm2 for aitviewer)
    xvfb-run ... python -m visualize.rebuild_gt_aligned --mode render \
        --output_dir debug_vis/scoop_cheese/gt_aligned \
        --output_path debug_vis/scoop_cheese/gt_aligned.mp4
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

SCENE_MAP = {
    'scoop_cheese': {
        'chain_dir': 'output/ar_chain_scoop_cheese',
        'seq_id': 'scene_01__A003++seq__a7a1a0cf7d90a9083013__2023-04-21-20-13-04',
        'seg_super_indices': [0, 2, 3],  # super-segment indices used in AR chain
    },
    'lightbulb': {
        'chain_dir': 'output/ar_chain_lightbulb',
        'seq_id': 'scene_02__A001++seq__ac736a20fe59cf0a1df3__2023-04-15-19-49-11',
        'seg_super_indices': [0, 1, 2],
    },
}


def parse_interval(key_str):
    import ast
    try:
        parsed = ast.literal_eval(key_str)
        if isinstance(parsed, tuple) and len(parsed) == 2:
            lh, rh = parsed
            return (tuple(lh) if lh is not None else None,
                    tuple(rh) if rh is not None else None)
    except Exception:
        pass
    return None, None


def extract_super_segments(program_info, total_frames, gap_threshold=300):
    all_intervals = []
    for key, prim_data in program_info.items():
        lh_interval, rh_interval = parse_interval(key)
        starts, ends = [], []
        if lh_interval:
            starts.append(lh_interval[0]); ends.append(lh_interval[1])
        if rh_interval:
            starts.append(rh_interval[0]); ends.append(rh_interval[1])
        if not starts:
            continue
        all_intervals.append({'start': min(starts), 'end': max(ends), 'data': prim_data})

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


def recover_frame_ids(anno, segments, seg_super_indices, step=3, target_len=200, K=20):
    """Recover the exact OakInk2 frame IDs for each of the 560 generated frames."""
    mocap = anno['mocap_frame_id_list']
    all_frame_ids = []

    for i, si in enumerate(seg_super_indices):
        seg = segments[si]
        seg_start, seg_end = seg['start'], seg['end']
        seg_frames = mocap[seg_start:seg_end + 1]
        T_seg = len(seg_frames)

        # Downsample 30fps -> 10fps
        ds_indices = list(range(0, T_seg, step))
        T_ds = len(ds_indices)

        # Uniform subsample to target_len
        if T_ds > target_len:
            sub_step = T_ds / target_len
            sub_indices = [int(j * sub_step) for j in range(target_len)]
        elif T_ds < target_len:
            sub_indices = list(range(T_ds)) + [T_ds - 1] * (target_len - T_ds)
        else:
            sub_indices = list(range(T_ds))

        # Map to actual frame IDs
        seg_mapped_ids = [seg_frames[ds_indices[si_]] for si_ in sub_indices]

        # Skip first K frames for segments after the first (history overlap)
        if i == 0:
            all_frame_ids.extend(seg_mapped_ids)
        else:
            all_frame_ids.extend(seg_mapped_ids[K:])

    return all_frame_ids


def quat_wxyz_to_aa(quats):
    xyzw = np.concatenate([quats[..., 1:], quats[..., :1]], axis=-1)
    return R.from_quat(xyzw.reshape(-1, 4)).as_rotvec().reshape(quats.shape[:-1] + (3,))


def fit_aligned_gt(scene_name, output_dir, device='cpu'):
    """Extract aligned GT body + objects from raw annotations."""
    import torch
    import smplx as smplx_lib
    import trimesh

    cfg = SCENE_MAP[scene_name]
    os.makedirs(output_dir, exist_ok=True)

    # Load annotation and program info
    seq_id = cfg['seq_id']
    print(f"Loading annotation: {seq_id}")
    with open(os.path.join(ANNO_DIR, f"{seq_id}.pkl"), 'rb') as f:
        anno = pickle.load(f)
    with open(os.path.join(PROG_DIR, f"{seq_id}.json")) as f:
        prog = json.load(f)

    mocap = anno['mocap_frame_id_list']
    segments = extract_super_segments(prog, len(mocap))

    # Recover frame IDs
    frame_ids = recover_frame_ids(anno, segments, cfg['seg_super_indices'])
    T = len(frame_ids)
    print(f"Recovered {T} frame IDs (expected 560)")

    # Extract SMPL-X parameters
    print("Extracting SMPL-X parameters...")
    transl, orient, body_pose, lhand, rhand, shape = [], [], [], [], [], []
    for fid in frame_ids:
        rs = anno['raw_smplx'][fid]
        transl.append(rs['world_tsl'].squeeze().numpy())
        orient.append(quat_wxyz_to_aa(rs['world_rot'].squeeze().numpy()[None])[0])
        body_pose.append(quat_wxyz_to_aa(rs['body_pose'].squeeze().numpy()).reshape(-1))
        lhand.append(quat_wxyz_to_aa(rs['left_hand_pose'].squeeze().numpy()).reshape(-1))
        rhand.append(quat_wxyz_to_aa(rs['right_hand_pose'].squeeze().numpy()).reshape(-1))
        shape.append(rs['body_shape'].squeeze().numpy())

    transl = np.array(transl, dtype=np.float32)
    orient = np.array(orient, dtype=np.float32)
    body_pose = np.array(body_pose, dtype=np.float32)
    lhand = np.array(lhand, dtype=np.float32)
    rhand = np.array(rhand, dtype=np.float32)
    shape = np.array(shape, dtype=np.float32)

    # SMPL-X forward pass
    dev = torch.device(device if 'cuda' not in device or torch.cuda.is_available() else 'cpu')
    print(f"Running SMPL-X forward pass on {dev}...")
    CHUNK = 200
    all_verts = []
    faces = None
    for cs in range(0, T, CHUNK):
        ce = min(cs + CHUNK, T)
        model = smplx_lib.create(
            model_path=SMPLX_MODEL_PATH, model_type='smplx', gender='neutral',
            use_pca=False, flat_hand_mean=True, batch_size=ce-cs, num_betas=300,
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

    body_vertices = np.concatenate(all_verts, axis=0)
    body_faces = faces.astype(np.int32)
    np.save(os.path.join(output_dir, 'body_vertices.npy'), body_vertices)
    np.save(os.path.join(output_dir, 'body_faces.npy'), body_faces)
    print(f"Body vertices: {body_vertices.shape}")

    # Extract ALL object transforms
    obj_transf = anno.get('obj_transf', {})
    Rz_neg90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)

    # Exclude truly static/unused objects
    exclude = {'O02@0017@00001', 'O02@0026@00001'}
    obj_ids = sorted([oid for oid in obj_transf.keys() if oid not in exclude])
    print(f"Objects ({len(obj_ids)}): {obj_ids}")

    obj_files = []
    for obj_id in obj_ids:
        safe_name = obj_id.replace('@', '_')
        obj_files.append(safe_name)

        mesh_path = os.path.join(OBJ_MESH_DIR, obj_id, 'model.obj')
        if not os.path.exists(mesh_path):
            print(f"  Warning: mesh not found: {mesh_path}")
            continue

        mesh = trimesh.load(mesh_path, force='mesh')
        base_verts = mesh.vertices.astype(np.float32)
        obj_faces = mesh.faces.astype(np.int32)

        td = obj_transf[obj_id]
        obj_verts = np.zeros((T, len(base_verts), 3), dtype=np.float32)

        for t_idx, fid in enumerate(frame_ids):
            if fid in td:
                tf = td[fid]
                rot = tf[:3, :3].astype(np.float32)
                pos = tf[:3, 3].astype(np.float32)
                obj_verts[t_idx] = (base_verts @ Rz_neg90.T @ rot.T) + pos
            elif t_idx > 0:
                obj_verts[t_idx] = obj_verts[t_idx - 1]

        np.save(os.path.join(output_dir, f'obj_{safe_name}_verts.npy'), obj_verts)
        np.save(os.path.join(output_dir, f'obj_{safe_name}_faces.npy'), obj_faces)
        print(f"  {safe_name}: verts {obj_verts.shape}")

    # Save info
    info = {
        'text': f'GT aligned: {seq_id}',
        'obj_ids': obj_ids,
        'obj_files': obj_files,
        'num_frames': T,
        'frame_ids': frame_ids,
        'seg_super_indices': cfg['seg_super_indices'],
    }
    with open(os.path.join(output_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"Done! Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fit", "render"], default="fit")
    parser.add_argument("--scene", required=True, choices=list(SCENE_MAP.keys()))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_path", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolution", default="medium")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "fit":
        fit_aligned_gt(args.scene, args.output_dir, device=args.device)
    elif args.mode == "render":
        from visualize.render_full import render_video
        if not args.output_path:
            args.output_path = args.output_dir.rstrip('/') + '.mp4'
        render_video(args.output_dir, args.output_path, fps=args.fps, resolution=args.resolution)


if __name__ == "__main__":
    main()
