#!/usr/bin/env python3
"""
Export mesh data as OBJ files per frame for Blender import.

Exports 4 types of meshes (gen human, gen objects, gt human, gt objects)
frame-aligned, sampled every N frames.

Usage:
    python -m visualize.export_obj_frames --scene lightbulb --step 3
    python -m visualize.export_obj_frames --scene scoop_cheese --step 3
"""

import os
import json
import argparse
import numpy as np
import trimesh


SCENE_MAP = {
    'lightbulb': 'ar_chain_lightbulb',
    'scoop_cheese': 'ar_chain_scoop_cheese',
    'donut': 'ar_chain_donut',
    'rearrange_plate': 'ar_chain_rearrange_plate',
}


def export_mesh_sequence(verts, faces, output_dir, frame_indices, prefix="frame"):
    """Export a mesh sequence as OBJ files for selected frames."""
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for fi in frame_indices:
        if fi >= len(verts):
            break
        mesh = trimesh.Trimesh(vertices=verts[fi], faces=faces, process=False)
        obj_path = os.path.join(output_dir, f"{prefix}_{fi:04d}.obj")
        mesh.export(obj_path)
        count += 1
    return count


def export_scene(scene_name, step=3, output_root=None):
    """Export all 4 mesh types for a scene."""
    scene_dir = SCENE_MAP.get(scene_name, scene_name)
    base = os.path.join('output', scene_dir)

    gen_dir = os.path.join(base, 'render_with_objects')
    gt_dir = os.path.join(base, 'render_gt_objects')

    if output_root is None:
        output_root = os.path.join(base, 'obj_export')

    # Load gen info
    with open(os.path.join(gen_dir, 'info.json')) as f:
        gen_info = json.load(f)
    with open(os.path.join(gt_dir, 'info.json')) as f:
        gt_info = json.load(f)

    # Load body data
    gen_body_v = np.load(os.path.join(gen_dir, 'body_vertices.npy'))
    gen_body_f = np.load(os.path.join(gen_dir, 'body_faces.npy'))
    gt_body_v = np.load(os.path.join(gt_dir, 'body_vertices.npy'))
    gt_body_f = np.load(os.path.join(gt_dir, 'body_faces.npy'))

    T = min(len(gen_body_v), len(gt_body_v))
    frame_indices = list(range(0, T, step))
    print(f"Scene: {scene_name} | Total frames: {T} | Step: {step} | Exporting: {len(frame_indices)} frames")

    # Export gen human
    print("Exporting gen_human...")
    n = export_mesh_sequence(gen_body_v, gen_body_f,
                             os.path.join(output_root, 'gen_human'), frame_indices)
    print(f"  {n} OBJ files")

    # Export gt human
    print("Exporting gt_human...")
    n = export_mesh_sequence(gt_body_v, gt_body_f,
                             os.path.join(output_root, 'gt_human'), frame_indices)
    print(f"  {n} OBJ files")

    # Export gen objects
    print("Exporting gen_objects...")
    for safe_name in gen_info.get('obj_files', []):
        vp = os.path.join(gen_dir, f'obj_{safe_name}_verts.npy')
        fp = os.path.join(gen_dir, f'obj_{safe_name}_faces.npy')
        if os.path.exists(vp):
            obj_v = np.load(vp)
            obj_f = np.load(fp)
            obj_dir = os.path.join(output_root, 'gen_objects', safe_name)
            n = export_mesh_sequence(obj_v, obj_f, obj_dir, frame_indices)
            print(f"  {safe_name}: {n} OBJ files")

    # Export gt objects
    print("Exporting gt_objects...")
    for safe_name in gt_info.get('obj_files', []):
        vp = os.path.join(gt_dir, f'obj_{safe_name}_verts.npy')
        fp = os.path.join(gt_dir, f'obj_{safe_name}_faces.npy')
        if os.path.exists(vp):
            obj_v = np.load(vp)
            obj_f = np.load(fp)
            obj_dir = os.path.join(output_root, 'gt_objects', safe_name)
            n = export_mesh_sequence(obj_v, obj_f, obj_dir, frame_indices)
            print(f"  {safe_name}: {n} OBJ files")

    print(f"\nDone! All OBJ files saved to: {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Export OBJ files per frame")
    parser.add_argument("--scene", required=True, choices=list(SCENE_MAP.keys()),
                        help="Scene name")
    parser.add_argument("--step", type=int, default=3,
                        help="Frame sampling step (default: 3)")
    parser.add_argument("--output_root", default=None,
                        help="Output directory (default: output/<scene>/obj_export/)")
    args = parser.parse_args()

    export_scene(args.scene, step=args.step, output_root=args.output_root)


if __name__ == "__main__":
    main()
