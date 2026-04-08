#!/usr/bin/env python3
"""
Visualize OakInk2 transition data: boundary_before (10f) + transition (60f) + boundary_after (10f).

Renders phase-colored video with text overlays showing source/target actions.

Usage:
    # Single transition
    xvfb-run -s "-screen 0 1024x768x24" python visualize/visualize_transitions.py \
        --transition_idx 0 --output_path output/trans_000.mp4

    # Batch of transitions
    xvfb-run -s "-screen 0 1024x768x24" python visualize/visualize_transitions.py \
        --batch 0-9 --output_path output/transitions/
"""

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature layout constants (duplicated to avoid aitviewer import at top level)
IDX_BODY_TRANSL = (0, 3)
IDX_BODY_ORIENT = (3, 9)
IDX_BODY_POSE = (9, 135)
IDX_LHAND_QUAT = (198, 262)
IDX_RHAND_QUAT = (265, 329)
IDX_OBJ1_POS = (371, 374)
IDX_OBJ1_ROT = (374, 380)
IDX_OBJ2_POS = (380, 383)
IDX_OBJ2_ROT = (383, 389)
IDX_OBJ3_POS = (389, 392)
IDX_OBJ3_ROT = (392, 398)

BODY_MESH_COLOR = (0.7, 0.7, 0.8, 1.0)

# Phase colors (RGB for PIL)
PHASE_COLOR_BOUNDARY = (68, 119, 170)   # Blue #4477AA
PHASE_COLOR_TRANSITION = (204, 102, 51)  # Orange #CC6633

OBJ_MESH_DIR = "/hhd4/lizhe/dataset/OakInk2/data/object_repair/align_ds"


def load_transition_data(transition_idx, dataset_dir="dataset/OAKINK2"):
    """Load and denormalize a full 80-frame transition sequence."""
    idx_str = f"{transition_idx:06d}"

    # Load arrays
    transition = np.load(os.path.join(dataset_dir, "oakink2_transitions", f"{idx_str}.npy"))
    boundary_before = np.load(os.path.join(dataset_dir, "boundaries_before", f"{idx_str}.npy"))
    boundary_after = np.load(os.path.join(dataset_dir, "boundaries_after", f"{idx_str}.npy"))

    # Denormalize all with transition stats
    mean = np.load(os.path.join(dataset_dir, "Mean_transitions.npy"))
    std = np.load(os.path.join(dataset_dir, "Std_transitions.npy"))
    std_safe = np.where(std < 1e-8, 1.0, std)

    transition = transition * std_safe + mean
    boundary_before = boundary_before * std_safe + mean
    boundary_after = boundary_after * std_safe + mean

    # Concatenate: (10 + 60 + 10, 398) = (80, 398)
    motion = np.concatenate([boundary_before, transition, boundary_after], axis=0)

    # Phase labels per frame
    phase_labels = (
        ["boundary_before"] * boundary_before.shape[0]
        + ["transition"] * transition.shape[0]
        + ["boundary_after"] * boundary_after.shape[0]
    )

    # Parse text annotations
    text_path = os.path.join(dataset_dir, "texts_transitions", f"{idx_str}.txt")
    text_source, text_target = "", ""
    if os.path.exists(text_path):
        with open(text_path, "r") as f:
            lines = f.readlines()
        # Line 2 = source text, Line 3 = target text
        if len(lines) >= 3:
            text_source = lines[1].split("#")[0].strip()
            text_target = lines[2].split("#")[0].strip()

    # Load metadata for object info
    metadata = {}
    meta_path = os.path.join(dataset_dir, "transition_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        transitions_list = meta.get("transitions", [])
        if transition_idx < len(transitions_list):
            metadata = transitions_list[transition_idx]

    return {
        "motion": motion,
        "phase_labels": phase_labels,
        "text_source": text_source,
        "text_target": text_target,
        "metadata": metadata,
    }


def parse_motion_components(motion):
    """Extract body, hand, and object components from 398D motion array."""
    return {
        "body_transl": motion[:, IDX_BODY_TRANSL[0]:IDX_BODY_TRANSL[1]],
        "body_orient": motion[:, IDX_BODY_ORIENT[0]:IDX_BODY_ORIENT[1]],
        "body_pose": motion[:, IDX_BODY_POSE[0]:IDX_BODY_POSE[1]],
        "lhand_quat": motion[:, IDX_LHAND_QUAT[0]:IDX_LHAND_QUAT[1]],
        "rhand_quat": motion[:, IDX_RHAND_QUAT[0]:IDX_RHAND_QUAT[1]],
        "obj1_pos": motion[:, IDX_OBJ1_POS[0]:IDX_OBJ1_POS[1]],
        "obj1_rot": motion[:, IDX_OBJ1_ROT[0]:IDX_OBJ1_ROT[1]],
        "obj2_pos": motion[:, IDX_OBJ2_POS[0]:IDX_OBJ2_POS[1]],
        "obj2_rot": motion[:, IDX_OBJ2_ROT[0]:IDX_OBJ2_ROT[1]],
        "obj3_pos": motion[:, IDX_OBJ3_POS[0]:IDX_OBJ3_POS[1]],
        "obj3_rot": motion[:, IDX_OBJ3_ROT[0]:IDX_OBJ3_ROT[1]],
    }


def build_object_transforms(obj_pos, obj_rot_6d):
    """Convert object position (T,3) + rotation 6D (T,6) into (T,4,4) transform matrices."""
    from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix

    T = obj_pos.shape[0]
    # Check if object data is all near-zero (unused)
    if np.linalg.norm(obj_pos) < 0.01 and np.linalg.norm(obj_rot_6d) < 0.01:
        return None

    rot_6d_t = torch.tensor(obj_rot_6d, dtype=torch.float32)
    rot_mat = rotation_6d_to_matrix(rot_6d_t).numpy()  # (T, 3, 3)

    transforms = np.eye(4, dtype=np.float32)[None].repeat(T, axis=0)  # (T, 4, 4)
    transforms[:, :3, :3] = rot_mat
    transforms[:, :3, 3] = obj_pos
    return transforms


def add_phase_overlay(frame_path, frame_idx, total_frames, phase_label,
                      text_source, text_target, n_before, n_transition):
    """Add phase indicator bar, labels, and text overlay to a rendered frame."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(frame_path)
    draw = ImageDraw.Draw(img)
    W, H = img.size

    # Try to load a nice font, fall back to default
    font_size = max(14, H // 40)
    font_small = max(12, H // 50)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_small)
    except OSError:
        font = ImageFont.load_default()
        font_s = font

    bar_height = max(8, H // 60)

    # Phase indicator bar at bottom
    if phase_label == "transition":
        bar_color = PHASE_COLOR_TRANSITION
    else:
        bar_color = PHASE_COLOR_BOUNDARY

    draw.rectangle([0, H - bar_height, W, H], fill=bar_color)

    # Progress bar just above the phase bar
    progress_h = max(4, H // 120)
    progress_y = H - bar_height - progress_h - 2
    # Background
    draw.rectangle([0, progress_y, W, progress_y + progress_h], fill=(60, 60, 60))
    # Fill
    progress_frac = (frame_idx + 1) / total_frames
    draw.rectangle([0, progress_y, int(W * progress_frac), progress_y + progress_h], fill=(200, 200, 200))

    # Phase boundary markers on progress bar
    before_x = int(W * n_before / total_frames)
    after_x = int(W * (n_before + n_transition) / total_frames)
    for marker_x in (before_x, after_x):
        draw.line([(marker_x, progress_y), (marker_x, progress_y + progress_h)], fill=(255, 255, 255), width=2)

    # Phase label text (top-left)
    phase_display = {
        "boundary_before": "BOUNDARY BEFORE",
        "transition": "TRANSITION",
        "boundary_after": "BOUNDARY AFTER",
    }
    label = phase_display.get(phase_label, phase_label.upper())
    draw.text((10, 10), label, fill=bar_color, font=font)

    # Frame counter (top-right)
    counter_text = f"Frame {frame_idx + 1}/{total_frames}"
    bbox = draw.textbbox((0, 0), counter_text, font=font_s)
    text_w = bbox[2] - bbox[0]
    draw.text((W - text_w - 10, 10), counter_text, fill=(220, 220, 220), font=font_s)

    # Text descriptions (bottom area, above bars)
    text_y = progress_y - font_small * 3 - 10
    if text_source:
        src_label = f"From: {text_source}"
        # Truncate if too long
        if len(src_label) > 80:
            src_label = src_label[:77] + "..."
        draw.text((10, text_y), src_label, fill=(180, 210, 240), font=font_s)
    if text_target:
        tgt_label = f"To: {text_target}"
        if len(tgt_label) > 80:
            tgt_label = tgt_label[:77] + "..."
        draw.text((10, text_y + font_small + 4), tgt_label, fill=(240, 200, 160), font=font_s)

    img.save(frame_path)


def render_transition(
    transition_idx,
    dataset_dir="dataset/OAKINK2",
    output_path="output/transition_vis.mp4",
    smplx_path="/hhd4/lizhe/code/InterAct/models",
    resolution="medium",
    fps=30,
    no_overlay=False,
):
    """Render a single transition sequence to video."""
    res_map = {"high": (2000, 1500), "medium": (1280, 960), "low": (640, 480)}
    res = res_map.get(resolution, (1280, 960))

    # Import heavy rendering dependencies
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.meshes import Meshes
    from visualize.visualize_oakink2 import (
        create_body_mesh_sequence,
        load_object_meshes,
        create_object_mesh_sequence,
    )

    print(f"Loading transition {transition_idx}...")
    data = load_transition_data(transition_idx, dataset_dir)
    motion = data["motion"]
    T = motion.shape[0]
    print(f"  Total frames: {T} (boundary_before + transition + boundary_after)")
    print(f"  Source: {data['text_source']}")
    print(f"  Target: {data['text_target']}")

    components = parse_motion_components(motion)

    # Count frames per phase
    n_before = sum(1 for p in data["phase_labels"] if p == "boundary_before")
    n_transition = sum(1 for p in data["phase_labels"] if p == "transition")

    # Initialize renderer
    print("Initializing renderer...")
    v = HeadlessRenderer(size=res)
    v.scene.camera.fov = 60
    v.scene.camera.position = [0.0, 1.0, -3.5]
    v.scene.camera.target = [0.0, 0.9, 0.0]
    v.scene.camera.up = [0, 1.0, 0]
    v.scene.floor.position = [0.0, 0.0, 0.0]
    v.auto_set_camera_target = False
    v.auto_set_floor = False

    if len(v.scene.lights) > 0:
        v.scene.remove(v.scene.lights[0])
    v.scene.remove(v.scene.origin)

    # Load SMPL-X model
    print(f"Loading SMPL-X model from: {smplx_path}")
    import smplx
    smplx_model = smplx.create(
        model_path=smplx_path,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=True,
        batch_size=T,
    ).to(device)

    # Create body mesh sequence
    print("Computing body mesh vertices...")
    body_verts, body_faces, floor_offset = create_body_mesh_sequence(
        components["body_transl"],
        components["body_orient"],
        components["body_pose"],
        components["lhand_quat"],
        components["rhand_quat"],
        smplx_model,
        T,
    )
    print(f"  Body vertices: {body_verts.shape}")

    body_mesh = Meshes(body_verts, body_faces, z_up=False, color=BODY_MESH_COLOR, name="Body")
    v.scene.add(body_mesh)

    # Try to add object meshes
    meta = data["metadata"]
    obj_ids = set()
    for key in ("obj_source", "obj_target"):
        obj_ids.update(meta.get(key, []))

    if obj_ids:
        obj_meshes = load_object_meshes(list(obj_ids), OBJ_MESH_DIR)
        if obj_meshes:
            # Build transforms from feature vector object channels
            obj_channels = [
                (components["obj1_pos"], components["obj1_rot"]),
                (components["obj2_pos"], components["obj2_rot"]),
                (components["obj3_pos"], components["obj3_rot"]),
            ]
            obj_id_list = sorted(obj_ids)
            obj_transf = {}
            for i, oid in enumerate(obj_id_list[:3]):
                pos, rot = obj_channels[i]
                tf = build_object_transforms(pos, rot)
                if tf is not None:
                    obj_transf[oid] = tf

            if obj_transf:
                obj_seqs = create_object_mesh_sequence(obj_meshes, obj_transf, T, floor_offset)
                obj_colors = [
                    (0.8, 0.6, 0.3, 1.0),
                    (0.6, 0.8, 0.4, 1.0),
                    (0.4, 0.6, 0.8, 1.0),
                ]
                for i, (oid, verts_seq, faces) in enumerate(obj_seqs):
                    obj_mesh = Meshes(verts_seq, faces, z_up=False,
                                      color=obj_colors[i % len(obj_colors)],
                                      name=f"Object_{oid}")
                    v.scene.add(obj_mesh)
                    print(f"  Added object '{oid}' to scene")
    else:
        # Fallback: check if object channels have non-zero data
        for ch_idx, (pos, rot) in enumerate([
            (components["obj1_pos"], components["obj1_rot"]),
            (components["obj2_pos"], components["obj2_rot"]),
            (components["obj3_pos"], components["obj3_rot"]),
        ]):
            if np.linalg.norm(pos) > 0.01:
                print(f"  Object channel {ch_idx + 1} has data but no mesh ID — skipping")

    # Render frames
    os.makedirs(os.path.dirname(output_path) or "output", exist_ok=True)
    frame_dir = output_path.replace(".mp4", "_frames")
    os.makedirs(frame_dir, exist_ok=True)

    print(f"Rendering {T} frames...")
    v._init_scene()
    for fi in range(T):
        v.scene.current_frame_id = fi
        v.render(0, 0, export=True)
        frame_path = os.path.join(frame_dir, f"frame_{fi:06d}.png")
        v.export_frame(frame_path)

        if not no_overlay:
            add_phase_overlay(
                frame_path, fi, T,
                data["phase_labels"][fi],
                data["text_source"], data["text_target"],
                n_before, n_transition,
            )

        if (fi + 1) % 20 == 0:
            print(f"  Rendered frame {fi + 1}/{T}")

    # Assemble video with ffmpeg
    frame_pattern = os.path.join(frame_dir, "frame_%06d.png")
    system_ffmpeg = shutil.which("ffmpeg")
    ffmpeg_bin = system_ffmpeg or "ffmpeg"

    cmd = [
        ffmpeg_bin, "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        output_path,
    ]
    print("Creating video...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Fallback to mpeg4
        cmd = [
            ffmpeg_bin, "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "mpeg4",
            "-q:v", "5",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Video saved to: {output_path}")
    else:
        print(f"FFmpeg failed: {result.stderr}")
        print(f"Frames available at: {frame_dir}")


def render_motion_file(
    motion_path,
    output_path="output/motion_vis.mp4",
    smplx_path="/hhd4/lizhe/code/InterAct/models",
    resolution="medium",
    fps=30,
    no_overlay=False,
    label="GENERATED",
    text_source="",
    text_target="",
    n_before=10,
    n_transition=60,
):
    """Render a pre-computed denormalized .npy motion file to video."""
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.renderables.meshes import Meshes
    from visualize.visualize_oakink2 import create_body_mesh_sequence

    res_map = {"high": (2000, 1500), "medium": (1280, 960), "low": (640, 480)}
    res = res_map.get(resolution, (1280, 960))

    motion = np.load(motion_path)  # [T, 398], already denormalized
    T = motion.shape[0]
    print(f"Loaded motion from {motion_path}: {motion.shape}")

    components = parse_motion_components(motion)

    # Build phase labels from frame counts
    n_after = T - n_before - n_transition
    if n_after < 0:
        n_after = 0
        n_transition = T - n_before
    phase_labels = (
        ["boundary_before"] * n_before
        + ["transition"] * n_transition
        + ["boundary_after"] * n_after
    )

    # Initialize renderer
    print("Initializing renderer...")
    v = HeadlessRenderer(size=res)
    v.scene.camera.fov = 60
    v.scene.camera.position = [0.0, 1.0, -3.5]
    v.scene.camera.target = [0.0, 0.9, 0.0]
    v.scene.camera.up = [0, 1.0, 0]
    v.scene.floor.position = [0.0, 0.0, 0.0]
    v.auto_set_camera_target = False
    v.auto_set_floor = False

    if len(v.scene.lights) > 0:
        v.scene.remove(v.scene.lights[0])
    v.scene.remove(v.scene.origin)

    # Load SMPL-X model
    print(f"Loading SMPL-X model from: {smplx_path}")
    import smplx
    smplx_model = smplx.create(
        model_path=smplx_path,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=True,
        batch_size=T,
    ).to(device)

    print("Computing body mesh vertices...")
    body_verts, body_faces, floor_offset = create_body_mesh_sequence(
        components["body_transl"],
        components["body_orient"],
        components["body_pose"],
        components["lhand_quat"],
        components["rhand_quat"],
        smplx_model,
        T,
    )
    print(f"  Body vertices: {body_verts.shape}")

    body_mesh = Meshes(body_verts, body_faces, z_up=False, color=BODY_MESH_COLOR, name="Body")
    v.scene.add(body_mesh)

    # Render frames
    os.makedirs(os.path.dirname(output_path) or "output", exist_ok=True)
    frame_dir = output_path.replace(".mp4", "_frames")
    os.makedirs(frame_dir, exist_ok=True)

    print(f"Rendering {T} frames...")
    v._init_scene()
    for fi in range(T):
        v.scene.current_frame_id = fi
        v.render(0, 0, export=True)
        frame_path = os.path.join(frame_dir, f"frame_{fi:06d}.png")
        v.export_frame(frame_path)

        if not no_overlay:
            add_phase_overlay(
                frame_path, fi, T,
                phase_labels[fi] if fi < len(phase_labels) else "transition",
                text_source, text_target,
                n_before, n_transition,
            )

        if (fi + 1) % 20 == 0:
            print(f"  Rendered frame {fi + 1}/{T}")

    # Assemble video with ffmpeg
    frame_pattern = os.path.join(frame_dir, "frame_%06d.png")
    system_ffmpeg = shutil.which("ffmpeg")
    ffmpeg_bin = system_ffmpeg or "ffmpeg"

    cmd = [
        ffmpeg_bin, "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        output_path,
    ]
    print("Creating video...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        cmd = [
            ffmpeg_bin, "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "mpeg4",
            "-q:v", "5",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Video saved to: {output_path}")
    else:
        print(f"FFmpeg failed: {result.stderr}")
        print(f"Frames available at: {frame_dir}")


def parse_batch_arg(batch_str):
    """Parse batch argument like '0-9' or '0,5,100' into list of indices."""
    indices = []
    for part in batch_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return indices


def main():
    parser = argparse.ArgumentParser(description="Visualize OakInk2 transition data")
    parser.add_argument("--transition_idx", type=int, default=None,
                        help="Index of transition to visualize")
    parser.add_argument("--dataset_dir", type=str, default="dataset/OAKINK2",
                        help="Path to OakInk2 dataset directory")
    parser.add_argument("--output_path", type=str, default="output/transition_vis.mp4",
                        help="Output video path (or directory for batch mode)")
    parser.add_argument("--smplx_path", type=str,
                        default="/hhd4/lizhe/code/InterAct/models",
                        help="Path to SMPL-X model directory")
    parser.add_argument("--resolution", type=str, default="medium",
                        choices=["high", "medium", "low"],
                        help="Output video resolution")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--no_overlay", action="store_true",
                        help="Disable text/phase overlays")
    parser.add_argument("--batch", type=str, default=None,
                        help="Batch mode: range '0-9' or list '0,5,100'")
    parser.add_argument("--motion_dir", type=str, default=None,
                        help="Directory with pre-computed full_*.npy files (denormalized)")

    args = parser.parse_args()

    if args.motion_dir is not None:
        # Render pre-computed .npy files from generate_transition output
        import glob
        npy_files = sorted(glob.glob(os.path.join(args.motion_dir, "full_*.npy")))
        if not npy_files:
            parser.error(f"No full_*.npy files found in {args.motion_dir}")
        output_dir = args.output_path
        os.makedirs(output_dir, exist_ok=True)
        print(f"Rendering {len(npy_files)} motion files from {args.motion_dir}")

        # Try to load text descriptions from stats.txt
        texts = {}
        stats_path = os.path.join(args.motion_dir, "stats.txt")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                for line in f:
                    if line.startswith("Sample "):
                        parts = line.strip()
                        try:
                            idx = int(parts.split(":")[0].split()[-1])
                            src = parts.split("src='")[1].split("'")[0]
                            tgt = parts.split("tgt='")[1].split("'")[0]
                            texts[idx] = (src, tgt)
                        except (IndexError, ValueError):
                            pass

        for npy_path in npy_files:
            basename = os.path.basename(npy_path).replace(".npy", "")
            idx = int(basename.split("_")[-1])
            out = os.path.join(output_dir, f"{basename}.mp4")
            src, tgt = texts.get(idx, ("", ""))
            print(f"\n{'=' * 60}")
            print(f"Rendering {npy_path} -> {out}")
            print(f"{'=' * 60}")
            render_motion_file(
                motion_path=npy_path,
                output_path=out,
                smplx_path=args.smplx_path,
                resolution=args.resolution,
                fps=args.fps,
                no_overlay=args.no_overlay,
                text_source=src,
                text_target=tgt,
            )
    elif args.batch is not None:
        indices = parse_batch_arg(args.batch)
        output_dir = args.output_path
        os.makedirs(output_dir, exist_ok=True)
        print(f"Batch mode: rendering {len(indices)} transitions to {output_dir}")

        for idx in indices:
            out = os.path.join(output_dir, f"trans_{idx:06d}.mp4")
            print(f"\n{'=' * 60}")
            print(f"Rendering transition {idx} -> {out}")
            print(f"{'=' * 60}")
            render_transition(
                transition_idx=idx,
                dataset_dir=args.dataset_dir,
                output_path=out,
                smplx_path=args.smplx_path,
                resolution=args.resolution,
                fps=args.fps,
                no_overlay=args.no_overlay,
            )
    elif args.transition_idx is not None:
        render_transition(
            transition_idx=args.transition_idx,
            dataset_dir=args.dataset_dir,
            output_path=args.output_path,
            smplx_path=args.smplx_path,
            resolution=args.resolution,
            fps=args.fps,
            no_overlay=args.no_overlay,
        )
    else:
        parser.error("Provide either --transition_idx or --batch")


if __name__ == "__main__":
    main()
