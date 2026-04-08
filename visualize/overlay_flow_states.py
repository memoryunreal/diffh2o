#!/usr/bin/env python3
"""
Post-process rendered frames to add flow state labels as text overlays.

Usage:
    python visualize/overlay_flow_states.py \
        --frame_dir output/flow_verify/000000_frames \
        --state_json dataset/OAKINK2_FLOW/state_trajectories/000000.json \
        --anno_pkl /hhd4/lizhe/dataset/OakInk2/data/anno_preview/{seq}.pkl \
        --output_path output/flow_verify/000000_states.mp4
"""

import os
import json
import pickle
import argparse
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# State colors (RGB)
STATE_COLORS = {
    "[Idle]": (160, 160, 160),
    "[Reach]": (255, 220, 50),
    "[Grasped]": (50, 200, 50),
    "[Grasped_and_Interacting]": (200, 50, 200),
    "[Release]": (255, 150, 50),
    "[Static]": (120, 120, 120),
    "[Interacting]": (220, 50, 50),
}


def get_state_at_frame(trajectory, mocap_frame_id):
    """Look up entity state for a given mocap frame ID from the trajectory spans."""
    for span in trajectory:
        f_start, f_end = span["frames"]
        if f_start <= mocap_frame_id <= f_end:
            return span["state"], span.get("target", ""), span.get("master", "")
    return "[Idle]", "", ""


def overlay_states_on_frame(frame_path, frame_idx, state_json, mocap_frames):
    """Add state text overlay to a single rendered frame."""
    img = Image.open(frame_path)
    draw = ImageDraw.Draw(img)
    W, H = img.size

    font_size = max(11, H // 45)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Get mocap frame ID for this rendered frame
    if frame_idx < len(mocap_frames):
        mocap_fid = mocap_frames[frame_idx]
    else:
        mocap_fid = frame_idx

    trajectories = state_json["token_state_trajectories"]
    entities = state_json.get("entities", [])

    # Build lines to display (only active entities — skip [Static] objects and [Idle] hands for brevity)
    lines = []
    lines.append(f"Frame {frame_idx + 1}/{len(mocap_frames)}  (mocap: {mocap_fid})")

    for ename in entities[:8]:  # Show up to 8 entities
        traj = trajectories.get(ename, [])
        state, target, master = get_state_at_frame(traj, mocap_fid)

        # Shorten object names
        display_name = ename.replace("obj_", "").split("@")[-1] if "obj_" in ename else ename.upper()

        extra = ""
        if target:
            target_short = target.replace("obj_", "").split("@")[-1]
            extra = f" -> {target_short}"
        if master:
            extra = f" by {master.upper()}"

        lines.append((display_name, state, extra))

    # Draw background box (semi-transparent)
    line_h = font_size + 3
    box_h = (len(lines)) * line_h + 10
    box_w = min(W // 2, 320)
    box_x = W - box_w - 5
    box_y = 5

    # Draw semi-transparent background
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(0, 0, 0, 150))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(img)

    y = box_y + 4
    # Header line
    header = lines[0]
    draw.text((box_x + 5, y), header, fill=(220, 220, 220), font=font)
    y += line_h

    # Entity lines
    for item in lines[1:]:
        display_name, state, extra = item
        color = STATE_COLORS.get(state, (200, 200, 200))
        text = f"{display_name}: {state}{extra}"
        draw.text((box_x + 5, y), text, fill=color, font=font)
        y += line_h

    img.save(frame_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dir", required=True)
    parser.add_argument("--state_json", required=True)
    parser.add_argument("--anno_pkl", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # Load state trajectory
    with open(args.state_json) as f:
        state_json = json.load(f)

    # Load annotation to get mocap frame mapping
    with open(args.anno_pkl, 'rb') as f:
        anno = pickle.load(f)
    mocap_frames = anno['mocap_frame_id_list']

    # Get rendered frames
    frame_files = sorted([f for f in os.listdir(args.frame_dir) if f.endswith('.png')])
    print(f"Overlaying states on {len(frame_files)} frames...")

    for i, fname in enumerate(frame_files):
        fpath = os.path.join(args.frame_dir, fname)
        overlay_states_on_frame(fpath, i, state_json, mocap_frames)

    # Re-encode to mp4
    frame_pattern = os.path.join(args.frame_dir, "frame_%06d.png")
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [ffmpeg, "-y", "-framerate", str(args.fps), "-i", frame_pattern,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", args.output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        cmd[6] = "mpeg4"
        cmd[7:9] = ["-q:v", "5"]
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Video saved to: {args.output_path}")
    else:
        print(f"FFmpeg failed: {result.stderr}")


if __name__ == "__main__":
    main()
