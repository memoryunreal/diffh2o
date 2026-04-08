#!/usr/bin/env python3
"""
Pre-compute CLIP ViT-B/32 text embeddings for all samples in a flow dataset.

This decouples CLIP from the training loop, enabling DDP multi-GPU training
without NCCL deadlocks from frozen CLIP parameters.

Usage:
    python -m prepare.precompute_clip_embeddings --data_root dataset/OAKINK2_V3
"""

import os
import argparse
import numpy as np
import torch
import clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset/OAKINK2_V3")
    args = parser.parse_args()

    texts_dir = os.path.join(args.data_root, "texts")
    out_dir = os.path.join(args.data_root, "clip_embeddings")
    os.makedirs(out_dir, exist_ok=True)

    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    # Find all text files
    txt_files = sorted(f for f in os.listdir(texts_dir) if f.endswith(".txt"))
    print(f"Encoding {len(txt_files)} texts with CLIP ViT-B/32...")

    for i, fname in enumerate(txt_files):
        idx_str = fname.replace(".txt", "")
        out_path = os.path.join(out_dir, f"{idx_str}.npy")
        if os.path.exists(out_path):
            continue

        with open(os.path.join(texts_dir, fname)) as f:
            text = f.readline().split("#")[0].strip()

        tokens = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens).float().cpu().numpy()[0]  # [512]

        np.save(out_path, emb)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(txt_files)}")

    print(f"Done. Saved {len(txt_files)} embeddings to {out_dir}/")


if __name__ == "__main__":
    main()
