#!/usr/bin/env python3
"""
DDP test with actual CRRFlowDiT_V3 model (skip_clip=True).

Usage:
    torchrun --nproc_per_node=4 --master_port=29500 test_ddp_v3.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.flow_dit_v3 import CRRFlowDiT_V3


def main():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print(f"World size: {dist.get_world_size()}")

    # Create V3 model WITHOUT CLIP
    model = CRRFlowDiT_V3(
        latent_dim=512, num_layers=8, num_heads=8, ff_dim=2048,
        skip_clip=True,
    ).to(device)

    if local_rank == 0:
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model: {params:.1f}M params (no CLIP)")

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    if local_rank == 0:
        print("DDP wrapped successfully")

    optimizer = torch.optim.AdamW(model.module.parameters_wo_clip(), lr=1e-4)

    # Training loop
    for step in range(5):
        x = torch.randn(4, 200, 267, device=device)
        t = torch.rand(4, device=device)
        y = {
            'text_emb': torch.randn(4, 512, device=device),
            'state_labels': torch.zeros(4, 200, 7, dtype=torch.long, device=device),
        }

        out = model(x, t, y=y)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if local_rank == 0:
            print(f"Step {step}: loss={loss.item():.4f}, GPU mem={torch.cuda.memory_allocated(device)/1e9:.1f}GB")

    if local_rank == 0:
        print("DDP V3 test PASSED!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
