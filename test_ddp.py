#!/usr/bin/env python3
"""
Minimal DDP test to diagnose NCCL issues.

Usage:
    torchrun --nproc_per_node=4 --master_port=29500 test_ddp.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    # Init process group
    print(f"[PID {os.getpid()}] Initializing process group...")
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {local_rank}/{world_size}] Process group initialized on GPU {local_rank}")

    # Simple model
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    ).to(device)

    print(f"[Rank {local_rank}] Model created, wrapping with DDP...")
    model = DDP(model, device_ids=[local_rank])
    print(f"[Rank {local_rank}] DDP wrapped successfully")

    # Simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(10):
        x = torch.randn(8, 256, device=device)
        y = model(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if local_rank == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    print(f"[Rank {local_rank}] Test PASSED - DDP works correctly!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
