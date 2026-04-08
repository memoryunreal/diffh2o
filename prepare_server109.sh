#!/bin/bash
# Transfer HIMO data, code, and checkpoints to server 109
#
# PREREQUISITES:
#   1. Set up SSH key: ssh-copy-id lizhe@172.18.36.109
#   2. Verify: ssh lizhe@172.18.36.109 hostname
#
# Usage: bash prepare_server109.sh

set -e
USER_HOST="lizhe@172.18.36.109"
DEST_BASE="/hdd0/lizhe/code/diffh2o"
DEST="${USER_HOST}:${DEST_BASE}"

echo "=========================================="
echo "  Transfer to Server 109"
echo "  $(date)"
echo "=========================================="

# Step 1: Create directories on server 109
echo ""
echo "[1/5] Creating directories on server 109..."
ssh ${USER_HOST} "mkdir -p ${DEST_BASE}/{dataset/HIMO_V4_2o,dataset/HIMO_V4_3o,save/himo_2o_v4_ar,save/himo_3o_v4_ar}"

# Step 2: Sync code (exclude large files)
echo ""
echo "[2/5] Syncing code..."
rsync -avz --progress \
  --exclude='save/' --exclude='dataset/' --exclude='wandb/' \
  --exclude='__pycache__/' --exclude='.git/' --exclude='*.pyc' \
  --exclude='*.pt' --exclude='*.tar' \
  /hhd4/lizhe/code/diffh2o/ ${DEST}/

# Step 3: Sync HIMO V4 preprocessed data
echo ""
echo "[3/5] Syncing HIMO V4 preprocessed data..."
rsync -avz --progress /hhd4/lizhe/code/diffh2o/dataset/HIMO_V4_2o/ ${DEST}/dataset/HIMO_V4_2o/
rsync -avz --progress /hhd4/lizhe/code/diffh2o/dataset/HIMO_V4_3o/ ${DEST}/dataset/HIMO_V4_3o/

# Step 4: Sync pretrained checkpoints for AR fine-tuning
echo ""
echo "[4/5] Syncing checkpoints..."
# 2o: use 200k non-AR checkpoint as init
scp /hhd4/lizhe/code/diffh2o/save/himo_2o_v4/model000200000.pt ${DEST}/save/himo_2o_v4_ar/model_init.pt
# 3o: use 150k non-AR checkpoint as init
scp /hhd4/lizhe/code/diffh2o/save/himo_3o_v4/model000150000.pt ${DEST}/save/himo_3o_v4_ar/model_init.pt

# Step 5: Verify
echo ""
echo "[5/5] Verifying transfer..."
ssh ${USER_HOST} "
echo 'Code files:' && ls ${DEST_BASE}/train/train_flow.py && \
echo 'HIMO 2o data:' && ls ${DEST_BASE}/dataset/HIMO_V4_2o/Mean_flow.npy && \
echo 'HIMO 3o data:' && ls ${DEST_BASE}/dataset/HIMO_V4_3o/Mean_flow.npy && \
echo '2o checkpoint:' && ls -lh ${DEST_BASE}/save/himo_2o_v4_ar/model_init.pt && \
echo '3o checkpoint:' && ls -lh ${DEST_BASE}/save/himo_3o_v4_ar/model_init.pt && \
echo 'GPUs available:' && nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
"

echo ""
echo "=========================================="
echo "  Transfer complete!"
echo "  Now on server 109, run:"
echo "    cd ${DEST_BASE}"
echo "    bash train_himo_v4_ar_ddp.sh"
echo "=========================================="
