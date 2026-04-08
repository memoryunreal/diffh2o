#!/bin/bash
# Wait for PID 830215 (luojingnan's process) to disappear, then launch train_v4_autoreg_ddp.sh
# Checks every 10 minutes

TARGET_PID=830215
INTERVAL=600  # 10 minutes
SCRIPT="/hdd0/lizhe/code/diffh2o/train_v4_autoreg_ddp.sh"
LOG="/hdd0/lizhe/code/diffh2o/wait_and_train.log"

echo "[$(date)] Monitoring PID $TARGET_PID every 10 min..." | tee -a "$LOG"

while true; do
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qw "$TARGET_PID"; then
        echo "[$(date)] PID $TARGET_PID still running. Next check in 10 min..." | tee -a "$LOG"
        sleep $INTERVAL
    else
        echo "[$(date)] PID $TARGET_PID gone. Launching training..." | tee -a "$LOG"
        nohup bash "$SCRIPT" > /hdd0/lizhe/code/diffh2o/train_v4_autoreg_ddp.log 2>&1 &
        echo "[$(date)] Training launched with PID $!" | tee -a "$LOG"
        exit 0
    fi
done
