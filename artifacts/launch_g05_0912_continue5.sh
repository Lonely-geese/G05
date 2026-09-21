#!/usr/bin/env bash
set -uo pipefail
cd /mnt/cfs/l1x67e/kele/GalaxeaVLA
unset DRY_RUN MAX_DATASETS MAX_EMBODIMENTS EXP_NAME WORLD_SIZE RANK LOCAL_RANK
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
printf 'Started: %s\n' "$(date -Is)"
bash ./run_g05_tianji_0912_continue5.sh 2>&1 | tee artifacts/g05_0912_extra5_lr2e5_train.log
train_status=${PIPESTATUS[0]}
printf 'Finished: %s; exit status: %s\n' "$(date -Is)" "$train_status" | tee artifacts/g05_0912_extra5_lr2e5_exit_status.txt
exit "$train_status"
