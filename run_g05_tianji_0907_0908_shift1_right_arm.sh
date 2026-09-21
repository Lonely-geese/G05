#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/cfs/l1x67e/kele/GalaxeaVLA
WORKSPACE=/mnt/cfs/l1x67e/kele/GenieSim3.0-Dataset/checkpoints
WEIGHT="$WORKSPACE/G05/model_state_dict.pt"
PROCESSOR="$WORKSPACE/qwen3_5_2b_base_processor"
DATA_CONFIG="$REPO/configs/data/tianji_0907_0908_shift1_right_arm.yaml"
NPP_LIB="$REPO/.venv/lib/python3.10/site-packages/nvidia/npp/lib"
NCCL_LIB="$WORKSPACE/nccl_runtime/nvidia/nccl/lib/libnccl.so.2"
# This filename is intentionally new: never reuse norm stats from 0907-only data.
DATA_STATS="$WORKSPACE/g05_runs/stats/tianji_0907_0908_shift1_right_arm_dataset_stats.json"

for required in "$WEIGHT" "$PROCESSOR" "$DATA_CONFIG" "$NPP_LIB" "$NCCL_LIB"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

source "$REPO/.venv/bin/activate"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$WORKSPACE/.hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export G05_OUTPUT_DIR="$WORKSPACE/g05_runs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LD_LIBRARY_PATH="$NPP_LIB:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$NCCL_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$WORKSPACE/compat_sdpa:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export EXP_NAME="${EXP_NAME:-tianji_0907_0908_shift1_right_arm_8gpu_gbs128_ep5_epoch_ckpt}"

cd "$REPO"
# Global batch size = 8 GPUs * 16 samples/GPU * 1 accumulation step = 128.
exec bash scripts/run/finetune.sh 8 r1pro \
  --mixture tianji_0907_0908_shift1_right_arm \
  "model.pretrained_ckpt=$WEIGHT" \
  "model.model_arch.hf_processor_path=$PROCESSOR" \
  "datastatics_path=$DATA_STATS" \
  model.enable_bf16_training=true \
  model.batch_size=16 \
  model.grad_accumulation_steps=1 \
  model.num_workers=8 \
  model.max_epochs=5 \
  checkpointing_steps=3249 \
  eval_steps=3249 \
  logger.mode=offline \
  "$@"
