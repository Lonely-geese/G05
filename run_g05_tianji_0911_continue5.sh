#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/cfs/l1x67e/kele/GalaxeaVLA
WORKSPACE=/mnt/cfs/l1x67e/kele/GenieSim3.0-Dataset/checkpoints
PARENT_RUN="$WORKSPACE/g05_runs/r1pro/tianji_0907_0908_0910_shift1_right_arm_from_0907_0908_ep5_8gpu_gbs128_ep5"
# Fine-tune from the final model weights, but start a fresh optimizer/scheduler.
WEIGHT="$PARENT_RUN/checkpoints/step_21505.pt"
PROCESSOR="$WORKSPACE/qwen3_5_2b_base_processor"
DATA_CONFIG="$REPO/configs/data/tianji_0907_0908_0910_shift1_right_arm.yaml"
NPP_LIB="$REPO/.venv/lib/python3.10/site-packages/nvidia/npp/lib"
NCCL_LIB="$WORKSPACE/nccl_runtime/nvidia/nccl/lib/libnccl.so.2"
# Reuse the parent run normalization statistics for the same dataset.
DATA_STATS="$PARENT_RUN/dataset_stats.json"

for required in "$WEIGHT" "$PROCESSOR" "$DATA_CONFIG" "$DATA_STATS" "$NPP_LIB" "$NCCL_LIB"; do
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
export EXP_NAME="${EXP_NAME:-tianji_0907_0908_0910_shift1_right_arm_from_0911_step21505_8gpu_gbs128_extra5ep}"

if [[ -e "$G05_OUTPUT_DIR/r1pro/$EXP_NAME" ]]; then
  echo "Output already exists: $G05_OUTPUT_DIR/r1pro/$EXP_NAME" >&2
  exit 1
fi
cd "$REPO"
# 550,438 train samples / (8 GPUs * 16 samples/GPU) => 4,301 optimizer steps/epoch.
# Global batch size = 8 * 16 * 1 accumulation step = 128.
exec bash scripts/run/finetune.sh 8 r1pro \
  --mixture tianji_0907_0908_0910_shift1_right_arm \
  "model.pretrained_ckpt=$WEIGHT" \
  "model.model_arch.hf_processor_path=$PROCESSOR" \
  "datastatics_path=$DATA_STATS" \
  model.enable_bf16_training=true \
  model.batch_size=16 \
  model.grad_accumulation_steps=1 \
  model.num_workers=8 \
  resume_ckpt=null \
  model.max_epochs=5 \
  checkpointing_steps=4301 \
  eval_steps=4301 \
  logger.mode=offline \
  "$@"
