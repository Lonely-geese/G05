#!/usr/bin/env bash
set -euo pipefail

# Fine-tune G0.5 on the cleaned/trimmed Tianji 0907 dataset.
# Defaults: physical GPUs 4-7, per-GPU micro-batch 16 with two-step
# gradient accumulation (effective batch size 32 per GPU), 5 epochs.

REPO=/mnt/cfs/l1x67e/kele/GalaxeaVLA
WORKSPACE=/mnt/cfs/l1x67e/kele/GenieSim3.0-Dataset/checkpoints
WEIGHT="$WORKSPACE/G05/model_state_dict.pt"
PROCESSOR="$WORKSPACE/qwen3_5_2b_base_processor"
DATA_CONFIG="$REPO/configs/data/tianji_0907_trimmed.yaml"
NPP_LIB="$REPO/.venv/lib/python3.10/site-packages/nvidia/npp/lib"
NCCL_LIB="$WORKSPACE/nccl_runtime/nvidia/nccl/lib/libnccl.so.2"
DATA_STATS="$WORKSPACE/g05_runs/r1pro/tianji_0907_trimmed_g05_joint_bs32x4_ep5/dataset_stats.json"

for required in "$WEIGHT" "$PROCESSOR" "$DATA_CONFIG" "$NPP_LIB" "$NCCL_LIB" "$DATA_STATS"; do
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export LD_LIBRARY_PATH="$NPP_LIB:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$NCCL_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$WORKSPACE/compat_sdpa:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export EXP_NAME="${EXP_NAME:-tianji_0907_trimmed_g05_joint_effbs32x4_ep5}"

cd "$REPO"

exec bash scripts/run/finetune.sh 4 r1pro \
  --mixture tianji_0907_trimmed \
  "model.pretrained_ckpt=$WEIGHT" \
  "model.model_arch.hf_processor_path=$PROCESSOR" \
  "datastatics_path=$DATA_STATS" \
  model.enable_bf16_training=true \
  model.batch_size=16 \
  model.grad_accumulation_steps=2 \
  model.num_workers=8 \
  model.max_epochs=5 \
  logger.mode=offline \
  "$@"
