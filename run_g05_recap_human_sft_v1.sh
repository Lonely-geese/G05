#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE=/mnt/cfs/l1x67e/kele/GenieSim3.0-Dataset/checkpoints
# User-selected 0912 morning model, also used for 194 collection episodes.
PARENT_RUN="$WORKSPACE/g05_runs/r1pro/tianji_0907_0908_0910_shift1_right_arm_from_0911_step21505_8gpu_gbs128_extra5ep_formal"
WEIGHT="${RECAP_INIT_WEIGHT:-$PARENT_RUN/checkpoints/step_21505.pt}"
DATA_STATS="${RECAP_INIT_STATS:-$PARENT_RUN/dataset_stats.json}"
PROCESSOR="$WORKSPACE/qwen3_5_2b_base_processor"
DATA_CONFIG="$REPO/configs/data/tianji_recap_human_sft_v1.yaml"
PREPARED=/mnt/cfs/l1x67e/kele/data/recap_human_sft_shift1_v1
NPP_LIB="$REPO/.venv/lib/python3.10/site-packages/nvidia/npp/lib"
NCCL_LIB="$WORKSPACE/nccl_runtime/nvidia/nccl/lib/libnccl.so.2"

for required in "$WEIGHT" "$DATA_STATS" "$PROCESSOR" "$DATA_CONFIG" "$PREPARED/READY.json" "$NPP_LIB" "$NCCL_LIB"; do
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
export PYTHONPATH="$WORKSPACE/compat_sdpa:$REPO/src:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export DRY_RUN=0 # The explicit --dry-run option below can set this to 1.
export EXP_NAME="${EXP_NAME:-tianji_recap_human_sft_v1_from_0912_morning_mix30_lr5e6_steps2000}"

if [[ -e "$G05_OUTPUT_DIR/r1pro/$EXP_NAME" ]]; then
  echo "Output already exists: $G05_OUTPUT_DIR/r1pro/$EXP_NAME; choose a new EXP_NAME." >&2
  exit 1
fi
cd "$REPO"
exec bash scripts/run/finetune.sh "${RECAP_GPUS:-8}" r1pro \
  --mixture tianji_recap_human_sft_v1 \
  "model.pretrained_ckpt=$WEIGHT" \
  "model.model_arch.hf_processor_path=$PROCESSOR" \
  "datastatics_path=$DATA_STATS" \
  model.enable_bf16_training=true \
  model.batch_size=16 \
  model.grad_accumulation_steps=1 \
  model.num_workers=8 \
  resume_ckpt=null \
  model.max_epochs=null \
  model.max_steps=2000 \
  model.learning_rate=5.0e-6 \
  model.warmup_ratio=null \
  model.warmup_steps=100 \
  model.lr_scheduler_type=cosine \
  checkpointing_steps=500 \
  eval_steps=500 \
  logger.mode=offline \
  "$@"
