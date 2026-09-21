#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_g05_tianji_0903.sh smoke       # 1 train step + 1 eval step
#   ./run_g05_tianji_0903.sh train       # full fine-tuning
#
# Full-training knobs can be overridden with environment variables:
#   GPU_COUNT=8 BATCH_SIZE=8 NUM_WORKERS=8 MAX_EPOCHS=10 EXP_NAME=my_run \
#     ./run_g05_tianji_0903.sh train

MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

REPO=/mnt/cfs/l1x67e/kele/GalaxeaVLA
WORKSPACE=/mnt/cfs/l1x67e/kele/GenieSim3.0-Dataset/checkpoints
WEIGHT="$WORKSPACE/G05/model_state_dict.pt"
PROCESSOR="$WORKSPACE/qwen3_5_2b_base_processor"
STATS="$WORKSPACE/g05_data/tianji_0903_stats.json"
DATA_CONFIG="$REPO/configs/data/tianji_0903_full.yaml"
NPP_LIB="$REPO/.venv/lib/python3.10/site-packages/nvidia/npp/lib"
NCCL_LIB="$WORKSPACE/nccl_runtime/nvidia/nccl/lib/libnccl.so.2"

for required in "$WEIGHT" "$PROCESSOR" "$STATS" "$DATA_CONFIG" "$NPP_LIB" "$NCCL_LIB"; do
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
export LD_LIBRARY_PATH="$NPP_LIB:${LD_LIBRARY_PATH:-}"
# PyTorch's bundled NCCL 2.26.2 crashes on this host's SM120 GPUs. The isolated
# NCCL 2.31.2 runtime passes the eight-GPU all-reduce test with P2P enabled.
export LD_PRELOAD="$NCCL_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
# FA4 4.0.0b15 + CUTLASS DSL 4.5.2 fails to compile the vision kernel on
# this SM120 GPU. This compatibility shim activates G0.5's built-in SDPA path.
export PYTHONPATH="$WORKSPACE/compat_sdpa:${PYTHONPATH:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

cd "$REPO"

COMMON_ARGS=(
  --mixture tianji_0903_full
  "model.pretrained_ckpt=$WEIGHT"
  "model.model_arch.hf_processor_path=$PROCESSOR"
  "datastatics_path=$STATS"
  model.enable_bf16_training=true
)

case "$MODE" in
  smoke)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    exec bash scripts/run/finetune.sh 1 r1pro \
      --dry-run --overfit_batch 1 \
      "${COMMON_ARGS[@]}" \
      model.batch_size=1 model.num_workers=1 batch_size_val=1 \
      "$@"
    ;;
  train)
    GPU_COUNT="${GPU_COUNT:-8}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    NUM_WORKERS="${NUM_WORKERS:-8}"
    MAX_EPOCHS="${MAX_EPOCHS:-10}"
    export EXP_NAME="${EXP_NAME:-tianji_0903_g05_$(date +%Y%m%d_%H%M%S)}"
    exec bash scripts/run/finetune.sh "$GPU_COUNT" r1pro \
      "${COMMON_ARGS[@]}" \
      "model.batch_size=$BATCH_SIZE" \
      "model.num_workers=$NUM_WORKERS" \
      "model.max_epochs=$MAX_EPOCHS" \
      logger.mode=offline \
      "$@"
    ;;
  *)
    echo "Usage: $0 {smoke|train} [Hydra overrides...]" >&2
    exit 2
    ;;
esac
