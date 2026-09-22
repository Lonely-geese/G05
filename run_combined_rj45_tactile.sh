#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
# G05 may share GalaxeaVLA's environment, but must import THIS branch's source.
PYTHON_BIN="${G05_PYTHON:-$REPO/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" && -z "${G05_PYTHON:-}" ]]; then
  PYTHON_BIN="$REPO/../GalaxeaVLA/.venv/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo 'Set G05_PYTHON to a Python environment with G05 dependencies.' >&2
  exit 1
fi
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG="${G05_NCCL_DEBUG:-WARN}"
# This host has CUDA 13 globally; its TorchCodec/FFmpeg requires CUDA 12 NPP.
# Reuse the existing CUDA 12 runtime without changing the system installation.
TACTILE_CUDA_LIB="${G05_CUDA_LIB:-$REPO/../GalaxeaVLA/.cache/cuda-12.8/usr/local/cuda-12.8/targets/x86_64-linux/lib}"
if [[ -d "$TACTILE_CUDA_LIB" ]]; then
  export LD_LIBRARY_PATH="$TACTILE_CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-$REPO/outputs}"
export G05_TACTILE_STATS="${G05_TACTILE_STATS:-$REPO/artifacts/combined_rj45_tactile_shift3/dataset_stats.json}"
MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
  prepare) exec "$PYTHON_BIN" tools/prepare_combined_rj45_tactile.py "$@" ;;
  check) exec "$PYTHON_BIN" tools/prepare_combined_rj45_tactile.py --check "$@" ;;
  smoke)
    export DRY_RUN=1 DRY_RUN_STEPS="${DRY_RUN_STEPS:-2}"
    NPROC="${NPROC:-1}"
    PER_GPU="${PER_GPU:-1}"
    ;;
  train)
    export DRY_RUN=0
    NPROC="${NPROC:-8}"
    PER_GPU="${PER_GPU:-4}"
    ;;
  *) echo 'Usage: bash run_combined_rj45_tactile.sh [prepare|check|smoke|train] [overrides...]' >&2; exit 2 ;;
esac
if [[ ! -s "$G05_TACTILE_STATS" ]]; then
  echo 'Run: bash run_combined_rj45_tactile.sh prepare' >&2
  exit 1
fi
"$PYTHON_BIN" tools/prepare_combined_rj45_tactile.py --check --stats-only
export EXP_NAME="${EXP_NAME:-ftp1_tactile_${MODE}_$(date -u +%Y%m%d_%H%M%S)}"
if [[ -e "$G05_OUTPUT_DIR/combined_rj45_tactile_shift3/$EXP_NAME" ]]; then
  echo "Refusing to reuse run directory for EXP_NAME=$EXP_NAME" >&2
  exit 1
fi
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NPROC" \
  scripts/finetune.py task=combined_rj45_tactile_shift3 model.batch_size="$PER_GPU" \
  batch_size_val=1 "$@"
