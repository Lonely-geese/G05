#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
source .venv/bin/activate
source .env
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 WANDB_MODE=offline OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TASK=combined_rj45_0917_0918
MODE="${1:-train}"
GBS="${GBS:-128}"
case "$GBS" in
  256) PER_GPU=32; EPOCH_STEPS=3717 ;;
  128) PER_GPU=16; EPOCH_STEPS=7433 ;;
  *) echo 'GBS must be 256 or 128' >&2; exit 2 ;;
esac
case "$MODE" in
  smoke) export DRY_RUN=1 DRY_RUN_STEPS=3 ;;
  train) export DRY_RUN=0 ;;
  *) echo 'Usage: bash run_combined_rj45.sh [smoke|train]' >&2; exit 2 ;;
esac
export EXP_NAME="${EXP_NAME:-left_arm_base_gbs${GBS}_ep10_${MODE}}"
if [[ -e "$G05_OUTPUT_DIR/$TASK/$EXP_NAME" ]]; then
  echo "Refusing to overwrite existing run: $G05_OUTPUT_DIR/$TASK/$EXP_NAME" >&2
  exit 1
fi
python - <<'PY'
import hashlib,json,torch
from pathlib import Path
p=Path('artifacts/combined_rj45_0917_0918/dataset_stats.json')
r=json.loads(p.with_name('norm_provenance.json').read_text())
assert r['frames']==951339 and r['episodes']==2615 and r['stats_downsample_rate']==1
assert hashlib.sha256(p.read_bytes()).hexdigest()==r['stats_sha256']
assert torch.cuda.device_count()==8
assert Path('base_checkpoint/g05-base/checkpoints/model_state_dict.pt').is_file()
PY
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/finetune.py task="$TASK" model.batch_size="$PER_GPU" \
  model.max_epochs=10 model.grad_accumulation_steps=1 resume_ckpt=null \
  checkpointing_steps="$EPOCH_STEPS" eval_steps="$EPOCH_STEPS"
