#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
source .venv/bin/activate
source .env
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
MODE="${1:-train}"
case "$MODE" in
  smoke)
    export DRY_RUN=1
    export DRY_RUN_STEPS="${DRY_RUN_STEPS:-3}"
    export EXP_NAME="${EXP_NAME:-cleaned_rj45_base_gbs128_smoke3}"
    ;;
  train)
    export DRY_RUN=0
    export EXP_NAME="${EXP_NAME:-cleaned_rj45_left_arm_base_gbs128_ep5_20260918}"
    ;;
  *) echo 'Usage: bash run_cleaned_rj45.sh [smoke|train]' >&2; exit 2 ;;
esac
if [[ -e "$G05_OUTPUT_DIR/cleaned_rj45/$EXP_NAME" ]]; then
    echo "Output already exists; choose a new EXP_NAME: $G05_OUTPUT_DIR/cleaned_rj45/$EXP_NAME" >&2
    exit 1
fi
python - <<'PY'
import hashlib,json,torch
from pathlib import Path
p=Path('artifacts/cleaned_rj45_setup/dataset_stats.json')
r=json.loads(p.with_name('norm_provenance.json').read_text())
assert r['frames']==535558 and r['stats_downsample_rate']==1
assert hashlib.sha256(p.read_bytes()).hexdigest()==r['stats_sha256']
assert torch.cuda.device_count()==8, 'This configuration requires eight GPUs for global batch size 128'
PY
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    scripts/finetune.py task=cleaned_rj45
