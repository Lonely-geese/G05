"""Detached training supervisor; records exit status and expected epoch checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts/combined_rj45_0917_0918'
GBS = int(os.environ.get('GBS', '128'))
assert GBS in (128, 256)
STEPS = {256: 3717, 128: 7433}[GBS]
NAME = os.environ.get('EXP_NAME', f'left_arm_base_gbs{GBS}_ep10_20260919')
OUTPUT = ROOT / 'outputs/combined_rj45_0917_0918' / NAME
STATUS = ARTIFACTS / 'training_status.json'


def save(state):
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(STATUS)


if __name__ == '__main__':
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    state = dict(status='starting', supervisor_pid=os.getpid(), started_unix=time.time(),
                 global_batch_size=GBS, per_gpu_batch_size=GBS // 8, gpu_count=8,
                 epochs=10, steps_per_epoch=STEPS, total_steps=STEPS * 10,
                 output_dir=str(OUTPUT), log=str(ARTIFACTS / 'train.log'),
                 checkpoint_by_epoch={str(i): str(OUTPUT / 'checkpoints' / f'step_{i*STEPS}.pt')
                                      for i in range(1, 11)})
    save(state)
    env = dict(os.environ, GBS=str(GBS), EXP_NAME=NAME)
    with (ARTIFACTS / 'train.log').open('x') as log:
        process = subprocess.Popen(['bash', str(ROOT / 'run_combined_rj45.sh'), 'train'],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        state.update(status='running', launcher_pid=process.pid)
        save(state)
        code = process.wait()
    checkpoints_exist = all(Path(p).is_file() and Path(p).stat().st_size > 0
                            for p in state['checkpoint_by_epoch'].values())
    state.update(status='complete' if code == 0 and checkpoints_exist else 'failed',
                 returncode=code, finished_unix=time.time(),
                 all_epoch_checkpoints_exist=checkpoints_exist)
    save(state)
