"""Compute full-data normalization and validate the fixed-right-arm training setup."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from g05.utils.config.config_resolvers import register_default_resolvers
from g05.utils.data.processor_utils import build_processors, instantiate_dataset
from g05.utils.data.normalizer import save_dataset_stats_to_json


def main():
    root = Path(__file__).resolve().parents[1]
    register_default_resolvers()
    with initialize_config_dir(str(root / 'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=['task=cleaned_rj45'])
    path = Path(cfg.datastatics_path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite or reuse existing statistics: {path}')
    processor = build_processors(cfg)
    dataset = instantiate_dataset(cfg, is_training_set=True)
    assert len(dataset) == 535558, len(dataset)
    for ds in dataset.datasets:
        assert ds.stats_downsample_rate == 1
        assert ds.val_set_proportion == 0
    filt = processor['galaxea_r1pro'].action_filter
    assert filt.inactive_keys == {'right_arm', 'right_gripper'}
    assert filt.hold_inactive_parts
    stats = dataset.get_dataset_stats(processor)
    def check(value):
        if isinstance(value, dict):
            for v in value.values(): check(v)
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            assert np.isfinite(np.asarray(value)).all(), 'Non-finite normalization statistic'
    check(stats)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_stats_to_json(stats, path)
    provenance = {
        'dataset': '/mnt/cfs/7rnh3z/kele/0917/cleaned_rj45',
        'frames': len(dataset), 'episodes': 1286,
        'stats_downsample_rate': 1, 'action_horizon': 32,
        'relative_joint_keys': ['left_arm','right_arm'],
        'inactive_action_keys': sorted(filt.inactive_keys),
        'source': 'fresh computation from all cleaned parquet rows; no checkpoint statistics',
        'stats_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_name('norm_provenance.json').write_text(json.dumps(provenance, indent=2))
    processor.set_normalizer_from_stats(stats)
    dataset.set_processor(processor)
    sample = dataset[0]
    mask = sample['action_op_mask']
    assert mask.shape[-1] == 27, mask.shape
    assert mask[..., :7].all() and mask[..., 9].all(), mask
    assert not mask[..., 10:20].any(), mask
    print('FULL DATA NORMALIZATION AND RIGHT-ARM MASK VERIFIED', flush=True)
    print(json.dumps(provenance, indent=2), flush=True)
    print('Sample action mask:', mask.tolist(), flush=True)


if __name__ == '__main__':
    main()
