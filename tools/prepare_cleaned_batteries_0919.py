"""Compute full-data normalization and validate the fixed-left-arm training setup."""
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
        cfg = compose(config_name='train', overrides=['task=cleaned_batteries_0919'])
    path = Path(cfg.datastatics_path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite or reuse existing statistics: {path}')
    processor = build_processors(cfg)
    dataset = instantiate_dataset(cfg, is_training_set=True)
    assert len(dataset) == 117689, len(dataset)
    for ds in dataset.datasets:
        assert ds.stats_downsample_rate == 1
        assert ds.val_set_proportion == 0
    filt = processor['galaxea_r1pro'].action_filter
    assert filt.inactive_keys == {'left_arm', 'left_gripper'}
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
        'datasets': ['/mnt/cfs/7rnh3z/kele/0919/mt/cleaned_batteries'],
        'frames': len(dataset), 'episodes': 227,
        'stats_downsample_rate': 1, 'action_horizon': 32,
        'relative_joint_keys': ['left_arm','right_arm'],
        'inactive_action_keys': sorted(filt.inactive_keys),
        'source': 'fresh computation from all cleaned parquet rows; no checkpoint statistics',
        'stats_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_name('norm_provenance.json').write_text(json.dumps(provenance, indent=2))
    processor.set_normalizer_from_stats(stats)
    dataset.set_processor(processor)
    from collections import Counter
    import pyarrow.parquet as pq
    task_counts = Counter()
    raw_parts = {'observation.state': [], 'action': []}
    data_root = Path('/mnt/cfs/7rnh3z/kele/0919/mt/cleaned_batteries')
    tasks = {x['task_index']: x['task'] for x in map(json.loads, (data_root / 'meta/tasks.jsonl').read_text().splitlines())}
    for f in sorted(data_root.glob('data/chunk-*/*.parquet')):
        t = pq.read_table(f)
        task_counts.update(t['task_index'].to_pylist())
        assert all(p == tasks[i] for p, i in zip(t['prompt'].to_pylist(), t['task_index'].to_pylist()))
        for key in raw_parts:
            raw_parts[key].append(np.stack(t[key].to_pylist()))
    raw_stats = {}
    for key, parts in raw_parts.items():
        x = np.concatenate(parts)
        raw_stats[key] = {k: v.tolist() for k, v in dict(mean=x.mean(0), std=x.std(0), min=x.min(0), max=x.max(0), q01=np.quantile(x,.01,axis=0), q99=np.quantile(x,.99,axis=0)).items()}
    path.with_name('raw_30d_stats.json').write_text(json.dumps(raw_stats, indent=2))
    assert set(task_counts) == {0, 1} and sum(task_counts.values()) == 117689
    ds = dataset.datasets[0]
    boundary = int(ds.segment_ends[0]) - 1
    saved_processor = ds.processor
    ds.processor = None
    cut_sample = ds[boundary]
    assert cut_sample['action_is_pad'][1:].all()
    assert not cut_sample['action_is_pad'][0]
    for value in cut_sample['action'].values():
        assert torch.equal(value, value[:1].expand_as(value))
    ds.processor = saved_processor
    provenance.update(task_counts=dict(task_counts), continuous_segments=len(ds.segment_ends), sequence_boundary_policy='clamp at audited segment end and mask padded steps; same boundaries for norm', cameras=['head_rgb', 'left_wrist_rgb', 'right_wrist_rgb'])
    path.with_name('norm_provenance.json').write_text(json.dumps(provenance, indent=2))
    sample = dataset[0]
    for index in [0, 50000, 100000, 117688]:
        check_sample=dataset[index]
        check_mask=check_sample['action_op_mask']
        assert check_mask[..., 10:17].all() and check_mask[..., 19].all()
        assert not check_mask[..., :10].any()
    mask = sample['action_op_mask']
    assert mask.shape[-1] == 27, mask.shape
    assert mask[..., 10:17].all() and mask[..., 19].all(), mask
    assert not mask[..., :10].any(), mask
    path.with_name('preflight_passed.json').write_text(json.dumps({'passed': True, 'norm_sha256': provenance['stats_sha256'], 'tasks': tasks}, indent=2))
    print('FULL DATA NORMALIZATION AND LEFT-ARM MASK VERIFIED', flush=True)
    print(json.dumps(provenance, indent=2), flush=True)
    print('Sample action mask:', mask.tolist(), flush=True)


if __name__ == '__main__':
    main()
