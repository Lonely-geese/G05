#!/usr/bin/env python3
"""Prepare indexed HUMAN-only shift1 supervision without re-encoding videos.

Backing episodes retain original frame numbers/timestamps and symlink videos.
Only CorrectionWindowDataset may train on them; generic loading is rejected.
The raw collection is never modified. Destination must not exist.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from g05.data.recap_windows import WindowRules, select_human_runs, shifted_human_actions


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in values))


def numeric_stats(values):
    a = np.asarray(values)
    return {**{k: np.atleast_1d(v).tolist() for k, v in {
        'min':a.min(axis=0), 'max':a.max(axis=0),
        'mean':a.mean(axis=0), 'std':a.std(axis=0)}.items()}, 'count':[len(a)]}


def inspect_rollout(path, rules):
    man = json.loads((path/'recap_manifest.json').read_text())
    info = json.loads((path/'meta/info.json').read_text())
    episode = json.loads((path/'meta/episodes.jsonl').read_text())
    required = {'finalized':True, 'outcome':'success', 'raw_action_semantics':'measured_state',
                'supervision':'human_segments_only', 'manual_mode':'drag', 'error':None}
    if any(man.get(k) != v for k,v in required.items()):
        raise ValueError(f"Unsupported manifest: {path}")
    if info['total_episodes'] != 1 or info['fps'] != 30:
        raise ValueError(f"Expected one 30Hz episode: {path}")
    if not all(episode.get(k) is True for k in ['structurally_valid','data_valid','quality_accounting_complete']):
        raise ValueError(f"Invalid episode: {path}")
    table = pq.read_table(path/'data/chunk-000/episode_000000.parquet')
    audit = pq.read_table(path/'audit/frames/chunk-000/episode_000000.parquet').to_pydict()
    frames, digest = [], hashlib.sha256()
    with (path/'recap.jsonl').open('rb') as f:
        for line in f:
            digest.update(line)
            d = json.loads(line)
            if d.get('event') == 'frame': frames.append(d)
    n = len(table)
    if n != info['total_frames'] or n != episode['length'] or len(frames) != n:
        raise ValueError(f"Frame count mismatch: {path}")
    if table['frame_index'].to_pylist() != list(range(n)):
        raise ValueError(f"Parquet frame_index mismatch: {path}")
    if not np.allclose(table['timestamp'].to_numpy(), np.arange(n)/30, atol=1e-5):
        raise ValueError(f"Unexpected nominal timestamps: {path}")
    state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    if not np.array_equal(state, action, equal_nan=True):
        raise ValueError(f"Action is not unshifted measured state: {path}")
    runs, counts = select_human_runs(frames, audit, state, rules)
    video_keys = [k for k,v in info['features'].items() if v['dtype']=='video']
    for key in video_keys:
        if not (path/f'videos/chunk-000/{key}/episode_000000.mp4').is_file():
            raise FileNotFoundError(f"Missing camera {key}: {path}")
    return {'path':str(path), 'rollout':path.name, 'table':table, 'states':state,
            'runs':runs, 'counts':counts, 'info':info, 'episode':episode,
            'recap_sha256':digest.hexdigest(), 'checkpoint':man.get('policy_metadata',{}).get('checkpoint')}


def export_split(root, items, rules):
    if not items:
        raise ValueError(f"Empty correction split: {root}")
    episodes, episode_stats, windows = [], [], []
    global_index = 0
    for ei, item in enumerate(items):
        path = Path(item['path']); table = item['table']; n = len(table)
        action = shifted_human_actions(item['states'], item['runs'])
        for name, values in [('action', action.tolist()), ('episode_index',[ei]*n),
                             ('index',list(range(global_index,global_index+n))), ('task_index',[0]*n)]:
            field = table.schema.field(name)
            table = table.set_column(table.schema.get_field_index(name), field, pa.array(values, type=field.type))
        chunk = ei//1000
        parquet = root/f'data/chunk-{chunk:03d}/episode_{ei:06d}.parquet'
        parquet.parent.mkdir(parents=True, exist_ok=True); pq.write_table(table, parquet)
        for key, feature in item['info']['features'].items():
            if feature['dtype'] != 'video': continue
            dest = root/f'videos/chunk-{chunk:03d}/{key}/episode_{ei:06d}.mp4'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(path/f'videos/chunk-000/{key}/episode_000000.mp4')
        ep = copy.deepcopy(item['episode']); ep.update(episode_index=ei, source_rollout=item['rollout'])
        episodes.append(ep)
        # Backing-file stats are accurate, but policy normalization MUST come
        # from the initialization checkpoint, not these unrestricted rows.
        stats = json.loads((path/'meta/episodes_stats.jsonl').read_text())['stats']
        for col in table.column_names:
            stats[col] = numeric_stats(table[col].to_pylist())
        episode_stats.append({'episode_index':ei, 'stats':stats})
        windows.append({'episode_index':ei, 'length':n, 'source_rollout':item['rollout'],
                        'source_path':item['path'], 'recap_sha256':item['recap_sha256'],
                        'checkpoint':item['checkpoint'], 'human_runs':item['runs'],
                        **item['counts']})
        global_index += n
    info = copy.deepcopy(items[0]['info'])
    info.pop('audit', None)
    info.update(total_episodes=len(items), total_frames=global_index,
                total_videos=3*len(items), total_chunks=(len(items)+999)//1000,
                splits={'train':f'0:{len(items)}'}, requires_correction_window_index=True)
    write_json(root/'meta/info.json', info)
    write_jsonl(root/'meta/episodes.jsonl', episodes)
    write_jsonl(root/'meta/episodes_stats.jsonl', episode_stats)
    write_jsonl(root/'meta/tasks.jsonl', [{'task_index':0, 'task':episodes[0]['tasks'][0]}])
    write_json(root/'meta/correction_windows.json', {'schema_version':1, 'horizon':rules.horizon,
        'action_shift':1, 'rules':rules.as_dict(), 'episodes':windows})
    return {'rollouts':len(items), 'backing_frames':global_index,
            'valid_windows':sum(x['counts']['valid_windows'] for x in items),
            'human_frames':sum(x['counts']['human_frames'] for x in items)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--config-output', type=Path, required=True)
    parser.add_argument('--val-rollouts', type=int, default=20)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--human-probability', type=float, default=0.30)
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--max-interval-ms', type=float, default=50)
    parser.add_argument('--max-state-age-ms', type=float, default=20)
    parser.add_argument('--max-camera-skew-ms', type=float, default=45)
    args = parser.parse_args()
    args.source = args.source.resolve(); args.output = args.output.resolve()
    if args.output.exists() or args.config_output.exists():
        parser.error('Output data/config must not exist; use a new versioned destination')
    if not 0 < args.human_probability < 1 or args.val_rollouts < 1:
        parser.error('Require 0 < human-probability < 1 and at least one validation rollout')
    rules = WindowRules(args.horizon, args.max_interval_ms, args.max_state_age_ms, args.max_camera_skew_ms)
    items, excluded = [], []
    for i, path in enumerate(sorted(args.source.glob('rollout_*'))):
        item = inspect_rollout(path, rules)
        if item['counts']['valid_windows']:
            items.append(item)
        else:
            excluded.append({'rollout':path.name, **item['counts'], 'reason':'no_valid_human_window'})
        if (i+1)%40 == 0: print(f'Inspected {i+1} rollouts', flush=True)
    if len(items) <= args.val_rollouts:
        raise ValueError('Not enough valid human rollouts for requested split')
    if len({tuple(i['episode']['tasks']) for i in items}) != 1:
        raise ValueError('This converter expects a single shared task')
    ids = [i['rollout'] for i in items]; random.Random(args.seed).shuffle(ids)
    val_ids = set(ids[:args.val_rollouts])
    splits = {k:[i for i in items if (i['rollout'] in val_ids)==(k=='val')] for k in ['train','val']}
    config = yaml.safe_load(args.base_config.read_text())
    if config['action_size'] != rules.horizon:
        raise ValueError('Base config horizon mismatch')
    args.output.mkdir(parents=True)
    summary = {k:export_split(args.output/k, v, rules) for k,v in splits.items()}
    old = config['embodiment_datasets']['galaxea_r1pro']
    old_root = Path(old['dataset_groups'][0]['dataset_dirs'][0])
    old_eps = [json.loads(l) for l in (old_root/'meta/episodes.jsonl').read_text().splitlines()]
    val_proportion = float(old.get('val_set_proportion',config['val_set_proportion']))
    train_ep_count = int(len(old_eps)*(1-val_proportion)) if val_proportion>=1e-6 else len(old_eps)
    old_length = sum(e['length'] for e in old_eps[:train_ep_count])
    human_weight = args.human_probability/(1-args.human_probability)*old_length/summary['train']['valid_windows']
    config['use_weight_for_sampling'] = True
    config['use_weight_normalization'] = True
    correction = copy.deepcopy(old)
    correction.update(type='g05.data.correction_window_dataset.CorrectionWindowDataset',
                      val_set_proportion=0.0,
                      dataset_groups=[{'weight':human_weight,'dataset_dirs':[str(args.output)]}])
    old['dataset_groups'][0]['weight'] = 1.0
    # Put correction first: the existing periodic evaluator consumes a single
    # validation batch, so it must not repeatedly evaluate only the old demo.
    config['embodiment_datasets'] = {'galaxea_r1pro_correction':correction, 'galaxea_r1pro':old}
    summary.update(source=str(args.source), rules=rules.as_dict(), seed=args.seed,
                   excluded=excluded, old_train_frames=old_length, human_weight=human_weight,
                   target_human_probability=args.human_probability,
                   validation_rollouts=sorted(val_ids),
                   backing_storage='shifted parquet; original video symlinks; mandatory window index',
                   semantic_video_review='not implied by structural filtering; see review artifact')
    write_json(args.output/'summary.json', summary)
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text('# Generated by prepare_recap_sft.py; weights use valid training windows.\n'+yaml.safe_dump(config, sort_keys=False))
    write_json(args.output/'READY.json', {'schema_version':1, 'config':str(args.config_output.resolve()),
                                        'summary_sha256':hashlib.sha256((args.output/'summary.json').read_bytes()).hexdigest()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
