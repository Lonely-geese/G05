from pathlib import Path
from collections import Counter, defaultdict
import json, csv
import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/mnt/cfs/l1x67e/kele/data/recap_assisted_success_rollouts')
OUT = Path(__file__).resolve().parent
counts = defaultdict(Counter)
metrics = defaultdict(list)
rows, issues = [], []
total_bytes = 0

def stats(x):
    a = np.asarray(x)
    return dict(zip(['min', 'p50', 'p95', 'p99', 'max'], np.quantile(a, [0,.5,.95,.99,1]).tolist())) if len(a) else {}

for k, p in enumerate(sorted(ROOT.glob('rollout_*'))):
    info = json.loads((p/'meta/info.json').read_text())
    ep = json.loads((p/'meta/episodes.jsonl').read_text())
    man = json.loads((p/'recap_manifest.json').read_text())
    tab = pq.read_table(p/'data/chunk-000/episode_000000.parquet').to_pydict()
    aud = pq.read_table(p/'audit/frames/chunk-000/episode_000000.parquet').to_pydict()
    n = len(tab['frame_index'])
    frames, source_events, commands = [], [], set()
    for line in (p/'recap.jsonl').open():
        d = json.loads(line)
        e = d.get('event', 'MISSING'); counts['events'][e] += 1
        if e == 'frame': frames.append(d)
        elif e == 'source': source_events.append(d)
        elif e == 'executed_command': commands.add(d['stamp_ns'])
        elif e == 'prediction':
            metrics['inference_latency_s'].append(d['latency_s'])
            counts['prediction_shapes'][str(np.asarray(d['model']).shape)] += 1
        elif e == 'config':
            for key in ['policy_hz','output_hz','execution_mode','execute_steps','slowdown']:
                counts[key][str(d['args'].get(key))] += 1
    s = np.array(tab['observation.state']); a = np.array(tab['action'])
    eq = np.all(s == a, axis=1)
    counts['state_action']['equal_rows'] += int(eq.sum())
    counts['state_action']['nonfinite_values'] += int((~np.isfinite(s)).sum()+(~np.isfinite(a)).sum())
    for key in ['outcome','human_intervened','raw_action_semantics','supervision','finalized','error','manual_mode']:
        counts[key][str(man.get(key))] += 1
    counts['checkpoint'][str(man.get('policy_metadata',{}).get('checkpoint'))] += 1
    for key in ['operator_verdict','quality_status','structurally_valid','quality_accounting_complete','data_valid']:
        counts[key][str(ep.get(key))] += 1
    counts['tasks'][str(ep['tasks'])] += 1
    counts['fps'][str(info['fps'])] += 1
    counts['quality_event_types'].update(ep.get('quality_counts',{}))
    counts['parquet_columns'][str(list(tab))] += 1
    counts['feature_shapes'][str({key:val['shape'] for key,val in info['features'].items()})] += 1
    sources = Counter(f['source'] for f in frames)
    counts['frame_sources'].update(sources)
    same_segment_h32 = 0
    human_segments = 0
    for i,f in enumerate(frames):
        if f['source'] == 'HUMAN':
            if i == 0 or frames[i-1]['segment_id'] != f['segment_id']: human_segments += 1
            if i+32 < len(frames) and all(g['source']=='HUMAN' and g['segment_id']==f['segment_id'] for g in frames[i:i+33]): same_segment_h32 += 1
        cmd = f.get('policy_command_joints_deg')
        counts['command_presence'][f['source']+('_present' if cmd is not None else '_missing')] += 1
        stamp = f.get('policy_command_stamp_ns')
        if stamp is not None:
            counts['command_join']['matched' if stamp in commands else 'missing'] += 1
            metrics['command_age_ms_'+f['source']].append((f['stamp_ns']-stamp)/1e6)
    bad = []
    if n != info['total_frames'] or n != ep['length'] or n != len(frames) or n != len(aud['frame_index']): bad.append('row_count')
    if tab['frame_index'] != list(range(n)): bad.append('frame_index')
    if [f['frame_index'] for f in frames] != tab['frame_index']: bad.append('recap_frame_join')
    if [f['observation_sequence'] for f in frames] != aud['observation_sequence']: bad.append('sequence_join')
    if aud['frame_index'] != tab['frame_index']: bad.append('audit_frame_join')
    if not np.allclose(tab['timestamp'],np.arange(n)/info['fps'],atol=1e-5):bad.append('nominal_timestamp')
    ts = np.array([f['stamp_ns'] for f in frames], dtype=np.int64)
    dt = np.diff(ts)/1e6
    metrics['frame_interval_ms'].extend(dt.tolist())
    counts['time']['nonpositive_intervals'] += int((dt<=0).sum())
    counts['time']['intervals_over_50ms'] += int((dt>50).sum())
    for key in ['state_age_ns','command_age_ns','camera_skew_ns']:
        metrics[key+'_ms'].extend((np.array(aud[key])/1e6).tolist())
    qm = np.array(aud['quality_mask'])
    counts['quality_masks'].update(map(str,qm.tolist()))
    for src in sources:
        ix = np.array([f['source']==src for f in frames])
        counts['quality_nonzero_by_source'][src] += int((qm[ix]!=0).sum())
    counts['quality']['camera_skew_over_45ms'] += int((np.array(aud['camera_skew_ns'])>45000000).sum())
    gaps = np.diff(aud['observation_sequence'])
    counts['time']['missing_observation_sequences'] += int(np.maximum(gaps-1,0).sum())
    for camera in ['headDepthCamera','leftCamera','rightCamera']:
        seq = np.array(aud['camera_sample_sequence.'+camera])
        counts['camera_repeated_adjacent'][camera] += int((np.diff(seq)==0).sum())
    duration = (ts[-1]-ts[0])/1e9 if len(ts) else 0
    row = {'rollout':p.name,'frames':n,'nominal_seconds':n/info['fps'],'recorded_span_seconds':duration,'human_segments':human_segments,'human_h32_shift1_windows':same_segment_h32,'quality_nonzero_frames':int((qm!=0).sum()),'source_gaps':int(np.maximum(gaps-1,0).sum()),'action_equal_state':bool(eq.all()),**{key+'_frames':sources[key] for key in ['HUMAN','POLICY','TRANSITION']}}
    rows.append(row)
    metrics['episode_seconds'].append(n/info['fps'])
    if bad:issues.append({'rollout':p.name,'issues':bad})
    for f in p.rglob('*'):
        if f.is_file():total_bytes += f.stat().st_size
    if (k+1)%40 == 0:print('audited',k+1,flush=True)

summary = {'root':str(ROOT),'episodes':len(rows),'frames':sum(r['frames'] for r in rows),'bytes':total_bytes,'nominal_hours':sum(r['nominal_seconds'] for r in rows)/3600,'human_segments':sum(r['human_segments'] for r in rows),'human_h32_shift1_windows':sum(r['human_h32_shift1_windows'] for r in rows),'counts':dict(counts),'metrics':{k:stats(v) for k,v in metrics.items()},'issues':issues}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
with (OUT/'episodes.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
