#!/usr/bin/env python3
"""Independently verify every exported correction window against raw collection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stats',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    ref=json.loads(args.stats.read_text())['galaxea_r1pro']['action']['right_arm']
    result={'splits':{},'video_reference_checks':0,'right_arm_target_values':0,
            'outside_pretrained_min_max':0,'outside_pretrained_q01_q99':0,
            'max_absolute_right_arm_delta_deg':0.0}
    split_ids={}
    for split in ['train','val']:
        root=args.root/split
        m=json.loads((root/'meta/correction_windows.json').read_text())
        h=m['horizon'];rules=m['rules'];windows=0;split_ids[split]=set()
        for ep in m['episodes']:
            source=Path(ep['source_path']);ei=ep['episode_index']
            split_ids[split].add(ep['source_rollout'])
            raw=pq.read_table(source/'data/chunk-000/episode_000000.parquet')
            out=pq.read_table(root/f'data/chunk-{ei//1000:03d}/episode_{ei:06d}.parquet')
            states=np.array(raw['observation.state'].to_pylist(),dtype=np.float32)
            np.testing.assert_array_equal(states,np.array(out['observation.state'].to_pylist()))
            np.testing.assert_array_equal(raw['timestamp'].to_numpy(),out['timestamp'].to_numpy())
            frames=[];digest=hashlib.sha256()
            with (source/'recap.jsonl').open('rb') as f:
                for line in f:
                    digest.update(line);d=json.loads(line)
                    if d.get('event')=='frame':frames.append(d)
            assert digest.hexdigest()==ep['recap_sha256']
            audit=pq.read_table(source/'audit/frames/chunk-000/episode_000000.parquet').to_pydict()
            action=np.array(out['action'].to_pylist(),dtype=np.float32)
            for start,end in ep['human_runs']:
                selected=frames[start:end]
                assert all(f['source']=='HUMAN' for f in selected)
                assert len({f['segment_id'] for f in selected})==1
                assert all(selected[i+1]['observation_sequence']==selected[i]['observation_sequence']+1 for i in range(len(selected)-1))
                dt=np.diff([f['stamp_ns'] for f in selected])
                assert ((dt>0)&(dt<=rules['max_interval_ms']*1e6)).all()
                for col,threshold in [('state_age_ns',rules['max_state_age_ms']),('camera_skew_ns',rules['max_camera_skew_ms'])]:
                    x=np.array(audit[col][start:end]);assert ((x>=0)&(x<=threshold*1e6)).all()
                assert np.isin(audit['quality_mask'][start:end],rules['allowed_quality_masks']).all()
                assert np.isfinite(states[start:end]).all()
                np.testing.assert_array_equal(action[start:end-1],states[start+1:end])
                starts=np.arange(start,end-h)
                # Check full chunks, independently of dataset's index expansion.
                target_idx=starts[:,None]+np.arange(h)[None,:]
                np.testing.assert_array_equal(action[target_idx],states[target_idx+1])
                delta=action[target_idx,15:22]-states[starts,None,15:22]
                result['right_arm_target_values']+=delta.size
                result['outside_pretrained_min_max']+=int(((delta<np.array(ref['stepwise_min']))|(delta>np.array(ref['stepwise_max']))).sum())
                result['outside_pretrained_q01_q99']+=int(((delta<np.array(ref['stepwise_q01']))|(delta>np.array(ref['stepwise_q99']))).sum())
                result['max_absolute_right_arm_delta_deg']=max(result['max_absolute_right_arm_delta_deg'],float(np.abs(delta).max()))
                windows+=len(starts)
            for key in ['headDepthCamera','leftCamera','rightCamera']:
                video=root/f'videos/chunk-{ei//1000:03d}/observation.images.{key}/episode_{ei:06d}.mp4'
                assert video.resolve()==(source/f'videos/chunk-000/observation.images.{key}/episode_000000.mp4').resolve()
                assert video.is_file();result['video_reference_checks']+=1
        assert windows==sum(e['valid_windows'] for e in m['episodes'])
        result['splits'][split]={'rollouts':len(split_ids[split]),'verified_windows':windows}
    assert not split_ids['train']&split_ids['val']
    result['rollout_overlap']=0
    for name in ['outside_pretrained_min_max','outside_pretrained_q01_q99']:
        result[name+'_fraction']=result[name]/result['right_arm_target_values']
    result['passed']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
