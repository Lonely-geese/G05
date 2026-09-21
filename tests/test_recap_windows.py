import numpy as np
import pytest

from g05.data.recap_windows import (
    WindowRules, expand_window_starts, select_human_runs, shifted_human_actions,
)


def trajectory(n=12):
    frames = [dict(frame_index=i, observation_sequence=100+i, stamp_ns=i*33333333,
                   source='HUMAN', segment_id=2) for i in range(n)]
    audit = dict(frame_index=list(range(n)), observation_sequence=list(range(100,100+n)),
                 state_age_ns=[3_000_000]*n, camera_skew_ns=[20_000_000]*n,
                 quality_mask=[64]*n)
    state = np.arange(n*30, dtype=np.float32).reshape(n,30)
    return frames, audit, state


def test_shifted_targets_stay_inside_human_segment():
    frames, audit, states = trajectory()
    frames[0]['source'] = 'POLICY'
    frames[-1]['source'] = 'TRANSITION'
    runs, counts = select_human_runs(frames,audit,states,WindowRules(horizon=4))
    assert runs == [(1,11)]
    assert counts['valid_windows'] == 6
    actions = shifted_human_actions(states,runs)
    np.testing.assert_array_equal(actions[1:10], states[2:11])
    np.testing.assert_array_equal(actions[[0,10,11]], states[[0,10,11]])
    starts = expand_window_starts([dict(episode_index=0,length=12,human_runs=runs)],4)
    for t in starts:
        np.testing.assert_array_equal(actions[t:t+4], states[t+1:t+5])
        assert all(f['source']=='HUMAN' for f in frames[t:t+5])


@pytest.mark.parametrize('break_type',['source','segment','sequence','time','future_quality','unknown_mask','nan'])
def test_windows_never_cross_a_future_discontinuity(break_type):
    frames, audit, states = trajectory()
    if break_type == 'source':frames[6]['source']='POLICY'
    if break_type == 'segment':
        for f in frames[6:]:f['segment_id']=3
    if break_type == 'sequence':
        for f in frames[6:]:f['observation_sequence']+=1
        audit['observation_sequence']=[f['observation_sequence'] for f in frames]
    if break_type == 'time':
        for f in frames[6:]:f['stamp_ns']+=1000000000
    if break_type == 'future_quality':audit['camera_skew_ns'][6]=50000000
    if break_type == 'unknown_mask':audit['quality_mask'][6]=128
    if break_type == 'nan':states[6,15]=np.nan
    runs,_ = select_human_runs(frames,audit,states,WindowRules(horizon=4))
    starts=expand_window_starts([dict(episode_index=0,length=12,human_runs=runs)],4)
    assert len(starts)>0
    assert all(not(t<6<=t+4) for t in starts)


def test_audit_join_mismatch_is_fatal():
    frames,audit,states=trajectory()
    audit['observation_sequence'][5]+=1
    with pytest.raises(ValueError,match='mismatch'):
        select_human_runs(frames,audit,states)


def test_exact_horizon_needs_one_extra_measured_state():
    for n,expected in [(4,0),(5,1)]:
        frames,audit,states=trajectory(n)
        _,counts=select_human_runs(frames,audit,states,WindowRules(horizon=4))
        assert counts['valid_windows']==expected


def test_merged_episode_offsets_and_invalid_ranges():
    episodes=[dict(episode_index=0,length=10,human_runs=[(2,8)]),
              dict(episode_index=1,length=12,human_runs=[(3,10)])]
    np.testing.assert_array_equal(expand_window_starts(episodes,4),[2,3,13,14,15])
    episodes[1]['human_runs']=[(3,13)]
    with pytest.raises(ValueError):expand_window_starts(episodes,4)


def test_corrupt_correction_frame_never_retries_on_policy_frames():
    from g05.data.correction_window_dataset import CorrectionWindowDataset

    fetched = []

    class BrokenBackingDataset:
        def __getitem__(self, idx):
            fetched.append(idx)
            raise OSError("Broken video frame")

    ds = CorrectionWindowDataset.__new__(CorrectionWindowDataset)
    ds._valid_raw_indices = np.array([17, 90])
    ds.multi_dataset = BrokenBackingDataset()
    ds._record_invalid_sample = lambda **kwargs: None
    ds._locate_sample = lambda idx: str(idx)
    with pytest.raises(RuntimeError, match='refusing unrestricted retry'):
        ds[1]
    assert fetched == [90]


def test_overfit_keeps_filtered_raw_indices_and_bounds():
    from g05.data.correction_window_dataset import CorrectionWindowDataset

    ds = CorrectionWindowDataset.__new__(CorrectionWindowDataset)
    ds._valid_raw_indices = np.array([17, 40, 90])
    ds.enable_overfit(2)
    assert len(ds) == 2
    assert [ds._resolve_sample_index(i) for i in range(2)] == [17, 90]
    with pytest.raises(IndexError):
        ds[2]
