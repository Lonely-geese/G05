"""Pure NumPy selection of continuous human-correction supervision windows."""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class WindowRules:
    horizon: int = 32
    max_interval_ms: float = 50.0
    max_state_age_ms: float = 20.0
    max_camera_skew_ms: float = 45.0
    # Bit 6 is retained only when independent observation checks pass. Its
    # producer definition is unresolved; command age is not a drag-state target.
    allowed_quality_masks: tuple = (0, 64)

    def __post_init__(self):
        if self.horizon < 1 or min(self.max_interval_ms, self.max_state_age_ms, self.max_camera_skew_ms) <= 0:
            raise ValueError("Horizon and quality thresholds must be positive")

    def as_dict(self):
        return asdict(self)


def select_human_runs(frames, audit, states, rules=WindowRules()):
    """Return maximal [start, end) runs; each needs H+1 states for shift1.

    All states/observations, not just the initial frame, must pass quality
    checks. Sequence gaps and source/segment switches break continuity.
    No stationary filtering: legitimate contact/insertion pauses are retained.
    """
    n = len(frames)
    if n == 0:
        return [], {"human_frames": 0, "quality_human_frames": 0, "valid_windows": 0}
    states = np.asarray(states)
    if states.shape != (n, 30):
        raise ValueError(f"Expected ({n}, 30) measured states, got {states.shape}")
    expected = np.arange(n)
    if not np.array_equal([f['frame_index'] for f in frames], expected):
        raise ValueError("recap frame_index is not contiguous and zero-based")
    if not np.array_equal(audit['frame_index'], expected):
        raise ValueError("audit frame_index mismatch")
    seq = np.asarray([f['observation_sequence'] for f in frames], dtype=np.int64)
    if not np.array_equal(audit['observation_sequence'], seq):
        raise ValueError("audit/recap observation_sequence mismatch")
    stamp = np.asarray([f['stamp_ns'] for f in frames], dtype=np.int64)
    source = np.asarray([f['source'] for f in frames])
    segment = np.asarray([f['segment_id'] for f in frames])
    age = np.asarray(audit['state_age_ns'])
    skew = np.asarray(audit['camera_skew_ns'])
    human = source == 'HUMAN'
    good = (human & np.isfinite(states).all(axis=1)
            & (age >= 0) & (age <= rules.max_state_age_ms * 1e6)
            & (skew >= 0) & (skew <= rules.max_camera_skew_ms * 1e6)
            & np.isin(audit['quality_mask'], rules.allowed_quality_masks))
    dt = np.diff(stamp)
    continuous = ((np.diff(seq) == 1) & (dt > 0)
                  & (dt <= rules.max_interval_ms * 1e6)
                  & (segment[1:] == segment[:-1]))
    runs, start = [], None
    for i in range(n):
        if start is not None and (not good[i] or not continuous[i-1]):
            if i-start >= rules.horizon+1:
                runs.append((start, i))
            start = None
        if good[i] and start is None:
            start = i
    if start is not None and n-start >= rules.horizon+1:
        runs.append((start, n))
    return runs, {
        'human_frames': int(human.sum()),
        'quality_human_frames': int(good.sum()),
        'valid_windows': sum(e-s-rules.horizon for s, e in runs),
    }


def shifted_human_actions(states, runs):
    """Shift exactly once within accepted runs; other rows aren't trainable."""
    action = np.array(states, copy=True)
    for start, end in runs:
        action[start:end-1] = states[start+1:end]
    return action


def expand_window_starts(episodes, horizon):
    """Validate ranges and return raw global indices in a merged dataset."""
    starts, offset = [], 0
    for expected_episode, ep in enumerate(episodes):
        if ep['episode_index'] != expected_episode:
            raise ValueError("Window episodes must be ordered and contiguous")
        previous_end = 0
        for start, end in ep['human_runs']:
            if not (previous_end <= start < end <= ep['length']) or end-start <= horizon:
                raise ValueError("Invalid, overlapping or short human run")
            starts.extend(range(offset+start, offset+end-horizon))
            previous_end = end
        offset += ep['length']
    return np.asarray(starts, dtype=np.int64)
