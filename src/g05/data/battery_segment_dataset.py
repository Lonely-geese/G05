"""Keep action chunks and their normalization inside audited continuous segments."""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from g05.data.base_lerobot_datasetV3 import BaseLerobotDatasetV3


class BatterySegmentDataset(BaseLerobotDatasetV3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(self.dataset_dirs) == 1 and self.obs_size == 1
        assert all(int(m.get("time_offset", 0)) == 0 for m in self.action_meta)
        ends, starts, offset = [], [], 0
        for path in sorted(Path(self.dataset_dirs[0]).glob("audit/frames/chunk-*/*.parquet")):
            table = pq.read_table(path, columns=["frame_index", "source_segment_index"])
            frames = table["frame_index"].to_numpy()
            assert np.array_equal(frames, np.arange(len(frames)))
            segments = table["source_segment_index"].to_numpy()
            boundaries = np.r_[0, np.flatnonzero(segments[1:] != segments[:-1]) + 1, len(segments)]
            starts.extend((boundaries[:-1] + offset).tolist())
            ends.extend((boundaries[1:] + offset).tolist())
            offset += len(segments)
        assert offset == self.multi_dataset.num_frames, (offset, self.multi_dataset.num_frames)
        self.segment_starts = torch.tensor(starts, dtype=torch.long)
        self.segment_ends = torch.tensor(ends, dtype=torch.long)
        self.frame_segment_end = np.repeat(np.asarray(ends), np.asarray(ends) - np.asarray(starts))

    def stats_sequence_boundaries(self):
        return self.segment_starts, self.segment_ends, len(self.frame_segment_end)

    def _get_additional_data(self, sample, lerobot_sample):
        sample = super()._get_additional_data(sample, lerobot_sample)
        idx = int(sample["idx"])
        valid = min(self.action_size, int(self.frame_segment_end[idx]) - idx)
        assert valid > 0
        if valid < self.action_size:
            sample["action_is_pad"][valid:] = True
            for value in sample["action"].values():
                value[valid:] = value[valid - 1].clone()
        return sample
