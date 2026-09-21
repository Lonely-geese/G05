"""Explicit train/val splits and fail-closed HUMAN-only action chunk sampling."""
import json
from pathlib import Path
import numpy as np

from g05.data.base_lerobot_datasetV3 import BaseLerobotDatasetV3
from g05.data.recap_windows import expand_window_starts


class CorrectionWindowDataset(BaseLerobotDatasetV3):
    supports_correction_windows = True

    def __init__(self, dataset_dirs, action_size, is_training_set=False,
                 val_set_proportion=0.0, **kwargs):
        if len(dataset_dirs) != 1:
            raise ValueError("Pass one prepared correction root containing train/ and val/")
        root = Path(dataset_dirs[0])
        if not (root/'READY.json').is_file():
            raise ValueError(f"Correction conversion is incomplete: {root}")
        split = 'train' if is_training_set else 'val'
        split_root = root/split
        manifest = json.loads((split_root/'meta/correction_windows.json').read_text())
        if manifest['horizon'] != action_size or manifest['action_shift'] != 1:
            raise ValueError("Correction horizon/shift does not match training configuration")
        shape_meta = kwargs['shape_meta']
        if any(int(m.get('time_offset', 0)) != 0 for group in ['action','state','images'] for m in shape_meta.get(group, [])):
            raise ValueError("Prepared correction data already has shift1; offsets must be zero")
        if kwargs.get('obs_size', 1) != 1 or kwargs.get('past_action_size', 0) != 0:
            raise ValueError("Correction windows currently require one observation and no past actions")
        self._valid_raw_indices = expand_window_starts(manifest['episodes'], action_size)
        if len(self._valid_raw_indices) == 0:
            raise ValueError(f"No correction windows in {split}")
        self.correction_manifest = manifest
        kwargs['fast_stats_computation'] = False
        super().__init__(dataset_dirs=[str(split_root)], action_size=action_size,
                         is_training_set=is_training_set, val_set_proportion=0.0, **kwargs)
        lengths = (self.episode_data_index['to']-self.episode_data_index['from']).tolist()
        if lengths != [ep['length'] for ep in manifest['episodes']]:
            raise ValueError("Correction manifest lengths differ from loaded episodes")

    def __len__(self):
        return getattr(self, '_overfit_len', len(self._valid_raw_indices))

    def _resolve_sample_index(self, idx):
        if hasattr(self, '_overfit_indices'):
            return int(self._overfit_indices[idx])
        return int(self._valid_raw_indices[idx])

    def _resample_random_idx(self):
        # A broken sample must not silently retry on arbitrary POLICY frames.
        raise RuntimeError("Correction sample failed; refusing unrestricted retry. Inspect invalid-sample records.")

    def enable_overfit(self, n_samples):
        take = min(int(n_samples), len(self._valid_raw_indices))
        if take < 1:
            raise ValueError("overfit sample count must be positive")
        idx = np.linspace(0, len(self._valid_raw_indices)-1, take, dtype=int)
        self._overfit_indices = self._valid_raw_indices[idx].tolist()
        self._overfit_len = take

    def get_dataset_stats(self, *args, **kwargs):
        raise ValueError("Correction baseline requires pretrained checkpoint dataset_stats.json; do not normalize unrestricted backing frames")
