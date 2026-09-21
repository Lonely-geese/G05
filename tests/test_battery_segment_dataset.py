import numpy as np
import torch

from g05.data.battery_segment_dataset import BatterySegmentDataset
from g05.data.base_lerobot_datasetV3 import _build_stats_sampling_plan


def test_internal_cut_clamps_targets_and_masks_future_steps():
    ds = BatterySegmentDataset.__new__(BatterySegmentDataset)
    ds.action_size = 4
    ds.frame_segment_end = np.array([3, 3, 3, 6, 6, 6])
    sample = {"idx": 1, "action_is_pad": torch.zeros(4, dtype=torch.bool),
              "action": {"right_arm": torch.tensor([[1.], [2.], [100.], [101.]])}}
    actual = ds._get_additional_data(sample, {})
    assert actual["action_is_pad"].tolist() == [False, False, True, True]
    assert actual["action"]["right_arm"].flatten().tolist() == [1., 2., 2., 2.]


def test_norm_sampling_plan_uses_identical_cut_boundaries():
    _, _, actions = _build_stats_sampling_plan(
        torch.arange(6), 4, [], [{"key": "right_arm", "time_offset": 0}],
        ep_starts=torch.tensor([0, 3]), ep_ends=torch.tensor([3, 6]))
    assert torch.stack(actions["right_arm"], 1).tolist() == [
        [0, 1, 2, 2], [1, 2, 2, 2], [2, 2, 2, 2],
        [3, 4, 5, 5], [4, 5, 5, 5], [5, 5, 5, 5]]
