import torch

from g05.data_processor.transforms.action_filter import FixedInactivePartsActionFilter
from g05.models.g05.helpers.fm_helper import FMHelper


def test_fixed_inactive_parts_filter_preserves_layout_and_masks_selected_part():
    action_filter = FixedInactivePartsActionFilter(inactive_keys=["left_arm", "left_gripper"])
    action_filter.set_shape_meta(
        {
            "action": [
                {"key": "left_arm", "raw_shape": 2},
                {"key": "left_gripper", "raw_shape": 1},
                {"key": "right_arm", "raw_shape": 2},
            ],
            "state": [],
        }
    )
    action = {
        "left_arm": torch.randn(4, 2),
        "left_gripper": torch.randn(4, 1),
        "right_arm": torch.randn(4, 2),
    }
    result = action_filter.forward({"action": action})

    assert result["action"]["left_arm"] is action["left_arm"]
    assert not result["action_op_mask"]["left_arm"].any()
    assert not result["action_op_mask"]["left_gripper"].any()
    assert result["action_op_mask"]["right_arm"].all()


def test_fm_loss_excludes_inactive_dimensions_from_numerator_and_denominator():
    helper = FMHelper.__new__(FMHelper)
    helper.time_convention = "pi_convention"
    helper.padding_action_weight = 0.0
    helper.zero_pad_action_target = False

    x0 = torch.zeros(1, 2, 3)
    x1 = torch.zeros_like(x0)
    prediction = torch.tensor([[[100.0, 1.0, 3.0], [100.0, 1.0, 3.0]]])
    action_op_mask = torch.tensor([[False, True, True]])

    loss = helper.cal_fm_loss(
        prediction,
        x0,
        x1,
        action_pad_masks=torch.zeros(1, 2, dtype=torch.bool),
        action_op_mask=action_op_mask,
    )

    # Mean of active squared errors: (1 + 9 + 1 + 9) / 4.
    torch.testing.assert_close(loss, torch.tensor(5.0))


def test_fixed_right_arm_holds_latest_raw_state_at_inference():
    filt = FixedInactivePartsActionFilter(
        inactive_keys=['right_arm', 'right_gripper'], hold_inactive_parts=True
    )
    left = torch.randn(2, 32, 7)
    state = {'right_arm': torch.randn(2, 3, 7), 'right_gripper': torch.randn(2, 3, 1)}
    batch = {'state': state, 'action': {'left_arm': left,
             'right_arm': torch.randn(2, 32, 7), 'right_gripper': torch.randn(2, 32, 1)}}
    out = filt.backward(batch)
    assert out['action']['left_arm'] is left
    for key in state:
        torch.testing.assert_close(out['action'][key], state[key][:, -1:, :].expand_as(out['action'][key]))


def test_fixed_right_arm_hold_requires_current_state():
    import pytest
    filt = FixedInactivePartsActionFilter(inactive_keys=['right_arm'], hold_inactive_parts=True)
    with pytest.raises(ValueError, match='current state'):
        filt.backward({'action': {'right_arm': torch.zeros(2, 32, 7)}})
