# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

import torch

from g05.data_processor import BaseActionStateTransform


class BaseActionFilter(BaseActionStateTransform):
    """
    Base class for action filters that mark operational dimensions.

    Action filters are NOT invertible - they modify action_op_mask which
    cannot be recovered from the action values alone.
    """

    invertible = False

    def __init__(
        self,
        joint_threshold: float | None = None,
        gripper_threshold: float | None = None,
        velocity_threshold: float | None = None,
        eef_threshold: float | None = 1e-3,
        dim_thresholds: dict | None = None,
        inactive_keys: list[str] | None = None,
    ):
        self.joint_threshold = joint_threshold
        self.gripper_threshold = gripper_threshold
        self.velocity_threshold = velocity_threshold
        self.eef_threshold = eef_threshold
        self.dim_thresholds = dim_thresholds or {}
        self.inactive_keys = set(inactive_keys or [])

    def set_shape_meta(self, shape_meta):
        processed_action_meta, processed_state_meta = [], []
        for meta in shape_meta["action"]:
            if meta["key"] is not None:
                processed_action_meta.append(meta)
        for meta in shape_meta["state"]:
            if meta["key"] is not None:
                processed_state_meta.append(meta)
        self.action_meta = processed_action_meta
        self.state_meta = processed_state_meta
        known_keys = {meta["key"] for meta in self.action_meta}
        unknown_keys = self.inactive_keys - known_keys
        if unknown_keys:
            raise ValueError(
                f"inactive_keys contains unknown action keys {sorted(unknown_keys)}; "
                f"available keys: {sorted(known_keys)}"
            )

    def forward(self, batch):
        if "action" not in batch:
            return batch

        action_op_mask = {}
        for meta in self.action_meta:
            k, meta_shape = meta["key"], meta["raw_shape"]
            actual_shape = batch["action"][k].shape[-1]
            flag = torch.ones(actual_shape, dtype=torch.bool)
            if k in self.inactive_keys:
                flag.fill_(False)
            action_op_mask[k] = flag
        batch["action_op_mask"] = action_op_mask  # A
        return batch

    def backward(self, batch):
        return batch


class DummyActionFilter(BaseActionFilter):
    """
    Action filter that marks all dimensions as operational (no masking).

    Sets action_op_mask to all True, equivalent to R1LiteJointActionFilter
    with all thresholds set to 0.
    """

    invertible = True

    def forward(self, batch):
        return super().forward(batch)

    def backward(self, batch):
        return batch


class FixedInactivePartsActionFilter(BaseActionFilter):
    """Mark configured action parts as inactive for tokenization and loss masking.

    The action tensor and its layout are left unchanged.  This is useful for datasets
    where a body part is present in the robot schema but was intentionally stationary
    during collection (for example, a single-arm task recorded with a dual-arm robot).
    """

    def __init__(self, hold_inactive_parts: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.hold_inactive_parts = hold_inactive_parts

    def backward(self, batch):
        # Called after inverse normalization and inverse relative-action transforms:
        # both action and state are in the original robot coordinates here.
        if not self.hold_inactive_parts or "action" not in batch:
            return batch
        for key in self.inactive_keys:
            if key not in batch["action"]:
                continue
            if key not in batch.get("state", {}):
                raise ValueError(f"Cannot hold inactive part {key!r} without its current state")
            target = batch["action"][key]
            current = batch["state"][key][..., -1:, :]
            batch["action"][key] = current.expand_as(target).clone()
        return batch
