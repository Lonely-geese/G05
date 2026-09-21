#!/usr/bin/env python3
"""Expand a 16D dual-arm Cartesian action to the dataset's 30D state layout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    return parser.parse_args()


def atomic_write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(tmp, path)


def atomic_write_jsonl(path: Path, values: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    info_path = root / "meta" / "info.json"
    stats_path = root / "meta" / "episodes_stats.jsonl"
    parquet_paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    with info_path.open(encoding="utf-8") as stream:
        info = json.load(stream)
    state_feature = info["features"]["observation.state"]
    action_feature = info["features"]["action"]
    if state_feature.get("shape") != [30] or action_feature.get("shape") != [16]:
        raise ValueError(
            f"Expected state/action shapes [30]/[16], got "
            f"{state_feature.get('shape')}/{action_feature.get('shape')}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = root / f".action16_backup_{stamp}"
    (backup / "data").mkdir(parents=True)
    (backup / "meta").mkdir()
    for source in parquet_paths:
        relative = source.relative_to(root / "data")
        destination = backup / "data" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
    shutil.copy2(info_path, backup / "meta" / "info.json")
    shutil.copy2(stats_path, backup / "meta" / "episodes_stats.jsonl")

    total_rows = 0
    for index, path in enumerate(parquet_paths, 1):
        table = pq.read_table(path)
        state = table["observation.state"]
        action = table["action"]
        state_lengths = pc.list_value_length(state)
        action_lengths = pc.list_value_length(action)
        if not pc.all(pc.equal(state_lengths, 30)).as_py():
            raise ValueError(f"Non-30D observation.state in {path}")
        if not pc.all(pc.equal(action_lengths, 16)).as_py():
            raise ValueError(f"Non-16D action in {path}")

        # The existing 16D action must match the Cartesian pose/gripper fields
        # in state before it is replaced by the full state-layout action.
        states = state.to_pylist()
        actions = action.to_pylist()
        for row_index, (state_row, action_row) in enumerate(zip(states, actions, strict=True)):
            expected = state_row[7:15] + state_row[22:30]
            if action_row != expected:
                raise ValueError(f"Action/state mismatch in {path}, row {row_index}")

        action_index = table.schema.get_field_index("action")
        action_field = pa.field("action", state.type)
        expanded = table.set_column(action_index, action_field, state)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        pq.write_table(expanded, tmp, compression="snappy")
        os.replace(tmp, path)
        total_rows += len(table)
        if index % 100 == 0 or index == len(parquet_paths):
            print(f"updated {index}/{len(parquet_paths)} parquet files", flush=True)

    stats_values = []
    with stats_path.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            value["stats"]["action"] = value["stats"]["observation.state"]
            stats_values.append(value)
    atomic_write_jsonl(stats_path, stats_values)

    info["features"]["action"] = {
        "dtype": state_feature["dtype"],
        "shape": list(state_feature["shape"]),
        "names": list(state_feature["names"]),
    }
    atomic_write_json(info_path, info)

    report = {
        "operation": "expand_action_with_state_joints",
        "timestamp_utc": stamp,
        "parquet_files": len(parquet_paths),
        "frames": total_rows,
        "old_action_shape": [16],
        "new_action_shape": [30],
        "new_action_layout": "observation.state layout",
        "backup": str(backup),
    }
    atomic_write_json(root / "meta" / "action_expansion_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
