"""Compute shift3-only full-data normalization; validate real five-camera samples."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from g05.utils.config.config_resolvers import register_default_resolvers
from g05.utils.data.processor_utils import build_processors, instantiate_dataset
from g05.utils.data.normalizer import load_dataset_stats_from_json, save_dataset_stats_to_json


def check_finite(value):
    if isinstance(value, dict):
        for v in value.values():
            check_finite(v)
    elif isinstance(value, (torch.Tensor, np.ndarray)):
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError("Non-finite normalization statistic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify existing statistics and real data; never recompute")
    parser.add_argument("--stats-only", action="store_true", help="With --check: validate provenance without loading the dataset")
    args = parser.parse_args()
    if args.stats_only and not args.check:
        parser.error("--stats-only requires --check")
    # Fail before expensive statistics or the dataset's random-index retry loop.
    from torchcodec.decoders import VideoDecoder  # noqa: F401
    root = Path(__file__).resolve().parents[1]
    register_default_resolvers()
    with initialize_config_dir(str(root / "configs"), version_base="1.3"):
        cfg = compose("train", ["task=combined_rj45_tactile_shift3"])
    path = Path(cfg.datastatics_path)
    source = Path(cfg.data.embodiment_datasets.galaxea_r1pro.dataset_groups[0].dataset_dirs[0]).resolve()
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text())
    provenance_path = path.with_name("norm_provenance.json")
    if not args.check and (path.exists() or provenance_path.exists()):
        raise FileExistsError(f"Refusing to overwrite statistics: {path}; use --check to validate")
    identity = {
        "dataset": str(source), "frames": info["total_frames"],
        "episodes": info["total_episodes"],
        "info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
        "stats_downsample_rate": 1, "action_horizon": int(cfg.data.action_size),
        "inactive_action_keys": ["right_arm", "right_gripper"],
        "extra_time_offset": 0,
    }
    if args.check:
        provenance = json.loads(provenance_path.read_text())
        for key, expected in identity.items():
            if provenance.get(key) != expected:
                raise ValueError(f"Statistics provenance mismatch for {key}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["stats_sha256"]:
            raise ValueError("Statistics checksum mismatch")
        stats = load_dataset_stats_from_json(path)
        check_finite(stats)
        if args.stats_only:
            print(f"Statistics provenance and video decoder verified: {path}", flush=True)
            return
    processors = build_processors(cfg)
    dataset = instantiate_dataset(cfg, is_training_set=True)
    assert len(dataset) == identity["frames"], (len(dataset), identity)
    assert all(ds.stats_downsample_rate == 1 and ds.val_set_proportion == 0 for ds in dataset.datasets)
    if not args.check:
        stats = dataset.get_dataset_stats(processors)
        check_finite(stats)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset_stats_to_json(stats, path)
        provenance = dict(identity, stats_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          source="fresh full-data parquet statistics; no base-checkpoint statistics")
        with provenance_path.open("x") as f:
            json.dump(provenance, f, indent=2)
    check_finite(stats)
    processors.set_normalizer_from_stats(stats)
    processors.eval()
    dataset.set_processor(processors)
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[index]
        assert int(sample["idx"]) == index, "Dataset silently retried a different frame"
        assert set(sample["pixel_values"]) == {"exterior", "wrist_left", "wrist_right"}
        assert set(sample["tactile_pixel_values"]) == {"tactile_left_1", "tactile_left_2"}
        assert all(tuple(x.shape) == (1, 3, 224, 224) for x in sample["tactile_pixel_values"].values())
        check_finite(sample)
        mask = sample["action_op_mask"]
        assert mask[..., :7].all() and mask[..., 9].all()
        assert not mask[..., 10:20].any()
        print(f"Verified real sample {index}: 3 visual + 2 tactile, action {tuple(sample['action'].shape)}", flush=True)
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
