"""Summarize all-reduced losses from finetune's console log (read-only input)."""
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


def parse_losses(text):
    pattern = r"Epoch (\d+), Step (\d+), Loss: ([-+\d.eE]+|nan|[-+]?inf)"
    values = {}
    for epoch, step, loss in re.findall(pattern, text, flags=re.IGNORECASE):
        # The progress label is zero-based, while optimizer steps are one-based.
        step = int(step) + 1
        loss = float(loss)
        if step in values and math.isfinite(loss) and abs(values[step] - loss) > 1e-5:
            raise ValueError(f"Conflicting cross-rank losses at step {step}")
        values[step] = loss
    return sorted(values.items())


def read_wandb_history(run_dir):
    """Read local offline history; never contact or sync to the W&B service."""
    from wandb.sdk.internal.datastore import DataStore
    from wandb.proto import wandb_internal_pb2
    history = []
    for path in sorted(Path(run_dir).rglob("run-*.wandb")):
        store = DataStore()
        store.open_for_scan(str(path))
        try:
            while True:
                data = store.scan_data()
                if data is None:
                    break
                record = wandb_internal_pb2.Record()
                record.ParseFromString(data)
                if record.HasField("history"):
                    row = {}
                    for item in record.history.item:
                        key = item.key or "/".join(item.nested_key)
                        row[key] = json.loads(item.value_json)
                    history.append(row)
        finally:
            store.close()
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    rows = parse_losses(args.log.read_text(errors="replace"))
    if not rows:
        raise ValueError("No completed training steps found")
    if args.expected_steps is not None and [s for s, _ in rows] != list(range(1, args.expected_steps + 1)):
        raise ValueError(f"Incomplete step sequence: {len(rows)} found, expected {args.expected_steps}")
    losses = [v for _, v in rows]
    if not all(math.isfinite(v) for v in losses):
        raise ValueError("Nonfinite training loss detected")
    window = min(10, len(losses))
    first = statistics.mean(losses[:window])
    last = statistics.mean(losses[-window:])
    summary = {
        "source_log": str(args.log.resolve()),
        "loss_kind": "all-reduced mean FM loss over all ranks; console precision 4 decimals",
        "steps": len(rows), "first": losses[0], "last": losses[-1],
        "min": min(losses), "min_step": rows[losses.index(min(losses))][0],
        "max": max(losses), "mean": statistics.mean(losses),
        "window_size": window, "first_window_mean": first, "last_window_mean": last,
        "window_mean_reduction_pct": 100 * (first - last) / first if first else None,
        "all_losses_finite": True,
        "blocks": [
            {"steps": [rows[i][0], rows[min(i + window, len(rows)) - 1][0]],
             "mean": statistics.mean(losses[i:i + window]),
             "min": min(losses[i:i + window]), "max": max(losses[i:i + window])}
            for i in range(0, len(rows), window)
        ],
    }
    history = read_wandb_history(args.run_dir) if args.run_dir else []
    grads = [r["grad_norm"] for r in history if "grad_norm" in r]
    if grads:
        summary["grad_norm_before_clipping"] = {
            "count": len(grads), "all_finite": all(math.isfinite(g) for g in grads),
            "min": min(grads), "max": max(grads), "last": grads[-1],
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse overwrites: repeated analyses should have their own output directory.
    with (args.output_dir / "loss.csv").open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer_step", "all_rank_mean_fm_loss"])
        writer.writerows(rows)
    with (args.output_dir / "summary.json").open("x") as f:
        json.dump(summary, f, indent=2)
    if history:
        with (args.output_dir / "offline_history.json").open("x") as f:
            json.dump(history, f, indent=2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = [s for s, _ in rows]
    rolling = [statistics.mean(losses[max(0, i - 4):i + 1]) for i in range(len(losses))]
    ax.plot(steps, losses, alpha=0.55, linewidth=1, label="Per-step mean across GPUs")
    ax.plot(steps, rolling, linewidth=2, label="5-step moving average")
    ax.set(xlabel="Optimizer step", ylabel="FM loss", title="G05 tactile short training run")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "loss_curve.png", dpi=160)
    fig.savefig(args.output_dir / "loss_curve.svg")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
