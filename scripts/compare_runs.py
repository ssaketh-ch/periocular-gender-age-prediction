from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare training runs from metrics.json files.")
    parser.add_argument("--runs-dir", required=True, type=Path, help="Directory to scan for metrics.json files.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of runs to display.")
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include metrics found under directories named legacy.",
    )
    return parser.parse_args()


def load_run(metrics_path: Path) -> dict | None:
    try:
        data = json.loads(metrics_path.read_text())
    except Exception:
        return None

    history = data.get("history", [])
    if not history:
        return None

    metric_key = None
    if "test_acc" in history[0]:
        metric_key = "test_acc"
    elif "test" in history[0] and "joint_acc" in history[0]["test"]:
        metric_key = "joint_acc"
    else:
        return None

    if metric_key == "test_acc":
        best = max(history, key=lambda row: row["test_acc"])
        final = history[-1]
        best_acc = best["test_acc"]
        final_acc = final["test_acc"]
        best_loss = best.get("test_loss")
        best_epoch = best["epoch"]
    else:
        best = max(history, key=lambda row: row["test"]["joint_acc"])
        final = history[-1]
        best_acc = best["test"]["joint_acc"]
        final_acc = final["test"]["joint_acc"]
        best_loss = best["test"].get("loss")
        best_epoch = best["epoch"]

    return {
        "run_dir": str(metrics_path.parent),
        "model": data.get("model"),
        "task": data.get("task"),
        "lr": data.get("lr"),
        "epochs": data.get("epochs"),
        "best_epoch": best_epoch,
        "best_acc": best_acc,
        "final_acc": final_acc,
        "best_loss": best_loss,
    }


def main() -> None:
    args = parse_args()
    runs = []
    for metrics_path in sorted(args.runs_dir.rglob("metrics.json")):
        if not args.include_legacy and "legacy" in metrics_path.parts:
            continue
        run = load_run(metrics_path)
        if run is not None:
            runs.append(run)

    runs.sort(key=lambda row: row["best_acc"], reverse=True)
    if not runs:
        print("No readable metrics.json files found.")
        return

    header = f"{'rank':<4} {'best_acc':<10} {'final_acc':<10} {'epoch':<6} {'model':<12} {'lr':<10} run_dir"
    print(header)
    print("-" * len(header))
    for idx, run in enumerate(runs[: args.top_k], start=1):
        lr = run["lr"] if run["lr"] is not None else "-"
        print(
            f"{idx:<4} "
            f"{run['best_acc']:<10.4f} "
            f"{run['final_acc']:<10.4f} "
            f"{run['best_epoch']:<6} "
            f"{str(run['model']):<12} "
            f"{str(lr):<10} "
            f"{run['run_dir']}"
        )


if __name__ == "__main__":
    main()
