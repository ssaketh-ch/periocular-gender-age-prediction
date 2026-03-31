from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def best_metric(metrics_path: Path) -> dict:
    data = load_json(metrics_path)
    history = data["history"]
    best = max(history, key=lambda row: row["test_acc"])
    return {
        "model": data["model"],
        "lr": data.get("lr"),
        "metrics_path": metrics_path,
        "history": history,
        "best_epoch": best["epoch"],
        "best_acc": best["test_acc"],
        "final_acc": history[-1]["test_acc"],
        "best_loss": best["test_loss"],
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def newest_metrics(run_dir: Path) -> Path | None:
    metrics = sorted(run_dir.rglob("metrics.json"))
    return metrics[-1] if metrics else None


def plot_history(series: list[tuple[str, dict]], out_path: Path, title: str) -> None:
    plt.figure(figsize=(9, 5))
    for label, info in series:
        epochs = [row["epoch"] for row in info["history"]]
        acc = [row["test_acc"] for row in info["history"]]
        plt.plot(epochs, acc, label=label, linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_bar(labels: list[str], values: list[float], out_path: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_confusion(eval_path: Path, out_path: Path, title: str) -> None:
    data = load_json(eval_path)
    cm = np.array(data["image_confusion_matrix"])
    labels = [data["idx_to_class"][str(i)] if isinstance(next(iter(data["idx_to_class"].keys())), str) else data["idx_to_class"][i] for i in range(len(cm))]
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.yticks(range(len(labels)), labels)
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    gender_dir = FIGURES_DIR / "gender"
    age_dir = FIGURES_DIR / "age"
    ensure_dir(gender_dir)
    ensure_dir(age_dir)

    # Gender selections
    gender_runs = {
        "ResNet18": REPO_ROOT / "runs/gender/ubipr/baselines/resnet18_e30/20260330_212202/metrics.json",
        "ResNet34": REPO_ROOT / "runs/gender/ubipr/baselines/resnet34_e30/20260330_214519/metrics.json",
        "ResNet50": REPO_ROOT / "runs/gender/ubipr/baselines/resnet50_e30/20260330_220911/metrics.json",
        "PeriGenderV2": REPO_ROOT / "runs/gender/ubipr/custom/perigender_v2_adamw/20260331_004845/metrics.json",
    }
    gender_infos = {label: best_metric(path) for label, path in gender_runs.items()}
    plot_bar(list(gender_infos.keys()), [info["best_acc"] for info in gender_infos.values()], gender_dir / "gender_best_accuracy.png", "UBIPr Gender: Best Test Accuracy", "Best Test Accuracy")
    plot_history(list(gender_infos.items()), gender_dir / "gender_accuracy_curves.png", "UBIPr Gender Accuracy Curves")
    for name, eval_rel in {
        "resnet18": REPO_ROOT / "runs/gender/ubipr/baselines/resnet18_e30/20260330_212202/eval.json",
        "resnet34": REPO_ROOT / "runs/gender/ubipr/baselines/resnet34_e30/20260330_214519/eval.json",
        "perigender_v2": REPO_ROOT / "runs/gender/ubipr/custom/perigender_v2_adamw/20260331_004845/eval.json",
    }.items():
        if eval_rel.exists():
            plot_confusion(eval_rel, gender_dir / f"{name}_confusion.png", f"UBIPr Gender Confusion Matrix: {name}")

    # Age selections
    age_runs = {
        "ResNet18": REPO_ROOT / "runs/age/periocular/baselines/resnet18/20260330_150429/metrics.json",
        "ResNet34": REPO_ROOT / "runs/age/periocular/baselines/resnet34/20260330_151158/metrics.json",
        "ResNet50": REPO_ROOT / "runs/age/periocular/baselines/resnet50/20260330_152227/metrics.json",
        "PeriAge": REPO_ROOT / "runs/age/periocular/custom/periage_lr5e4/20260330_165623/metrics.json",
        "PeriAgeV2": REPO_ROOT / "runs/age/periocular/custom/periage_v2_bs32/20260330_174946/metrics.json",
        "PeriAgeResNet34": REPO_ROOT / "runs/age/periocular/hybrid/periage_resnet34_ft/20260330_202332/metrics.json",
    }
    age_infos = {label: best_metric(path) for label, path in age_runs.items()}
    plot_bar(list(age_infos.keys()), [info["best_acc"] for info in age_infos.values()], age_dir / "age_best_accuracy.png", "Periocular Age: Best Test Accuracy", "Best Test Accuracy")
    plot_history(list(age_infos.items()), age_dir / "age_accuracy_curves.png", "Periocular Age Accuracy Curves")
    for name, eval_rel in {
        "resnet34": REPO_ROOT / "runs/age/periocular/baselines/resnet34/20260330_151158/eval.json",
        "periage_v2": REPO_ROOT / "runs/age/periocular/custom/periage_v2_bs32/20260330_174946/eval.json",
        "periage_resnet34": REPO_ROOT / "runs/age/periocular/hybrid/periage_resnet34_ft/20260330_202332/eval.json",
    }.items():
        if eval_rel.exists():
            plot_confusion(eval_rel, age_dir / f"{name}_confusion.png", f"Periocular Age Confusion Matrix: {name}")


if __name__ == "__main__":
    main()
