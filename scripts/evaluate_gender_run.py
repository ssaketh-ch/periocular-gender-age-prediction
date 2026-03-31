from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.perigender import PeriGender, PeriGenderV2
from scripts.common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a gender checkpoint on the processed UBIPr test split.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path, help="Prepared dataset root containing train/ and test/.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-json", type=Path, default=None, help="Optional output path for evaluation JSON.")
    return parser.parse_args()


def load_model(payload: dict):
    model_name = payload["model_name"]
    task = payload["task"]
    if task != "gender":
        raise ValueError(f"Checkpoint task must be gender, got {task}")

    if model_name == "perigender":
        model = PeriGender()
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif model_name == "perigender_v2":
        model = PeriGenderV2()
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        weights = ResNet18_Weights.DEFAULT
        mean, std = weights.transforms().mean, weights.transforms().std
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        weights = ResNet34_Weights.DEFAULT
        mean, std = weights.transforms().mean, weights.transforms().std
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        weights = ResNet50_Weights.DEFAULT
        mean, std = weights.transforms().mean, weights.transforms().std
    else:
        raise ValueError(f"Unsupported model_name={model_name}")

    model.load_state_dict(payload["state_dict"])
    transform = transforms.Compose(
        [
            transforms.Resize((224, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return model, transform


def subject_id_from_path(path: str) -> str:
    return Path(path).stem.split("_")[0]


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    model, transform = load_model(payload)
    model = model.to(args.device).eval()

    test_ds = datasets.ImageFolder(args.data_dir / "test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    idx_to_class = {idx: label for label, idx in test_ds.class_to_idx.items()}

    all_true = []
    all_pred = []
    subject_votes: dict[str, list[tuple[int, int]]] = defaultdict(list)

    with torch.no_grad():
        sample_idx = 0
        for images, labels in test_loader:
            images = images.to(args.device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            labels_list = labels.tolist()

            for pred, true in zip(preds, labels_list):
                all_pred.append(pred)
                all_true.append(true)
                image_path, _ = test_ds.samples[sample_idx]
                subject_votes[subject_id_from_path(image_path)].append((pred, true))
                sample_idx += 1

    image_acc = sum(int(p == t) for p, t in zip(all_pred, all_true)) / len(all_true)
    bal_acc = balanced_accuracy_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred).tolist()

    subject_true = []
    subject_pred = []
    for _, votes in sorted(subject_votes.items()):
        pred_counts = Counter(pred for pred, _ in votes)
        true_counts = Counter(true for _, true in votes)
        subject_pred.append(pred_counts.most_common(1)[0][0])
        subject_true.append(true_counts.most_common(1)[0][0])

    subject_acc = sum(int(p == t) for p, t in zip(subject_pred, subject_true)) / len(subject_true)
    subject_bal_acc = balanced_accuracy_score(subject_true, subject_pred)
    subject_cm = confusion_matrix(subject_true, subject_pred).tolist()

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "num_test_images": len(all_true),
        "num_test_subjects": len(subject_true),
        "class_to_idx": test_ds.class_to_idx,
        "idx_to_class": idx_to_class,
        "image_accuracy": image_acc,
        "image_balanced_accuracy": subject_bal_acc if False else bal_acc,
        "image_confusion_matrix": cm,
        "subject_majority_accuracy": subject_acc,
        "subject_majority_balanced_accuracy": subject_bal_acc,
        "subject_majority_confusion_matrix": subject_cm,
    }

    print(json.dumps(result, indent=2))
    if args.save_json is not None:
        save_json(result, args.save_json)


if __name__ == "__main__":
    main()
