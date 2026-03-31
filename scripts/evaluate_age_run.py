from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.periage import PeriAge, PeriAgeResNet34, PeriAgeV2
from scripts.common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an age checkpoint on the processed UTKFace periocular test split.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--print-every", type=int, default=0, help="Print progress every N batches. Disabled when 0.")
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def load_model(payload: dict):
    model_name = payload["model_name"]
    task = payload["task"]
    if task != "age":
        raise ValueError(f"Checkpoint task must be age, got {task}")

    if model_name == "periage":
        model = PeriAge()
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif model_name == "periage_v2":
        model = PeriAgeV2()
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif model_name == "periage_resnet34":
        model = PeriAgeResNet34(weights=None)
        weights = ResNet34_Weights.DEFAULT
        mean, std = weights.transforms().mean, weights.transforms().std
    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        weights = ResNet18_Weights.DEFAULT
        mean, std = weights.transforms().mean, weights.transforms().std
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        weights = ResNet34_Weights.DEFAULT
        mean, std = weights.transforms().mean, weights.transforms().std
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)
        weights = ResNet50_Weights.DEFAULT
        mean, std = weights.transforms().mean, weights.transforms().std
    else:
        raise ValueError(f"Unsupported model_name={model_name}")

    model.load_state_dict(payload["state_dict"])
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return model, transform


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.num_threads))
    payload = torch.load(args.checkpoint, map_location="cpu")
    model, transform = load_model(payload)
    model = model.to(args.device).eval()

    test_ds = datasets.ImageFolder(args.data_dir / "test", transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    idx_to_class = {idx: label for label, idx in test_ds.class_to_idx.items()}

    all_true = []
    all_pred = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader, start=1):
            images = images.to(args.device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_pred.extend(preds)
            all_true.extend(labels.tolist())
            if args.print_every and (batch_idx % args.print_every == 0 or batch_idx == len(test_loader)):
                print(f"batch={batch_idx}/{len(test_loader)}")

    image_acc = sum(int(p == t) for p, t in zip(all_pred, all_true)) / len(all_true)
    bal_acc = balanced_accuracy_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred).tolist()

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "num_test_images": len(all_true),
        "class_to_idx": test_ds.class_to_idx,
        "idx_to_class": idx_to_class,
        "image_accuracy": image_acc,
        "image_balanced_accuracy": bal_acc,
        "image_confusion_matrix": cm,
    }

    print(json.dumps(result, indent=2))
    if args.save_json is not None:
        save_json(result, args.save_json)


if __name__ == "__main__":
    main()
