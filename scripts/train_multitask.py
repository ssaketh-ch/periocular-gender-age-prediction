from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.periocular import PeriOcular
from scripts.common import age_to_bucket, parse_utkface_filename, resolve_run_dir, save_json, set_seed, split_items


class UTKFaceMultiTaskDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform: transforms.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        age, gender = parse_utkface_filename(image_path)
        age_bucket = age_to_bucket(age)
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), age_bucket, gender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multitask PeriOcular model on UTKFace.")
    parser.add_argument("--raw-dir", required=True, type=Path, help="Directory containing UTKFace .jpg files.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-timestamp", action="store_true", help="Write outputs directly into --output-dir instead of a timestamped subfolder.")
    return parser.parse_args()


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    gender_correct = 0
    age_correct = 0
    joint_correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, age_labels, gender_labels in loader:
            images = images.to(device)
            age_labels = age_labels.to(device)
            gender_labels = gender_labels.to(device)

            gender_logits, age_logits = model(images)
            gender_loss = criterion(gender_logits, gender_labels)
            age_loss = criterion(age_logits, age_labels)
            loss = gender_loss + age_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            gender_preds = torch.argmax(gender_logits, dim=1)
            age_preds = torch.argmax(age_logits, dim=1)
            gender_correct += (gender_preds == gender_labels).sum().item()
            age_correct += (age_preds == age_labels).sum().item()
            joint_correct += ((gender_preds == gender_labels) & (age_preds == age_labels)).sum().item()
            total += images.size(0)

    return {
        "loss": total_loss / total,
        "gender_acc": gender_correct / total,
        "age_acc": age_correct / total,
        "joint_acc": joint_correct / total,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = resolve_run_dir(args.output_dir, use_timestamp=not args.no_timestamp)

    image_paths = sorted([path for path in args.raw_dir.rglob("*.jpg") if path.is_file()])
    image_paths = [path for path in image_paths if len(path.name.split("_")) >= 2]
    train_paths, test_paths = split_items(image_paths, args.test_size, args.seed)

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    train_ds = UTKFaceMultiTaskDataset(train_paths, train_transform)
    test_ds = UTKFaceMultiTaskDataset(test_paths, test_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = PeriOcular().to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_joint = -1.0
    best_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, args.device, optimizer=optimizer)
        test_metrics = run_epoch(model, test_loader, criterion, args.device)
        history.append({"epoch": epoch, "train": train_metrics, "test": test_metrics})
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_gender_acc={train_metrics['gender_acc']:.4f} "
            f"train_age_acc={train_metrics['age_acc']:.4f} "
            f"train_joint_acc={train_metrics['joint_acc']:.4f} "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_gender_acc={test_metrics['gender_acc']:.4f} "
            f"test_age_acc={test_metrics['age_acc']:.4f} "
            f"test_joint_acc={test_metrics['joint_acc']:.4f}"
        )

        if test_metrics["joint_acc"] > best_joint:
            best_joint = test_metrics["joint_acc"]
            torch.save(
                {
                    "model_name": "periocular",
                    "task": "multitask",
                    "state_dict": model.state_dict(),
                },
                best_path,
            )

    save_json(
        {
            "task": "multitask",
            "model": "periocular",
            "raw_dir": str(args.raw_dir.resolve()),
            "output_dir": str(output_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "best_joint_acc": best_joint,
            "history": history,
            "num_train_images": len(train_ds),
            "num_test_images": len(test_ds),
        },
        output_dir / "metrics.json",
    )
    print(f"Run directory: {output_dir}")
    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
