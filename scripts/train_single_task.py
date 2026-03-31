from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.periage import PeriAge, PeriAgeResNet34, PeriAgeV2
from models.perigender import PeriGender, PeriGenderV2
from scripts.common import resolve_run_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single-task gender or age model.")
    parser.add_argument("--task", choices=["gender", "age"], required=True)
    parser.add_argument("--model", choices=["resnet18", "resnet34", "resnet50", "perigender", "perigender_v2", "periage", "periage_v2", "periage_resnet34"], required=True)
    parser.add_argument("--data-dir", required=True, type=Path, help="Prepared dataset root containing train/ and test/.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--backbone-lr-mult", type=float, default=1.0, help="Multiplier for pretrained backbone LR when the model exposes backbone/head parameter groups.")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0, help="Freeze pretrained backbone for the first N epochs when supported by the model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-timestamp", action="store_true", help="Write outputs directly into --output-dir instead of a timestamped subfolder.")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weighting", choices=["none", "balanced"], default="none")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    return parser.parse_args()


def build_model(name: str, task: str) -> tuple[nn.Module, transforms.Compose, transforms.Compose, float]:
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        default_lr = 1e-3
    elif name == "resnet34":
        weights = ResNet34_Weights.DEFAULT
        model = models.resnet34(weights=weights)
        default_lr = 1e-3
    elif name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        default_lr = 1e-3
    elif name == "perigender":
        if task != "gender":
            raise ValueError("perigender can only be used with --task gender")
        model = PeriGender(num_classes=2)
        weights = None
        default_lr = 1e-4
    elif name == "perigender_v2":
        if task != "gender":
            raise ValueError("perigender_v2 can only be used with --task gender")
        model = PeriGenderV2(num_classes=2)
        weights = None
        default_lr = 3e-4
    elif name == "periage":
        if task != "age":
            raise ValueError("periage can only be used with --task age")
        model = PeriAge(num_classes=10)
        weights = None
        default_lr = 1e-3
    elif name == "periage_v2":
        if task != "age":
            raise ValueError("periage_v2 can only be used with --task age")
        model = PeriAgeV2(num_classes=10)
        weights = None
        default_lr = 3e-4
    elif name == "periage_resnet34":
        if task != "age":
            raise ValueError("periage_resnet34 can only be used with --task age")
        weights = ResNet34_Weights.DEFAULT
        model = PeriAgeResNet34(num_classes=10, weights=weights)
        default_lr = 3e-4
    else:
        raise ValueError(name)

    if name.startswith("resnet"):
        num_classes = 2 if task == "gender" else 10
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    if task == "gender":
        size = (224, 112)
    else:
        size = (224, 224)

    if weights is not None:
        mean, std = weights.transforms().mean, weights.transforms().std
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    train_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return model, train_transform, test_transform, default_lr


def build_class_weights(dataset: datasets.ImageFolder, device: str) -> torch.Tensor:
    counts = torch.zeros(len(dataset.classes), dtype=torch.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    return weights.to(device)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str, optimizer=None) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    running_loss = 0.0
    running_correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, running_correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = resolve_run_dir(args.output_dir, use_timestamp=not args.no_timestamp)

    model, train_transform, test_transform, default_lr = build_model(args.model, args.task)
    lr = args.lr if args.lr is not None else default_lr

    train_ds = datasets.ImageFolder(args.data_dir / "train", transform=train_transform)
    test_ds = datasets.ImageFolder(args.data_dir / "test", transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = model.to(args.device)
    class_weights = None
    if args.class_weighting == "balanced":
        class_weights = build_class_weights(train_ds, args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    if hasattr(model, "backbone_parameters") and hasattr(model, "head_parameters"):
        param_groups = [
            {"params": list(model.backbone_parameters()), "lr": lr * args.backbone_lr_mult},
            {"params": list(model.head_parameters()), "lr": lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": lr}]

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    history = []
    best_acc = -1.0
    best_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        if hasattr(model, "set_backbone_trainable"):
            model.set_backbone_trainable(epoch > args.freeze_backbone_epochs)
        train_loss, train_acc = run_epoch(model, train_loader, criterion, args.device, optimizer=optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, args.device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_name": args.model,
                    "task": args.task,
                    "class_to_idx": train_ds.class_to_idx,
                    "state_dict": model.state_dict(),
                },
                best_path,
            )
        if scheduler is not None:
            scheduler.step()

    save_json(
        {
            "task": args.task,
            "model": args.model,
            "data_dir": str(args.data_dir.resolve()),
            "output_dir": str(output_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": lr,
            "optimizer": args.optimizer,
            "backbone_lr_mult": args.backbone_lr_mult,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "label_smoothing": args.label_smoothing,
            "class_weighting": args.class_weighting,
            "scheduler": args.scheduler,
            "min_lr": args.min_lr,
            "best_test_acc": best_acc,
            "history": history,
            "class_to_idx": train_ds.class_to_idx,
        },
        output_dir / "metrics.json",
    )
    print(f"Run directory: {output_dir}")
    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
