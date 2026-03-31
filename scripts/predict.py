from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.periage import PeriAge, PeriAgeResNet34, PeriAgeV2
from models.perigender import PeriGender, PeriGenderV2
from models.periocular import PeriOcular
from scripts.common import bucket_description


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference from a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(payload: dict):
    model_name = payload["model_name"]
    task = payload["task"]

    if model_name == "perigender":
        model = PeriGender()
        transform = transforms.Compose(
            [
                transforms.Resize((224, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    elif model_name == "perigender_v2":
        model = PeriGenderV2()
        transform = transforms.Compose(
            [
                transforms.Resize((224, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    elif model_name == "periage":
        model = PeriAge()
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    elif model_name == "periage_v2":
        model = PeriAgeV2()
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    elif model_name == "periage_resnet34":
        model = PeriAgeResNet34(weights=None)
        weights = ResNet34_Weights.DEFAULT
        mean, std = weights.transforms().mean, weights.transforms().std
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif model_name == "periocular":
        model = PeriOcular()
        transform = transforms.Compose(
            [
                transforms.Resize((224, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    elif model_name == "resnet18":
        model = models.resnet18(weights=None)
        weights = ResNet18_Weights.DEFAULT
        mean, std = weights.transforms().mean, weights.transforms().std
        size = (224, 112) if task == "gender" else (224, 224)
        model.fc = torch.nn.Linear(model.fc.in_features, 2 if task == "gender" else 10)
        transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
        weights = ResNet34_Weights.DEFAULT
        mean, std = weights.transforms().mean, weights.transforms().std
        size = (224, 112) if task == "gender" else (224, 224)
        model.fc = torch.nn.Linear(model.fc.in_features, 2 if task == "gender" else 10)
        transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean, std)])
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        weights = ResNet50_Weights.DEFAULT
        mean, std = weights.transforms().mean, weights.transforms().std
        size = (224, 112) if task == "gender" else (224, 224)
        model.fc = torch.nn.Linear(model.fc.in_features, 2 if task == "gender" else 10)
        transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        raise ValueError(f"Unsupported checkpoint model_name={model_name}")

    model.load_state_dict(payload["state_dict"])
    return model, transform, task, payload.get("class_to_idx")


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    model, transform, task, class_to_idx = load_model(payload)
    model = model.to(args.device).eval()

    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(args.device)

    with torch.no_grad():
        output = model(tensor)

    if task == "multitask":
        gender_logits, age_logits = output
        gender_idx = torch.argmax(gender_logits, dim=1).item()
        age_idx = torch.argmax(age_logits, dim=1).item()
        gender_label = "male" if gender_idx == 0 else "female"
        print(f"gender={gender_label}")
        print(f"age_bucket={bucket_description(age_idx)}")
        return

    pred_idx = torch.argmax(output, dim=1).item()
    if task == "gender" and class_to_idx is not None:
        idx_to_class = {idx: label for label, idx in class_to_idx.items()}
        print(f"prediction={idx_to_class[pred_idx]}")
    elif task == "age":
        print(f"prediction={bucket_description(pred_idx)}")
    else:
        print(f"prediction={pred_idx}")


if __name__ == "__main__":
    main()
