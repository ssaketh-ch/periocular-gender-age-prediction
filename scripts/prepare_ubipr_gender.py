from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from common import ensure_dir, parse_ubipr_gender, save_json, set_seed, split_items


def list_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(raw_dir.glob("*.jpg")):
        txt_path = image_path.with_suffix(".txt")
        if txt_path.exists():
            pairs.append((image_path, txt_path))
    if not pairs:
        raise FileNotFoundError(f"No .jpg/.txt pairs found in {raw_dir}")
    return pairs


def copy_pair(image_path: Path, txt_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, destination / image_path.name)
    shutil.copy2(txt_path, destination / txt_path.name)


def subject_id_from_path(image_path: Path) -> str:
    return image_path.stem.split("_")[0]


def augment_female_samples(train_female_dir: Path, target_count: int) -> int:
    image_paths = sorted(train_female_dir.glob("*.jpg"))
    if not image_paths:
        return 0

    created = 0
    idx = 0
    while len(image_paths) + created < target_count:
        source = image_paths[idx % len(image_paths)]
        image = Image.open(source).convert("RGB")
        augmented = ImageOps.mirror(image) if idx % 2 == 0 else ImageOps.autocontrast(image)
        out_name = f"{source.stem}_aug_{created:05d}.jpg"
        augmented.save(train_female_dir / out_name)
        created += 1
        idx += 1
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UBIPeriocular gender dataset into ImageFolder format.")
    parser.add_argument("--raw-dir", required=True, type=Path, help="Directory containing raw UBIPeriocular .jpg/.txt files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write train/test splits.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Fraction of examples to reserve for test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--balance-train", action="store_true", help="Augment the female class in train to match male count.")
    parser.add_argument("--split-by-subject", action="store_true", help="Split by subject ID instead of by image to avoid identity leakage.")
    args = parser.parse_args()

    set_seed(args.seed)
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()

    pairs = list_pairs(raw_dir)
    pair_lookup = {image: txt for image, txt in pairs}
    split_strategy = "image"

    if args.split_by_subject:
        subject_to_images: dict[str, list[Path]] = {}
        for image_path, _ in pairs:
            subject_to_images.setdefault(subject_id_from_path(image_path), []).append(image_path)
        train_subjects, test_subjects = split_items(sorted(subject_to_images.keys()), args.test_size, args.seed)
        train_subjects = set(train_subjects)
        test_subjects = set(test_subjects)
        train_pairs = []
        test_pairs = []
        for subject_id, image_paths in subject_to_images.items():
            if subject_id in train_subjects:
                train_pairs.extend(image_paths)
            elif subject_id in test_subjects:
                test_pairs.extend(image_paths)
        split_strategy = "subject"
    else:
        train_pairs, test_pairs = split_items([image for image, _ in pairs], args.test_size, args.seed)

    summary = {"train": Counter(), "test": Counter(), "augmented_female_images": 0}

    for split_name, image_paths in (("train", train_pairs), ("test", test_pairs)):
        for image_path in image_paths:
            txt_path = pair_lookup[image_path]
            label = parse_ubipr_gender(txt_path)
            copy_pair(image_path, txt_path, output_dir / split_name / label)
            summary[split_name][label] += 1

    if args.balance_train:
        male_count = summary["train"]["male"]
        female_count = summary["train"]["female"]
        if female_count < male_count:
            created = augment_female_samples(output_dir / "train" / "female", male_count)
            summary["augmented_female_images"] = created
            summary["train"]["female"] += created

    save_json(
        {
            "raw_dir": str(raw_dir),
            "output_dir": str(output_dir),
            "seed": args.seed,
            "test_size": args.test_size,
            "split_strategy": split_strategy,
            "train_counts": dict(summary["train"]),
            "test_counts": dict(summary["test"]),
            "augmented_female_images": summary["augmented_female_images"],
        },
        output_dir / "summary.json",
    )
    print(f"Prepared UBIPeriocular split at {output_dir}")
    print(f"Train counts: {dict(summary['train'])}")
    print(f"Test counts: {dict(summary['test'])}")


if __name__ == "__main__":
    main()
