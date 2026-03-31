from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from common import age_to_bucket, bucket_name, ensure_dir, parse_utkface_filename, save_json, set_seed, split_items


def list_images(raw_dir: Path) -> list[Path]:
    images = sorted([path for path in raw_dir.rglob("*.jpg") if path.is_file()])
    if not images:
        raise FileNotFoundError(f"No .jpg files found under {raw_dir}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare UTKFace age dataset into ImageFolder bucket directories.")
    parser.add_argument("--raw-dir", required=True, type=Path, help="Directory containing UTKFace .jpg files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write train/test splits.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of examples to reserve for test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--min-age", type=int, default=0, help="Discard images younger than this age.")
    parser.add_argument("--max-age", type=int, default=120, help="Discard images older than this age.")
    args = parser.parse_args()

    set_seed(args.seed)
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()

    images = []
    for image_path in list_images(raw_dir):
        try:
            age, _ = parse_utkface_filename(image_path)
        except Exception:
            continue
        if args.min_age <= age <= args.max_age:
            images.append(image_path)

    if not images:
        raise RuntimeError("No valid UTKFace images remained after filtering")

    train_images, test_images = split_items(images, args.test_size, args.seed)
    summary = {"train": Counter(), "test": Counter()}

    for split_name, split_images in (("train", train_images), ("test", test_images)):
        for image_path in split_images:
            age, _ = parse_utkface_filename(image_path)
            bucket_idx = age_to_bucket(age)
            target_dir = ensure_dir(output_dir / split_name / bucket_name(bucket_idx))
            shutil.copy2(image_path, target_dir / image_path.name)
            summary[split_name][bucket_name(bucket_idx)] += 1

    save_json(
        {
            "raw_dir": str(raw_dir),
            "output_dir": str(output_dir),
            "seed": args.seed,
            "test_size": args.test_size,
            "min_age": args.min_age,
            "max_age": args.max_age,
            "train_counts": dict(summary["train"]),
            "test_counts": dict(summary["test"]),
        },
        output_dir / "summary.json",
    )
    print(f"Prepared UTKFace age split at {output_dir}")
    print(f"Train counts: {dict(summary['train'])}")
    print(f"Test counts: {dict(summary['test'])}")


if __name__ == "__main__":
    main()

