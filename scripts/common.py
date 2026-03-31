from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch


AGE_BUCKETS = [
    (0, 10, "00_10"),
    (11, 20, "11_20"),
    (21, 30, "21_30"),
    (31, 40, "31_40"),
    (41, 50, "41_50"),
    (51, 60, "51_60"),
    (61, 70, "61_70"),
    (71, 80, "71_80"),
    (81, 90, "81_90"),
    (91, 120, "91_120"),
]


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_items(items: Iterable[Path], test_size: float, seed: int) -> tuple[list[Path], list[Path]]:
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    split_idx = int(len(items) * (1.0 - test_size))
    split_idx = max(1, min(split_idx, len(items) - 1))
    return items[:split_idx], items[split_idx:]


def parse_ubipr_gender(txt_path: str | Path) -> str:
    lines = Path(txt_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 7:
        raise ValueError(f"{txt_path} has fewer than 7 lines")
    label = lines[6].replace(";", "").strip().lower()
    if label not in {"male", "female"}:
        raise ValueError(f"Unexpected gender label '{label}' in {txt_path}")
    return label


def parse_utkface_filename(image_path: str | Path) -> tuple[int, int]:
    parts = Path(image_path).name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected UTKFace filename format: {image_path}")
    age = int(parts[0])
    gender = int(parts[1])
    if gender not in {0, 1}:
        raise ValueError(f"Unexpected UTKFace gender label '{gender}' in {image_path}")
    return age, gender


def age_to_bucket(age: int) -> int:
    for idx, (lower, upper, _) in enumerate(AGE_BUCKETS):
        if lower <= age <= upper:
            return idx
    return len(AGE_BUCKETS) - 1


def bucket_name(bucket_idx: int) -> str:
    return AGE_BUCKETS[bucket_idx][2]


def bucket_description(bucket_idx: int) -> str:
    lower, upper, _ = AGE_BUCKETS[bucket_idx]
    return f"{lower}-{upper}"


def save_json(payload: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_run_dir(base_dir: str | Path, use_timestamp: bool = True) -> Path:
    base_dir = Path(base_dir).resolve()
    if not use_timestamp:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    run_dir = base_dir / timestamp_slug()
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
