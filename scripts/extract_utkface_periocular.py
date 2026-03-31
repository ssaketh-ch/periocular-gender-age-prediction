from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2

from common import ensure_dir, save_json


class PeriocularCropper:
    def __init__(self, strict: bool = False) -> None:
        cascades = Path(cv2.data.haarcascades)
        self.face_cascade = cv2.CascadeClassifier(str(cascades / "haarcascade_frontalface_default.xml"))
        self.eye_cascade = cv2.CascadeClassifier(str(cascades / "haarcascade_eye_tree_eyeglasses.xml"))
        self.strict = strict
        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError("Failed to load OpenCV Haar cascades")

    def _clip_box(self, image, x1, y1, x2, y2):
        height, width = image.shape[:2]
        x1 = max(0, min(width - 2, x1))
        y1 = max(0, min(height - 2, y1))
        x2 = max(x1 + 1, min(width - 1, x2))
        y2 = max(y1 + 1, min(height - 1, y2))
        return image[y1:y2, x1:x2]

    def _crop_from_eyes(self, image, face_box):
        x, y, w, h = face_box
        face_gray = cv2.cvtColor(image[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        eye_boxes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(12, 12),
        )

        eye_candidates = []
        for ex, ey, ew, eh in eye_boxes:
            center_x = x + ex + ew / 2.0
            center_y = y + ey + eh / 2.0
            if center_y > y + h * 0.65:
                continue
            eye_candidates.append((center_x, center_y, ew, eh))

        if len(eye_candidates) < 2:
            return None

        eye_candidates.sort(key=lambda item: item[2] * item[3], reverse=True)
        eye_candidates = eye_candidates[:6]

        best_pair = None
        best_score = None
        for i in range(len(eye_candidates)):
            for j in range(i + 1, len(eye_candidates)):
                left, right = sorted([eye_candidates[i], eye_candidates[j]], key=lambda item: item[0])
                dx = right[0] - left[0]
                dy = abs(right[1] - left[1])
                if dx < w * 0.12:
                    continue
                if dy > h * 0.18:
                    continue
                score = dx - 2.5 * dy
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (left, right)

        if best_pair is None:
            return None

        left, right = best_pair
        interocular = right[0] - left[0]
        center_x = (left[0] + right[0]) / 2.0
        center_y = (left[1] + right[1]) / 2.0

        crop_half_width = interocular * 1.15
        crop_half_height = interocular * 0.65

        x1 = int(round(center_x - crop_half_width))
        x2 = int(round(center_x + crop_half_width))
        y1 = int(round(center_y - crop_half_height))
        y2 = int(round(center_y + crop_half_height * 0.85))
        return self._clip_box(image, x1, y1, x2, y2)

    def _fallback_crop(self, image, face_box=None):
        height, width = image.shape[:2]
        if face_box is not None:
            x, y, w, h = face_box
            x1 = int(x + 0.10 * w)
            x2 = int(x + 0.90 * w)
            y1 = int(y + 0.12 * h)
            y2 = int(y + 0.52 * h)
        else:
            x1 = int(width * 0.12)
            x2 = int(width * 0.88)
            y1 = int(height * 0.16)
            y2 = int(height * 0.52)
        return self._clip_box(image, x1, y1, x2, y2)

    def extract(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(60, 60),
        )
        face_box = None
        if len(faces) > 0:
            face_box = max(faces, key=lambda box: box[2] * box[3])
            crop = self._crop_from_eyes(image, face_box)
            if crop is not None and crop.size > 0:
                return crop, "eye_detector"
        if self.strict:
            return None, "strict_skip"
        return self._fallback_crop(image, face_box), "fallback"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract periocular crops from UTKFace images.")
    parser.add_argument("--raw-dir", required=True, type=Path, help="Directory containing raw UTKFace images.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write periocular crops.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--strict", action="store_true", help="Keep only images where the eye detector found a periocular crop.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    cropper = PeriocularCropper(strict=args.strict)

    image_paths = sorted([path for path in raw_dir.rglob("*.jpg") if path.is_file()])
    if not image_paths:
        raise FileNotFoundError(f"No .jpg files found under {raw_dir}")

    stats = Counter()
    for idx, image_path in enumerate(image_paths, start=1):
        relative_path = image_path.relative_to(raw_dir)
        target_path = output_dir / relative_path
        ensure_dir(target_path.parent)
        if target_path.exists() and not args.overwrite:
            stats["skipped_existing"] += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            stats["failed_read"] += 1
            continue

        crop, method = cropper.extract(image)
        if crop is None or crop.size == 0:
            stats["failed_crop"] += 1
            stats[method] += 1
            continue

        ok = cv2.imwrite(str(target_path), crop)
        if not ok:
            stats["failed_write"] += 1
            continue

        stats["written"] += 1
        stats[method] += 1
        if idx % 5000 == 0:
            print(f"processed={idx}/{len(image_paths)} written={stats['written']} eye_detector={stats['eye_detector']} fallback={stats['fallback']}")

    save_json(
        {
            "raw_dir": str(raw_dir),
            "output_dir": str(output_dir),
            "stats": dict(stats),
        },
        output_dir / "periocular_summary.json",
    )
    print(f"Finished periocular extraction to {output_dir}")
    print(dict(stats))


if __name__ == "__main__":
    main()
