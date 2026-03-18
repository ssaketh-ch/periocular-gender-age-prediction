# Model Architecture

## Background

The periocular region (the area surrounding the eye within the orbital socket) is a highly discriminative biometric region. It contains rich texture information from the eyelids, eyebrows, eye corners, and surrounding skin — making it effective for attributes like gender and age even under partial face occlusion.

---

## Shared Building Blocks

### ResBlock

The core residual block used across all custom models:

```
Input
  │
  ├─ Conv2d(k=3, stride) → BatchNorm → ReLU
  │
  └─ Conv2d(k=3, stride=1) → BatchNorm → ReLU
  │
Output
```

No identity/shortcut skip connection is used here (unlike standard ResNet). The residual signal is instead captured by the dedicated `SkipConnection` modules that bypass the ResBlocks entirely.

### SkipConnection Modules

Four skip connections capture multi-scale features from progressively deeper layers:

| Module | Input → Operation | Purpose |
|--------|-------------------|---------|
| `SkipConnection1` | `Conv(3×3) → MaxPool(8, s=8)` | Early low-level features |
| `SkipConnection2` | `Conv(1×1) → MaxPool(5, s=4)` | Mid-level features |
| `SkipConnection3` | `Conv(1×1) → MaxPool(4, s=2)` | High-level features |
| `SkipConnection4` | `Conv(1×1) → MaxPool((1,2))` | Deep features |

All skip connections output **3 channels** and are designed to produce a `7×7` spatial map, enabling concatenation before the average pool.

---

## PeriGender

**Task**: Binary gender classification (female / male)  
**Input**: `(B, 3, 224, 112)` — periocular images with 2:1 aspect ratio  
**Output**: `(B, 2)` logits

```
Input (3, 224, 112)
│
Conv2d(64, k=3, s=2) → MaxPool(3, s=2)          → (64, 56, 28)
│
├─ Skip1 ────────────────────────────────────────→ (3, 7, 7)
└─ ResBlock(64→64, s=2) → ResBlock(64→128)       → (128, 14, 14)
   │
   ├─ Skip2 ──────────────────────────────────── → (3, 7, 7)
   └─ ResBlock(128→128, s=2) → ResBlock(128→256) → (256, 7, 7)
      │
      ├─ Skip3 ────────────────────────────────  → (3, 7, 7)
      └─ ResBlock(256→256, s=2) → ResBlock(256→512) → (512, 4, 4)
         │
         ├─ Skip4 ──────────────────────────────→ (3, 7, 7)
         └─ MaxPool((1,2)) ───────────────────── → (512, 7, 7)
            │
Concat[skip1, skip2, skip3, skip4, main]        → (524, 7, 7)
AdaptiveAvgPool(1,1) → Dropout(0.5) → FC(524→2)
```

**Total concatenated channels**: 3 + 3 + 3 + 3 + 512 = **524**

---

## PeriAge

**Task**: Age-range classification (10 classes, decade buckets)  
**Input**: `(B, 3, 224, 224)` — square crop  
**Output**: `(B, 10)` logits

Identical backbone to PeriGender, with two differences:
1. `SkipConnection4` includes an `Upsample(7×7)` step to handle the different spatial resolution from 224×224 input.
2. The main path includes an additional `Upsample` (transposed conv) before concatenation.

**Age bucket mapping**:
```
0 → 1–10    1 → 11–20   2 → 21–30   3 → 31–40   4 → 41–50
5 → 51–60   6 → 61–70   7 → 71–80   8 → 81–90   9 → 91–100
```

---

## PeriOcular (Multi-Task)

**Task**: Joint gender + age prediction  
**Input**: `(B, 3, 224, 112)`  
**Output**: `(gender_logits (B,2), age_logits (B,10))`

Identical backbone to PeriGender. Two independent FC heads branch off the shared feature vector after `AdaptiveAvgPool + Dropout`:

```
Shared backbone (same as PeriGender)
         │
   feature vector (524,)
       │       │
    FC(→2)   FC2(→10)
    gender    age
```

**Training Loss**: `L = CrossEntropy(gender) + CrossEntropy(age)`  
**Optimizer**: Adam, `lr=0.01`

---

## Comparison to Standard ResNet

| Property | PeriGender | ResNet-18/34/50 |
|----------|-----------|----------------|
| Pretrained weights | ✗ | ✓ (ImageNet) |
| Skip connections | Multi-scale (4 branches) | Per-block identity shortcuts |
| Input size | 224×112 | 224×224 (default) |
| Params (est.) | ~8M | 11M / 21M / 25M |
| Multi-task heads | ✓ (PeriOcular) | ✗ (single head) |
