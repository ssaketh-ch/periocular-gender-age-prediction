# Periocular Gender & Age Prediction: Model Architecture

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Context](#project-context)
3. [Overview](#overview)
4. [Preprocessing Steps](#preprocessing-steps-from-project-report-and-paper)
5. [Model Building Blocks](#shared-building-blocks)
6. [Model Architectures](#model-architectures)
7. [Comparison to Standard ResNet](#comparison-to-standard-resnet)

---

## Executive Summary

This document details the architecture and methodology for periocular-based gender and age prediction using deep convolutional neural networks. The approach is inspired by the PeriGender model ([Hussain et al., 2023](https://www.techscience.com/iasc/v35n3/49392/html)), which fuses multi-scale features via skip connections. The pipeline includes robust preprocessing, custom model design, and multi-task learning, with all steps and design choices documented for reproducibility and clarity.

---

## Project Context

This repository is the result of an undergraduate research project focused on advancing gender and age prediction from periocular images. The work is based on and extends the methodology of Hussain et al. (2023): ["Unconstrained Gender Recognition from Periocular Region Using Multiscale Deep Features"](https://www.techscience.com/iasc/v35n3/49392/html).


## Overview

The periocular region (area surrounding the eye) is a robust biometric source, rich in texture and structure, enabling reliable gender and age prediction even under occlusion. This project implements a custom deep CNN (PeriGender) that fuses multi-scale features via skip connections, benchmarked against standard ResNets and extended for age and multi-task learning.

### Key Figures (from original paper)

- **PeriGender Model Architecture**:  
   ![PeriGender Architecture](https://cdn.techscience.cn/ueditor/files/iasc/TSP_IASC-35-3/TSP_IASC_30036/TSP_IASC_30036/Images/IASC_30036-fig-6.png/origin_webp)
- **Preprocessing Pipeline**:  
   ![Preprocessing Pipeline](https://cdn.techscience.cn/ueditor/files/iasc/TSP_IASC-35-3/TSP_IASC_30036/TSP_IASC_30036/Images/IASC_30036-fig-5.png/origin_webp)
- **Face Normalization**:  
   ![Face Normalization](https://cdn.techscience.cn/ueditor/files/iasc/TSP_IASC-35-3/TSP_IASC_30036/TSP_IASC_30036/Images/IASC_30036-fig-4.png/origin_webp)

See [figures/](../figures/) for more, and the [original paper](https://www.techscience.com/iasc/v35n3/49392/html) for full details.

## Background

The periocular region (the area surrounding the eye within the orbital socket) is a highly discriminative biometric region. It contains rich texture information from the eyelids, eyebrows, eye corners, and surrounding skin — making it effective for attributes like gender and age even under partial face occlusion.

---



## Preprocessing Steps (from Project Report and Paper)

1. **Face Normalization**: Align faces using eye center coordinates to correct for scale and rotation ([see Fig. 4](https://cdn.techscience.cn/ueditor/files/iasc/TSP_IASC-35-3/TSP_IASC_30036/TSP_IASC_30036/Images/IASC_30036-fig-4.png/origin_webp)).
2. **Periocular Region Extraction**: Crop the region of interest (ROI) around the eyes, either as left/right or whole periocular region, using empirically determined bounds ([see Fig. 5](https://cdn.techscience.cn/ueditor/files/iasc/TSP_IASC-35-3/TSP_IASC_30036/TSP_IASC_30036/Images/IASC_30036-fig-5.png/origin_webp)).
3. **Resizing**: Standardize periocular images to 224×112 (gender/multi-task) or 224×224 (age) to preserve aspect ratio and spatial features.
4. **Label Extraction**:
   - **Gender**: For UBIPeriocular, read the 7th line of each `.txt` metadata file (male/female).
   - **Age**: For UTKFace, parse the age from the filename and map to decade buckets.
5. **Data Augmentation**: Random horizontal flip, invert, solarize, crop, and normalization to balance classes and improve generalization.
6. **Class Balancing**: Augment minority class (female) to match the number of male samples in training.

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

Unlike standard ResNet, no identity/shortcut skip connection is used here. Instead, the residual signal is captured by dedicated `SkipConnection` modules that bypass the ResBlocks entirely.

### SkipConnection Modules

Four skip connections capture multi-scale features from progressively deeper layers:

| Module            | Input → Operation                | Purpose                |
|-------------------|----------------------------------|------------------------|
| `SkipConnection1` | `Conv(3×3) → MaxPool(8, s=8)`    | Early low-level        |
| `SkipConnection2` | `Conv(1×1) → MaxPool(5, s=4)`    | Mid-level              |
| `SkipConnection3` | `Conv(1×1) → MaxPool(4, s=2)`    | High-level             |
| `SkipConnection4` | `Conv(1×1) → MaxPool((1,2))`     | Deep features          |

All skip connections output **3 channels** and are designed to produce a `7×7` spatial map, enabling concatenation before the average pool.

---

## Model Architectures


### PeriGender

**Task**: Binary gender classification (female / male)  
**Input**: `(B, 3, 224, 112)` — periocular images, 2:1 aspect ratio  
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

### PeriAge

**Task**: Age-range classification (10 classes, decade buckets)  
**Input**: `(B, 3, 224, 224)` — square crop  
**Output**: `(B, 10)` logits

Backbone is identical to PeriGender, with two differences:
1. `SkipConnection4` includes an `Upsample(7×7)` step to handle the different spatial resolution from 224×224 input.
2. The main path includes an additional `Upsample` (transposed conv) before concatenation.

**Age bucket mapping**:
```
0 → 1–10    1 → 11–20   2 → 21–30   3 → 31–40   4 → 41–50
5 → 51–60   6 → 61–70   7 → 71–80   8 → 81–90   9 → 91–100
```

---

### PeriOcular (Multi-Task)

**Task**: Joint gender + age prediction  
**Input**: `(B, 3, 224, 112)`  
**Output**: `(gender_logits (B,2), age_logits (B,10))`

Backbone is identical to PeriGender. Two independent FC heads branch off the shared feature vector after `AdaptiveAvgPool + Dropout`:

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

| Property           | PeriGender                | ResNet-18/34/50         |
|--------------------|---------------------------|-------------------------|
| Pretrained weights | ✗                         | ✓ (ImageNet)            |
| Skip connections   | Multi-scale (4 branches)  | Per-block identity      |
| Input size         | 224×112                   | 224×224 (default)       |
| Params (est.)      | ~8M                       | 11M / 21M / 25M         |
| Multi-task heads   | ✓ (PeriOcular)            | ✗ (single head)         |

---

## References

Hussain, M., Alrabiah, R., & AboAlSamh, H. A. (2023). Unconstrained Gender Recognition from Periocular Region Using Multiscale Deep Features. *Intelligent Automation & Soft Computing*, 35(3), 2941-2962. [https://doi.org/10.32604/iasc.2023.030036](https://doi.org/10.32604/iasc.2023.030036)
