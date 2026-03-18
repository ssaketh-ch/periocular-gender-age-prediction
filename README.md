
# Periocular Region Gender & Age Prediction

This repository contains the code and documentation for an **undergraduate research project** on gender and age prediction from periocular images, inspired by the work of Hussain et al. (2023) — ["Unconstrained Gender Recognition from Periocular Region Using Multiscale Deep Features"](https://www.techscience.com/iasc/v35n3/49392/html).

A deep learning project that predicts **gender** and **age** from periocular images — the region surrounding the eye within the orbit. This biometrically rich area has shown strong performance in deep learning classification tasks, especially in unconstrained environments where the face may be partially occluded.

---

## Overview

This project implements and benchmarks multiple CNN architectures for two classification tasks:

| Task | Output | Classes |
|------|--------|---------|
| Gender Classification | Male / Female | 2 |
| Age Classification | Age range (decade buckets) | 10 |

Both tasks are ultimately unified into a **single multi-task model** (`PeriOcular`) that predicts gender and age simultaneously.

---

## Models

### Baseline — Pretrained ResNets (Transfer Learning)
Fine-tuned ImageNet-pretrained weights on the periocular dataset:
- **ResNet-18**
- **ResNet-34**
- **ResNet-50**


### Custom — PeriGender
A custom residual architecture inspired by the paper:
> *"Unconstrained Gender Recognition from Periocular Region Using Multiscale Deep Features"*  
> ([Paper PDF](https://file.techscience.com/ueditor/files/iasc/TSP_IASC-35-3/TSP_IASC_30036/TSP_IASC_30036.pdf))

Architecture: `Conv → MaxPool → 4× [SkipConnection + ResBlock pairs] → AdaptiveAvgPool → FC(2)`

See [docs/architecture.md](docs/architecture.md) for a detailed breakdown and [figures/](figures/) for architecture diagrams from the original paper.

### Custom — PeriAge
A modified PeriGender model adapted for 10-class age-range prediction, with an added `Upsample` layer to handle spatial dimension matching.

### Custom — PeriOcular (Multi-Task)
Combines PeriGender and PeriAge into a single shared-backbone model with **two output heads**:
- `fc` → 2 classes (gender)
- `fc2` → 10 classes (age range)

---

## Results

### Gender Classification (Test Accuracy)

| Model | Best Test Acc |
|-------|--------------|
| ResNet-18 | ~93.4% |
| ResNet-34 | ~96.8% |
| ResNet-50 | ~96.9% |
| PeriGender | ~93.2% |

### Age Classification (Test Accuracy)

| Model | Best Test Acc |
|-------|--------------|
| ResNet-18 | ~60.1% |
| ResNet-34 | ~60.8% |
| ResNet-50 | ~60.6% |
| PeriAge | ~57.2% |

### Combined PeriOcular (Gender + Age, Test Accuracy)

| Model | Best Test Acc |
|-------|--------------|
| PeriOcular | ~32.7% |

> Note: The combined accuracy requires *both* gender and age to be correct simultaneously, so lower values are expected.

See [`results/`](results/) for accuracy-vs-epoch plots and tables.

---

## Dataset

- **Gender**: [UBIPeriocular Dataset](https://ubilab-security.github.io/) — images organized into `male/` and `female/` class folders.
- **Age**: [UTKFace Dataset](https://susanqq.github.io/UTKFace/) — filenames encode age, gender, race labels.

### Dataset Statistics (after balancing)
| Split | Male | Female | Total |
|-------|------|--------|-------|
| Train | 6584 | 6584\* | 13168 |
| Test | 709 | 311 | 1020 |

\*Female images were augmented to balance the dataset.


### Preprocessing & Augmentation
- **Face normalization**: Align faces using eye center coordinates (see [docs/architecture.md](docs/architecture.md) and original paper, Fig. 4)
- **Periocular region extraction**: Crop ROI using empirically determined bounds (see Fig. 5)
- **Resize**: `224×112` (gender/multi-task) or `224×224` (age)
- **Label extraction**: Gender from UBIPeriocular `.txt` files (7th line), age from UTKFace filename
- **Class balancing**: Augment minority class (female) to match male samples
- **Augmentation**: Random horizontal flip, invert, solarize, crop
- **Normalize**: `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`

---


## Project Report

The full project report, detailing methodology, experiments, results, and discussion, is available here:

**[Periocular Gender & Age Prediction Project Report (PDF)](./Periocular_Gender_Age_Project_Report.pdf)**

This document provides a comprehensive overview of the research, including background, dataset details, model architectures, training procedures, results, and analysis.

---

## Project Structure

```
periocular-gender-age-prediction/
├── notebooks/
│   └── periocular-final.ipynb    # Full training & evaluation notebook
├── models/
│   ├── perigender.py             # PeriGender architecture
│   ├── periage.py                # PeriAge architecture
│   └── periocular.py             # PeriOcular multi-task architecture
├── docs/
│   ├── architecture.md           # Detailed model architecture notes
│   └── dataset.md                # Dataset description & preprocessing
├── results/
│   └── accuracy_plots.md         # Accuracy results and comparisons
├── figures/                      # Architecture diagrams (see README inside)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ssaketh-ch/periocular-gender-age-prediction.git
cd periocular-gender-age-prediction
pip install -r requirements.txt
```

---

## Usage

Open the notebook to run the full pipeline:

```bash
jupyter notebook notebooks/periocular-final.ipynb
```

Or to use a saved model for inference:

```python
from models.perigender import PeriGender
import torch
from PIL import Image
from torchvision import transforms

model = PeriGender()
model.load_state_dict(torch.load('perigender_v2.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

image = Image.open('your_image.jpg')
tensor = transform(image).unsqueeze(0)
output = model(tensor)
predicted = torch.argmax(output).item()
print(['female', 'male'][predicted])
```

---

## Requirements

See [`requirements.txt`](requirements.txt)

---


## References

- [Unconstrained Gender Recognition from Periocular Region Using Multiscale Deep Features](https://file.techscience.com/ueditor/files/iasc/TSP_IASC-35-3/TSP_IASC_30036/TSP_IASC_30036.pdf)
- [Periocular Gender & Age Prediction Project Report (PDF)](./Periocular_Gender_Age_Project_Report.pdf)
- [UBIPeriocular Dataset](https://ubilab-security.github.io/)
- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- [Deep Residual Learning for Image Recognition (He et al.)](https://arxiv.org/abs/1512.03385)

---

## Author

**Sai Saketh Cherukuri**
