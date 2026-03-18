# Periocular Region Gender & Age Prediction

A deep learning project that predicts **gender** and **age** from periocular images — the region surrounding the eye within the orbit. This biometrically rich area has shown strong performance in deep learning classification tasks.

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
- Resize to `224×112` (periocular aspect ratio)
- Random horizontal flip, random invert, random solarize, random crop
- Normalize: `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`

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
├── figures/                      # Architecture diagrams
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
- [UBIPeriocular Dataset](https://ubilab-security.github.io/)
- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- [Deep Residual Learning for Image Recognition (He et al.)](https://arxiv.org/abs/1512.03385)

---

## Author

**Sai Saketh Cherukuri**
