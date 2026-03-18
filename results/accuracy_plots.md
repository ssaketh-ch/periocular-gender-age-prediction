# Results & Model Comparisons

All models trained on GPU (CUDA) with batch size 64 for 25–50 epochs on Kaggle.

---

## Gender Classification

**Dataset**: UBIPeriocular (balanced, 13168 train / 1020 test)  
**Metric**: Test accuracy (correct gender predictions / total)

### Accuracy over 25 Epochs

| Epoch | ResNet-18 | ResNet-34 | ResNet-50 | PeriGender |
|-------|-----------|-----------|-----------|------------|
| 1 | 81.5% | 96.3% | 94.1% | 84.6% |
| 5 | 87.5% | 94.9% | 81.5% | 89.7% |
| 10 | 90.2% | 94.7% | 89.8% | 90.2% |
| 15 | 92.4% | 93.1% | 93.6% | 90.8% |
| 20 | 93.4% | 94.7% | 87.8% | 91.8% |
| 25 | 91.9% | 92.5% | 91.5% | 93.2% |

**Training configuration**:
- Optimizer: SGD, momentum=0.9, weight\_decay=0.01
- lr: 0.01 (ResNet) / 0.0001 (PeriGender)
- Loss: CrossEntropyLoss

**Key observations**:
- ResNet-34 and ResNet-50 converge faster due to ImageNet pretraining
- PeriGender catches up but shows more epoch-to-epoch variance
- All models achieve >90% test accuracy by epoch 15–20

---

## Age Classification

**Dataset**: UTKFace (10 age-decade buckets)  
**Metric**: Test accuracy (correct age-bucket / total)

### Accuracy over ~27 Epochs

| Epoch | ResNet-18 | ResNet-34 | ResNet-50 | PeriAge |
|-------|-----------|-----------|-----------|---------|
| 1 | 54.7% | 55.6% | 55.0% | 56.8% |
| 5 | 59.7% | 60.8% | 58.5% | 40.2% |
| 10 | 58.5% | 59.8% | 60.1% | 57.2% |
| 15 | 57.7% | 58.4% | 60.1% | 46.6% |
| 20 | 58.6% | 58.6% | 59.2% | 49.2% |
| 27 | 57.9% | 58.9% | 58.5% | 50.0% |

**Training configuration**:
- Optimizer: SGD, momentum=0.9
- lr: 0.001
- Loss: CrossEntropyLoss

**Key observations**:
- Age prediction is significantly harder than gender — even ResNet plateaus ~60%
- PeriAge shows high variance across epochs, suggesting the architecture needs further tuning for age
- The decade-bucket formulation collapses fine-grained age information

---

## Combined PeriOcular (Gender + Age)

**Metric**: Test accuracy where **both** gender AND age must be correct

| Epoch | PeriOcular |
|-------|-----------|
| 1 | 16.4% |
| 5 | 28.2% |
| 10 | 30.5% |
| 15 | 31.5% |
| 20 | 33.1% |
| 25 | 32.9% |

**Training configuration**:
- Optimizer: Adam, lr=0.01
- Loss: `CrossEntropy(gender) + CrossEntropy(age)`

**Key observations**:
- Combined accuracy is lower because it requires both tasks to be simultaneously correct
- Steady improvement through 25 epochs
- The architecture successfully learns both tasks jointly from a single backbone

---

## Summary Table

| Model | Task | Best Acc | Epochs |
|-------|------|----------|--------|
| ResNet-18 | Gender | 93.4% | 25 |
| ResNet-34 | Gender | 96.8% | 1 |
| ResNet-50 | Gender | 96.9% | 4 |
| PeriGender | Gender | 93.2% | 23 |
| ResNet-18 | Age | 60.1% | 9 |
| ResNet-34 | Age | 60.8% | 5 |
| ResNet-50 | Age | 60.6% | 16 |
| PeriAge | Age | 57.2% | 11 |
| PeriOcular | Gender+Age | 33.4% | 19 |
