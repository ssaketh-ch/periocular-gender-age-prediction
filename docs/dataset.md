# Dataset Description & Preprocessing

## Datasets Used

### 1. UBIPeriocular Dataset (Gender Task)

- **Source**: University of Beira Interior Periocular Dataset
- **Content**: Near-infrared and visible spectrum periocular images
- **Format**: `.jpg` images with accompanying `.txt` metadata files
- **Labels**: The 7th line of each `.txt` file contains the gender label (`male` / `female`)

**Raw distribution (before balancing)**:
| Split | Male | Female |
|-------|------|--------|
| Train | 6584 | 2584 |
| Test | 709 | 311 |

The dataset is significantly imbalanced. Female images are augmented to balance training.

---

### 2. UTKFace Dataset (Age Task)

- **Source**: [UTKFace Large Scale Face Dataset](https://susanqq.github.io/UTKFace/)
- **Content**: Face images in the wild with age, gender, race labels
- **Format**: Filename-encoded labels: `[age]_[gender]_[race]_[datetime].jpg`
- **Labels extracted**: Age integer parsed from filename, mapped to decade bucket

**Age bucket mapping**:
```python
def class_labels_reassign(age):
    buckets = [(1,10,0), (11,20,1), (21,30,2), (31,40,3), (41,50,4),
               (51,60,5), (61,70,6), (71,80,7), (81,90,8), (91,100,9)]
    for lo, hi, label in buckets:
        if lo <= age <= hi:
            return label
    return 9  # default for age > 100
```

---

## Preprocessing Pipeline

### Gender Task (UBIPeriocular)

**Step 1 — Train/Test split** (90/10):
```python
random.shuffle(image_files)
split_point = int(len(image_files) * 0.9)
train_files = image_files[:split_point]
test_files  = image_files[split_point:]
```

**Step 2 — Class folder creation**: Move images + text files into `train/male/`, `train/female/`, `test/male/`, `test/female/` by reading the gender label from each `.txt` file.

**Step 3 — Augmentation** (female images only, to reach 6584):
```python
transforms.Compose([
    transforms.Resize((224, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomInvert(),
    transforms.RandomSolarize(threshold=192.0),
    transforms.RandomCrop((224, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

**Final training transform** (used during training):
```python
transforms.Compose([
    transforms.Resize((224, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
```

---

### Age Task (UTKFace)

Custom `Dataset` class reads all images from a directory and parses the age from the filename:
```python
age_label = int(img_path.split('/')[-1].split('_')[0])
```

Then maps age to a decade bucket using `class_labels_reassign()`.

**Transform**:
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
```

---

## DataLoaders

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Shuffle (train) | True |
| Workers | 2 |
| Framework | `torchvision.datasets.ImageFolder` (gender), custom `Dataset` (age) |
