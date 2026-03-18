# Notebooks

This folder contains the main Jupyter notebook for the periocular gender and age prediction project.

> **Note on old repository:** The previous development repository contained 4 older notebooks (`Periocular_GC.ipynb`, `periocular-gc-v2.ipynb`, `periage-resnet.ipynb`, `combined-model.ipynb`). These were intermediate development iterations and all of their content has been consolidated and superseded by `periocular-final.ipynb`. Nothing from the old notebooks was omitted.

---

## periocular-final.ipynb

This is the complete, final notebook. It covers the entire pipeline from raw data to trained multi-task model. Below is a section-by-section walkthrough with code snippets.

---

### 1. Library Imports and Device Setup

All required libraries are imported upfront. PyTorch and torchvision handle model building, training, and data loading. `torchinfo` provides model summaries. Device-agnostic code ensures the same script runs on CPU or GPU.

```python
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms, models
from torchvision.models import resnet
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from PIL import Image
import os, random, shutil
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

### 2. Data Preprocessing: Gender (UBIPeriocular Dataset)

The UBIPeriocular dataset provides each image alongside a `.txt` metadata file. The 7th line of each `.txt` file is the gender label (`male` or `female`).

**Step 1 - Train/test split.** Images are shuffled and split 90/10:

```python
data_folder = '/kaggle/working/my-data/ubipr/UBIPeriocular'
train_percent = 0.9
image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpg')]
random.shuffle(image_files)
split_point = int(len(image_files) * train_percent)
train_files, test_files = image_files[:split_point], image_files[split_point:]
```

**Step 2 - Label extraction and sorting.** Each image is moved into a `male/` or `female/` subfolder based on its metadata:

```python
folder_path = '/kaggle/working/my-data/ubipr/UBIPeriocular/train'
line_num = 7
os.makedirs(os.path.join(folder_path, 'male'), exist_ok=True)
os.makedirs(os.path.join(folder_path, 'female'), exist_ok=True)

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        with open(os.path.join(folder_path, filename), 'r') as file:
            lines = file.readlines()
            label = lines[line_num - 1].strip().replace(';', '')
            img_filename = filename.replace('.txt', '.jpg')
            shutil.move(img_filename, os.path.join(folder_path, label.lower(), img_filename))
```

**Step 3 - Class balancing.** The dataset is imbalanced (6584 male, 2584 female in train). Augmentation is applied to increase female samples to match the male count:

```python
transform = transforms.Compose([
    transforms.Resize((224, 112)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomInvert(),
    transforms.RandomSolarize(threshold=192.0),
    transforms.RandomCrop((224, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

female_dir = '/kaggle/input/cleaned-data/UBIPeriocular/train/female'
target_num_images = 6584  # match male count
num_female_images = len(os.listdir(female_dir))

while num_female_images < target_num_images:
    female_image_path = os.path.join(female_dir, random.choice(os.listdir(female_dir)))
    female_image = Image.open(female_image_path)
    augmented_image = transform(female_image)
    transforms.ToPILImage()(augmented_image).save(
        os.path.join(augmented_dir, f'augmented_female_{num_female_images}.jpg')
    )
    num_female_images += 1
```

---

### 3. Baseline: Pretrained ResNet Models (Gender Classification)

ResNet-18, 34, and 50 pretrained on ImageNet are fine-tuned for 2-class gender prediction. The final fully connected layer is replaced automatically by the framework since both the source and target are 2-class classification (ImageNet has 1000, but we swap it):

```python
resnet_18 = resnet.resnet18()
resnet_18.load_state_dict(torch.load('/kaggle/input/resnet-18/resnet18-5c106cde.pth'))

transform = transforms.Compose([
    transforms.Resize((224, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
```

The training loop uses SGD with momentum and weight decay, running for 10 epochs. After each epoch, the model is put in eval mode (`model.eval()`) and test accuracy is printed:

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

for epoch in tqdm(range(num_epochs)):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, preds = torch.max(outputs, 1)
            acc += torch.sum(preds == labels.to(device))
    model.train()
```

---

### 4. Custom Model: PeriGender Architecture

The PeriGender model is implemented from scratch based on the research paper. The key design choices are:

- **ResBlock**: Two convolutional layers each followed by batch normalization and ReLU. This forms the basic feature extraction unit.
- **SkipConnections (1-4)**: Each skip connection captures features at a different scale using a Conv + MaxPool combination. The kernel/stride parameters vary between skip connections to produce feature maps of matching spatial dimensions for later concatenation.
- **PeriGender**: Wraps all the above into a full model. The forward pass extracts skip outputs at 4 different stages and concatenates them with the final feature map before the classifier head.

```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out

class PeriGender(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv     = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.skip1    = SkipConnection1(64, 64)
        self.resblock1 = nn.Sequential(ResBlock(64, 64, stride=2), ResBlock(64, 128))
        self.skip2    = SkipConnection2(128, 128)
        self.resblock2 = nn.Sequential(ResBlock(128, 128, stride=2), ResBlock(128, 256))
        self.skip3    = SkipConnection3(256, 256)
        self.resblock3 = nn.Sequential(ResBlock(256, 256, stride=2), ResBlock(256, 512))
        self.skip4    = SkipConnection4(512, 4)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout  = nn.Dropout(p=0.5)
        self.fc       = nn.Linear(524, num_classes)  # 512 + 3+3+3+3 from skip outputs

    def forward(self, x):
        out   = self.maxpool1(self.conv(x))
        skip1 = self.skip1(out);  out = self.resblock1(out)
        skip2 = self.skip2(out);  out = self.resblock2(out)
        skip3 = self.skip3(out);  out = self.resblock3(out)
        skip4 = self.skip4(out);  out = self.maxpool2(out)
        out   = self.avg_pool(torch.cat([skip1, skip2, skip3, skip4, out], dim=1))
        out   = self.fc(self.dropout(out.squeeze()))
        return out
```

PeriGender is trained with SGD (lr=0.0001, momentum=0.9) and CrossEntropyLoss for 10 epochs. The same training/eval loop as the ResNet baseline is used.

---

### 5. Custom Model: PeriAge Architecture

PeriAge is a direct adaptation of PeriGender for 10-class age-range prediction (decade buckets 1-10, 11-20, ..., 91-100). The key addition is an `Upsample` module in the forward pass to handle the different spatial dimensions that arise from using 224x224 input (age uses square crops) versus 224x112 (gender uses portrait crops):

```python
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.pad(x, (0, 1, 0, 0))  # pad to match target spatial size
        return x

class PeriAge(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # ... same blocks as PeriGender ...
        self.Upsample = Upsample(512, 512)
        self.fc = nn.Linear(524, num_classes)
```

A custom `Dataset` class handles the UTKFace dataset, where age is encoded in the filename (e.g. `25_0_0_...jpg` = age 25). Ages are mapped to decade class indices:

```python
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir  = data_dir
        self.img_paths = sorted(os.listdir(data_dir))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        img_path  = os.path.join(self.data_dir, self.img_paths[idx])
        img       = Image.open(img_path).convert('RGB')
        age_label = int(img_path.split('/')[-1].split('_')[0])
        return self.transform(img), age_label
```

The decade mapping function:

```python
def class_labels_reassign(age):
    if   1 <= age <= 10:  return 0
    elif 11 <= age <= 20: return 1
    elif 21 <= age <= 30: return 2
    # ... and so on up to class 9 for 91-100
```

PeriAge is trained with SGD (lr=0.001, momentum=0.9) and CrossEntropyLoss for 50 epochs.

---

### 6. Combined Multi-Task Model: PeriOcular

PeriOcular shares the exact same backbone as PeriGender/PeriAge but has two output heads: one for gender (2 classes) and one for age (10 classes). This is classic multi-task learning.

```python
class PeriOcular(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # ... same backbone blocks ...
        self.fc  = nn.Linear(524, num_classes)   # gender head
        self.fc2 = nn.Linear(524, 10)            # age head

    def forward(self, x):
        # ... same forward pass up to avg_pool ...
        out  = self.avg_pool(torch.cat([skip1, skip2, skip3, skip4, out], dim=1))
        out  = self.dropout(out.squeeze())
        return self.fc(out), self.fc2(out)  # returns (gender, age) tuple
```

The multi-task training loop combines both losses:

```python
def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    for images, age_labels, gender_labels in train_loader:
        images, age_labels, gender_labels = images.to(device), age_labels.to(device), gender_labels.to(device)
        optimizer.zero_grad()
        gender_outputs, age_outputs = model(images)
        loss = loss_fn(gender_outputs, gender_labels) + loss_fn(age_outputs, age_labels)
        loss.backward()
        optimizer.step()
```

Combined accuracy requires **both** gender and age to be correct simultaneously, which is why it is lower (~32.7%) than either individual task:

```python
train_acc += (predicted_age.eq(age_labels) & predicted_gender.eq(gender_labels)).sum().item()
```

PeriOcular is trained using Adam (lr=0.01) for 25 epochs.

---

### 7. Results & Accuracy Plots

Test accuracy data recorded over training runs is stored as Python lists and plotted using matplotlib:

```python
# Gender classification results (25 epochs)
test_acc_18 = [0.8151, 0.8333, ..., 0.9189]
test_acc_34 = [0.9630, 0.9678, ..., 0.9248]
test_acc_50 = [0.9412, 0.9678, ..., 0.9148]
test_acc_PeriGender = [0.8457, 0.7878, ..., 0.9320]

plt.plot(range(1, len(test_acc_34)+1), test_acc_34, label='ResNet-34')
plt.plot(range(1, len(test_acc_PeriGender)+1), test_acc_PeriGender, label='PeriGender')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Summary of best test accuracies:

| Task | Model | Best Test Acc |
|------|-------|--------------|
| Gender | ResNet-34 | ~96.8% |
| Gender | ResNet-50 | ~96.9% |
| Gender | PeriGender | ~93.2% |
| Age | ResNet-50 | ~60.6% |
| Age | PeriAge | ~57.2% |
| Gender + Age | PeriOcular | ~32.7% |

---

### 8. Inference Example

A helper function loads and transforms a single image and runs it through the loaded model:

```python
def predict(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        gender_output, age_output = model(image)
    gender = ['female', 'male'][torch.argmax(gender_output).item()]
    age_range = get_age_range(torch.argmax(age_output).item())
    return gender, age_range
```

---

## Usage

```bash
jupyter notebook notebooks/periocular-final.ipynb
```

For details on the model architectures, see [../docs/architecture.md](../docs/architecture.md).
For dataset details, see [../docs/dataset.md](../docs/dataset.md).
