# CLI Workflow

This repo now has a notebook-free workflow for:

- preparing `UBIPeriocular` for gender classification
- extracting periocular crops from `UTKFace`
- preparing `UTKFace` for age-bucket classification
- training `ResNet` and custom periocular models
- training the joint `PeriOcular` multitask model
- running inference from saved checkpoints

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Data Download

### UTKFace

The official UTKFace homepage is:

- https://susanqq.github.io/UTKFace/

Practical local download option with Kaggle CLI:

```bash
pip install kaggle
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle
# place kaggle.json in ~/.kaggle/kaggle.json first

mkdir -p data/raw/utkface
kaggle datasets download -d jangedoo/utkface-new -p data/raw/utkface
unzip data/raw/utkface/utkface-new.zip -d data/raw/utkface
```

### UBIPeriocular / UBIPr

The official UBIPr page is:

- https://iris.di.ubi.pt/ubipr.html

The source site exposes the dataset directly, but the exact archive URL can change. The safest path is:

1. Download the original UBIPr archive manually from the official page.
2. Extract it into `data/raw/ubipr/`.

Example after you download the archive:

```bash
mkdir -p data/raw/ubipr
tar -xf ~/Downloads/UBIPr*.tar* -C data/raw/ubipr
```

If your download is a zip instead:

```bash
mkdir -p data/raw/ubipr
unzip ~/Downloads/UBIPr*.zip -d data/raw/ubipr
```

## 3. Prepare Datasets

### Prepare UBIPr for gender

This creates:

- `data/processed/ubipr_gender/train/male`
- `data/processed/ubipr_gender/train/female`
- `data/processed/ubipr_gender/test/male`
- `data/processed/ubipr_gender/test/female`

```bash
python scripts/prepare_ubipr_gender.py \
  --raw-dir data/raw/ubipr/UBIPeriocular \
  --output-dir data/processed/ubipr_gender \
  --test-size 0.10 \
  --seed 42 \
  --balance-train \
  --split-by-subject
```

### Extract periocular crops from UTKFace

This converts full UTKFace face images into periocular-only crops. It uses OpenCV face/eye detection.

If you want a strict periocular-only experiment, use `--strict` so images without a reliable eye-based crop are skipped instead of falling back to a looser upper-face crop.

```bash
python scripts/extract_utkface_periocular.py \
  --raw-dir data/raw/utkface \
  --output-dir data/processed/utkface_periocular_raw \
  --strict
```

### Prepare periocular UTKFace for age buckets

This creates bucket directories such as `00_10`, `11_20`, `21_30`, etc.

```bash
python scripts/prepare_utkface_age.py \
  --raw-dir data/processed/utkface_periocular_raw \
  --output-dir data/processed/utkface_age \
  --test-size 0.20 \
  --seed 42 \
  --min-age 0 \
  --max-age 120
```

## 4. Train Models

### Gender baselines

```bash
python scripts/train_single_task.py \
  --task gender \
  --model resnet18 \
  --data-dir data/processed/ubipr_gender \
  --output-dir runs/gender/ubipr/baselines/resnet18_e30 \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device cuda
```

```bash
python scripts/train_single_task.py \
  --task gender \
  --model resnet34 \
  --data-dir data/processed/ubipr_gender \
  --output-dir runs/gender/ubipr/baselines/resnet34_e30 \
  --epochs 30 \
  --batch-size 64 \
  --lr 0.001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device cuda
```

```bash
python scripts/train_single_task.py \
  --task gender \
  --model resnet50 \
  --data-dir data/processed/ubipr_gender \
  --output-dir runs/gender/ubipr/baselines/resnet50_e30 \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device cuda
```

### Best custom gender model

```bash
python scripts/train_single_task.py \
  --task gender \
  --model perigender_v2 \
  --data-dir data/processed/ubipr_gender \
  --output-dir runs/gender/ubipr/custom/perigender_v2_adamw \
  --epochs 40 \
  --batch-size 64 \
  --lr 0.0003 \
  --optimizer adamw \
  --weight-decay 0.0001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device cuda
```

### Age baselines

```bash
python scripts/train_single_task.py \
  --task age \
  --model resnet34 \
  --data-dir data/processed/utkface_age \
  --output-dir runs/age/periocular/baselines/resnet34 \
  --epochs 15 \
  --batch-size 64 \
  --lr 0.001 \
  --device cuda
```

```bash
python scripts/train_single_task.py \
  --task age \
  --model resnet50 \
  --data-dir data/processed/utkface_age \
  --output-dir runs/age/periocular/baselines/resnet50 \
  --epochs 15 \
  --batch-size 32 \
  --lr 0.001 \
  --device cuda
```

### Best custom scratch-style age model

```bash
python scripts/train_single_task.py \
  --task age \
  --model periage_v2 \
  --data-dir data/processed/utkface_age \
  --output-dir runs/age/periocular/custom/periage_v2_bs32 \
  --epochs 35 \
  --batch-size 32 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --label-smoothing 0.05 \
  --class-weighting balanced \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device cuda
```

### Best overall age model

```bash
python scripts/train_single_task.py \
  --task age \
  --model periage_resnet34 \
  --data-dir data/processed/utkface_age \
  --output-dir runs/age/periocular/hybrid/periage_resnet34_ft \
  --epochs 40 \
  --batch-size 64 \
  --lr 0.0003 \
  --optimizer adamw \
  --weight-decay 0.0001 \
  --backbone-lr-mult 0.1 \
  --freeze-backbone-epochs 5 \
  --label-smoothing 0.05 \
  --class-weighting balanced \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device cuda
```

### Multitask model on periocular UTKFace

This uses periocular-cropped UTKFace filenames for both age and gender labels and trains `PeriOcular`.

```bash
python scripts/train_multitask.py \
  --raw-dir data/processed/utkface_periocular_raw \
  --output-dir runs/periocular_multitask \
  --epochs 25 \
  --batch-size 64 \
  --lr 0.001 \
  --test-size 0.20 \
  --seed 42
```

## 5. Inference

### Gender checkpoint

```bash
python scripts/predict.py \
  --checkpoint runs/gender/ubipr/custom/perigender_v2_adamw/<timestamp>/best.pt \
  --image /absolute/path/to/example.jpg
```

### Age checkpoint

```bash
python scripts/predict.py \
  --checkpoint runs/age/periocular/hybrid/periage_resnet34_ft/<timestamp>/best.pt \
  --image /absolute/path/to/example.jpg
```

### Multitask checkpoint

```bash
python scripts/predict.py \
  --checkpoint runs/periocular_multitask/best.pt \
  --image /absolute/path/to/example.jpg
```

## 6. Compare and Sweep

Compare completed experiments:

```bash
python scripts/compare_runs.py --runs-dir runs/gender/ubipr
python scripts/compare_runs.py --runs-dir runs/age/periocular
```

Run the long sweeps:

```bash
bash run_gender_baselines.sh
bash run_age_baselines.sh
```

Or submit them through Slurm:

```bash
sbatch run_gender_baselines.sh
sbatch run_age_baselines.sh
```

## 7. Files Written By Training

By default, training scripts create a timestamped subdirectory inside `--output-dir`.

Example:

```bash
python scripts/train_single_task.py \
  --task age \
  --model periage \
  --data-dir data/processed/utkface_age \
  --output-dir runs/age/periocular/periage
```

This writes to something like:

```text
runs/age/periocular/periage/20260330_142355/
```

If you want the old behavior, add `--no-timestamp`.

Every run writes:

- `best.pt`
- `metrics.json`

Compare a set of runs with:

```bash
python scripts/compare_runs.py \
  --runs-dir runs/age/periocular
```

Example:

```bash
cat runs/perigender/metrics.json
cat runs/periocular_multitask/metrics.json
```

## 7. Suggested First Reproduction Order

```bash
python scripts/prepare_ubipr_gender.py --raw-dir data/raw/ubipr/UBIPeriocular --output-dir data/processed/ubipr_gender --balance-train
python scripts/train_single_task.py --task gender --model resnet34 --data-dir data/processed/ubipr_gender --output-dir runs/gender_resnet34 --epochs 10
python scripts/train_single_task.py --task gender --model perigender --data-dir data/processed/ubipr_gender --output-dir runs/perigender --epochs 15
python scripts/extract_utkface_periocular.py --raw-dir data/raw/utkface --output-dir data/processed/utkface_periocular_raw --strict
python scripts/prepare_utkface_age.py --raw-dir data/processed/utkface_periocular_raw --output-dir data/processed/utkface_age
python scripts/train_single_task.py --task age --model resnet50 --data-dir data/processed/utkface_age --output-dir runs/age_resnet50 --epochs 15
python scripts/train_single_task.py --task age --model periage --data-dir data/processed/utkface_age --output-dir runs/periage --epochs 25
python scripts/train_multitask.py --raw-dir data/processed/utkface_periocular_raw --output-dir runs/periocular_multitask --epochs 25
```
