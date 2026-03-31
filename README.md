# Periocular Gender and Age Prediction

This repository contains the refreshed CLI-first version of an older undergraduate periocular biometrics project. The original notebook-driven work has now been reproduced, tightened up, and improved with cleaner data handling, stronger baselines, stricter evaluation, and better custom model variants.

## New Results First

The strongest results in the refreshed workflow are:

| Task | Best Model | Best Result | Notes |
|---|---|---:|---|
| UBIPr gender | `ResNet18` | `98.10%` image accuracy | subject-level split, no identity leakage |
| UBIPr gender, custom | `PeriGenderV2` | `97.11%` image accuracy | best custom gender model |
| UTKFace periocular age | `PeriAgeResNet34` | `86.68%` test accuracy | strict periocular-only age pipeline |
| UTKFace periocular age baseline | `ResNet34` | `85.08%` test accuracy | strongest plain age baseline |

The most important improvements over the old version are:

- the project can now be run end to end from the CLI
- periocular age experiments now use actual periocular crops instead of full-face images
- UBIPr gender is now split by subject rather than by image
- custom v2 and hybrid models were added to make the periocular-specific architectures more competitive

## What This Project Does

The project studies whether the periocular region alone, the area around the eye, contains enough information for:

- gender classification
- age-bucket classification
- eventually joint multitask prediction

The repo currently has three families of models:

- pretrained ResNet baselines
- custom multiscale periocular models
- hybrid models that combine pretrained backbones with periocular-specific fusion

## Repository Guide

- [CLI Workflow](docs/cli_workflow.md)
  End-to-end command-line setup, data prep, training, and inference.
- [Scripts](scripts/README.md)
  What each CLI script does.
- [Experiment Runs](runs/README.md)
  Organized run folders and canonical checkpoints.
- [Age Experiments](runs/age/README.md)
  Detailed interpretation of the periocular age work.
- [Gender Experiments](runs/gender/README.md)
  Detailed interpretation of the UBIPr gender work.
- [Custom Model Families](docs/custom_models/README.md)
  Technical notes for the v1, v2, and hybrid architectures.
- [Architecture Notes](docs/architecture.md)
  High-level architecture context from the original project.
- [Dataset Notes](docs/dataset.md)
  Background on the datasets and preprocessing assumptions.

## CLI Quick Start

### 1. Clone and create an environment

```bash
git clone https://github.com/ssaketh-ch/periocular-gender-age-prediction.git
cd periocular-gender-age-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download the datasets

For UTKFace, the repo supports Kaggle CLI download:

```bash
mkdir -p ~/.kaggle
# place kaggle.json in ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

mkdir -p data/raw/utkface
kaggle datasets download -d jangedoo/utkface-new -p data/raw/utkface
unzip data/raw/utkface/utkface-new.zip -d data/raw/utkface
```

For UBIPr, download the official archive and extract it into `data/raw/ubipr/`.

### 3. Prepare the periocular datasets

Gender on UBIPr:

```bash
python scripts/prepare_ubipr_gender.py \
  --raw-dir data/raw/ubipr/UBIPeriocular \
  --output-dir data/processed/ubipr_gender \
  --test-size 0.10 \
  --seed 42 \
  --balance-train \
  --split-by-subject
```

Age on UTKFace:

```bash
python scripts/extract_utkface_periocular.py \
  --raw-dir data/raw/utkface \
  --output-dir data/processed/utkface_periocular_raw \
  --strict

python scripts/prepare_utkface_age.py \
  --raw-dir data/processed/utkface_periocular_raw \
  --output-dir data/processed/utkface_age \
  --test-size 0.20 \
  --seed 42
```

### 4. Run the main sweeps

Age sweep:

```bash
bash run_age_baselines.sh
```

Gender sweep:

```bash
bash run_gender_baselines.sh
```

If you are on Slurm, both sweep scripts are `sbatch`-ready:

```bash
sbatch run_age_baselines.sh
sbatch run_gender_baselines.sh
```

### 5. Compare finished runs

```bash
python scripts/compare_runs.py --runs-dir runs/age/periocular
python scripts/compare_runs.py --runs-dir runs/gender/ubipr
```

### 6. Run inference from a saved checkpoint

```bash
python scripts/predict.py \
  --checkpoint runs/age/periocular/hybrid/periage_resnet34_ft/20260330_202332/best.pt \
  --image /absolute/path/to/image.jpg \
  --device cpu
```

## Current Model Story

### Gender

On UBIPr, transfer learning remains very strong. `ResNet18` produced the best overall gender result, while `PeriGenderV2` became the strongest custom periocular model. The current gender pipeline is much more reliable than the older notebook version because:

- labels are read directly from the paired metadata files
- the split is performed by subject, not by image
- subject-level evaluation was explicitly checked

### Age

Age prediction is the harder problem. Once the project was converted to strict periocular crops, the task became more faithful to the original goal and the numbers became more realistic. Plain `ResNet34` remained a strong baseline, but the hybrid `PeriAgeResNet34` fine-tune ultimately became the best model in the repo.

## Figures

The repo can generate comparison plots and confusion matrices from the saved run artifacts:

```bash
python scripts/generate_report_artifacts.py
```

Generated outputs are written under `results/figures/`.

## Custom Models

The custom architecture documentation is split by family:

- [PeriGender Family](docs/custom_models/perigender/README.md)
- [PeriAge Family](docs/custom_models/periage/README.md)

These docs explain:

- the original v1 scratch models
- the v2 fusion upgrades
- the age hybrid that combines a pretrained backbone with periocular-specific multiscale fusion

## Notebook Usage

The repo now treats the CLI workflow as the primary interface.

If you still want to explore the project interactively, the notebook remains available:

```bash
jupyter notebook notebooks/periocular-final.ipynb
```

The recommended order is:

1. run the CLI pipeline first
2. use the notebook afterward for exploration or presentation

## References

- Hussain, M., Alrabiah, R., & AboAlSamh, H. A. (2023). *Unconstrained Gender Recognition from Periocular Region Using Multiscale Deep Features*. Intelligent Automation & Soft Computing, 35(3), 2941-2962.
- Chandrashekhar Padole, Hugo Proenca. *Periocular Recognition: Analysis of Performance Degradation Factors*. Proceedings of the Fifth IAPR/IEEE International Conference on Biometrics, ICB 2012, New Delhi, India, March 30-April 1, 2012.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*.
- [Project Report PDF](Periocular_Gender_Age_Project_Report.pdf)

## Original Results From The Older Work

The original notebook-era project reported approximately:

- gender: `93-97%` depending on backbone
- age: `57-61%`
- multitask joint accuracy: `~32.7%`

Those numbers were useful as the starting point for the project, but the refreshed workflow is more trustworthy because it fixes evaluation issues, makes the periocular extraction explicit, and records the full training history for each experiment.
