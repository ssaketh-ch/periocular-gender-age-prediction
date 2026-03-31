# Scripts

This folder contains the notebook-free workflow for the refreshed repo. The scripts are organized around the same lifecycle used in the new experiments:

- prepare raw datasets
- train baselines and custom models
- evaluate saved checkpoints
- compare experiments
- generate report figures

## Data Preparation

### `prepare_ubipr_gender.py`

Builds the UBIPr gender dataset from raw `.jpg` / `.txt` pairs.

Key behaviors:

- reads gender from line 7 of the paired metadata file
- can split by subject with `--split-by-subject`
- can balance the training set with female-only augmentation using `--balance-train`

This is the script that fixes the old image-level leakage issue in the gender pipeline.

### `extract_utkface_periocular.py`

Converts full-face UTKFace images into periocular crops.

Key behaviors:

- uses OpenCV face and eye detection
- writes cropped periocular images to a new folder
- supports `--strict` to discard images without a reliable eye-based crop

This script is what turns the age experiments into true periocular-only experiments.

### `prepare_utkface_age.py`

Builds an ImageFolder-style age classification dataset from periocular UTKFace crops.

Key behaviors:

- parses age from the filename
- maps each image into a decade bucket
- creates `train/` and `test/` directories with one folder per age bucket

## Training

### `train_single_task.py`

General trainer for gender and age experiments.

Supported models:

- `resnet18`
- `resnet34`
- `resnet50`
- `perigender`
- `perigender_v2`
- `periage`
- `periage_v2`
- `periage_resnet34`

Important training features:

- timestamped output directories by default
- SGD or AdamW
- cosine scheduler
- label smoothing
- balanced class weighting
- staged backbone fine-tuning for hybrid models

### `train_multitask.py`

Trains the shared `PeriOcular` multitask model for joint age and gender prediction.

This script is still closer to the original project line of work than the new age and gender pipelines, but it has been kept as the bridge toward a cleaner multitask setup.

## Evaluation and Analysis

### `compare_runs.py`

Scans a run tree and prints a leaderboard ranked by best recorded test accuracy.

Useful for:

- quick experiment triage
- comparing sweeps
- building README tables

By default it skips directories named `legacy/`. Use `--include-legacy` if you want the historical snapshots too.

### `evaluate_gender_run.py`

Evaluates a gender checkpoint on the processed UBIPr test split.

Outputs:

- image-level accuracy
- balanced accuracy
- confusion matrix
- subject-level majority-vote accuracy

### `evaluate_age_run.py`

Evaluates an age checkpoint on the processed periocular UTKFace test split.

Outputs:

- image-level accuracy
- balanced accuracy
- confusion matrix

### `generate_report_artifacts.py`

Creates publication-style plots for the repository documentation.

Current artifacts include:

- best-accuracy bar charts
- test-accuracy curves
- selected confusion matrices

## Inference and Shared Helpers

### `predict.py`

Loads a saved checkpoint and runs inference on a single image.

Useful for:

- sanity checks
- quick demos
- verifying that a checkpoint can still be loaded after refactors

### `common.py`

Shared helpers for:

- label parsing
- age bucket mapping
- run directory creation
- JSON saving

## Sweep Scripts At Repo Root

The longer overnight sweeps live in the repository root rather than in `scripts/`:

- `run_age_baselines.sh`
- `run_gender_baselines.sh`

Both are:

- CLI-runnable
- `sbatch`-ready
- relative-path safe
- configured to compare all completed runs at the end
