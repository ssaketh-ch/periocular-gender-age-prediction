# Notebooks

The notebook in this folder is now a companion to the CLI workflow, not the primary implementation.

## Current Role Of The Notebook

`periocular-final.ipynb` now exists for:

- interactive inspection of saved runs
- quick checkpoint loading and demo inference
- presenting the refreshed workflow in a notebook-friendly format

## What Changed

The original project was driven mostly from Kaggle-style notebooks with hardcoded paths and experiment logic mixed together in one place. The repo has now been reorganized so that:

- data preparation lives in `scripts/prepare_*`
- training lives in `scripts/train_*`
- evaluation lives in `scripts/evaluate_*`
- reporting lives in `scripts/generate_report_artifacts.py`

The notebook therefore mirrors the new workflow instead of trying to remain the authoritative implementation.

## Recommended Usage

1. Run the CLI pipeline first.
2. Use the notebook afterward to inspect saved metrics and checkpoints.

If you are starting fresh, begin with:

- [CLI Workflow](../docs/cli_workflow.md)
- [Scripts README](../scripts/README.md)

Then open:

- `notebooks/periocular-final.ipynb`
