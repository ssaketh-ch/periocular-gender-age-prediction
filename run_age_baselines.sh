#!/usr/bin/env bash
#SBATCH --job-name=periage-sweep
#SBATCH --partition=mig_nodes
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p slurm_logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "job_id=${SLURM_JOB_ID:-local}"
echo "host=$(hostname)"
echo "start_time=$(date -Is)"

DATA_DIR="data/processed/utkface_age"
DEVICE="cuda"
WORKERS=4

run() {
  echo
  echo "============================================================"
  echo "RUN: $*"
  echo "============================================================"
  "$@"
}

# 1. ResNet baselines
run python scripts/train_single_task.py \
  --task age \
  --model resnet18 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/baselines/resnet18 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.001 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task age \
  --model resnet34 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/baselines/resnet34 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.001 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task age \
  --model resnet50 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/baselines/resnet50 \
  --epochs 50 \
  --batch-size 32 \
  --num-workers "$WORKERS" \
  --lr 0.001 \
  --device "$DEVICE"

# 2. Original PeriAge variants
run python scripts/train_single_task.py \
  --task age \
  --model periage \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/custom/periage_lr1e3 \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.001 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task age \
  --model periage \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/custom/periage_lr5e4 \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0005 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task age \
  --model periage \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/custom/periage_lr3e4 \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task age \
  --model periage \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/custom/periage_lr1e4 \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0001 \
  --device "$DEVICE"

# 3. New PeriAgeV2 architecture sweeps
run python scripts/train_single_task.py \
  --task age \
  --model periage_v2 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/custom/periage_v2_base \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task age \
  --model periage_v2 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/custom/periage_v2_balanced \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --label-smoothing 0.05 \
  --class-weighting balanced \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task age \
  --model periage_v2 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/custom/periage_v2_bs32 \
  --epochs 40 \
  --batch-size 32 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --label-smoothing 0.05 \
  --class-weighting balanced \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task age \
  --model periage_v2 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/custom/periage_v2_lr1e4 \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --label-smoothing 0.05 \
  --class-weighting balanced \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task age \
  --model periage_v2 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/custom/periage_v2_wd5e4 \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --weight-decay 0.0005 \
  --label-smoothing 0.05 \
  --class-weighting balanced \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task age \
  --model periage_resnet34 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/hybrid/periage_resnet34_ft \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --optimizer adamw \
  --weight-decay 0.0001 \
  --backbone-lr-mult 0.1 \
  --freeze-backbone-epochs 5 \
  --label-smoothing 0.05 \
  --class-weighting balanced \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task age \
  --model periage_resnet34 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/hybrid/periage_resnet34_ft_repeat \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --optimizer adamw \
  --weight-decay 0.0001 \
  --backbone-lr-mult 0.1 \
  --freeze-backbone-epochs 5 \
  --label-smoothing 0.05 \
  --class-weighting balanced \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task age \
  --model periage_resnet34 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/age/periocular/hybrid/periage_resnet34_ft_e50 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --optimizer adamw \
  --weight-decay 0.0001 \
  --backbone-lr-mult 0.1 \
  --freeze-backbone-epochs 5 \
  --label-smoothing 0.05 \
  --class-weighting balanced \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

echo
echo "============================================================"
echo "COMPARING ALL PERIOCULAR AGE RUNS"
echo "============================================================"
python scripts/compare_runs.py --runs-dir runs/age/periocular

echo "end_time=$(date -Is)"
