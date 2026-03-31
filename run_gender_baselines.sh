#!/usr/bin/env bash
#SBATCH --job-name=perigender-sweep
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

DATA_DIR="data/processed/ubipr_gender"
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
  --task gender \
  --model resnet18 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/baselines/resnet18_e30 \
  --epochs 30 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model resnet18 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/baselines/resnet18_e50 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model resnet34 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/baselines/resnet34_e30 \
  --epochs 30 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model resnet34 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/baselines/resnet34_e50 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model resnet50 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/baselines/resnet50_e30 \
  --epochs 30 \
  --batch-size 32 \
  --num-workers "$WORKERS" \
  --lr 0.001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model resnet50 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/baselines/resnet50_e50 \
  --epochs 50 \
  --batch-size 32 \
  --num-workers "$WORKERS" \
  --lr 0.001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

# 2. Original PeriGender sweeps
run python scripts/train_single_task.py \
  --task gender \
  --model perigender \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_lr1e3 \
  --epochs 30 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model perigender \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_lr5e4 \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0005 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model perigender \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_lr3e4 \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model perigender \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_lr1e4 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model perigender \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_smooth \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --label-smoothing 0.02 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model perigender \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_bs32 \
  --epochs 40 \
  --batch-size 32 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

# 3. PeriGenderV2 sweeps
run python scripts/train_single_task.py \
  --task gender \
  --model perigender_v2 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_v2_base \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model perigender_v2 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_v2_wd1e4 \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model perigender_v2 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_v2_lr1e4 \
  --epochs 50 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model perigender_v2 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_v2_bs32 \
  --epochs 40 \
  --batch-size 32 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model perigender_v2 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_v2_smooth \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --label-smoothing 0.02 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model perigender_v2 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_v2_adamw \
  --epochs 40 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --optimizer adamw \
  --weight-decay 0.0001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model perigender_v2 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_v2_long \
  --epochs 60 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

run python scripts/train_single_task.py \
  --task gender \
  --model perigender_v2 \
  --data-dir "$DATA_DIR" \
  --output-dir runs/gender/ubipr/custom/perigender_v2_long_smooth \
  --epochs 60 \
  --batch-size 64 \
  --num-workers "$WORKERS" \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --label-smoothing 0.02 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --device "$DEVICE"

echo
echo "============================================================"
echo "COMPARING ALL UBIPr GENDER RUNS"
echo "============================================================"
python scripts/compare_runs.py --runs-dir runs/gender/ubipr

echo "end_time=$(date -Is)"
