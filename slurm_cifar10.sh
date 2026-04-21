#!/bin/bash
#SBATCH --job-name=qkformer_cifar10
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1               # single V100 or A100 is enough for CIFAR-10
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00          # 12 hours is more than enough
#SBATCH --output=logs/qkformer_cifar10_%j.out
#SBATCH --error=logs/qkformer_cifar10_%j.err

# ─── Environment setup ──────────────────────────────────────────────────────
module load cuda/11.7
# conda activate qkformer
# source /path/to/venv/bin/activate

mkdir -p logs

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"

# ─── Training ───────────────────────────────────────────────────────────────
cd "$(dirname "$0")/cifar10"

# Data path is set inside cifar10.yml — edit that file if needed.
# Default config already uses 400 epochs; reduce here if desired.
python train.py \
  --config cifar10.yml \
  --epochs 200

echo "End: $(date)"
