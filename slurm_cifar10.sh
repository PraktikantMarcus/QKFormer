#!/bin/bash
#SBATCH --job-name=qkformer_cifar10
#SBATCH --partition=normal
#SBATCH --exclude=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               # single V100 or A100 is enough for CIFAR-10
#SBATCH --time=0-12:00:00          # 12 hours is more than enough
#SBATCH --output=logs/qkformer_cifar10_%j.out
#SBATCH --error=logs/qkformer_cifar10_%j.err

# ─── Environment setup ──────────────────────────────────────────────────────
module load cuda/12.3

source /home/fritzsche/qkformer/bin/activate

mkdir -p logs

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Start:     $(date)"

# ─── Training ───────────────────────────────────────────────────────────────
# Run download_cifar10.sh on the login node before submitting this job.
# Dataset is expected at /home/fritzsche/cifar10/ (set in cifar10.yml).
cd /work/fritzsche6/qkformer/repo/cifar10

python train.py \
  --config cifar10.yml \
  --epochs 200

echo "End: $(date)"
