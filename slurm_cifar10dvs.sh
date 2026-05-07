#!/bin/bash
#SBATCH --job-name=qkformer_cifar10dvs
#SBATCH --partition=normal
#SBATCH --exclude=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1               # single GPU is enough; DVS train.py is not multi-GPU
#SBATCH --time=0-12:00:00          # 12 hours is more than enough for 96 epochs
#SBATCH --output=logs/qkformer_cifar10dvs_%j.out
#SBATCH --error=logs/qkformer_cifar10dvs_%j.err

# ─── Environment setup ──────────────────────────────────────────────────────
module load cuda/12.3

source /home/fritzsche/qkformer/bin/activate

mkdir -p logs

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Start:     $(date)"

# ─── Training ───────────────────────────────────────────────────────────────
# Run download_cifar10dvs.sh on the login node before submitting this job.
# Processed frames are expected at /nfsscratch/fritzsche/cifar10dvs/.
cd /home/fritzsche/QKFormer/cifar10-dvs

torchrun --standalone --nproc_per_node=1 train.py \
  --model QKFormer \
  --data-path /nfsscratch/fritzsche/cifar10dvs \
  --epochs 96 \
  --batch-size 16 \
  --T 16 \
  --device cuda

echo "End: $(date)"
