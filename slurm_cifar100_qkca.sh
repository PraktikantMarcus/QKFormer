#!/bin/bash
#SBATCH --job-name=qkformer_cifar100_qkca
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/qkformer_cifar100_qkca_%j.out
#SBATCH --error=logs/qkformer_cifar100_qkca_%j.err

# ─── Environment setup ──────────────────────────────────────────────────────
module load system/CUDA/11.6.0

# Conda needs explicit initialisation in non-interactive SLURM shells
source $(conda info --base)/etc/profile.d/conda.sh
conda activate qkformer

mkdir -p logs

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
echo "Start:    $(date)"

# ─── Training ───────────────────────────────────────────────────────────────
cd /work/fritzsche6/qkformer/repo/cifar100

python train_qkca.py --config cifar100.yml

echo "End: $(date)"
