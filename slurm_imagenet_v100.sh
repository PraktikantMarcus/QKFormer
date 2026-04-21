#!/bin/bash
#SBATCH --job-name=qkformer_imagenet
#SBATCH --partition=gpu            # n-hpc-ga[1-4] live in the gpu partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30         # 10 workers × 3 GPUs
#SBATCH --gres=gpu:3               # 3× V100 16GB per node
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/qkformer_v100_%j.out
#SBATCH --error=logs/qkformer_v100_%j.err

# ─── Environment setup ──────────────────────────────────────────────────────
module load cuda/11.7              # adjust to your HPC
# conda activate qkformer
# source /path/to/venv/bin/activate

# ─── Paths — set these before submitting ────────────────────────────────────
DATA_PATH=/path/to/imagenet        # must contain train/ and val/ subdirectories
OUTPUT_DIR=./output_imagenet_v100_$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUTPUT_DIR" logs

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Output dir:  $OUTPUT_DIR"
echo "Start time:  $(date)"

# ─── Training ───────────────────────────────────────────────────────────────
cd "$(dirname "$0")/imagenet"

# V100 has only 16GB — keep batch_size=16 to stay safe.
# Effective batch size: 16 × 3 = 48.  The blr will be auto-scaled.
torchrun \
  --standalone \
  --nproc_per_node=3 \
  train.py \
    --model QKFormer_10_384 \
    --time_step 4 \
    --input_size 224 \
    --batch_size 16 \
    --epochs 100 \
    --accum_iter 2 \
    --blr 6e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --drop_path 0.1 \
    --smoothing 0.1 \
    --aa rand-m9-mstd0.5-inc1 \
    --reprob 0.25 \
    --remode pixel \
    --num_workers 10 \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$OUTPUT_DIR" \
    --dist_eval

echo "End time: $(date)"
