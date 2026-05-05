#!/bin/bash
#SBATCH --job-name=qkformer_imagenet
#SBATCH --partition=gpu            
#SBATCH --nodelist=n-hpc-gz6       # 4× A100 SXM4 40GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40         # 10 workers × 4 GPUs
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --output=/work/fritzsche6/qkformer/repo/imagenet/logs/qkformer_a100_%j.out
#SBATCH --error=/work/fritzsche6/qkformer/repo/imagenet/logs/qkformer_a100_%j.err

# ─── Environment setup ──────────────────────────────────────────────────────
# Adjust module names to match your HPC's module system
module load system/CUDA/11.6.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qkformer

# ─── Paths — set these before submitting ────────────────────────────────────
DATA_PATH=/work/fritzsche6/qkformer/imagenet
# Use a fixed output dir so --resume always points to the right checkpoint
OUTPUT_DIR=/work/fritzsche6/qkformer/repo/imagenet/output_imagenet_a100

mkdir -p "$OUTPUT_DIR" /work/fritzsche6/qkformer/repo/imagenet/logs

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Output dir:  $OUTPUT_DIR"
echo "Start time:  $(date)"

# ─── Training ───────────────────────────────────────────────────────────────
cd /work/fritzsche6/qkformer/repo/imagenet

# Only pass --resume if the checkpoint actually exists
RESUME_ARG=""
if [ -f "$OUTPUT_DIR/checkpoint-latest.pth" ]; then
    RESUME_ARG="--resume $OUTPUT_DIR/checkpoint-latest.pth"
    echo "Resuming from $OUTPUT_DIR/checkpoint-latest.pth"
else
    echo "No checkpoint found, starting from scratch"
fi

torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py \
    --model QKFormer_10_384 \
    --time_step 4 \
    --input_size 224 \
    --batch_size 32 \
    --epochs 200 \
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
    $RESUME_ARG \
    --dist_eval

echo "End time: $(date)"
