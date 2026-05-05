#!/bin/bash
#SBATCH --job-name=qkformer_multinode
#SBATCH --partition=gpu
# ── Pick however many V100 nodes you can get (1-4 available: n-hpc-ga[1-4]) ─
#SBATCH --nodes=2
# Uncomment to pin specific nodes, or leave out to let SLURM pick any free ones:
##SBATCH --nodelist=n-hpc-ga[1-2]
#SBATCH --ntasks-per-node=1          # one torchrun launcher per node
#SBATCH --cpus-per-task=15           # 5 workers × 3 GPUs
#SBATCH --gres=gpu:3                 # 3× V100 16GB per node
#SBATCH --mem=90G
#SBATCH --time=5-00:00:00
#SBATCH --output=/work/fritzsche6/qkformer/repo/imagenet/logs/qkformer_multinode_%j.out
#SBATCH --error=/work/fritzsche6/qkformer/repo/imagenet/logs/qkformer_multinode_%j.err

# ─── Environment ────────────────────────────────────────────────────────────
module load system/CUDA/11.6.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qkformer

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH=/work/fritzsche6/qkformer/imagenet
# Same output dir as the A100 run — checkpoint-latest.pth will be used to resume
OUTPUT_DIR=/work/fritzsche6/qkformer/repo/imagenet/output_imagenet_a100

mkdir -p "$OUTPUT_DIR" /work/fritzsche6/qkformer/repo/imagenet/logs

# ─── Batch size: keep effective batch ≈ 256 regardless of node count ────────
# Original (A100): batch_size=32 × accum_iter=2 × 4 GPUs = 256
# Formula: BATCH_SIZE = 256 / (NODES × GPUS_PER_NODE × ACCUM_ITER)
#   2 nodes × 3 GPUs, accum=3 → 14 × 3 × 6 = 252  ✓
#   3 nodes × 3 GPUs, accum=3 →  9 × 3 × 9 = 243  ✓
#   4 nodes × 3 GPUs, accum=3 →  7 × 3 × 12 = 252 ✓
GPUS_PER_NODE=3
ACCUM_ITER=3
WORLD_SIZE=$(( SLURM_NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( 256 / (WORLD_SIZE * ACCUM_ITER) ))

# ─── Rendezvous setup ───────────────────────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $SLURM_NNODES  ($SLURM_JOB_NODELIST)"
echo "World size:    $WORLD_SIZE GPUs"
echo "Batch/GPU:     $BATCH_SIZE  (eff_batch ≈ $(( BATCH_SIZE * ACCUM_ITER * WORLD_SIZE )))"
echo "Master:        ${MASTER_ADDR}:${MASTER_PORT}"
echo "Output dir:    $OUTPUT_DIR"
echo "Start time:    $(date)"

# ─── Checkpoint resume ──────────────────────────────────────────────────────
RESUME_ARG=""
if [ -f "$OUTPUT_DIR/checkpoint-latest.pth" ]; then
    RESUME_ARG="--resume $OUTPUT_DIR/checkpoint-latest.pth"
    echo "Resuming from $OUTPUT_DIR/checkpoint-latest.pth"
else
    echo "No checkpoint found, starting from scratch"
fi

# ─── Launch: srun starts one torchrun per node ──────────────────────────────
cd /work/fritzsche6/qkformer/repo/imagenet

# $SLURM_NODEID is per-task (0, 1, 2, …) — must be escaped so srun expands it
srun bash -c "
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=\$SLURM_NODEID \
    train.py \
      --model QKFormer_10_384 \
      --time_step 4 \
      --input_size 224 \
      --batch_size $BATCH_SIZE \
      --epochs 200 \
      --accum_iter $ACCUM_ITER \
      --blr 6e-4 \
      --weight_decay 0.05 \
      --warmup_epochs 5 \
      --drop_path 0.1 \
      --smoothing 0.1 \
      --aa rand-m9-mstd0.5-inc1 \
      --reprob 0.25 \
      --remode pixel \
      --num_workers 5 \
      --data_path $DATA_PATH \
      --output_dir $OUTPUT_DIR \
      --log_dir $OUTPUT_DIR \
      $RESUME_ARG \
      --dist_eval
"

echo "End time: $(date)"
