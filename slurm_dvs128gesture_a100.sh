#!/bin/bash
#SBATCH --job-name=qkformer_dvs128
#SBATCH --partition=gpu
#SBATCH --nodelist=n-hpc-gz6       # 4× A100 SXM4 40GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # 4 workers for single GPU
#SBATCH --gres=gpu:1               # single GPU — dataset is tiny (~1.3k samples)
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00          # ~3-5h expected; 12h is a safe ceiling
#SBATCH --output=/work/fritzsche6/qkformer/repo/dvs128-gesture/logs/qkformer_dvs128_%j.out
#SBATCH --error=/work/fritzsche6/qkformer/repo/dvs128-gesture/logs/qkformer_dvs128_%j.err

# ─── Environment setup ──────────────────────────────────────────────────────
module load system/CUDA/11.6.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qkformer

# ─── Paths — set DATA_PATH before submitting ────────────────────────────────
# Download/process the dataset on the login node first (compute nodes block
# internet). Run: python -c "from spikingjelly.datasets import dvs128_gesture;
#   dvs128_gesture.DVS128Gesture(root='$DATA_PATH', train=True, ...)"
# The raw IBMGestureNew/ folder must be inside DATA_PATH beforehand.
DATA_PATH=/work/fritzsche6/qkformer/dvs128gesture

# Fixed output dir so --resume always resolves to the same checkpoint.
# train.py appends a subdirectory based on model/batch/T/wd/opt/lr args —
# this must match the args passed below exactly.
BASE_OUTPUT=/work/fritzsche6/qkformer/repo/dvs128-gesture/output_dvs128gesture
CKPT_DIR=${BASE_OUTPUT}/QKFormer_b16_T16_wd0.06_adamw_cnf_ADD/lr0.001

mkdir -p "$BASE_OUTPUT" /work/fritzsche6/qkformer/repo/dvs128-gesture/logs

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Data:       $DATA_PATH"
echo "Output:     $CKPT_DIR"
echo "Start time: $(date)"

# ─── Training ───────────────────────────────────────────────────────────────
cd /work/fritzsche6/qkformer/repo/dvs128-gesture

# Resume from latest checkpoint if one exists (crash/requeue recovery)
RESUME_ARG=""
if [ -f "${CKPT_DIR}/checkpoint_latest.pth" ]; then
    RESUME_ARG="--resume ${CKPT_DIR}/checkpoint_latest.pth"
    echo "Resuming from ${CKPT_DIR}/checkpoint_latest.pth"
else
    echo "No checkpoint found — starting from scratch"
fi

python train.py \
    --model QKFormer \
    --dataset DVS128Gesture \
    --num-classes 11 \
    --data-path "$DATA_PATH" \
    --device cuda \
    --T 16 \
    --epochs 192 \
    --batch-size 16 \
    --workers 4 \
    --opt adamw \
    --lr 1e-3 \
    --weight-decay 0.06 \
    --sched cosine \
    --warmup-epochs 10 \
    --min-lr 2e-5 \
    --mixup 0.5 \
    --smoothing 0.1 \
    --connect_f ADD \
    --amp \
    --output-dir "$BASE_OUTPUT" \
    --print-freq 50 \
    $RESUME_ARG

echo "End time: $(date)"
