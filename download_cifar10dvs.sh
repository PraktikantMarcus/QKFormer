#!/bin/bash
# Run this script on the IHP HPC login node BEFORE submitting slurm_cifar10dvs.sh.
# Downloads CIFAR10-DVS (~3.7 GB) and pre-processes T=16 event frames.
# Expect this to take 20-40 minutes on first run.
#
# Usage:
#   bash download_cifar10dvs.sh

set -e

DATA_DIR="/home/fritzsche/cifar10dvs"
mkdir -p "$DATA_DIR"

source /home/fritzsche/qkformer/bin/activate

python - <<EOF
from spikingjelly.datasets import cifar10_dvs
print("Downloading and converting CIFAR10-DVS (T=16 frames) — this will take a while...")
dataset = cifar10_dvs.CIFAR10DVS(root="$DATA_DIR", data_type='frame', frames_number=16, split_by='number')
print(f"Done! Dataset size: {len(dataset)} samples")
print(f"Frames cached at: $DATA_DIR/CIFAR10DVS/frames_number_16_split_by_number/")
EOF
