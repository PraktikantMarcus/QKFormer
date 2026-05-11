#!/bin/bash
#SBATCH --job-name=dvs128_prep
#SBATCH --partition=gpu
#SBATCH --nodelist=n-hpc-gz6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=/work/fritzsche6/qkformer/repo/download_dvs128gesture_%j.out
#SBATCH --error=/work/fritzsche6/qkformer/repo/download_dvs128gesture_%j.err
# Submit with: sbatch download_dvs128gesture.sh
# Run this script BEFORE submitting slurm_dvs128gesture_a100.sh.
#
# DVS128 Gesture cannot be downloaded automatically — IBM requires a manual
# request. Follow these steps:
#
#   1. Request the dataset at:
#      https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8
#      Download the file: DvsGesture.tar.gz  (~1.1 GB)
#
#   2. Transfer it to the login node, e.g. from your local machine:
#      scp DvsGesture.tar.gz fritzsche6@<up-hpc-login>:/work/fritzsche6/qkformer/dvs128gesture/
#
#   3. Then run this script:
#      bash download_dvs128gesture.sh
#
# The script will extract the archive and pre-process T=16 event frames.
# Expect 10-20 minutes on first run (frame conversion is CPU-bound).

set -e

DATA_DIR="/work/fritzsche6/qkformer/dvs128gesture"
# spikingjelly looks for the tarball specifically in the download/ subdirectory
DOWNLOAD_DIR="$DATA_DIR/download"
TARBALL="$DOWNLOAD_DIR/DvsGesture.tar.gz"

mkdir -p "$DOWNLOAD_DIR"

# ─── Sanity check ───────────────────────────────────────────────────────────
# Also handle the case where the file was placed directly in DATA_DIR
if [ -f "$DATA_DIR/DvsGesture.tar.gz" ] && [ ! -f "$TARBALL" ]; then
    echo "Moving DvsGesture.tar.gz into download/ subdirectory..."
    mv "$DATA_DIR/DvsGesture.tar.gz" "$TARBALL"
fi

if [ ! -f "$TARBALL" ]; then
    echo "ERROR: $TARBALL not found."
    echo ""
    echo "Please download DvsGesture.tar.gz from IBM Box first:"
    echo "  https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8"
    echo ""
    echo "Then transfer it to the HPC into the download/ subdirectory:"
    echo "  scp DvsGesture.tar.gz fritzsche6@<up-hpc-login>:$DOWNLOAD_DIR/"
    exit 1
fi

# ─── Environment ────────────────────────────────────────────────────────────
module load system/CUDA/11.6.0

source $(conda info --base)/etc/profile.d/conda.sh
conda activate qkformer

# ─── Frame conversion ───────────────────────────────────────────────────────
# spikingjelly will extract DvsGesture.tar.gz and build the frame cache.
# The frames_number_16 cache is what train.py expects (T=16).
echo "Converting DVS128 Gesture events to T=16 frames..."
echo "Data dir: $DATA_DIR"
echo "Start: $(date)"

python - <<'EOF'
import os, sys, threading, time, contextlib
from pathlib import Path
from tqdm import tqdm
from spikingjelly.datasets import dvs128_gesture

DATA_DIR = "/work/fritzsche6/qkformer/dvs128gesture"
FRAMES_DIR = os.path.join(DATA_DIR, "DVS128Gesture", "frames_number_16_split_by_number")

def count_npy(d):
    return sum(1 for _ in Path(d).rglob("*.npy")) if os.path.isdir(d) else 0

stop = threading.Event()

def monitor(desc):
    with tqdm(desc=desc, unit=" files", dynamic_ncols=True, file=sys.stdout) as bar:
        prev = 0
        while not stop.is_set():
            n = count_npy(FRAMES_DIR)
            if n > prev:
                bar.update(n - prev)
                prev = n
            time.sleep(1)
        # Final flush
        n = count_npy(FRAMES_DIR)
        if n > prev:
            bar.update(n - prev)

# ── Train split ──────────────────────────────────────────────────────────────
tqdm.write("Processing train split...")
stop.clear()
t = threading.Thread(target=monitor, args=("  train frames",), daemon=True)
t.start()
with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
    train_set = dvs128_gesture.DVS128Gesture(
        root=DATA_DIR, train=True, data_type="frame", frames_number=16, split_by="number"
    )
stop.set(); t.join()
tqdm.write(f"  => {len(train_set)} samples")

# ── Test split ───────────────────────────────────────────────────────────────
tqdm.write("Processing test split...")
stop.clear()
t = threading.Thread(target=monitor, args=("  test frames",), daemon=True)
t.start()
with open(os.devnull, "w") as null, contextlib.redirect_stdout(null):
    test_set = dvs128_gesture.DVS128Gesture(
        root=DATA_DIR, train=False, data_type="frame", frames_number=16, split_by="number"
    )
stop.set(); t.join()
tqdm.write(f"  => {len(test_set)} samples")

tqdm.write(f"\nFrames cached at: {FRAMES_DIR}")
EOF

echo "End: $(date)"
echo ""
echo "You can now submit the training job:"
echo "  sbatch slurm_dvs128gesture_a100.sh"
