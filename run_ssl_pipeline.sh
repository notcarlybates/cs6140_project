#!/bin/bash
#SBATCH --job-name=ssl_har_pipeline
#SBATCH --output=logs/ssl_pipeline_%j.out
#SBATCH --error=logs/ssl_pipeline_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================="
echo "SSL HAR Pipeline"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "=============================="

source "${SCRIPT_DIR}/.venv/bin/activate"

LOCATIONS=("LeftWrist" "RightAnkle" "RightThigh")

# Create all required directories before anything runs
mkdir -p "${SCRIPT_DIR}/logs"
for LOCATION in "${LOCATIONS[@]}"; do
    # RF pipeline dirs (needed for labeled data sync/slice)
    mkdir -p "/scratch/bates.car/datasets/paaws_fl_synced/${LOCATION}"
    mkdir -p "/scratch/bates.car/datasets/paaws_fl_trimmed/${LOCATION}"
    # SSL-specific dirs
    mkdir -p "/scratch/bates.car/datasets/paaws_ssl_preprocessed/${LOCATION}"
    mkdir -p "/scratch/bates.car/models/ssl_pretrained/${LOCATION}"
    mkdir -p "/scratch/bates.car/datasets/paaws_ssl_results/${LOCATION}"
done

for LOCATION in "${LOCATIONS[@]}"; do
    echo ""
    echo "=============================="
    echo "Processing location: ${LOCATION}"
    echo "=============================="

    # Step 1: Sync labeled accelerometer data with activity labels
    # (produces the trimmed CSVs that ssl_0_preprocess reads for labeled data)
    echo ""
    echo "--- [${LOCATION}] Step 1: Sync labeled data ($(date)) ---"
    python "${SCRIPT_DIR}/sync_paaws.py" --location "${LOCATION}"
    echo "--- [${LOCATION}] Step 1 complete ($(date)) ---"

    # Step 2: Slice (remove before/after collection rows from labeled data)
    echo ""
    echo "--- [${LOCATION}] Step 2: Slice labeled data ($(date)) ---"
    python "${SCRIPT_DIR}/slice_data.py" --location "${LOCATION}"
    echo "--- [${LOCATION}] Step 2 complete ($(date)) ---"

    # Step 3: SSL preprocessing
    # - Reads unlabeled data from /scratch/mazzucchelli.u/paaws_fl_data/acc_data/
    # - Reads labeled data from /scratch/bates.car/datasets/paaws_fl_trimmed/
    # - Excludes the 20 labeled participants from the unlabeled set
    # - Outputs: unlabeled_windows.npy (pre-training) + labeled_windows.npy (fine-tuning)
    echo ""
    echo "--- [${LOCATION}] Step 3: SSL preprocessing ($(date)) ---"
    python "${SCRIPT_DIR}/ssl_0_preprocess.py" --location "${LOCATION}"
    echo "--- [${LOCATION}] Step 3 complete ($(date)) ---"

    # Step 4: SSL pre-training (on unlabeled participants only)
    echo ""
    echo "--- [${LOCATION}] Step 4: SSL pre-training ($(date)) ---"
    python "${SCRIPT_DIR}/ssl_1_pretrain.py" \
        --location "${LOCATION}" \
        --epochs 20 \
        --n-subjects-per-batch 4 \
        --n-windows-per-subject 1500
    echo "--- [${LOCATION}] Step 4 complete ($(date)) ---"

    # Step 5: Downstream fine-tuning + evaluation (on 20 labeled participants)
    echo ""
    echo "--- [${LOCATION}] Step 5: Fine-tuning ($(date)) ---"
    python "${SCRIPT_DIR}/ssl_2_finetune.py" \
        --location "${LOCATION}" \
        --checkpoint best_backbone.pt
    echo "--- [${LOCATION}] Step 5 complete ($(date)) ---"

done

echo ""
echo "=============================="
echo "Pipeline complete: $(date)"
echo "=============================="
