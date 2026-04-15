#!/bin/bash
#SBATCH --job-name=ssl_har_pipeline
#SBATCH --output=logs/ssl_pipeline_%j.out
#SBATCH --error=logs/ssl_pipeline_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1

set -euo pipefail

SCRIPT_DIR="/home/bates.car/github/cs6140_project"

# echo "=============================="
# echo "SSL HAR Pipeline"
# echo "Job ID: ${SLURM_JOB_ID}"
# echo "Node:   $(hostname)"
# echo "Start:  $(date)"
# echo "=============================="

source "${SCRIPT_DIR}/.venv/bin/activate"

LOCATIONS=("LeftWrist") #  "RightAnkle" "RightThigh"

# # Create all required directories before anything runs
# mkdir -p "${SCRIPT_DIR}/logs"
# for LOCATION in "${LOCATIONS[@]}"; do
#     mkdir -p "/scratch/bates.car/datasets/paaws_fl_synced/${LOCATION}"
#     mkdir -p "/scratch/bates.car/datasets/paaws_fl_trimmed/${LOCATION}"
#     mkdir -p "/scratch/bates.car/datasets/paaws_fl_preprocessed/${LOCATION}"
#     mkdir -p "/scratch/bates.car/models/ssl_pretrained/${LOCATION}"
#     mkdir -p "/scratch/bates.car/datasets/paaws_ssl_results/${LOCATION}"
# done

for LOCATION in "${LOCATIONS[@]}"; do
#     echo ""
#     echo "=============================="
#     echo "Processing location: ${LOCATION}"
#     echo "=============================="

#     # Step 1: Sync accelerometer data with labels
#     echo ""
#     echo "--- [${LOCATION}] Step 1: Sync ($(date)) ---"
#     python "${SCRIPT_DIR}/sync_paaws.py" --location "${LOCATION}"
#     echo "--- [${LOCATION}] Step 1 complete ($(date)) ---"

#     # Step 2: Slice (remove before/after collection rows)
#     echo ""
#     echo "--- [${LOCATION}] Step 2: Slice ($(date)) ---"
#     python "${SCRIPT_DIR}/slice_data.py" --location "${LOCATION}"
#     echo "--- [${LOCATION}] Step 2 complete ($(date)) ---"

#     # Step 3: Preprocessing (resample to 30 Hz + 10-second windowing)
#     # Shared with the RF pipeline — skip if windows.npy already exists
#     echo ""
#     echo "--- [${LOCATION}] Step 3: Preprocessing ($(date)) ---"
#     WINDOWS_FILE="/scratch/bates.car/datasets/paaws_fl_preprocessed/${LOCATION}/windows.npy"
#     if [ -f "${WINDOWS_FILE}" ]; then
#         echo "  windows.npy already exists, skipping preprocessing"
#     else
#         python "${SCRIPT_DIR}/rf_1_preprocess.py" --location "${LOCATION}"
#     fi
#     echo "--- [${LOCATION}] Step 3 complete ($(date)) ---"

#     # Step 4: SSL pre-training
#     echo ""
#     echo "--- [${LOCATION}] Step 4: SSL pre-training ($(date)) ---"
#     python "${SCRIPT_DIR}/ssl_1_pretrain.py" \
#         --location "${LOCATION}" \
#         --epochs 20 \
#         --n-subjects-per-batch 4 \
#         --n-windows-per-subject 1500
#     echo "--- [${LOCATION}] Step 4 complete ($(date)) ---"

    # Step 5: Downstream fine-tuning + evaluation
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
