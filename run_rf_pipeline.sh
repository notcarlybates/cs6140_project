#!/bin/bash
#SBATCH --job-name=rf_har_pipeline
#SBATCH --output=logs/rf_pipeline_%j.out
#SBATCH --error=logs/rf_pipeline_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================="
echo "RF HAR Pipeline"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "=============================="

# Activate virtual environment
source "${SCRIPT_DIR}/.venv/bin/activate"

LOCATIONS=("LeftWrist" "RightAnkle" "RightThigh")

# Create all required directories before anything runs
mkdir -p "${SCRIPT_DIR}/logs"
for LOCATION in "${LOCATIONS[@]}"; do
    mkdir -p "/scratch/bates.car/datasets/paaws_fl_trimmed/${LOCATION}"
    mkdir -p "/scratch/bates.car/datasets/paaws_fl_preprocessed/${LOCATION}"
    mkdir -p "/scratch/bates.car/datasets/paaws_fl_features/${LOCATION}"
    mkdir -p "/scratch/bates.car/datasets/paaws_fl_results/${LOCATION}"
done

for LOCATION in "${LOCATIONS[@]}"; do
    echo ""
    echo "=============================="
    echo "Processing location: ${LOCATION}"
    echo "=============================="

    # Step 1: Preprocessing
    echo ""
    echo "--- [${LOCATION}] Step 1: Preprocessing ($(date)) ---"
    python "${SCRIPT_DIR}/rf_1_preprocess.py" --location "${LOCATION}"
    echo "--- [${LOCATION}] Step 1 complete ($(date)) ---"

    # Step 2: Feature extraction
    echo ""
    echo "--- [${LOCATION}] Step 2: Feature extraction ($(date)) ---"
    python "${SCRIPT_DIR}/rf_2_features.py" --location "${LOCATION}"
    echo "--- [${LOCATION}] Step 2 complete ($(date)) ---"

    # Step 3: Training
    echo ""
    echo "--- [${LOCATION}] Step 3: Training ($(date)) ---"
    python "${SCRIPT_DIR}/rf_3_train.py" --location "${LOCATION}"
    echo "--- [${LOCATION}] Step 3 complete ($(date)) ---"

done

echo ""
echo "=============================="
echo "Pipeline complete: $(date)"
echo "=============================="
