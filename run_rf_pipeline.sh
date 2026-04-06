#!/bin/bash
#SBATCH --job-name=rf_har_pipeline
#SBATCH --error=/scratch/bates.car/jobs/rf/%j/rf_pipeline_%j.err
#SBATCH --error=/scratch/bates.car/jobs/rf/%j/rf_pipeline_%j.err
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

mkdir -p "${SCRIPT_DIR}/logs"

git checkout other_limbs

# Commented out b/c step One complete
# # Step 1: Data syncing
# echo ""
# echo "--- Step 1: Syncing accelerometer and label files ($(date)) ---"
# python "${SCRIPT_DIR}/sync_paaws.py"
# echo "--- Step 1 complete ($(date)) ---"

# Step 2: Slicing data
echo ""
echo "--- Step 2: Slicing ends of data ($(date)) ---"
python "${SCRIPT_DIR}/slice_data.py"
echo "--- Step 2 complete ($(date)) ---"

# Step 1: Preprocessing
echo ""
echo "--- Step 3: Preprocessing ($(date)) ---"
python "${SCRIPT_DIR}/rf_1_preprocess.py"
echo "--- Step 3 complete ($(date)) ---"

# Step 2: Feature extraction
echo ""
echo "--- Step 4: Feature extraction ($(date)) ---"
python "${SCRIPT_DIR}/rf_2_features.py"
echo "--- Step 4 complete ($(date)) ---"

# Step 3: Training
echo ""
echo "--- Step 5: Training ($(date)) ---"
python "${SCRIPT_DIR}/rf_3_train.py"
echo "--- Step 5 complete ($(date)) ---"

echo ""
echo "=============================="
echo "Pipeline complete: $(date)"
echo "=============================="
