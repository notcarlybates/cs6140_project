#!/bin/bash
#SBATCH --job-name=ssl_har_finetune
#SBATCH --output=logs/ssl_finetune_%j.out
#SBATCH --error=logs/ssl_finetune_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --mem=16GB
#SBATCH --ntasks=1

set -euo pipefail

SCRIPT_DIR="/home/bates.car/github/cs6140_project"

echo "=============================="
echo "SSL HAR Fine-tuning Only"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "=============================="

source "${SCRIPT_DIR}/.venv/bin/activate"
export PYTHONUNBUFFERED=1

LOCATIONS=("LeftWrist")   # "RightAnkle" "RightThigh"

mkdir -p "${SCRIPT_DIR}/logs"

for LOCATION in "${LOCATIONS[@]}"; do
    echo ""
    echo "=============================="
    echo "Processing location: ${LOCATION}"
    echo "=============================="

    # Step 5: Downstream fine-tuning + evaluation on existing backbone
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
