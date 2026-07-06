#!/bin/bash

# Exit immediately if any command fails
set -e

# Path to the data directory (default matching train.sh/train.py)
DATA_ROOT="/mnt/1t/mit-project/Dataset/full_body"
CSV_PATH="crash_relevant_parts.csv"

# Get the script's directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "========================================================================"
echo " Launching 1-Epoch Smoke Test (Training + Validation + Inference)"
echo "   CSV Filter : $CSV_PATH"
echo "   Data Root  : $DATA_ROOT"
echo "========================================================================"
echo ""

# Run the wrapper train.sh script with 1 epoch and save weights
# Passes --filter_parts_csv to train on the sliced part geometry only
"$SCRIPT_DIR/train.sh" \
    --encoder enhanced \
    --epochs 1 \
    --log_every 1 \
    --n_steps 1 \
    --filter_parts_csv "$CSV_PATH" \
    --data_root "$DATA_ROOT"

echo ""
echo "========================================================================"
echo " Smoke test completed successfully!"
echo " All pipelines are fully functional."
echo "========================================================================"
