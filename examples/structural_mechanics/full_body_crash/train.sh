#!/bin/bash

# Exit immediately if any command fails
set -e

# Default values matching train.py
ENCODER="enhanced"
DATA_ROOT="/mnt/1t/mit-project/Dataset/full_body"

# Parse encoder and data_root arguments without destroying the input array
for ((i=1; i<=$#; i++)); do
    j=$((i+1))
    if [ "${!i}" == "--encoder" ]; then
        ENCODER="${!j}"
    elif [ "${!i}" == "--data_root" ]; then
        DATA_ROOT="${!j}"
    fi
done

# Resolve the script directory path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "========================================================================"
echo " Launching Unified Train-and-Inference Pipeline"
echo "   Encoder Model: $ENCODER"
echo "   Data Root    : $DATA_ROOT"
echo "========================================================================"
echo ""

# 1. Run training
echo "[Step 1/2] Running train.py..."
python "$SCRIPT_DIR/train.py" "$@"

# 2. Retrieve WandB run ID for resuming
WANDB_RUN_ID=""
WANDB_ID_FILE="$SCRIPT_DIR/outputs/$ENCODER/wandb_run_id.txt"
if [ -f "$WANDB_ID_FILE" ]; then
    WANDB_RUN_ID=$(cat "$WANDB_ID_FILE")
    echo ""
    echo "Detected WandB Run ID: $WANDB_RUN_ID"
else
    echo ""
    echo "Warning: No WandB Run ID file found at $WANDB_ID_FILE."
    echo "Inference evaluation will run as a standalone WandB experiment."
fi

# 3. Run inference on test simulations using the same WandB run ID
echo ""
echo "[Step 2/2] Running inference evaluation and visualization..."
"$SCRIPT_DIR/inference/inference.sh" "$DATA_ROOT" "$ENCODER" "$WANDB_RUN_ID" "$@"

echo ""
echo "========================================================================"
echo " Train-and-Inference Pipeline finished successfully!"
echo "========================================================================"
