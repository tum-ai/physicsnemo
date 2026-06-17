#!/bin/bash

# Resolve the absolute path to the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Navigate to the crash workspace directory (one level up from scripts)
cd "$SCRIPT_DIR/.."

# Define variables matching your notebook parameters
RAW_DATA_DIR="/content/drive/MyDrive/physicsnemo/simulations"
GLOBAL_FEATURES="/content/drive/MyDrive/physicsnemo/global_features.json"
MASTER_CSV="/content/drive/MyDrive/physicsnemo/bumper_beam_master_with_split.csv"

DRIVE_CKPT_DIR="/content/drive/MyDrive/physicsnemo/checkpoints"
DRIVE_TB_DIR="/content/drive/MyDrive/physicsnemo/tensorboard_logs"
DRIVE_STATS_DIR="/content/drive/MyDrive/physicsnemo/stats"

# Epoch count override
EPOCHS=1000

# Ensure output directories exist on Google Drive
mkdir -p "$DRIVE_CKPT_DIR"
mkdir -p "$DRIVE_TB_DIR"
mkdir -p "$DRIVE_STATS_DIR"

# Generate global features if not already present
if [ ! -f "$GLOBAL_FEATURES" ]; then
    echo "Global features file not found at $GLOBAL_FEATURES. Generating it from $MASTER_CSV..."
    python make_global_features.py --master-csv "$MASTER_CSV" --out "$GLOBAL_FEATURES"
fi

# Run training with the correct config namespace hierarchies
HYDRA_FULL_ERROR=1 python train.py --config-name=bumper_vtkhdf_geotransolver_oneshot \
    training.raw_data_dir="$RAW_DATA_DIR" \
    training.raw_data_dir_validation="$RAW_DATA_DIR" \
    training.global_features_filepath="$GLOBAL_FEATURES" \
    inference.raw_data_dir_test="$RAW_DATA_DIR" \
    reader.master_csv="$MASTER_CSV" \
    training.ckpt_path="$DRIVE_CKPT_DIR" \
    training.tensorboard_log_dir="$DRIVE_TB_DIR" \
    datapipe.stats_dir="$DRIVE_STATS_DIR" \
    training.num_training_samples=20 \
    training.num_validation_samples=3 \
    training.epochs="$EPOCHS"
