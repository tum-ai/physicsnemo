#!/bin/bash

# Resolve the absolute path to the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Navigate to the crash workspace directory (one level up from scripts)
cd "$SCRIPT_DIR/.."

# ==============================================================================
# CONFIGURATION BLOCK
# Edit these parameters to match your target experiment.
# ==============================================================================

# 1. Hydra Experiment Configuration Name
CONFIG_NAME="crash_geotransolver_custom_oneshot.yaml"
EXP_NAME="custom_geotransolver"

# 2. Base Paths (no need to change this)
DATA_DIR="/mnt/1t/mit-project/Dataset/full_body"
PROJ_DIR="/mnt/1t/mit-project/physicsnemo/examples/structural_mechanics/crash"
OUT_DIR="./outputs/${EXP_NAME}/"

# 3. Config Paths
SIM_DIR="${DATA_DIR}"   

# 4. Dataset Size Overrides (Matches the size of your downloaded data)
NUM_TRAIN_SAMPLES=6
NUM_VAL_SAMPLES=1

# 5. Training Epoch Count Override
EPOCHS=200

# ==============================================================================
# RUN TIME EXECUTION
# ==============================================================================


# Launch training with configuration overrides
python train.py --config-name="$CONFIG_NAME" \
    training.raw_data_dir="$SIM_DIR" \
    training.raw_data_dir_validation="$SIM_DIR" \
    training.global_features_filepath="$GLOBAL_FEATURES" \
    inference.raw_data_dir_test="$SIM_DIR" \
    reader.master_csv="$MASTER_CSV" \
    training.num_training_samples="$NUM_TRAIN_SAMPLES" \
    training.num_validation_samples="$NUM_VAL_SAMPLES" \
    hydra.run.dir="$OUT_DIR" \
    training.epochs="$EPOCHS"
