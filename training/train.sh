#!/bin/bash

# ===== SLURM Job Template: Training =====
#SBATCH --job-name=<JOB_NAME>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=<NUM_GPUS>
#SBATCH --gpus-per-task=<GPUS_PER_TASK>
#SBATCH --cpus-per-task=<CPUS_PER_TASK>
#SBATCH --mem=<MEMORY>
#SBATCH --partition=<PARTITION>
#SBATCH --time=<HH:MM:SS>
#SBATCH --output=%x.%j.out
#SBATCH --mail-type=<NONE|END|FAIL|ALL>
#SBATCH --mail-user=<EMAIL@EXAMPLE.COM>
#SBATCH --account=<ACCOUNT>
#SBATCH --qos=<QOS>

set -euo pipefail

# ===== Container: Apptainer/Singularity =====
module purge
module load apptainer

# Environment cleanup similar to working example
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PIP_DISABLE_PIP_VERSION_CHECK=1

# Container image and flags
SIF="<PATH_TO_CONTAINER>.sif"
APPTAINER_FLAGS="--nv --bind <BIND_DIRS> --cleanenv"

# ===== Paths =====
VCA_ROOT="<PATH_TO_REPO_ROOT>"
PY_SCRIPT="$VCA_ROOT/training/train.py"
PYTHON_BIN=python3

DATA_DIR="<PATH_TO_DATA_ROOT>"
TRAIN_DATA="<TRAIN_FOLDER_NAME>"
TRAIN_CSV="<TRAIN_CSV_NAME>"
TEST_DATA="<VAL_OR_TEST_FOLDER_NAME>"
TEST_CSV="<VAL_OR_TEST_CSV_NAME>"

IMAGE_SIZE=<IMAGE_SIZE>
BATCH_SIZE=<BATCH_SIZE>
EPOCHS=<EPOCHS>
LR=<LEARNING_RATE>
STEP_SIZE=<STEP_SIZE>

# LR schedule
LR_SCHEDULE_TYPE="<none|step|plateau|cyclic_plateau>"
LR_MIN=<LR_MIN>
LR_PEAK_EPOCH=<LR_PEAK_EPOCH>
LR_CYCLE_END_EPOCH=<LR_CYCLE_END_EPOCH>

MODEL_DIR="<PATH_TO_MODEL_DIR>/$(date +%b%d_%H%M%S)_<RUN_TAG>/"
MODEL_NAME="<MODEL_TAG>_$(date +%b%d%Y).pth"

RESULT_DIR="<PATH_TO_RESULTS_DIR>"
LOG_FOLDER="<EXPERIMENT_TAG>_$(date +%b%d%Y)"
VAL_CSV="${LOG_FOLDER}.csv"

# Early stopping configuration
EARLY_STOP_PATIENCE=<EARLY_STOP_PATIENCE>

SPLIT_RATIO=<TRAIN_VAL_SPLIT>
REPEAT=<REPEAT>
IS_LOG=<True|False>
RESUME=<None|PATH_TO_CHECKPOINT>
START_EPOCH=<START_EPOCH>
MODEL_TO_RUN=<MODEL_ID>
IS_PREDICT_TWO=<True|False>
IS_AROUSAL=<True|False>
IS_SAVE_EPOCH=<True|False>
CLIP_MODEL_NAME="<CLIP_MODEL_NAME>"

mkdir -p "$MODEL_DIR"
mkdir -p "$RESULT_DIR/$LOG_FOLDER"

# Build container-executed command with python resolver
RUN_CMD=$(cat <<BASH
set -e
PYTHON_BIN=\$(command -v python3 || command -v python || true)
if [ -z "\$PYTHON_BIN" ]; then echo "ERROR: No python interpreter found inside container"; exit 127; fi
echo "=== Using python: \$PYTHON_BIN ==="
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VCA_ROOT:\$PYTHONPATH"
cd "$VCA_ROOT"
"\$PYTHON_BIN" "$PY_SCRIPT" --data_dir "$DATA_DIR" --train_data "$TRAIN_DATA" --train_csv_data "$TRAIN_CSV" --test_data "$TEST_DATA" --test_csv_data "$TEST_CSV" --batch_size $BATCH_SIZE --epoch $EPOCHS --lr $LR --lr_schedule_type $LR_SCHEDULE_TYPE --lr_min $LR_MIN --lr_peak_epoch $LR_PEAK_EPOCH --lr_cycle_end_epoch $LR_CYCLE_END_EPOCH --step_size $STEP_SIZE --model_dir "$MODEL_DIR" --model_name "$MODEL_NAME" --logfolder "$LOG_FOLDER" --resultfolder "$RESULT_DIR" --validation_performances "$VAL_CSV" --is_log $IS_LOG --resume $RESUME --start_epoch $START_EPOCH --model_to_run $MODEL_TO_RUN --image_size $IMAGE_SIZE --is_predict_two $IS_PREDICT_TWO --isarousal $IS_AROUSAL --clip_model_name "$CLIP_MODEL_NAME" --split_ratio $SPLIT_RATIO --repeat $REPEAT --is_saveModel_epoch $IS_SAVE_EPOCH --early_stop_patience $EARLY_STOP_PATIENCE
BASH
)

# ===== Run inside Apptainer container =====
apptainer exec $APPTAINER_FLAGS "$SIF" bash -lc "$RUN_CMD"

# Direct run fallback (disabled)
# "${CMD[@]}"
