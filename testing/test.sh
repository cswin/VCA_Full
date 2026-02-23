#!/bin/bash

# ===== SLURM Job Template: Testing =====
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

export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PIP_DISABLE_PIP_VERSION_CHECK=1

SIF="<PATH_TO_CONTAINER>.sif"
APPTAINER_FLAGS="--nv --bind <BIND_DIRS> --cleanenv"

# ===== Paths =====
VCA_ROOT="<PATH_TO_REPO_ROOT>"
PY_SCRIPT="$VCA_ROOT/testing/test.py"

# Test Data 
DATA_DIR="<PATH_TO_DATA_ROOT>"
TEST_DATA="<TEST_FOLDER_NAME>" # Test data folder name
TEST_CSV="<TEST_CSV_NAME>" # Test CSV file name --labels
TEST_CSV_PATH="$DATA_DIR/$TEST_CSV"
TEST_IMG_DIR="$DATA_DIR/$TEST_DATA"

# Model checkpoint: set explicitly or via env MODEL_PTH
MODEL_PTH_DEFAULT="<PATH_TO_MODEL_CHECKPOINT>.pth"
MODEL_PTH="${MODEL_PTH:-$MODEL_PTH_DEFAULT}"

# Results
RESULT_DIR="<PATH_TO_RESULTS_DIR>"
LOG_FOLDER="<EXPERIMENT_TAG>_${TEST_DATA}_$(date +%b%d%Y_%H%M%S)"
OUT_DIR="$RESULT_DIR/$LOG_FOLDER"

# Hyperparameters/settings aligned with training
IMAGE_SIZE=<IMAGE_SIZE>
BATCH_SIZE=<BATCH_SIZE>
MODEL_TO_RUN=<MODEL_ID>
IS_PREDICT_TWO=<True|False>
# IS_AROUSAL selects the target when running single-output regression.
# - False: use valence labels
# - True:  use arousal labels
# Note: When IS_PREDICT_TWO=True (two-head model), this flag is ignored.
IS_AROUSAL=<True|False>
CLIP_MODEL_NAME="<CLIP_MODEL_NAME>"
DROPOUT=<DROPOUT_RATE>

mkdir -p "$OUT_DIR"

if [[ ! -f "$MODEL_PTH" ]]; then
  echo "ERROR: Model checkpoint not found at: $MODEL_PTH" >&2
  echo "Set MODEL_PTH env var or update MODEL_PTH_DEFAULT in testing/HPG_test.sh" >&2
  exit 2
fi

if [[ ! -f "$TEST_CSV_PATH" ]]; then
  echo "ERROR: Test CSV not found at: $TEST_CSV_PATH" >&2
  exit 2
fi

if [[ ! -d "$TEST_IMG_DIR" ]]; then
  echo "ERROR: Test image dir not found at: $TEST_IMG_DIR" >&2
  exit 2
fi

# ===== Build command for container =====
RUN_CMD=$(cat <<BASH
set -e
PYTHON_BIN=\$(command -v python3 || command -v python || true)
if [ -z "\$PYTHON_BIN" ]; then echo "ERROR: No python interpreter found inside container"; exit 127; fi
echo "=== Using python: \$PYTHON_BIN ==="
export PYTHONNOUSERSITE=1
export PYTHONPATH="$VCA_ROOT:\$PYTHONPATH"
cd "$VCA_ROOT"
"\$PYTHON_BIN" "$PY_SCRIPT" \
  --model_path "$MODEL_PTH" \
  --csv_path "$TEST_CSV_PATH" \
  --img_dir "$TEST_IMG_DIR" \
  --batch_size $BATCH_SIZE \
  --image_size $IMAGE_SIZE \
  --model_to_run $MODEL_TO_RUN \
  --is_predict_two $IS_PREDICT_TWO \
  --isarousal $IS_AROUSAL \
  --clip_model_name "$CLIP_MODEL_NAME" \
  --dropout_rate $DROPOUT \
  --resultfolder "$OUT_DIR"
BASH
)

# ===== Execute inside Apptainer =====
apptainer exec $APPTAINER_FLAGS "$SIF" bash -lc "$RUN_CMD"

echo "Test complete. Outputs saved to: $OUT_DIR"


