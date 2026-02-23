#!/bin/bash

# ===== SLURM Job Template: Inference =====
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
PY_SCRIPT="$VCA_ROOT/inference/Inference.py"

# Inference Data (no CSV labels required)
DATA_DIR="<PATH_TO_DATA_ROOT>"
TEST_DATA="<TEST_FOLDER_NAME>"            # folder name containing images
TEST_IMG_DIR="$DATA_DIR/$TEST_DATA"

# Model checkpoint: set explicitly or via env MODEL_PTH
MODEL_PTH_DEFAULT="<PATH_TO_MODEL_CHECKPOINT>.pth"
MODEL_PTH="${MODEL_PTH:-$MODEL_PTH_DEFAULT}"

# Results
RESULT_DIR="<PATH_TO_RESULTS_DIR>"
LOG_FOLDER="<EXPERIMENT_TAG>_${TEST_DATA}_$(date +%b%d%Y_%H%M%S)"
OUT_DIR="$RESULT_DIR/$LOG_FOLDER"

# Hyperparameters/settings
IMAGE_SIZE=<IMAGE_SIZE>
BATCH_SIZE=<BATCH_SIZE>

mkdir -p "$OUT_DIR"

if [[ ! -f "$MODEL_PTH" ]]; then
  echo "ERROR: Model checkpoint not found at: $MODEL_PTH" >&2
  echo "Set MODEL_PTH env var or update MODEL_PTH_DEFAULT in inference/HPG_infer.sh" >&2
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
  --test_img_dir "$TEST_IMG_DIR" \
  --batch_size $BATCH_SIZE \
  --image_size $IMAGE_SIZE \
  --resultfolder "$OUT_DIR" \
  --model_to_run <MODEL_ID> \
  --is_predict_two <True|False> \
  --clip_model_name "<CLIP_MODEL_NAME>"
BASH
)

# ===== Execute inside Apptainer =====
apptainer exec $APPTAINER_FLAGS "$SIF" bash -lc "$RUN_CMD"

echo "Inference complete. Outputs saved to: $OUT_DIR"


