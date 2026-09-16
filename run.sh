#!/bin/bash
# ==============================================================================
# Helper Script: Run any Python / Bash command in Docker easily
# Syntax:  ./run.sh [command to run]
# Example: ./run.sh python3 sd3_5_visdial_baseline.py --skip_existing
#          ./run.sh bash
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

IMAGE_NAME="aiclub_visdial-core_baotg_sd35_v1:latest"
GPU_IDS="${GPU_IDS:-0}"

# Automatically build image if it does not exist locally
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "🔧 Image '${IMAGE_NAME}' does not exist. Building Docker Image..."
    docker build -t "${IMAGE_NAME}" .
fi

# Create cache directories if they do not exist
mkdir -p "$SCRIPT_DIR/.cache/huggingface" "$SCRIPT_DIR/.cache/torch"

if [ $# -eq 0 ]; then
    docker run --gpus "\"device=${GPU_IDS}\"" \
      --ipc=host \
      --shm-size=16g \
      -it --rm \
      -v "$SCRIPT_DIR":/workspace \
      -v "$SCRIPT_DIR/.cache/huggingface":/root/.cache/huggingface \
      -v "$SCRIPT_DIR/.cache/torch":/root/.cache/torch \
      -v /workingspace_aiclub/Datasets/CIR/unlabeled2017:/workspace/unlabeled2017 \
      -v /workingspace_aiclub/Datasets/CIR/VisDial:/workspace/VisDial \
      -v /workingspace_aiclub:/workingspace_aiclub \
      "${IMAGE_NAME}" bash
else
    docker run --gpus "\"device=${GPU_IDS}\"" \
      --ipc=host \
      --shm-size=16g \
      -it --rm \
      -v "$SCRIPT_DIR":/workspace \
      -v "$SCRIPT_DIR/.cache/huggingface":/root/.cache/huggingface \
      -v "$SCRIPT_DIR/.cache/torch":/root/.cache/torch \
      -v /workingspace_aiclub/Datasets/CIR/unlabeled2017:/workspace/unlabeled2017 \
      -v /workingspace_aiclub/Datasets/CIR/VisDial:/workspace/VisDial \
      -v /workingspace_aiclub:/workingspace_aiclub \
      "${IMAGE_NAME}" "$@"
fi
