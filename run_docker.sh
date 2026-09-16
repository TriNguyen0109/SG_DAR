#!/bin/bash

# ==============================================================================
# Script to automatically Build & Run Container for SD 3.5 VisDial Pipeline
# ==============================================================================

IMAGE_NAME="aiclub_visdial-core_baotg_sd35_v1"
CONTAINER_NAME="sd35_visdial_runner"

# Navigate to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 1. Build Docker Image (if not present or when updated)
echo "--------------------------------------------------------"
echo "🔧 Building Docker Image: ${IMAGE_NAME}..."
echo "--------------------------------------------------------"
docker build -t ${IMAGE_NAME} .

# 2. Select GPU to use (Default: GPU 0)
GPU_IDS="${1:-0}"

echo "--------------------------------------------------------"
echo "🚀 Running Docker Container on GPU(s): ${GPU_IDS}..."
echo "--------------------------------------------------------"

# Create cache directory if not present to avoid redownloading Hugging Face weights
mkdir -p "$SCRIPT_DIR/.cache/huggingface"

docker run --gpus "\"device=${GPU_IDS}\"" \
  --ipc=host \
  --shm-size=16g \
  -it --rm \
  --name ${CONTAINER_NAME} \
  -v "$SCRIPT_DIR":/workspace \
  -v "$SCRIPT_DIR/.cache/huggingface":/workspace/.cache/huggingface \
  ${IMAGE_NAME} \
  python3 sd3_5_visdial_baseline.py \
    --model_id "stabilityai/stable-diffusion-3.5-medium" \
    --dtype "bfloat16" \
    --skip_existing \
    --output_dir ./generated_images
