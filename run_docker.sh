#!/bin/bash

# ==============================================================================
# Script tự động Build & Run Container cho SD 3.5 VisDial Baseline
# Quy định naming: <Nhóm>_<task>-<user>_<model>_<version>
# ==============================================================================

IMAGE_NAME="aiclub_visdial-core_baotg_sd35_v1"
CONTAINER_NAME="sd35_visdial_runner"

# Chuyển tới thư mục script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# 1. Build Docker Image (nếu chưa có hoặc khi nâng cấp)
echo "--------------------------------------------------------"
echo "🔧 Building Docker Image: ${IMAGE_NAME}..."
echo "--------------------------------------------------------"
docker build -t ${IMAGE_NAME} .

# 2. Chọn GPU sử dụng (Mặc định chọn GPU 0 - NVIDIA RTX 3060 12GB)
GPU_IDS="${1:-0}"

echo "--------------------------------------------------------"
echo "🚀 Running Docker Container on GPU(s): ${GPU_IDS}..."
echo "--------------------------------------------------------"

# Tạo thư mục cache nếu chưa có để lưu weights Hugging Face không bị tải lại nhiều lần
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
    --cpu_offload \
    --skip_t5 \
    --skip_existing \
    --output_dir ./generated_images \
    --log_csv generation_log.csv
