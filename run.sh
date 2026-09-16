#!/bin/bash
# ==============================================================================
# Helper Script: Chạy bất kỳ lệnh Python / Bash nào trong Docker cực kỳ ngắn gọn
# Cú pháp: ./run.sh [lệnh cần chạy]
# Ví dụ:   ./run.sh python3 sd3_5_visdial_baseline.py --cpu_offload
#          ./run.sh bash
# ==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

IMAGE_NAME="aiclub_visdial-core_baotg_sd35_v1:latest"
GPU_IDS="${GPU_IDS:-0}"

# Tự động build image nếu chưa tồn tại cục bộ
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "🔧 Image '${IMAGE_NAME}' chưa có. Đang tiến hành Build Docker Image..."
    docker build -t "${IMAGE_NAME}" .
fi

# Tạo thư mục cache nếu chưa tồn tại
mkdir -p "$SCRIPT_DIR/.cache/huggingface"

# Nếu không truyền câu lệnh nào, mặc định vào môi trường bash
if [ $# -eq 0 ]; then
    CMD="bash"
else
    CMD="$@"
fi

docker run --gpus "\"device=${GPU_IDS}\"" \
  --ipc=host \
  --shm-size=16g \
  -it --rm \
  -v "$SCRIPT_DIR":/workspace \
  -v "$SCRIPT_DIR/.cache/huggingface":/workspace/.cache/huggingface \
  ${IMAGE_NAME} ${CMD}
