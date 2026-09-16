FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Environment settings
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# Install system dependencies & libraries needed for CV / Image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    htop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Upgrade pip & install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel ninja && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Default command
CMD ["python3", "sd3_5_visdial_baseline.py", "--help"]
