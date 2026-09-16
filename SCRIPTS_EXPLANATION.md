# Script Overview & Docker Execution Guide

This document details the purpose, workflow, data structure, command-line arguments, and **comprehensive Docker setup and execution guide** for the VisDial image generation project using Stable Diffusion 3.5.

---

## 📋 File List & Role Definitions

### 1. Python Files (`.py`)
| Python File | Purpose & Role |
| :--- | :--- |
| **`prepare_sketch_dataset.py`** | Reformat VisDial JSON dataset to add the sketch path field (`"sketch"`) for each dialogue turn. |
| **`sd3_5_visdial_baseline.py`** | **Text-Only Baseline:** Generates images solely from conversation text (`text`). Automatically ignores sketch fields even if present. Supports real-time progress logging, average speed, and ETA. |
| **`sd3_5_visdial_sketch.py`** | **Dual-Input (Text + Sketch):** Generates images conditioned on both text descriptions and sketch images. |

### 2. Shell Scripts & Docker Files
| File | Purpose & Role |
| :--- | :--- |
| **`Dockerfile`** | Defines container environment (`PyTorch 2.4.0`, `CUDA 12.1`, `Diffusers`, OpenCV/Image libraries). |
| **`run_docker.sh`** | Automatically builds the Docker image and runs the baseline pipeline on specified GPU(s). |
| **`run.sh`** | Versatile helper script: Executes any Python command or opens an interactive Bash terminal inside Docker. |
| **`compose.yaml`** | Docker Compose configuration file for GPU environment. |

---

## 🐍 1. Python Scripts Detail

### 1.1 `prepare_sketch_dataset.py`

#### 🎯 Functionality
Converts and formats VisDial v1.0 JSON files, extracting original image metadata and adding corresponding sketch paths (`sketches/{stem}_{turn_idx}.png`) for each dialogue turn.

#### 🔄 Data Flow (JSON Formatting)
* **Input (Original):**
  ```json
  [
      {
          "img": "unlabeled2017/000000185565.jpg",
          "dialog": [
              "a bedroom is filled with lots of posters and a busy computer desk",
              "a vibrant bedroom filled with colorful posters..."
          ]
      }
  ]
  ```
* **Output (With sketch paths):**
  ```json
  [
      {
          "img": "unlabeled2017/000000185565.jpg",
          "dialog": [
              {
                  "text": "a bedroom is filled with lots of posters and a busy computer desk",
                  "sketch": "sketches/000000185565_0.png"
              },
              {
                  "text": "a vibrant bedroom filled with colorful posters...",
                  "sketch": "sketches/000000185565_1.png"
              }
          ]
      }
  ]
  ```

---

### 1.2 `sd3_5_visdial_baseline.py` (Text-Only Baseline)

#### 🎯 Functionality & Real-time Progress
Executes pure **Text-to-Image baseline** generation from VisDial dialogues using Stable Diffusion 3.5.

#### 📊 Real-time Progress Log Format:
Each generated image displays:
- **Current image index / Total images** & Progress percentage `%`.
- **Processing duration per image** (in seconds).
- **Average speed** (`Avg: X.XXs/img`) and **Estimated Time of Arrival (ETA)**.

```text
[5/22 (22.7%)] Generating '0_4.jpg'...
✓ Saved '0_4.jpg' in 3.45s | Avg: 3.20s/img | ETA: 54s
```

---

### 1.3 `sd3_5_visdial_sketch.py` (Dual-Input Text + Sketch)

#### 🎯 Functionality
Executes **Dual-Input (Text + Sketch)** generation. Each dialogue turn pairs the text description with the corresponding sketch image as conditional input to the Stable Diffusion pipeline.

---

## 🐳 2. Docker Execution Guide

The Docker environment isolates dependencies, guarantees PyTorch / CUDA version compatibility, and automatically caches model weights without requiring complex host setup.

### 2.1 Environment Configuration & Cache Mechanism
- **Base Image**: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`
- **Image Name**: `aiclub_visdial-core_baotg_sd35_v1`
- **Volume Mounts**:
  - `$(pwd)` -> `/workspace`: Source code, JSON data, and generated images are synced directly to the host machine.
  - `$(pwd)/.cache/huggingface` -> `/workspace/.cache/huggingface`: Persists Hugging Face model weights so weights are downloaded only once.

---

### 2.2 Method 1: Run with `run_docker.sh` (Automatic Baseline Pipeline)

The `run_docker.sh` script automatically builds the image (if not already built) and launches the Text-Only Baseline generation process with optimized settings.

#### Syntax:
```bash
./run_docker.sh [GPU_ID]
```

#### Examples:
```bash
# 1. Run on GPU 0 (default)
./run_docker.sh

# 2. Run on GPU 1
./run_docker.sh 1

# 3. Run on GPU 2 or multiple GPUs (e.g., "0,1")
./run_docker.sh "0,1"
```

#### Internal Steps Executed:
1. Changes to script directory.
2. Automatically builds image `aiclub_visdial-core_baotg_sd35_v1`.
3. Creates `.cache/huggingface` directory to prevent re-downloading.
4. Executes `sd3_5_visdial_baseline.py` in `bfloat16` precision, enabling `--skip_existing` to resume progress.

---

### 2.3 Method 2: Run with Helper Script `run.sh` (Flexible & Recommended)

`run.sh` is the most versatile helper script, allowing you to pass **any Python or Bash command** into the Docker container.

#### Syntax:
```bash
./run.sh [command_to_run]
```

#### Common Usage Patterns:

1. **Open interactive Bash terminal inside container:**
   ```bash
   ./run.sh
   # or
   ./run.sh bash
   ```

2. **Run quick dry-run test without loading GPU/VRAM:**
   ```bash
   ./run.sh python3 sd3_5_visdial_baseline.py --dry_run --max_dialogs 2
   ```

3. **Run Text-Only Baseline pipeline:**
   ```bash
   ./run.sh python3 sd3_5_visdial_baseline.py \
     --model_id "stabilityai/stable-diffusion-3.5-medium" \
     --dtype "bfloat16" \
     --skip_existing
   ```

4. **Run Dual-Input (Sketch + Text) pipeline:**
   ```bash
   ./run.sh python3 sd3_5_visdial_sketch.py \
     --json_path "VisDial_v1_0_queries_val_sketch.json" \
     --model_id "stabilityai/stable-diffusion-3.5-medium" \
     --dtype "bfloat16" \
     --skip_existing
   ```

5. **Select GPU with `GPU_IDS` environment variable:**
   ```bash
   # Run on GPU 1
   GPU_IDS=1 ./run.sh python3 sd3_5_visdial_baseline.py --skip_existing

   # Run on GPU 2
   GPU_IDS=2 ./run.sh python3 sd3_5_visdial_sketch.py --skip_existing
   ```

---

### 2.4 Method 3: Run via Docker Compose

To manage the environment using Docker Compose:

```bash
# 1. Start background service
docker compose up -d

# 2. Access the running container
docker compose exec app bash

# 3. Execute script inside container
python3 sd3_5_visdial_baseline.py --skip_existing

# 4. Stop container when finished
docker compose down
```

---

### 2.5 Method 4: Manual Docker CLI Commands

To execute directly using standard Docker CLI:

```bash
# Step 1: Build Docker Image
docker build -t aiclub_visdial-core_baotg_sd35_v1 .

# Step 2: Run Container
docker run --gpus '"device=0"' \
  --ipc=host \
  --shm-size=16g \
  -it --rm \
  --name sd35_visdial_runner \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/.cache/huggingface":/workspace/.cache/huggingface \
  aiclub_visdial-core_baotg_sd35_v1 \
  python3 sd3_5_visdial_baseline.py --skip_existing
```

---

## ⚙️ 3. Key Pipeline Parameters

| Parameter | Meaning | Recommendation |
| :--- | :--- | :--- |
| `--skip_existing` | Skip already generated images | **Highly recommended** to pause and resume generation seamlessly. |
| `--dry_run` | Test data flow without loading GPU/model | Use to verify JSON input format and output filenames. |
| `--max_dialogs N` | Limit processing to `N` dialogues | Useful for quick smoke testing on small samples. |
| `--dtype` | Float precision (`bfloat16`, `float16`, `float32`) | `bfloat16` is recommended for SD 3.5 to optimize performance and VRAM. |
| `--cpu_offload` | Offload unused modules to system RAM | Use when GPU VRAM is constrained (prevents OOM). |
| `--strength` | Influence weight of sketch image (sketch script only) | Default is `0.85`. |

---

## 🛠️ 4. Troubleshooting

1. **`Permission denied` when running `.sh` scripts**:
   Grant execution permissions:
   ```bash
   chmod +x run.sh run_docker.sh
   ```

2. **CUDA Out of Memory (OOM)**:
   Add `--cpu_offload` to the execution command:
   ```bash
   ./run.sh python3 sd3_5_visdial_baseline.py --cpu_offload --skip_existing
   ```

3. **Check available GPUs before running**:
   ```bash
   nvidia-smi
   ```
