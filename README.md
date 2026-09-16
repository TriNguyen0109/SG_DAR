# Stable Diffusion 3.5 VisDial Generation Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/diffusers-0.31.0+-yellow.svg)](https://github.com/huggingface/diffusers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A clean, production-ready pipeline for generating images from the **Visual Dialog (VisDial v1.0)** dataset using **Stable Diffusion 3.5**. Supports both **Text-Only Baseline** generation and **Dual-Input (Text + Sketch Condition)** image synthesis.

---

## ✨ Key Features & Task Separation

- **Text-Only Baseline (`sd3_5_visdial_baseline.py`):** Pure Text-to-Image generation using dialog prompts. Automatically ignores sketch fields even if present in the JSON dataset to maintain pure text baseline evaluation. Saves outputs to `./generated_images_baseline`.
- **Dual-Input (`sd3_5_visdial_sketch.py`):** Dual-input image synthesis leveraging turn-level sketch conditions combined with dialog text via Img2Img pipeline. Saves outputs to `./generated_images_sketch`.
- **Fast Dry-Run Support:** Test dataset loading and pipeline execution in milliseconds without requiring GPUs or downloading multi-gigabyte model weights (`--dry_run`).
- **Resumable Generation:** Easily pause and resume progress with `--skip_existing`.
- **Docker-Ready:** Fully containerized setup with GPU passthrough, optimized shared memory, and single-command helper scripts (`./run.sh`).

---

## 📁 Repository Structure

```
diffusion/
├── Dockerfile                      # Production Docker container image configuration
├── compose.yaml                    # Docker Compose service manifest
├── run.sh                          # Helper wrapper for Docker commands
├── requirements.txt                # Python package dependencies
├── prepare_sketch_dataset.py       # Formats VisDial JSON to include per-turn sketch file paths
├── sd3_5_visdial_baseline.py       # Pure Text-Only Baseline generation script
├── sd3_5_visdial_sketch.py         # Dual-input (Text + Sketch Condition) generation script
├── VisDial_v1_0_queries_val.json   # VisDial v1.0 validation queries dataset
├── SCRIPTS_EXPLANATION.md          # Technical documentation for all Python scripts
└── README.md                       # Project documentation
```

For an in-depth breakdown of each `.py` file, see **[SCRIPTS_EXPLANATION.md](file:///d:/hocAI/research/diffusion2/diffusion/SCRIPTS_EXPLANATION.md)**.

---

## ⚡ Quick Start

### 1. Fast Dry-Run (No GPU or Weights Required)

Verify dataset reading and output file formatting instantly on any CPU machine:

```bash
# Test Text-Only Baseline pipeline
python3 sd3_5_visdial_baseline.py --dry_run --max_dialogs 2

# Test Dual-Input Text + Sketch pipeline
python3 sd3_5_visdial_sketch.py --dry_run --max_dialogs 2
```

---

### 2. Dataset JSON Preparation (Sketch Conditioning)

Format the VisDial JSON query dataset to pair each dialog turn with its corresponding sketch file path:

```bash
python3 prepare_sketch_dataset.py \
  --input_json VisDial_v1_0_queries_val.json \
  --output_json VisDial_v1_0_queries_val_sketch.json \
  --sketches_dir sketches
```

---

### 3. Running Generation on GPU

#### Option A: Native Python

```bash
# Run Text-Only Baseline Generation (Output: ./generated_images_baseline)
python3 sd3_5_visdial_baseline.py \
  --model_id "stabilityai/stable-diffusion-3.5-medium" \
  --dtype "bfloat16" \
  --skip_existing

# Run Dual-Input Sketch + Text Generation (Output: ./generated_images_sketch)
python3 sd3_5_visdial_sketch.py \
  --json_path "VisDial_v1_0_queries_val_sketch.json" \
  --model_id "stabilityai/stable-diffusion-3.5-medium" \
  --dtype "bfloat16" \
  --skip_existing
```

#### Option B: Docker Helper Script (`run.sh`)

```bash
# Run Text-Only baseline inside Docker container
./run.sh python3 sd3_5_visdial_baseline.py --skip_existing

# Run Dual-Input sketch inside Docker container
./run.sh python3 sd3_5_visdial_sketch.py --skip_existing

# Open interactive bash shell inside container
./run.sh
```

---

## 🔧 CLI Argument Reference

Common arguments supported across `sd3_5_visdial_baseline.py` and `sd3_5_visdial_sketch.py`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--json_path` | `str` | `VisDial_v1_0_queries_val.json` | Path to VisDial dataset JSON |
| `--output_dir` | `str` | `./generated_images_baseline` / `./generated_images_sketch` | Directory to save generated images |
| `--model_id` | `str` | `stabilityai/stable-diffusion-3.5-medium` | Hugging Face Model ID |
| `--device` | `str` | `cuda` | Target compute device (`cuda` or `cpu`) |
| `--dtype` | `str` | `bfloat16` | Precision (`bfloat16`, `float16`, `float32`) |
| `--height` | `int` | `1024` | Image height |
| `--width` | `int` | `1024` | Image width |
| `--num_inference_steps` | `int` | `28` | Denoising steps |
| `--guidance_scale` | `float` | `6.0` | Classifier-Free Guidance (CFG) scale |
| `--strength` | `float` | `0.85` | Sketch conditioning strength (used in `sd3_5_visdial_sketch.py`) |
| `--no_invert_sketch` | `flag` | `False` | Disable automatic sketch color inversion (used in `sd3_5_visdial_sketch.py`) |
| `--seed` | `int` | `42` | Random seed |
| `--start_idx` | `int` | `0` | Starting index in dataset |
| `--end_idx` | `int` | `None` | Ending index in dataset |
| `--max_dialogs` | `int` | `None` | Max dialog items to process |
| `--skip_existing` | `flag` | `False` | Skip already generated images |
| `--dry_run` | `flag` | `False` | Run mock test without loading GPU/weights |

---