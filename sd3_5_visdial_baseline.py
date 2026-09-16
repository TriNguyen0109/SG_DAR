#!/usr/bin/env python3
"""
Stable Diffusion 3.5 VisDial Baseline Image Generation Script (Text-Only)

Generates images purely from text dialog prompts using Stable Diffusion 3.5.
Ignores sketch inputs even if present in the JSON dataset file.

Features:
- Pure Text-to-Image baseline generation
- Real-time progress tracking (Current Image / Total Images, Percentage %, Time per Image, Average Time & ETA)
- Hugging Face `diffusers` integration for SD 3.5 (Medium / Large)
- Flexible CLI parameters (steps, guidance scale, precision, resolution, random seed)
- Flexible JSON dialog parsing (extracts text prompt from string list or dict format)
- Resumable generation (--skip_existing)
- Range filtering (--start_idx, --end_idx, --max_dialogs)
- Fast dry-run / Mock mode (--dry_run) for format verification without GPU/weights
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion 3.5 Text-Only Baseline on VisDial text dialogs."
    )
    
    # Input/Output paths
    parser.add_argument(
        "--json_path", 
        type=str, 
        default="VisDial_v1_0_queries_val.json",
        help="Path to VisDial JSON queries file."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./generated_images_baseline",
        help="Directory where generated images will be saved."
    )
    
    # Model configuration
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Hugging Face model ID."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run inference on (cuda/cpu)."
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        choices=["bfloat16", "float16", "float32"], 
        default="bfloat16",
        help="Torch data type for model weights."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face User Access Token."
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable Diffusers model CPU offloading to save VRAM."
    )
    
    # Generation hyperparameters
    parser.add_argument("--height", type=int, default=1024, help="Image height in pixels.")
    parser.add_argument("--width", type=int, default=1024, help="Image width in pixels.")
    parser.add_argument("--num_inference_steps", type=int, default=28, help="Number of denoising inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Classifier-Free Guidance (CFG) scale.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # Execution & subset controls
    parser.add_argument("--start_idx", type=int, default=0, help="Starting dialog index in JSON array.")
    parser.add_argument("--end_idx", type=int, default=None, help="Ending dialog index in JSON array (exclusive).")
    parser.add_argument("--max_dialogs", type=int, default=None, help="Maximum number of dialog items to process.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip generation if output file already exists.")
    parser.add_argument("--dry_run", action="store_true", help="Run script in dry-run mode without loading model weights.")
    
    return parser.parse_args()


def load_visdial_data(json_path):
    """Loads and validates the VisDial JSON queries file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")
    
    logging.info(f"Loading dataset from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logging.info(f"Loaded {len(data)} dialog items.")
    return data


def init_sd35_pipeline(args):
    """Initializes Stable Diffusion 3.5 Pipeline via diffusers."""
    if args.dry_run:
        logging.info("DRY RUN MODE: Skipping PyTorch and Diffusers model initialization.")
        return None, None

    try:
        import torch
        from diffusers import StableDiffusion3Pipeline
    except ImportError as e:
        logging.error("Failed to import torch or diffusers. Please install dependencies.")
        raise e

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map[args.dtype]

    logging.info(f"Loading Stable Diffusion 3.5 Text-to-Image model '{args.model_id}' ({args.dtype})...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        token=args.hf_token
    )
    should_offload = args.cpu_offload
    if not should_offload and args.device == "cuda" and torch.cuda.is_available():
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            if total_mem < 20 * (1024 ** 3):
                should_offload = True
                logging.info(f"Detected GPU VRAM ({total_mem / (1024**3):.2f} GB) < 20GB. Automatically enabling CPU Offloading to prevent OOM.")
        except Exception:
            pass

    if should_offload:
        logging.info("Enabling Model CPU Offloading to optimize VRAM usage...")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)
    logging.info("Pipeline loaded successfully.")
    
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    return pipe, generator


def create_dummy_image(prompt_text, width=1024, height=1024, save_path=None):
    """Generates a placeholder image during dry-run mode."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (width, height), color=(30, 41, 59))
        draw = ImageDraw.Draw(img)
        draw.text((30, 30), f"DRY RUN BASELINE (TEXT-ONLY)\n\nSave: {save_path}\nPrompt: {prompt_text[:100]}...", fill=(255, 255, 255))
        if save_path:
            img.save(save_path)
    except ImportError:
        if save_path:
            with open(save_path, "w") as f:
                f.write(f"Dry run placeholder for prompt: {prompt_text}")


def format_time(seconds):
    """Formats seconds into human-readable string (e.g., '2m 15s' or '45.2s')."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem_sec = int(seconds % 60)
    return f"{minutes}m {rem_sec}s"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_visdial_data(args.json_path)
    
    start_idx = args.start_idx
    end_idx = len(data) if args.end_idx is None else min(args.end_idx, len(data))
    if args.max_dialogs is not None:
        end_idx = min(start_idx + args.max_dialogs, end_idx)
        
    target_data = data[start_idx:end_idx]
    
    # Calculate total expected images across all dialog turns
    total_expected_images = sum(len(item.get("dialog", [])) for item in target_data)
    
    logging.info("=" * 60)
    logging.info(f"Target Dialog Range : [{start_idx} to {end_idx - 1}] ({len(target_data)} dialog items)")
    logging.info(f"Total Images to Process : {total_expected_images} images")
    logging.info("=" * 60)
    
    pipe, generator = init_sd35_pipeline(args)
    
    total_generated = 0
    total_skipped = 0
    current_image_idx = 0
    start_time = time.time()
    
    for local_i, item in enumerate(target_data):
        dialog_idx = start_idx + local_i
        dialog_turns = item.get("dialog", [])
        
        logging.info(f"\n>>> Processing Dialog [{dialog_idx}/{end_idx - 1}] ({len(dialog_turns)} turns) <<<")
        
        for turn_idx, turn_item in enumerate(dialog_turns):
            current_image_idx += 1
            progress_pct = (current_image_idx / total_expected_images) * 100
            
            # Extract TEXT ONLY (ignore sketch even if present in JSON dict)
            if isinstance(turn_item, dict):
                prompt = turn_item.get("text", "")
            else:
                prompt = str(turn_item)
                
            img_name = f"{dialog_idx}_{turn_idx}.jpg"
            img_path = os.path.join(args.output_dir, img_name)
            
            if args.skip_existing and os.path.exists(img_path):
                logging.info(f"[{current_image_idx}/{total_expected_images} - {progress_pct:.1f}%] Skipping existing file: {img_name}")
                total_skipped += 1
                continue
            
            t0 = time.time()
            if args.dry_run:
                create_dummy_image(prompt, width=args.width, height=args.height, save_path=img_path)
                gen_time = time.time() - t0
                logging.info(
                    f"[DRY-RUN {current_image_idx}/{total_expected_images} ({progress_pct:.1f}%)] "
                    f"Saved: '{img_name}' in {gen_time:.4f}s | Prompt: '{prompt[:50]}...'"
                )
            else:
                logging.info(f"[{current_image_idx}/{total_expected_images} ({progress_pct:.1f}%)] Generating '{img_name}'...")
                try:
                    output = pipe(
                        prompt=prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator
                    )
                    output.images[0].save(img_path, quality=95)
                    gen_time = time.time() - t0
                    
                    elapsed_so_far = time.time() - start_time
                    avg_time_per_img = elapsed_so_far / (total_generated + 1)
                    remaining_imgs = total_expected_images - current_image_idx
                    eta_seconds = avg_time_per_img * remaining_imgs
                    
                    logging.info(
                        f"✓ Saved '{img_name}' in {format_time(gen_time)} | "
                        f"Avg: {avg_time_per_img:.2f}s/img | ETA: {format_time(eta_seconds)}"
                    )
                except Exception as e:
                    logging.error(f"❌ Error generating '{img_name}': {e}")
                    continue
            
            total_generated += 1

    elapsed = time.time() - start_time
    logging.info("\n" + "=" * 60)
    logging.info("Execution Complete!")
    logging.info(f"Total Images Generated : {total_generated}")
    logging.info(f"Total Images Skipped   : {total_skipped}")
    logging.info(f"Total Time Elapsed     : {format_time(elapsed)}")
    if total_generated > 0:
        logging.info(f"Average Speed Per Image: {elapsed / total_generated:.2f}s/image")
    logging.info(f"Output Directory       : {os.path.abspath(args.output_dir)}")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
