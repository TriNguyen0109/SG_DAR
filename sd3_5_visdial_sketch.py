#!/usr/bin/env python3
"""
Stable Diffusion Dual-Input (Text + Sketch) Generation Script for VisDial Dataset

Generates images using dual inputs: dialog prompt text + sketch image condition per turn.

Features:
- Dual-input processing (Text prompt + Sketch image per turn via Img2Img pipeline)
- Real-time progress tracking (Current Image / Total Images, Percentage %, Time per Image, Average Time & ETA)
- Flexible CLI parameters (steps, guidance scale, conditioning strength, precision, resolution, random seed)
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
import numpy as np
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageOps
except ImportError:
    Image = None
    ImageDraw = None
    ImageOps = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion Dual-Input (Text + Sketch) Generation on VisDial queries."
    )
    
    # Input/Output paths
    parser.add_argument(
        "--json_path", 
        type=str, 
        default="VisDial_v1_0_queries_val_sketch.json",
        help="Path to formatted VisDial JSON queries file with text and sketch fields."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./generated_images_sketch",
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
    parser.add_argument("--strength", type=float, default=0.90, help="Sketch conditioning / Img2Img transformation strength.")
    parser.add_argument("--sketch_mode", type=str, choices=["neutral_gray", "binary_black", "invert", "raw"], default="neutral_gray", help="Sketch preprocessing mode (neutral_gray prevents dark/white color bias).")
    parser.add_argument("--lineart_threshold", type=int, default=220, help="Grayscale threshold (0-255) for lineart binarization.")
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
    """Initializes Stable Diffusion Image-to-Image / Condition Pipeline via diffusers."""
    if args.dry_run:
        logging.info("DRY RUN MODE: Skipping PyTorch and Diffusers model initialization.")
        return None, None

    try:
        import torch
        from diffusers import StableDiffusion3Img2ImgPipeline
    except ImportError as e:
        logging.error("Failed to import torch or diffusers. Please install dependencies.")
        raise e

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map[args.dtype]

    logging.info(f"Loading Stable Diffusion Img2Img model '{args.model_id}' ({args.dtype})...")
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
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


def create_dummy_image(prompt_text, sketch_path, width=1024, height=1024, save_path=None):
    """Generates a placeholder output image during dry-run mode."""
    if Image and ImageDraw:
        img = Image.new("RGB", (width, height), color=(30, 41, 59))
        draw = ImageDraw.Draw(img)
        msg = f"DRY RUN SKETCH + TEXT DUAL-INPUT\n\nSave: {save_path}\nSketch: {sketch_path}\nPrompt: {prompt_text[:80]}..."
        draw.text((30, 30), msg, fill=(255, 255, 255))
        if save_path:
            img.save(save_path)
    elif save_path:
        with open(save_path, "w") as f:
            f.write(f"Placeholder text={prompt_text}, sketch={sketch_path}")


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
            
            if isinstance(turn_item, dict):
                prompt = turn_item.get("text", "")
                sketch_path = turn_item.get("sketch", "")
            else:
                prompt = str(turn_item)
                sketch_path = ""
                
            img_name = f"{dialog_idx}_{turn_idx}.jpg"
            img_path = os.path.join(args.output_dir, img_name)
            
            if args.skip_existing and os.path.exists(img_path):
                logging.info(f"[{current_image_idx}/{total_expected_images} - {progress_pct:.1f}%] Skipping existing file: {img_name}")
                total_skipped += 1
                continue
            
            t0 = time.time()
            if args.dry_run:
                create_dummy_image(prompt, sketch_path, width=args.width, height=args.height, save_path=img_path)
                gen_time = time.time() - t0
                logging.info(
                    f"[DRY-RUN {current_image_idx}/{total_expected_images} ({progress_pct:.1f}%)] "
                    f"Saved: '{img_name}' in {gen_time:.4f}s | Sketch: {sketch_path} | Text: '{prompt[:40]}...'"
                )
            else:
                logging.info(f"[{current_image_idx}/{total_expected_images} ({progress_pct:.1f}%)] Generating '{img_name}' using Text + Sketch ({sketch_path})...")
                try:
                    sketch_img = None
                    if sketch_path and os.path.exists(sketch_path) and Image:
                        raw_img = Image.open(sketch_path).convert("L")
                        thresh = args.lineart_threshold
                        
                        if args.sketch_mode == "neutral_gray":
                            # Background -> 128 (Neutral Gray, Zero-centered VAE Latent), Lines -> 0 (Dark stroke)
                            arr = np.array(raw_img)
                            proc = np.where(arr >= thresh, 128, 0).astype(np.uint8)
                            sketch_img = Image.fromarray(proc).convert("RGB").resize((args.width, args.height))
                        elif args.sketch_mode == "binary_black":
                            # Background -> 0 (Pitch Black), Lines -> 255 (Bright White)
                            binary_mask = raw_img.point(lambda p: 255 if p < thresh else 0)
                            sketch_img = binary_mask.convert("RGB").resize((args.width, args.height))
                        elif args.sketch_mode == "invert":
                            # Inverted RGB sketch
                            sketch_img = raw_img.convert("RGB").resize((args.width, args.height))
                            if ImageOps:
                                sketch_img = ImageOps.invert(sketch_img)
                        else:
                            # Raw original sketch
                            sketch_img = raw_img.convert("RGB").resize((args.width, args.height))
                    
                    if sketch_img is not None:
                        output = pipe(
                            prompt=prompt,
                            image=sketch_img,
                            strength=args.strength,
                            height=args.height,
                            width=args.width,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            generator=generator
                        )
                    else:
                        logging.warning(f"Sketch file not found at '{sketch_path}'. Falling back to prompt text only.")
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
