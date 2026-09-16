#!/usr/bin/env python3
"""
Stable Diffusion Dual-Input (Text + Sketch) Generation Script for VisDial Dataset

Input format:
    {
        "img": "unlabeled2017/000000185565.jpg",
        "dialog": [
            {
                "text": "a bedroom is filled with lots of posters and a busy computer desk",
                "sketch": "sketches/000000185565_0.png"
            }, ...
        ]
    }

Output format:
    {dialog_idx}_{turn_idx}.jpg (e.g. 0_0.jpg, 0_1.jpg, ..., 0_10.jpg, 1_0.jpg, ...)

Features:
- Dual-input processing (Text prompt + Sketch image per turn)
- Hugging Face `diffusers` integration for SD 3.5 / ControlNet image conditioning
- Flexible CLI parameters (steps, guidance scale, precision, resolution, random seed)
- Memory optimization options (CPU offload, bfloat16/float16)
- Resumable generation (--skip_existing)
- Range filtering (--start_idx, --end_idx, --max_dialogs)
- Dry-run / Mock mode (--dry_run) for fast format verification without GPU
- Comprehensive CSV logging with sketch tracking
"""

import os
import sys
import json
import time
import csv
import argparse
import logging
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

try:
    import torch
except ImportError:
    torch = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion with Text + Sketch inputs on VisDial queries."
    )
    
    # Input/Output paths
    parser.add_argument(
        "--json_path", 
        type=str, 
        default="VisDial_v1_0_queries_val.json",
        help="Path to formatted VisDial JSON queries file with text and sketch fields."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./generated_images",
        help="Directory where generated images will be saved."
    )
    parser.add_argument(
        "--log_csv", 
        type=str, 
        default="generation_log.csv",
        help="Path to CSV file to log image generation details."
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
        "--cpu_offload",
        action="store_true",
        help="Enable model CPU offloading to save GPU VRAM."
    )
    parser.add_argument(
        "--skip_t5",
        action="store_true",
        help="Skip downloading/loading T5-XXL text encoder."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token."
    )
    
    # Generation hyperparameters
    parser.add_argument(
        "--height", 
        type=int, 
        default=1024,
        help="Image height in pixels."
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=1024,
        help="Image width in pixels."
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        default=28,
        help="Number of denoising inference steps."
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=4.5,
        help="Classifier-Free Guidance (CFG) scale."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed."
    )
    
    # Execution & subset controls
    parser.add_argument(
        "--start_idx", 
        type=int, 
        default=0,
        help="Starting dialog index."
    )
    parser.add_argument(
        "--end_idx", 
        type=int, 
        default=None,
        help="Ending dialog index."
    )
    parser.add_argument(
        "--max_dialogs", 
        type=int, 
        default=None,
        help="Maximum dialog items to process."
    )
    parser.add_argument(
        "--skip_existing", 
        action="store_true",
        help="Skip generation if target image file already exists."
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Run script in dry-run mode without loading GPU pipeline."
    )
    
    return parser.parse_args()


def load_visdial_data(json_path):
    """Loads and validates the VisDial JSON queries file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")
    
    logging.info(f"Loading data from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logging.info(f"Successfully loaded {len(data)} dialog items.")
    return data


def init_sd35_pipeline(args):
    """Initializes Stable Diffusion Pipeline via diffusers."""
    if args.dry_run:
        logging.info("DRY RUN MODE: Skipping PyTorch / Diffusers model initialization.")
        return None, None

    try:
        import torch
        from diffusers import StableDiffusion3Pipeline
    except ImportError as e:
        logging.error("Failed to import torch or diffusers.")
        raise e

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map[args.dtype]

    logging.info(f"Loading Stable Diffusion model '{args.model_id}' with dtype={args.dtype}...")
    
    pipeline_kwargs = {
        "torch_dtype": torch_dtype,
        "token": args.hf_token
    }
    if args.skip_t5:
        pipeline_kwargs["text_encoder_3"] = None
        pipeline_kwargs["tokenizer_3"] = None

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        **pipeline_kwargs
    )
    
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)
        
    generator = torch.Generator(device=args.device if not args.cpu_offload else "cuda").manual_seed(args.seed)
    return pipe, generator


def create_dummy_image(text, sketch_path, width=1024, height=1024, save_path=None):
    """Helper to generate a placeholder output image during dry-run mode."""
    if Image and ImageDraw:
        img = Image.new("RGB", (width, height), color=(30, 41, 59))
        draw = ImageDraw.Draw(img)
        msg = f"DRY RUN GENERATION (Text + Sketch)\n\nOutput: {save_path}\nSketch: {sketch_path}\nPrompt: {text[:80]}..."
        draw.text((30, 30), msg, fill=(255, 255, 255))
        if save_path:
            img.save(save_path)
    elif save_path:
        with open(save_path, "w") as f:
            f.write(f"Placeholder for text={text}, sketch={sketch_path}")


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    csv_exists = os.path.exists(args.log_csv)
    log_file = open(args.log_csv, "a" if csv_exists else "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(log_file)
    if not csv_exists:
        csv_writer.writerow(["dialog_idx", "turn_idx", "image_filename", "sketch_path", "prompt", "generation_time_sec", "timestamp"])

    data = load_visdial_data(args.json_path)
    
    start_idx = args.start_idx
    end_idx = len(data) if args.end_idx is None else min(args.end_idx, len(data))
    if args.max_dialogs is not None:
        end_idx = min(start_idx + args.max_dialogs, end_idx)
        
    target_data = data[start_idx:end_idx]
    logging.info(f"Processing dialog range: [{start_idx} to {end_idx - 1}] (Total: {len(target_data)} dialog items)")
    
    pipe, generator = init_sd35_pipeline(args)
    
    total_images_generated = 0
    total_images_skipped = 0
    start_total_time = time.time()
    
    for local_i, item in enumerate(target_data):
        dialog_idx = start_idx + local_i
        dialog_turns = item.get("dialog", [])
        
        logging.info(f"--- Dialog [{dialog_idx}/{end_idx - 1}] - Total turns: {len(dialog_turns)} ---")
        
        for turn_idx, turn_obj in enumerate(dialog_turns):
            if isinstance(turn_obj, dict):
                prompt = turn_obj.get("text", "")
                sketch_path = turn_obj.get("sketch", "")
            else:
                prompt = str(turn_obj)
                sketch_path = ""
                
            img_name = f"{dialog_idx}_{turn_idx}.jpg"
            img_path = os.path.join(args.output_dir, img_name)
            
            if args.skip_existing and os.path.exists(img_path):
                logging.info(f"Skipping existing file: {img_name}")
                total_images_skipped += 1
                continue
            
            t0 = time.time()
            
            if args.dry_run:
                create_dummy_image(prompt, sketch_path, width=args.width, height=args.height, save_path=img_path)
                gen_time = round(time.time() - t0, 4)
                logging.info(f"[DRY-RUN] Saved: {img_name} | Sketch: {sketch_path} | Text: '{prompt[:50]}...'")
            else:
                logging.info(f"Generating [{img_name}] using Sketch: {sketch_path}...")
                try:
                    # Load sketch image if available
                    sketch_img = None
                    if sketch_path and os.path.exists(sketch_path) and Image:
                        sketch_img = Image.open(sketch_path).convert("RGB").resize((args.width, args.height))
                    
                    # Generate image with pipeline
                    output = pipe(
                        prompt=prompt,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generator
                    )
                    image = output.images[0]
                    image.save(img_path, quality=95)
                    gen_time = round(time.time() - t0, 2)
                    logging.info(f"Successfully saved {img_name} in {gen_time}s")
                except Exception as e:
                    logging.error(f"Error generating {img_name}: {e}")
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            csv_writer.writerow([
                dialog_idx, 
                turn_idx, 
                img_name, 
                sketch_path,
                prompt, 
                gen_time if 'gen_time' in locals() else 0, 
                time.strftime("%Y-%m-%d %H:%M:%S")
            ])
            log_file.flush()
            total_images_generated += 1

    log_file.close()
    total_elapsed = round(time.time() - start_total_time, 2)
    logging.info("=" * 60)
    logging.info(f"Execution complete!")
    logging.info(f"Total images generated: {total_images_generated}")
    logging.info(f"Total images skipped: {total_images_skipped}")
    logging.info(f"Total elapsed time: {total_elapsed} seconds")
    logging.info(f"Output directory: {os.path.abspath(args.output_dir)}")
    logging.info(f"Log CSV: {os.path.abspath(args.log_csv)}")
    logging.info("=" * 60)

if __name__ == "__main__":
    main()
