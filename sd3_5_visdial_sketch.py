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
        "--use_flux",
        action="store_true",
        help="Use FLUX Img2Img pipeline instead of Stable Diffusion 3.5."
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
    parser.add_argument(
        "--no_cpu_offload",
        action="store_true",
        help="Force disable CPU offloading and keep all model weights in GPU VRAM."
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="./models/sketch_to_image_klein_4b/sketch_to_image_klein_4b.safetensors",
        help="Path or directory to custom LoRA weights (e.g. ./models/sketch_to_image_klein_4b/sketch_to_image_klein_4b.safetensors)."
    )
    
    # Generation hyperparameters
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for dual-input prompt & sketch inference.")
    parser.add_argument("--height", type=int, default=1024, help="Image height in pixels.")
    parser.add_argument("--width", type=int, default=1024, help="Image width in pixels.")
    parser.add_argument("--num_inference_steps", type=int, default=18, help="Number of denoising inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=8.5, help="Classifier-Free Guidance (CFG) scale for rich vibrant colors.")
    parser.add_argument("--strength", type=float, default=0.85, help="Sketch conditioning / Img2Img transformation strength.")
    parser.add_argument("--sketch_mode", type=str, choices=["neutral_gray", "binary_black", "invert", "raw"], default="neutral_gray", help="Sketch preprocessing mode (neutral_gray balances background tones for vibrant VAE colors).")
    parser.add_argument("--lineart_threshold", type=int, default=220, help="Grayscale threshold (0-255) for lineart binarization.")
    parser.add_argument("--prompt_prefix", type=str, default="a beautiful masterpiece, professional photograph, 8k resolution, vibrant vivid saturated colors, sharp focus, high contrast, cinematic lighting, ", help="Prefix to enforce photorealism and rich colors like baseline.")
    parser.add_argument("--negative_prompt", type=str, default="monochrome, grayscale, black and white, line art, sketch lines, cartoon, anime, 2d, desaturated, washed out, low contrast, dark tones, blurry, bad anatomy, draft", help="Negative prompt to prevent monochrome/sketch look.")
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
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except ImportError as e:
        logging.error("Failed to import torch or diffusers. Please install dependencies.")
        raise e

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map[args.dtype]

    is_flux = "flux" in args.model_id.lower() or args.use_flux
    if is_flux:
        from diffusers import FluxImg2ImgPipeline
        logging.info(f"Loading FLUX Img2Img model '{args.model_id}' ({args.dtype})...")
        pipe = FluxImg2ImgPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            token=args.hf_token
        )
    else:
        from diffusers import StableDiffusion3Img2ImgPipeline
        logging.info(f"Loading Stable Diffusion Img2Img model '{args.model_id}' ({args.dtype})...")
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            token=args.hf_token
        )
    should_offload = args.cpu_offload
    if not args.no_cpu_offload and not should_offload and args.device == "cuda" and torch.cuda.is_available():
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            if total_mem < 20 * (1024 ** 3):
                should_offload = True
                logging.info(f"Detected GPU VRAM ({total_mem / (1024**3):.2f} GB) < 20GB. Automatically enabling CPU Offloading to prevent OOM.")
        except Exception:
            pass

    if args.no_cpu_offload:
        should_offload = False
        logging.info("Forced disabling CPU Offloading (--no_cpu_offload). Loading full model directly into GPU VRAM.")

    if args.lora_path and os.path.exists(args.lora_path):
        logging.info(f"Loading custom LoRA weights from '{args.lora_path}'...")
        try:
            if os.path.isfile(args.lora_path):
                dir_name = os.path.dirname(args.lora_path) or "."
                file_name = os.path.basename(args.lora_path)
                pipe.load_lora_weights(dir_name, weight_name=file_name)
            else:
                pipe.load_lora_weights(args.lora_path)
            logging.info("✓ LoRA weights loaded successfully.")
        except Exception as e:
            logging.warning(f"Failed to load LoRA weights from '{args.lora_path}': {e}")

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
    
    # Flatten all generation tasks across dialogs
    all_tasks = []
    for local_i, item in enumerate(target_data):
        dialog_idx = start_idx + local_i
        dialog_turns = item.get("dialog", [])
        for turn_idx, turn_item in enumerate(dialog_turns):
            if isinstance(turn_item, dict):
                prompt = turn_item.get("text", "")
                sketch_path = turn_item.get("sketch", "")
            else:
                prompt = str(turn_item)
                sketch_path = ""
            img_name = f"{dialog_idx}_{turn_idx}.jpg"
            img_path = os.path.join(args.output_dir, img_name)
            all_tasks.append({
                "dialog_idx": dialog_idx,
                "turn_idx": turn_idx,
                "prompt": prompt,
                "sketch_path": sketch_path,
                "img_name": img_name,
                "img_path": img_path
            })

    total_expected_images = len(all_tasks)
    
    logging.info("=" * 60)
    logging.info(f"Target Dialog Range : [{start_idx} to {end_idx - 1}] ({len(target_data)} dialog items)")
    logging.info(f"Total Images to Process : {total_expected_images} images")
    logging.info(f"Batch Size : {args.batch_size}")
    logging.info("=" * 60)
    
    pipe, generator = init_sd35_pipeline(args)
    
    total_generated = 0
    total_skipped = 0
    start_time = time.time()

    pending_tasks = []
    for t in all_tasks:
        if args.skip_existing and os.path.exists(t["img_path"]):
            total_skipped += 1
        else:
            pending_tasks.append(t)

    if total_skipped > 0:
        logging.info(f"Skipping {total_skipped} already existing images.")

    current_image_idx = total_skipped
    batch_size = max(1, args.batch_size)

    def load_processed_sketch(sketch_path):
        if sketch_path and os.path.exists(sketch_path) and Image:
            raw_img = Image.open(sketch_path).convert("L")
            thresh = args.lineart_threshold
            if args.sketch_mode == "neutral_gray":
                arr = np.array(raw_img)
                proc = np.where(arr >= thresh, 128, 0).astype(np.uint8)
                return Image.fromarray(proc).convert("RGB").resize((args.width, args.height))
            elif args.sketch_mode == "binary_black":
                binary_mask = raw_img.point(lambda p: 255 if p < thresh else 0)
                return binary_mask.convert("RGB").resize((args.width, args.height))
            elif args.sketch_mode == "invert":
                img = raw_img.convert("RGB").resize((args.width, args.height))
                if ImageOps:
                    img = ImageOps.invert(img)
                return img
            else:
                return raw_img.convert("RGB").resize((args.width, args.height))
        return None

    for i in range(0, len(pending_tasks), batch_size):
        chunk = pending_tasks[i:i + batch_size]
        prompts = [t["prompt"] for t in chunk]
        batch_names = [t["img_name"] for t in chunk]

        t0 = time.time()
        if args.dry_run:
            for task in chunk:
                create_dummy_image(task["prompt"], task["sketch_path"], width=args.width, height=args.height, save_path=task["img_path"])
            gen_time = time.time() - t0
            current_image_idx += len(chunk)
            total_generated += len(chunk)
            progress_pct = (current_image_idx / total_expected_images) * 100
            logging.info(
                f"[DRY-RUN {current_image_idx}/{total_expected_images} ({progress_pct:.1f}%)] "
                f"Saved batch {batch_names} in {gen_time:.4f}s"
            )
        else:
            current_image_idx += len(chunk)
            progress_pct = (current_image_idx / total_expected_images) * 100
            logging.info(f"[{current_image_idx}/{total_expected_images} ({progress_pct:.1f}%)] Generating batch: {batch_names}...")
            try:
                sketch_imgs = [load_processed_sketch(t["sketch_path"]) for t in chunk]
                # If any sketch is missing, fallback to neutral gray image placeholder
                fallback_blank = Image.new("RGB", (args.width, args.height), color=(128, 128, 128)) if Image else None
                input_imgs = [s if s is not None else fallback_blank for s in sketch_imgs]

                full_prompts = [args.prompt_prefix + p for p in prompts]
                prompt_arg = full_prompts[0] if len(full_prompts) == 1 else full_prompts
                img_arg = input_imgs[0] if len(input_imgs) == 1 else input_imgs

                neg_prompts = [args.negative_prompt] * len(chunk)
                neg_arg = neg_prompts[0] if len(neg_prompts) == 1 else neg_prompts

                import torch
                batch_gens = [torch.Generator(device=args.device).manual_seed(args.seed + t["dialog_idx"]) for t in chunk]
                gen_arg = batch_gens[0] if len(batch_gens) == 1 else batch_gens

                kwargs = {
                    "prompt": prompt_arg,
                    "image": img_arg,
                    "strength": args.strength,
                    "height": args.height,
                    "width": args.width,
                    "num_inference_steps": args.num_inference_steps,
                    "guidance_scale": args.guidance_scale,
                    "generator": gen_arg
                }
                if args.negative_prompt:
                    kwargs["negative_prompt"] = neg_arg

                output = pipe(**kwargs)
                for idx_in_batch, out_img in enumerate(output.images):
                    out_img.save(chunk[idx_in_batch]["img_path"], quality=95)

                gen_time = time.time() - t0
                total_generated += len(chunk)

                elapsed_so_far = time.time() - start_time
                avg_time_per_img = elapsed_so_far / total_generated
                remaining_imgs = total_expected_images - current_image_idx
                eta_seconds = avg_time_per_img * remaining_imgs

                logging.info(
                    f"✓ Saved batch {batch_names} in {format_time(gen_time)} | "
                    f"Avg: {avg_time_per_img:.2f}s/img | ETA: {format_time(eta_seconds)}"
                )
            except Exception as e:
                logging.error(f"❌ Error generating batch {batch_names}: {e}")
                continue

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
