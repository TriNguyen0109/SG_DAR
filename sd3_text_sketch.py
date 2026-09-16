#!/usr/bin/env python3
"""
Stable Diffusion 3 + Softedge ControlNet (SG-DAR).

Each dialogue turn corresponds to:
- 1 separate text prompt (read from the 'text' field in JSON).
- 1 separate sketch image (read directly from the 'sketch' field in JSON, no memory carry).
- Independent random seed per image.
- Sequential single-image generation (batch_size = 1) ensuring 100% precision.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np

# Silence CLIP token length warnings (SD3 uses T5 with 512 tokens)
import transformers
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"
CONTROLNET = "alimama-creative/SD3-Controlnet-Softedge"


def parse_args():
    p = argparse.ArgumentParser("SG-DAR / SD3-medium + Softedge")

    p.add_argument("--json_path", default="VisDial_v1_0_queries_val_sketch.json")
    p.add_argument("--output_dir", default="generated_images_text_sketch")
    p.add_argument("--sketch_root", default="",
                   help="Root directory if sketch paths in JSON are relative.")

    p.add_argument("--model_id", default=BASE_MODEL)
    p.add_argument("--controlnet_id", default=CONTROLNET)
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--offload", choices=["none", "model", "sequential"], default="none",
                   help="CPU offload mode: 'none' (default - 100%% loaded on GPU, fastest), 'model' (~8GB VRAM), 'sequential' (<3GB VRAM).")

    # Image generation
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=7.0,
                   help="Default CFG scale (7.0).")
    p.add_argument("--max_sequence_length", type=int, default=512,
                   help="T5 token length. Default is 512.")
    p.add_argument("--negative_prompt", default="low quality, blurry, deformed, noisy, grainy, bad anatomy, artifacts, cut off",
                   help="Negative prompt to denoise and clean image details.")
    p.add_argument("--num_samples_per_turn", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # Sketch / ControlNet
    p.add_argument("--cond_scale", type=float, default=0.5,
                   help="ControlNet conditioning scale (0.5).")
    p.add_argument("--control_guidance_start", type=float, default=0.0,
                   help="Fraction of steps to start applying ControlNet (0.0 = from start).")
    p.add_argument("--control_guidance_end", type=float, default=0.7,
                   help="Fraction of steps to stop applying ControlNet (0.7 = stop ControlNet after 70%% of steps so SD3 refines details in final 30%%).")
    p.add_argument("--blur_radius", type=float, default=0.6,
                   help="Blur radius for stroke smoothing (0.6).")
    p.add_argument("--baseline", action="store_true",
                   help="Original DAR: forces cond_scale = 0 across all turns.")

    # Scope
    p.add_argument("--eval_turns", default="",
                   help="Turns to generate (1-based). Empty = all turns.")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--max_dialogs", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    # Probe
    p.add_argument("--probe", action="store_true",
                   help="Generate same turn at scales 0.0/0.5/0.9 and MEASURE pixel differences.")
    p.add_argument("--probe_scales", default="0.0,0.5,0.9")

    return p.parse_args()


# ------------------------------------------------------------- Preprocessing

def to_softedge(path, w, h, blur):
    """Sketch (black strokes/white background) -> softedge map (white strokes/black background, smooth grayscale)."""
    if not path or not os.path.exists(path):
        return None
    arr = 255.0 - np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if arr.max() <= 1e-6:
        return None
    arr = arr / arr.max() * 255.0

    img = Image.fromarray(arr.astype(np.uint8))
    img = img.resize((w, h), Image.BILINEAR)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))

    arr = np.asarray(img, dtype=np.float32)
    if arr.max() <= 1e-6:
        return None
    arr = arr / arr.max() * 255.0
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def resolve_sketch(raw, root):
    if not raw:
        return None
    cands = [raw] + ([os.path.join(root, raw)] if root else [])
    for c in list(cands):
        stem = os.path.splitext(c)[0]
        cands += [stem + e for e in (".png", ".jpg", ".jpeg", ".webp")]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def black(w, h):
    return Image.new("RGB", (w, h), (0, 0, 0))


# ------------------------------------------------------------- Pipeline

def init_pipe(args):
    if args.dry_run:
        logging.info("DRY RUN — skipping model loading.")
        return None

    import torch
    from diffusers import StableDiffusion3ControlNetPipeline
    from diffusers.models import SD3ControlNetModel

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    token = args.hf_token
    if not token:
        for tp in ("/workspace/.cache/huggingface/token",
                   os.path.expanduser("~/.cache/huggingface/token"),
                   ".cache/huggingface/token"):
            if os.path.exists(tp):
                try:
                    with open(tp) as f:
                        token = f.read().strip()
                    break
                except Exception:
                    pass

    DTYPE = torch.bfloat16

    logging.info(f"Loading ControlNet {args.controlnet_id} ({DTYPE}) ...")
    cn = SD3ControlNetModel.from_pretrained(
        args.controlnet_id, torch_dtype=DTYPE, token=token)

    logging.info(f"Loading base model {args.model_id} ({DTYPE}) ...")
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.model_id, controlnet=cn, torch_dtype=DTYPE, token=token)
    pipe.set_progress_bar_config(disable=True)

    # Optimize VAE to reduce peak VRAM during image decoding
    try:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except Exception:
        pass

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        logging.info(f"VRAM: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")

    if args.offload == "none":
        pipe = pipe.to(args.device)
        logging.info("Mode: Full model on GPU (offload=none).")
    elif args.offload == "sequential":
        logging.info("Mode: Sequential CPU Offload (offload=sequential, <3GB VRAM, slow).")
        pipe.enable_sequential_cpu_offload()
    else:  # "model"
        logging.info("Mode: Model CPU Offload (offload=model, ~8GB VRAM).")
        pipe.enable_model_cpu_offload()

    if torch.cuda.is_available():
        logging.info(f"Allocated VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return pipe


def fmt(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m {int(s%60)}s"


# ------------------------------------------------------------- Main

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.json_path):
        for alt in ("VisDial_v1_0_queries_val_sketch.json",
                    "../VisDial_v1_0_queries_val_sketch.json",
                    "dataset/VisDial_v1_0_queries_val_sketch.json"):
            if os.path.exists(alt):
                args.json_path = alt
                break

    with open(args.json_path) as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} dialogues.")

    end = len(data) if args.end_idx is None else min(args.end_idx, len(data))
    if args.max_dialogs:
        end = min(args.start_idx + args.max_dialogs, end)
    subset = data[args.start_idx:end]

    keep = None
    if args.eval_turns.strip():
        keep = {int(x) - 1 for x in args.eval_turns.split(",")}
        logging.info(f"Generating turns (1-based): {sorted(i+1 for i in keep)}")

    jobs = []
    for li, item in enumerate(subset):
        d = args.start_idx + li
        turns = item.get("dialog", [])
        for t, turn in enumerate(turns):
            if keep is not None and t not in keep:
                continue
            prompt = turn.get("text", "") if isinstance(turn, dict) else str(turn)
            raw_sketch = turn.get("sketch", "") if isinstance(turn, dict) else ""
            sketch_path = resolve_sketch(raw_sketch, args.sketch_root)
            jobs.append({
                "d": d,
                "t": t,
                "prompt": prompt,
                "sketch": sketch_path
            })

    probe_scales = [float(x) for x in args.probe_scales.split(",")] if args.probe else None
    per_job = args.num_samples_per_turn * (len(probe_scales) if args.probe else 1)
    total_img = len(jobs) * per_job
    with_sk = sum(1 for j in jobs if j["sketch"])

    logging.info("=" * 64)
    logging.info(f"Dialogues {args.start_idx}..{end-1} | {len(jobs)} turns | {total_img} images")
    logging.info(f"Turns with direct sketch from JSON: {with_sk}/{len(jobs)}")
    logging.info(f"steps={args.num_inference_steps} cfg={args.guidance_scale} "
                 f"T5={args.max_sequence_length} cond={args.cond_scale} "
                 f"baseline={args.baseline} probe={args.probe}")
    logging.info("=" * 64)

    if args.probe and with_sk == 0:
        logging.error("No turns have sketches — probe is meaningless. Use --eval_turns 3")
        return

    pipe = init_pipe(args)

    def out_path(d, t, s=0, tag=""):
        if args.num_samples_per_turn == 1 and not tag:
            return os.path.join(args.output_dir, f"{d}_{t}.jpg")
        return os.path.join(args.output_dir, f"{d}_{t}{tag}_s{s}.jpg" if args.num_samples_per_turn > 1 else f"{d}_{t}{tag}.jpg")

    t0 = time.time()
    made = skipped = 0
    probe_records = []

    tasks = []
    for k, job in enumerate(jobs):
        has_sk = bool(job["sketch"] and os.path.exists(job["sketch"]))

        if args.probe and not has_sk:
            continue

        scale_val = 0.0 if args.baseline else args.cond_scale
        scales = probe_scales if args.probe else [scale_val]
        rec = {}

        for cs in scales:
            tag = f"_c{cs}" if args.probe else ""
            paths = [out_path(job["d"], job["t"], s, tag)
                     for s in range(args.num_samples_per_turn)]
            rec[cs] = paths

            for s in range(args.num_samples_per_turn):
                p = paths[s]
                if args.skip_existing and os.path.exists(p):
                    skipped += 1
                    continue
                tasks.append({
                    "d": job["d"], "t": job["t"], "s": s,
                    "prompt": job["prompt"], "sketch": job["sketch"],
                    "has_sk": has_sk, "cs": float(cs),
                    "out_path": p,
                    "seed": args.seed + job["d"] * 1000 + job["t"] * 10 + s
                })

        if args.probe and len(rec) == len(scales):
            probe_records.append(rec)

    for k, tsk in enumerate(tasks):
        if args.dry_run:
            logging.info(f"[DRY {k+1}/{len(tasks)}] d{tsk['d']} turn{tsk['t']+1} sketch={tsk['has_sk']} scale={tsk['cs']:.2f}")
            made += 1
            continue

        try:
            import torch
            ts = time.time()

            ctrl = to_softedge(tsk["sketch"], args.width, args.height, args.blur_radius) \
                   if tsk["has_sk"] else black(args.width, args.height)
            if ctrl is None:
                ctrl = black(args.width, args.height)

            gen = torch.Generator(device="cpu").manual_seed(tsk["seed"])

            with torch.inference_mode():
                imgs = pipe(
                    prompt=tsk["prompt"],
                    negative_prompt=args.negative_prompt,
                    control_image=ctrl,
                    controlnet_conditioning_scale=tsk["cs"],
                    control_guidance_start=args.control_guidance_start,
                    control_guidance_end=args.control_guidance_end,
                    height=args.height, width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    max_sequence_length=args.max_sequence_length,
                    generator=gen,
                ).images

            imgs[0].save(tsk["out_path"], quality=95)
            made += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            el = time.time() - t0
            avg = el / made
            left = total_img - made - skipped
            logging.info(f"✓ [{made}/{total_img-skipped}] d{tsk['d']} turn{tsk['t']+1} "
                         f"sk={tsk['has_sk']} scale={tsk['cs']:.2f} | {fmt(time.time()-ts)} "
                         f"| {avg:.1f}s/img | ETA {fmt(avg*max(left,0))}")
        except Exception as e:
            logging.error(f"❌ Error at d{tsk['d']} turn{tsk['t']+1}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    el = time.time() - t0
    logging.info("=" * 64)
    logging.info(f"Generated {made} | skipped {skipped} | total {fmt(el)}")
    if made:
        logging.info(f"Average {el/made:.2f}s/img")
    logging.info(f"Directory: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
