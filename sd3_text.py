#!/usr/bin/env python3
"""
B3 — DAR Baseline Replication: generate intermediate representations using pure SD3-medium (Text-to-Image).

No ControlNet, no sketch. Prompt = cumulative conversation history,
matching the exact text string construction in Queries.__getitem__ from eval.py.

Output: {output_dir}/{dialog_idx}_{round_idx}.jpg, round_idx = 0..10
"""

import os, json, time, argparse, logging
import transformers

# Silence CLIP token length warnings to keep logs clean
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

BASE_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"


def parse_args():
    p = argparse.ArgumentParser("B3 / DAR baseline — SD3-medium (Text-to-Image)")

    p.add_argument("--json_path", default="VisDial_v1_0_queries_val_sketch.json",
                   help="Path to dialogue JSON file (or VisDial_v1_0_queries_val.json).")
    p.add_argument("--output_dir", default="generated_images_text",
                   help="Directory to save generated images.")

    p.add_argument("--model_id", default=BASE_MODEL)
    p.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--offload", choices=["none", "model", "sequential"], default="model",
                   help="CPU offload mode: 'none' (full on GPU), 'model' (default - fast, ~8GB VRAM), 'sequential' (slow, <3GB VRAM).")

    # Image generation
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=7.0,
                   help="Default CFG scale (7.0).")
    p.add_argument("--max_sequence_length", type=int, default=512,
                   help="T5 token length. Diffusers default is 256 -> long 10-turn captions would be truncated.")
    p.add_argument("--negative_prompt", default="low quality, blurry, deformed, noisy, grainy, bad anatomy, artifacts, cut off",
                   help="Negative prompt to denoise and clean image details.")
    p.add_argument("--sep_token", default=", ",
                   help="MUST match cfg['sep_token'] in eval.py.")
    p.add_argument("--seed", type=int, default=42)

    # Scope
    p.add_argument("--num_rounds", type=int, default=11,
                   help="dialog[0] is caption, followed by 10 Q&A rounds (total 11 rounds).")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--max_dialogs", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def turn_text(turn):
    return turn.get("text", "") if isinstance(turn, dict) else str(turn)


def build_prompts(turns, num_rounds):
    """
    Prompt for turn r = extract the standalone description sentence for turn r:
        prompt = turns[r]['text']
    """
    prompts = []
    for r in range(num_rounds):
        if r < len(turns):
            prompts.append(turn_text(turns[r]))
        else:
            prompts.append(turn_text(turns[-1]) if turns else "")
    return prompts


def init_pipe(args):
    if args.dry_run:
        logging.info("DRY RUN — skipping model loading.")
        return None

    import torch
    from diffusers import StableDiffusion3Pipeline

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
                    with open(tp, "r") as f:
                        token = f.read().strip() or None
                    if token:
                        break
                except Exception:
                    pass

    DTYPE = torch.bfloat16      # SD3 trained with bf16
    logging.info(f"Loading {args.model_id} ({DTYPE}) ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id, torch_dtype=DTYPE, token=token)
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
        logging.info("Mode: Model CPU Offload (offload=model, ~8GB VRAM, FAST).")
        pipe.enable_model_cpu_offload()

    if torch.cuda.is_available():
        logging.info(f"Allocated VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return pipe


def fmt(s):
    return f"{s:.1f}s" if s < 60 else f"{int(s//60)}m {int(s%60)}s"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Find JSON file if relative path
    json_path = args.json_path
    if not os.path.exists(json_path):
        for alt in ["dataset/VisDial_v1_0_queries_val_sketch.json", "VisDial_v1_0_queries_val_sketch.json",
                    "VisDial_v1_0_queries_val.json", "dataset/VisDial_v1_0_queries_val.json"]:
            if os.path.exists(alt):
                json_path = alt
                break

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} dialogues from {json_path}")

    end = len(data) if args.end_idx is None else min(args.end_idx, len(data))
    if args.max_dialogs:
        end = min(args.start_idx + args.max_dialogs, end)

    jobs = []
    short = 0
    for d in range(args.start_idx, end):
        turns = data[d].get("dialog", [])
        if len(turns) < args.num_rounds:
            short += 1
        prompts = build_prompts(turns, args.num_rounds)
        for r, prompt in enumerate(prompts):
            jobs.append({
                "d": d, "r": r, "prompt": prompt,
                "path": os.path.join(args.output_dir, f"{d}_{r}.jpg"),
            })

    if short:
        logging.warning(f"⚠ {short} dialogues have fewer than {args.num_rounds} turns — "
                        f"prompts for final turns will repeat.")

    logging.info("=" * 64)
    logging.info(f"Dialogues {args.start_idx}..{end-1} | {len(jobs)} images")
    logging.info(f"steps={args.num_inference_steps} cfg={args.guidance_scale} "
                 f"T5={args.max_sequence_length} {args.width}x{args.height}")
    if jobs:
        logging.info(f"Sample prompt turn 0 : {jobs[0]['prompt'][:120]}")
        if len(jobs) > 10:
            logging.info(f"Sample prompt turn 10: {jobs[10]['prompt'][:120]} ...")
    logging.info("=" * 64)

    pipe = init_pipe(args)

    t0 = time.time()
    made = skipped = 0

    for k, job in enumerate(jobs):
        if args.skip_existing and os.path.exists(job["path"]):
            skipped += 1
            continue

        if args.dry_run:
            logging.info(f"[DRY {k+1}/{len(jobs)}] {job['d']}_{job['r']}.jpg | {len(job['prompt'].split())} words")
            made += 1
            continue

        try:
            import torch
            gen = torch.Generator(device="cpu").manual_seed(
                args.seed + job["d"] * 1000 + job["r"] * 10)

            ts = time.time()
            with torch.inference_mode():
                img = pipe(
                    prompt=job["prompt"],
                    negative_prompt=args.negative_prompt,
                    height=args.height, width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    max_sequence_length=args.max_sequence_length,
                    generator=gen,
                ).images[0]

            img.save(job["path"], quality=95)
            made += 1

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            el = time.time() - t0
            avg = el / made
            left = len(jobs) - made - skipped
            logging.info(f"✓ [{k+1}/{len(jobs)}] {job['d']}_{job['r']} "
                         f"| {fmt(time.time()-ts)} | {avg:.1f}s/img | ETA {fmt(avg*left)}")
        except Exception as e:
            logging.error(f"❌ {job['d']}_{job['r']}: {e}")
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
