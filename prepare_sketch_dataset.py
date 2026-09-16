#!/usr/bin/env python3
"""
Script to prepare formatted sketch dataset and update VisDial JSON queries file.

Transform VisDial dialog items from:
"dialog": [
    "text 1",
    "text 2"
]
To:
"dialog": [
    {
        "text": "text 1",
        "sketch": "sketches/000000185565_0.png"
    },
    {
        "text": "text 2",
        "sketch": "sketches/000000185565_1.png"
    }
]

And populate `sketches/` directory with formatted sketch PNG images.
"""

import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def create_dry_sketch_image(save_path, stem, turn_idx):
    """Creates a dry/placeholder sketch PNG image when no source sketch exists or for dry format."""
    width, height = 512, 512
    # White background (typical for sketch/lineart)
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw simple sketch-like outlines / shapes
    draw.rectangle([20, 20, width - 20, height - 20], outline=(0, 0, 0), width=3)
    draw.line([20, 20, width - 20, height - 20], fill=(128, 128, 128), width=2)
    draw.line([20, height - 20, width - 20, 20], fill=(128, 128, 128), width=2)
    draw.ellipse([width//4, height//4, 3*width//4, 3*height//4], outline=(0, 0, 0), width=4)
    
    # Text label
    text = f"SKETCH: {stem}_{turn_idx}"
    draw.text((30, 30), text, fill=(0, 0, 0))
    
    img.save(save_path, quality=95)

def prepare_dataset(input_json, output_json, sketches_src_dir, sketches_dst_dir, max_items=None):
    os.makedirs(sketches_dst_dir, exist_ok=True)
    
    logging.info(f"Loading input JSON from {input_json}...")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    logging.info(f"Total items in input JSON: {len(data)}")
    
    if max_items:
        data = data[:max_items]
        logging.info(f"Processing subset of {max_items} items.")

    formatted_data = []
    total_turns = 0
    sketches_copied = 0
    sketches_created_dry = 0
    
    for dialog_idx, item in enumerate(data):
        img_path = item.get("img", "")
        # Extract stem e.g., '000000185565' from 'unlabeled2017/000000185565.jpg'
        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]
        
        # Source sketch PNG in output_sketches_pipeline directory
        src_sketch_png = os.path.join(sketches_src_dir, f"{stem}_sketch.png")
        src_exists = os.path.exists(src_sketch_png)
        
        raw_dialog = item.get("dialog", [])
        new_dialog = []
        
        for turn_idx, turn_item in enumerate(raw_dialog):
            total_turns += 1
            # Extract prompt text (support string or pre-existing dict)
            if isinstance(turn_item, dict):
                text_prompt = turn_item.get("text", "")
            else:
                text_prompt = str(turn_item)
                
            sketch_filename = f"{stem}_{turn_idx}.png"
            rel_sketch_path = os.path.join(sketches_dst_dir, sketch_filename)
            abs_sketch_path = os.path.abspath(rel_sketch_path)
            
            # Populate sketch file in sketches_dst_dir
            if not os.path.exists(abs_sketch_path):
                if src_exists:
                    shutil.copy2(src_sketch_png, abs_sketch_path)
                    sketches_copied += 1
                else:
                    create_dry_sketch_image(abs_sketch_path, stem, turn_idx)
                    sketches_created_dry += 1
            
            new_dialog.append({
                "text": text_prompt,
                "sketch": f"{sketches_dst_dir}/{sketch_filename}"
            })
            
        formatted_item = {
            "img": img_path,
            "dialog": new_dialog
        }
        formatted_data.append(formatted_item)

    logging.info(f"Writing formatted dataset to {output_json}...")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)

    logging.info("=" * 50)
    logging.info("Dataset Preparation Complete!")
    logging.info(f"Total dialog items processed: {len(formatted_data)}")
    logging.info(f"Total dialog turns processed: {total_turns}")
    logging.info(f"Sketches copied from source: {sketches_copied}")
    logging.info(f"Dry sketch images generated: {sketches_created_dry}")
    logging.info(f"Formatted JSON saved to: {os.path.abspath(output_json)}")
    logging.info(f"Sketches directory: {os.path.abspath(sketches_dst_dir)}")
    logging.info("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Prepare sketch dataset and update VisDial JSON query format.")
    parser.add_argument("--input_json", type=str, default="VisDial_v1_0_queries_val.json", help="Input VisDial JSON query file.")
    parser.add_argument("--output_json", type=str, default="VisDial_v1_0_queries_val_sketch.json", help="Output formatted JSON query file.")
    parser.add_argument("--sketches_src_dir", type=str, default="output_sketches_pipeline", help="Source directory containing sketches.")
    parser.add_argument("--sketches_dst_dir", type=str, default="sketches", help="Destination directory for formatted per-turn sketches.")
    parser.add_argument("--max_items", type=int, default=None, help="Optional max items to process for testing.")
    parser.add_argument("--overwrite_original", action="store_true", help="Also overwrite input_json with updated dataset.")
    
    args = parser.parse_args()
    
    prepare_dataset(
        input_json=args.input_json,
        output_json=args.output_json,
        sketches_src_dir=args.sketches_src_dir,
        sketches_dst_dir=args.sketches_dst_dir,
        max_items=args.max_items
    )
    
    if args.overwrite_original:
        logging.info(f"Overwriting original {args.input_json} with formatted data...")
        shutil.copy2(args.output_json, args.input_json)
        logging.info("Original file updated.")

if __name__ == "__main__":
    main()
