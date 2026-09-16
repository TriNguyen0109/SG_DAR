#!/usr/bin/env python3
"""
Script to format VisDial JSON queries file with per-turn sketch file paths.

Transforms VisDial dialog items from:
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
"""

import os
import sys
import json
import shutil
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def prepare_json_dataset(input_json, output_json, sketches_dir="sketches", max_items=None):
    logging.info(f"Loading input JSON from {input_json}...")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    logging.info(f"Total items in input JSON: {len(data)}")
    if max_items:
        data = data[:max_items]
        logging.info(f"Processing subset of {max_items} items.")

    formatted_data = []
    total_turns = 0
    
    for item in data:
        img_path = item.get("img", "")
        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]
        
        raw_dialog = item.get("dialog", [])
        new_dialog = []
        
        for turn_idx, turn_item in enumerate(raw_dialog):
            total_turns += 1
            if isinstance(turn_item, dict):
                text_prompt = turn_item.get("text", "")
            else:
                text_prompt = str(turn_item)
                
            sketch_filename = f"{stem}_{turn_idx}.png"
            rel_sketch_path = os.path.join(sketches_dir, sketch_filename).replace("\\", "/")
            
            new_dialog.append({
                "text": text_prompt,
                "sketch": rel_sketch_path
            })
            
        formatted_data.append({
            "img": img_path,
            "dialog": new_dialog
        })
        
    logging.info(f"Writing formatted dataset to {output_json}...")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)

    logging.info("=" * 50)
    logging.info("JSON Dataset Formatting Complete!")
    logging.info(f"Total dialog items: {len(formatted_data)} | Total turns: {total_turns}")
    logging.info(f"Output JSON saved to: {os.path.abspath(output_json)}")
    logging.info("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Format VisDial JSON queries file with text and sketch fields.")
    parser.add_argument("--input_json", type=str, default="VisDial_v1_0_queries_val.json", help="Input VisDial JSON query file.")
    parser.add_argument("--output_json", type=str, default="VisDial_v1_0_queries_val_sketch.json", help="Output formatted JSON query file.")
    parser.add_argument("--sketches_dir", type=str, default="sketches", help="Directory path prefix for sketch PNG files.")
    parser.add_argument("--max_items", type=int, default=None, help="Optional max items to process for testing.")
    parser.add_argument("--overwrite_original", action="store_true", help="Also overwrite input_json with updated dataset.")
    
    args = parser.parse_args()
    
    prepare_json_dataset(
        input_json=args.input_json,
        output_json=args.output_json,
        sketches_dir=args.sketches_dir,
        max_items=args.max_items
    )
    
    if args.overwrite_original:
        logging.info(f"Overwriting original {args.input_json} with formatted data...")
        shutil.copy2(args.output_json, args.input_json)
        logging.info("Original file updated.")

if __name__ == "__main__":
    main()
