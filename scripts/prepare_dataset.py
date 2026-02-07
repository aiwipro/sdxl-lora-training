#!/usr/bin/env python3
"""
Dataset Preparation Script for SDXL LoRA Training

This script helps prepare your dataset for training by:
1. Validating image formats
2. Creating metadata files
3. Optionally generating captions
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict
from PIL import Image


def validate_image(image_path: Path) -> bool:
    """Validate that an image file is valid and can be opened."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        print(f"Warning: Invalid image {image_path}: {e}")
        return False


def get_image_info(image_path: Path) -> Dict:
    """Get image information including dimensions."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            return {
                "width": width,
                "height": height,
                "format": img.format,
                "mode": img.mode
            }
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None


def create_metadata_jsonl(
    dataset_dir: Path,
    output_file: Path,
    caption_suffix: str = ".txt"
) -> None:
    """
    Create a metadata.jsonl file for the dataset.
    
    Each line contains a JSON object with:
    - file_name: path to image relative to dataset_dir
    - text: caption text (from .txt file or empty)
    """
    metadata_lines = []
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    # Find all images
    image_files = [
        f for f in dataset_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    print(f"Found {len(image_files)} images in {dataset_dir}")
    
    for image_file in sorted(image_files):
        if not validate_image(image_file):
            continue
        
        # Look for caption file
        caption_file = image_file.with_suffix(caption_suffix)
        caption_text = ""
        
        if caption_file.exists():
            try:
                with open(caption_file, "r", encoding="utf-8") as f:
                    caption_text = f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read caption file {caption_file}: {e}")
        
        # Create metadata entry
        metadata_entry = {
            "file_name": image_file.name,
            "text": caption_text
        }
        
        metadata_lines.append(json.dumps(metadata_entry))
    
    # Write metadata.jsonl
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))

    print(f"Created metadata file: {output_file}")
    print(f"Total entries: {len(metadata_lines)}")

    # Write metadata.json in kohya_ss format (keyed by absolute image path)
    kohya_metadata = {}
    for line in metadata_lines:
        entry = json.loads(line)
        image_key = str((dataset_dir / entry["file_name"]).resolve())
        kohya_metadata[image_key] = {"text": entry["text"]}

    kohya_file = output_file.with_suffix(".json")
    with open(kohya_file, "w", encoding="utf-8") as f:
        json.dump(kohya_metadata, f, indent=2, ensure_ascii=False)

    print(f"Created kohya_ss metadata: {kohya_file}")


def analyze_dataset(dataset_dir: Path) -> None:
    """Analyze dataset and print statistics."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [
        f for f in dataset_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    resolutions = []
    valid_images = 0
    
    for image_file in image_files:
        info = get_image_info(image_file)
        if info:
            valid_images += 1
            resolutions.append((info["width"], info["height"]))
    
    print(f"\nDataset Analysis:")
    print(f"Total images found: {len(image_files)}")
    print(f"Valid images: {valid_images}")
    
    if resolutions:
        widths = [r[0] for r in resolutions]
        heights = [r[1] for r in resolutions]
        print(f"Resolution range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
        print(f"Average resolution: {sum(widths)//len(widths)}x{sum(heights)//len(heights)}")
        
        # Check for SDXL compatibility (1024x1024 recommended)
        sdxl_compatible = sum(1 for w, h in resolutions if w >= 1024 and h >= 1024)
        print(f"SDXL-compatible (>=1024x1024): {sdxl_compatible}/{valid_images}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for SDXL LoRA training"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input_dir)"
    )
    parser.add_argument(
        "--create_metadata",
        action="store_true",
        help="Create metadata.jsonl file"
    )
    parser.add_argument(
        "--caption_suffix",
        type=str,
        default=".txt",
        help="Suffix for caption files (default: .txt)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze dataset and print statistics"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    if args.analyze:
        analyze_dataset(input_dir)
    
    if args.create_metadata:
        metadata_file = output_dir / "metadata.jsonl"
        create_metadata_jsonl(input_dir, metadata_file, args.caption_suffix)
        print(f"\nMetadata files created in: {output_dir}")
        print("\nNext steps:")
        print("1. Review and edit captions in metadata.jsonl, then re-run to regenerate metadata.json")
        print("2. Run training: bash scripts/train_sdxl_lora.sh")


if __name__ == "__main__":
    main()
