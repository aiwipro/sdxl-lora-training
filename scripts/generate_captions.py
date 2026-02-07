#!/usr/bin/env python3
"""
Automatic Caption Generation Script for SDXL LoRA Training

Uses BLIP (Bootstrapping Language-Image Pre-training) to automatically
generate captions for training images.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


def load_blip_model(device="cpu"):
    """Load BLIP model for image captioning."""
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    model.eval()
    print("BLIP model loaded successfully!")
    return processor, model


def generate_caption(image_path: Path, processor, model, device="cpu") -> str:
    """Generate a caption for a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Process image
        inputs = processor(image, return_tensors="pt").to(device)
        
        # Generate caption
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50, num_beams=3)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""


def generate_captions_for_dataset(
    dataset_dir: Path,
    output_format: str = "txt",
    device: str = "auto"
):
    """
    Generate captions for all images in a dataset.
    
    Args:
        dataset_dir: Directory containing images
        output_format: "txt" for .txt files, "jsonl" for metadata.jsonl
        device: "auto", "cpu", "mps" (for Mac), or "cuda"
    """
    # Detect device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS (Metal) backend on Mac")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA backend")
        else:
            device = "cpu"
            print("Using CPU backend")
    
    # Load model
    processor, model = load_blip_model(device)
    
    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [
        f for f in dataset_dir.iterdir()
        if f.suffix.lower() in image_extensions and f.is_file()
    ]
    
    print(f"\nFound {len(image_files)} images to caption")
    print("Generating captions...\n")
    
    captions_generated = 0
    
    if output_format == "txt":
        # Generate .txt files
        for i, image_file in enumerate(sorted(image_files), 1):
            print(f"[{i}/{len(image_files)}] Processing {image_file.name}...")
            caption = generate_caption(image_file, processor, model, device)
            
            if caption:
                txt_file = image_file.with_suffix(".txt")
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(caption)
                print(f"  ✓ Caption: {caption}")
                captions_generated += 1
            else:
                print(f"  ✗ Failed to generate caption")
    
    elif output_format == "jsonl":
        # Update metadata.jsonl
        metadata_file = dataset_dir / "metadata.jsonl"
        
        # Load existing metadata if it exists
        metadata_entries = []
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        metadata_entries.append(json.loads(line))
        
        # Create a mapping of file_name to entry
        metadata_dict = {entry["file_name"]: entry for entry in metadata_entries}
        
        # Generate captions and update metadata
        for i, image_file in enumerate(sorted(image_files), 1):
            print(f"[{i}/{len(image_files)}] Processing {image_file.name}...")
            caption = generate_caption(image_file, processor, model, device)
            
            if caption:
                file_name = image_file.name
                if file_name in metadata_dict:
                    metadata_dict[file_name]["text"] = caption
                else:
                    metadata_dict[file_name] = {
                        "file_name": file_name,
                        "text": caption
                    }
                print(f"  ✓ Caption: {caption}")
                captions_generated += 1
            else:
                print(f"  ✗ Failed to generate caption")
        
        # Write updated metadata
        with open(metadata_file, "w", encoding="utf-8") as f:
            for entry in sorted(metadata_dict.values(), key=lambda x: x["file_name"]):
                f.write(json.dumps(entry) + "\n")
        
        print(f"\nUpdated metadata.jsonl with {captions_generated} captions")
    
    print(f"\n✓ Successfully generated {captions_generated}/{len(image_files)} captions!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate automatic captions for training images using BLIP"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["txt", "jsonl"],
        default="jsonl",
        help="Output format: 'txt' for .txt files, 'jsonl' for metadata.jsonl"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Device to use (auto detects MPS/CUDA/CPU)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    print("=" * 60)
    print("Automatic Caption Generation with BLIP")
    print("=" * 60)
    
    generate_captions_for_dataset(
        input_dir,
        output_format=args.output_format,
        device=args.device
    )
    
    print("\n" + "=" * 60)
    print("Caption generation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the generated captions")
    print("2. Edit captions if needed for better training results")
    print("3. Run training: bash scripts/train_sdxl_lora.sh")


if __name__ == "__main__":
    main()
