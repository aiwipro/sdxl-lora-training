#!/usr/bin/env python3
"""Convert metadata.jsonl to kohya_ss format (dict with image paths as keys)"""

import json
import sys
from pathlib import Path

def convert_to_kohya_format(jsonl_path: Path, json_path: Path, dataset_dir: Path):
    """Convert JSONL to kohya_ss metadata format."""
    entries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    # Convert to dict format: {image_path: {text: caption}}
    # kohya_ss expects paths relative to train_data_dir or absolute paths
    metadata_dict = {}
    for entry in entries:
        file_name = entry['file_name']
        # Use absolute path
        image_key = str((dataset_dir / file_name).resolve())
        metadata_dict[image_key] = {
            'text': entry.get('text', '')
        }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(entries)} entries to kohya_ss format")
    print(f"Output: {json_path}")
    return json_path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: fix_metadata_format.py <metadata.jsonl> <dataset_dir> [metadata.json]")
        sys.exit(1)
    
    jsonl_path = Path(sys.argv[1])
    dataset_dir = Path(sys.argv[2])
    json_path = Path(sys.argv[3]) if len(sys.argv) > 3 else jsonl_path.with_suffix('.json')
    
    convert_to_kohya_format(jsonl_path, json_path, dataset_dir)
