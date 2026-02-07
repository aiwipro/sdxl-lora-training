#!/usr/bin/env python3
"""Convert metadata.jsonl to metadata.json (array format)"""

import json
import sys
from pathlib import Path

def convert_jsonl_to_json(jsonl_path: Path, json_path: Path = None):
    """Convert JSONL file to JSON array."""
    if json_path is None:
        json_path = jsonl_path.with_suffix('.json')
    
    entries = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(entries)} entries from {jsonl_path} to {json_path}")
    return json_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: convert_jsonl_to_json.py <metadata.jsonl> [metadata.json]")
        sys.exit(1)
    
    jsonl_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    convert_jsonl_to_json(jsonl_path, json_path)
