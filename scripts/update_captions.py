#!/usr/bin/env python3
"""
Update captions in metadata.jsonl to use a consistent trigger word.
"""

import argparse
import json
from pathlib import Path


def update_captions(metadata_file: Path, old_pattern: str, new_pattern: str):
    """Update captions by replacing old pattern with new pattern."""
    entries = []
    
    # Read existing metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    # Update captions
    updated_count = 0
    for entry in entries:
        old_text = entry["text"]
        if old_pattern.lower() in old_text.lower():
            entry["text"] = old_text.replace(old_pattern, new_pattern).replace(
                old_pattern.capitalize(), new_pattern.capitalize()
            ).replace(
                old_pattern.upper(), new_pattern.upper()
            )
            updated_count += 1
    
    # Write updated metadata
    with open(metadata_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Updated {updated_count}/{len(entries)} captions")
    print(f"Changed '{old_pattern}' to '{new_pattern}'")


def main():
    parser = argparse.ArgumentParser(
        description="Update captions in metadata.jsonl with a trigger word"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Path to metadata.jsonl file"
    )
    parser.add_argument(
        "--old_pattern",
        type=str,
        default="a woman",
        help="Pattern to replace (default: 'a woman')"
    )
    parser.add_argument(
        "--new_pattern",
        type=str,
        default="my_subject",
        help="New pattern/trigger word (default: 'my_subject')"
    )
    
    args = parser.parse_args()
    
    metadata_file = Path(args.metadata_file)
    if not metadata_file.exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        return
    
    update_captions(metadata_file, args.old_pattern, args.new_pattern)
    
    print("\n✓ Captions updated successfully!")
    print("\nExample updated captions:")
    with open(metadata_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            entry = json.loads(line)
            print(f"  {entry['file_name']}: {entry['text']}")


if __name__ == "__main__":
    main()
