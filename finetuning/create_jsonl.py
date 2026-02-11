# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Script to create JSONL dataset from a directory of audio files with transcripts.
# Each audio file should have a corresponding .txt file with the same base name.
#
# Example directory structure:
#   data/
#     utt0001.wav
#     utt0001.txt
#     utt0002.flac
#     utt0002.txt
#
# Output JSONL format:
#   {"audio": "./data/utt0001.wav", "text": "transcript content"}
#   {"audio": "./data/utt0002.flac", "text": "transcript content"}

import argparse
import json
import os
from pathlib import Path


# Supported audio extensions
AUDIO_EXTENSIONS = {'.flac'}


def find_audio_transcript_pairs(input_dir):
    """Find all audio files with corresponding transcript files (recursive)."""
    pairs = []
    missing_transcripts = []
    
    for root, dirs, files in os.walk(input_dir):
        root_path = Path(root)
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                audio_path = root_path / filename
                txt_path = audio_path.with_suffix('.txt')
                
                if txt_path.exists():
                    pairs.append((audio_path, txt_path))
                else:
                    missing_transcripts.append(audio_path)
    
    pairs.sort(key=lambda x: x[0])
    return pairs, missing_transcripts


def read_transcript(txt_path):
    """Read transcript from a text file."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        # Read and strip whitespace, join multiple lines with space
        lines = [line.strip() for line in f.readlines()]
        return ' '.join(line for line in lines if line)


def main():
    parser = argparse.ArgumentParser(
        description='Create JSONL dataset from audio files and transcripts.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python create_jsonl.py --input_dir ./data --output_jsonl ./train.jsonl
        """
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing audio files and transcripts')
    parser.add_argument('--output_jsonl', type=str, required=True,
                        help='Output JSONL file path')
    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1

    # Find audio-transcript pairs
    print(f"Scanning recursively in: {args.input_dir}")
    pairs, missing = find_audio_transcript_pairs(args.input_dir)

    if missing:
        print(f"\nWarning: {len(missing)} audio files have no matching .txt transcript:")
        for path in missing[:10]:  # Show first 10
            print(f"  - {path}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        print()

    if not pairs:
        print("Error: No audio-transcript pairs found!")
        return 1

    print(f"Found {len(pairs)} audio-transcript pairs")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write JSONL
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for audio_path, txt_path in pairs:
            transcript = read_transcript(txt_path)
            audio_str = str(audio_path.resolve())
            
            entry = {
                'audio': audio_str,
                'text': transcript
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Created: {args.output_jsonl}")
    print(f"Total entries: {len(pairs)}")
    return 0


if __name__ == '__main__':
    exit(main())
