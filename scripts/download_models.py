#!/usr/bin/env python3
"""
Script to download Qwen3-TTS models from Hugging Face.
Usage: python download_models.py [--models MODEL1 MODEL2 ...] [--output-dir ./models]
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download


# Available models
DEFAULT_MODELS = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
]


def download_models(models, output_dir="./models"):
    """Download models from Hugging Face."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading models to {output_dir.resolve()}")
    print("-" * 60)

    failed = []
    for model_id in models:
        model_name = model_id.split("/")[-1]
        local_dir = output_dir / model_name

        print(f"\nDownloading {model_name}...")
        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                repo_type="model",
            )
            print(f"✓ Successfully downloaded {model_name}")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
            failed.append(model_name)

    print("\n" + "=" * 60)
    print(f"Download completed!")
    print(f"Models location: {output_dir.resolve()}")

    if failed:
        print(f"\nFailed downloads ({len(failed)}):")
        for model in failed:
            print(f"  - {model}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen3-TTS models from Hugging Face"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to download (default: all available models)",
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Output directory for models (default: ./models)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for model in DEFAULT_MODELS:
            print(f"  - {model}")
        return

    success = download_models(args.models, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
