#!/usr/bin/env python3
"""
Integration test script for Qwen3-TTS REST API.

This script tests the API endpoints from the host machine.
Run this after starting the container with: docker compose up

Usage:
    python scripts/test_api.py [--host HOST] [--port PORT]
"""

import argparse
import sys
import wave
from io import BytesIO
from pathlib import Path

import requests


def test_health(base_url: str) -> bool:
    """Test the health endpoint."""
    print("Testing GET /health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code != 200:
            print(f"  FAIL: Status code {response.status_code}")
            return False
        data = response.json()
        if "status" not in data or "model_loaded" not in data:
            print(f"  FAIL: Missing fields in response: {data}")
            return False
        print(f"  OK: status={data['status']}, model_loaded={data['model_loaded']}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_speakers(base_url: str) -> bool:
    """Test the speakers endpoint."""
    print("Testing GET /v1/tts/speakers...")
    try:
        response = requests.get(f"{base_url}/v1/tts/speakers", timeout=10)
        if response.status_code != 200:
            print(f"  FAIL: Status code {response.status_code}")
            return False
        data = response.json()
        if "speakers" not in data:
            print(f"  FAIL: Missing 'speakers' field: {data}")
            return False
        print(f"  OK: speakers={data['speakers']}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_languages(base_url: str) -> bool:
    """Test the languages endpoint."""
    print("Testing GET /v1/tts/languages...")
    try:
        response = requests.get(f"{base_url}/v1/tts/languages", timeout=10)
        if response.status_code != 200:
            print(f"  FAIL: Status code {response.status_code}")
            return False
        data = response.json()
        if "languages" not in data:
            print(f"  FAIL: Missing 'languages' field: {data}")
            return False
        print(f"  OK: languages={data['languages']}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def test_synthesize(base_url: str, output_path: Path) -> bool:
    """Test the synthesize endpoint."""
    print("Testing POST /v1/tts/synthesize...")
    try:
        payload = {
            "text": "Hello, this is a test of the Qwen3 text to speech API.",
            "speaker": "Vivian",
            "language": "English",
        }
        response = requests.post(
            f"{base_url}/v1/tts/synthesize",
            json=payload,
            timeout=120,  # TTS can take a while
        )
        if response.status_code != 200:
            print(f"  FAIL: Status code {response.status_code}")
            print(f"  Response: {response.text}")
            return False

        content_type = response.headers.get("content-type", "")
        if "audio/wav" not in content_type:
            print(f"  FAIL: Wrong content-type: {content_type}")
            return False

        # Validate WAV data
        wav_data = response.content
        if len(wav_data) < 44:  # WAV header is 44 bytes
            print(f"  FAIL: WAV data too small: {len(wav_data)} bytes")
            return False

        # Check WAV header
        if wav_data[:4] != b"RIFF":
            print("  FAIL: Invalid WAV header (no RIFF)")
            return False

        # Try to parse the WAV file
        try:
            wav_buffer = BytesIO(wav_data)
            with wave.open(wav_buffer, "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                nframes = wav_file.getnframes()
                duration = nframes / framerate
                print(f"  WAV: {channels}ch, {sample_width * 8}bit, {framerate}Hz, {duration:.2f}s")
        except Exception as e:
            print(f"  FAIL: Could not parse WAV: {e}")
            return False

        # Save the output file
        output_path.write_bytes(wav_data)
        print(f"  OK: Saved audio to {output_path}")
        return True

    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-TTS REST API")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--output", default="test_output.wav", help="Output WAV file")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    output_path = Path(args.output)

    print(f"Testing API at {base_url}\n")

    results = {
        "health": test_health(base_url),
        "speakers": test_speakers(base_url),
        "languages": test_languages(base_url),
        "synthesize": test_synthesize(base_url, output_path),
    }

    print("\n--- Summary ---")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
