#!/bin/bash
# REQ-005 End-to-End Verification Script
# This script verifies that the Docker container runs inference correctly.
#
# Prerequisites:
# - Docker daemon must be running
# - NVIDIA Container Toolkit must be installed
# - GPU must be available
#
# Usage:
#   ./scripts/verify_req005.sh
#
# The script will:
# 1. Build and start the Docker container
# 2. Wait for the server to become healthy
# 3. Run the basic_inference example inside the container
# 4. Hit /v1/tts/synthesize from the host
# 5. Validate the returned audio
# 6. Clean up

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${GREEN}=== STEP: $1 ===${NC}"
}

cleanup() {
    log_info "Cleaning up..."
    docker compose down --timeout 10 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

# Step 1: Check prerequisites
log_step "Checking prerequisites"

if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    exit 1
fi

if ! docker info &> /dev/null; then
    log_error "Docker daemon is not running"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not available"
    exit 1
fi

# Check for NVIDIA Container Toolkit
if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    log_error "NVIDIA Container Toolkit is not configured or GPU is not available"
    exit 1
fi

log_info "All prerequisites met"

# Step 2: Build the Docker image
log_step "Building Docker image"
docker compose build

# Step 3: Start the container
log_step "Starting container"
docker compose up -d

# Step 4: Wait for server to become healthy
log_step "Waiting for server to become healthy"
MAX_WAIT=300  # 5 minutes max wait for model loading
WAIT_INTERVAL=5
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(docker compose ps --format json 2>/dev/null | jq -r '.Health // "starting"' 2>/dev/null || echo "starting")

    # Also try curl directly
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        log_info "Server is healthy!"
        break
    fi

    log_info "Waiting for server... ($ELAPSED/$MAX_WAIT seconds)"
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    log_error "Server did not become healthy within $MAX_WAIT seconds"
    docker compose logs
    exit 1
fi

# Step 5: Run basic_inference example inside container
log_step "Running basic_inference example inside container"

# Create a test script to run inside the container
docker compose exec -T qwen3-tts-server python -c "
import torch
from qwen_tts import Qwen3TTSModel

print('Loading model...')
tts = Qwen3TTSModel.from_pretrained(
    'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
    device_map='cuda:0',
    dtype=torch.bfloat16,
    attn_implementation='flashinfer',
)
print('Model loaded successfully!')

print('Running inference...')
wavs, sr = tts.generate_custom_voice(
    text='Hello, this is a test.',
    language='English',
    speaker='Vivian',
)
print(f'Generated audio: {len(wavs[0])} samples at {sr}Hz')
print('Basic inference test PASSED!')
"

if [ $? -ne 0 ]; then
    log_error "Basic inference test failed"
    exit 1
fi
log_info "Basic inference test passed"

# Step 6: Test /v1/tts/synthesize from host
log_step "Testing /v1/tts/synthesize from host"

OUTPUT_FILE="test_output_req005.wav"
HTTP_CODE=$(curl -s -o "$OUTPUT_FILE" -w "%{http_code}" \
    -X POST http://localhost:8000/v1/tts/synthesize \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, this is a test of the text to speech API.", "speaker": "Vivian", "language": "English"}')

if [ "$HTTP_CODE" != "200" ]; then
    log_error "API returned HTTP $HTTP_CODE"
    cat "$OUTPUT_FILE"
    exit 1
fi

# Validate WAV file
if [ ! -f "$OUTPUT_FILE" ]; then
    log_error "Output file not created"
    exit 1
fi

FILE_SIZE=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || stat -f%z "$OUTPUT_FILE")
if [ "$FILE_SIZE" -lt 44 ]; then
    log_error "Output file too small: $FILE_SIZE bytes"
    exit 1
fi

# Check WAV header
HEADER=$(xxd -l 4 "$OUTPUT_FILE" | awk '{print $2$3}')
if [ "$HEADER" != "5249" ]; then  # "RI" in hex (RIFF)
    log_error "Invalid WAV header"
    exit 1
fi

log_info "API test passed - generated WAV file: $OUTPUT_FILE ($FILE_SIZE bytes)"

# Step 7: Run full test suite
log_step "Running full API test suite"
python scripts/test_api.py --output test_output_full.wav

# Summary
log_step "REQ-005 VERIFICATION COMPLETE"
echo ""
echo "All verification steps passed:"
echo "  [PASS] Docker container builds and starts"
echo "  [PASS] Server becomes healthy"
echo "  [PASS] Basic inference runs inside container"
echo "  [PASS] /v1/tts/synthesize accessible from host"
echo "  [PASS] Returned audio is valid WAV format"
echo ""
echo "Output files:"
echo "  - test_output_req005.wav"
echo "  - test_output_full.wav"
echo ""
echo "To play the audio:"
echo "  aplay test_output_req005.wav"
echo "  # or"
echo "  ffplay test_output_req005.wav"
echo ""

log_info "REQ-005 verification PASSED!"
