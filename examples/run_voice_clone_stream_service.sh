#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_voice_clone_stream_service.sh [options] [-- extra-args]

Options:
  -g, --gpu ID           GPU id to use (default: 4)
  -p, --port PORT        Port to bind (default: 8002)
  -m, --model MODEL      Model id or path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)
  -s, --voice-store DIR  Voice store directory (default: <repo>/voice_store)
  -d, --device DEVICE    Device for device_map (default: cuda:0)
      --ip IP            Bind IP (default: 0.0.0.0)
  -h, --help             Show this help

Env overrides: GPU_ID, PORT, MODEL, VOICE_STORE, DEVICE, IP, PYTHON_BIN
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-5}"
PORT="${PORT:-8002}"
MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
VOICE_STORE="${VOICE_STORE:-${PROJECT_ROOT}/voice_store}"
DEVICE="${DEVICE:-cuda:0}"
IP="${IP:-0.0.0.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpu)
      GPU_ID="$2"
      shift 2
      ;;
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -m|--model)
      MODEL="$2"
      shift 2
      ;;
    -s|--voice-store)
      VOICE_STORE="$2"
      shift 2
      ;;
    -d|--device)
      DEVICE="$2"
      shift 2
      ;;
    --ip)
      IP="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p "$VOICE_STORE"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
  "$SCRIPT_DIR/voice_clone_stream_service.py" \
  --device "$DEVICE" \
  --ip "$IP" \
  --port "$PORT" \
  --model "$MODEL" \
  --voice-store "$VOICE_STORE" \
  "${EXTRA_ARGS[@]}"
