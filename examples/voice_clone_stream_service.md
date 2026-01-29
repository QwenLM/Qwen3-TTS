# Voice Clone Streaming Service (SSE)

This service streams voice-clone audio chunks as Server-Sent Events (SSE).

Constraints
- Single-sample only (one `text`, one `voice_id` per request).
- Base 12Hz models only (e.g. `Qwen/Qwen3-TTS-12Hz-1.7B-Base`).
- SSE payload is base64-encoded `int16` PCM audio.

Start
```bash
./Qwen3-TTS/examples/run_voice_clone_stream_service.sh
```

Test page
```text
http://localhost:8002/
```

Request example
```bash
curl -N -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "YOUR_VOICE_ID",
    "text": "Hello, this is streaming TTS.",
    "chunk_size": 8,
    "left_context_size": 25
  }' \
  http://localhost:8002/voice/stream
```

Model switching
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"model_id":"Qwen/Qwen3-TTS-12Hz-0.6B-Base","safe_mode":true}' \
  http://localhost:8002/model/load
```

Available models from the voice store
```bash
curl http://localhost:8002/models
```

Notes
- Lower `chunk_size` yields faster first audio but more overhead.
- `left_context_size` improves continuity at chunk boundaries.
- FlashAttention is enabled by default; use `--no-flash-attn` to disable.
- TF32 matmul/cudnn is enabled by default on CUDA; use `--no-tf32` to disable.
- Model switching returns HTTP 409 if a stream is active.
- Safe mode disables sub-talker sampling for stability.
