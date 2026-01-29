# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
FastAPI service for streaming voice-clone audio chunks from stored prompts.
"""

import argparse
import base64
import gc
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem


MODEL_LOCK = threading.Lock()
CURRENT_TTS: Optional[Qwen3TTSModel] = None
CURRENT_MODEL_ID: Optional[str] = None
CURRENT_CONFIG: Dict[str, Any] = {
    "model": None,
    "device": None,
    "dtype": None,
    "flash_attn": None,
    "tf32": None,
    "safe_mode": False,
}

VOICE_INDEX_LOCK = threading.Lock()
VOICE_INDEX: Dict[str, Dict[str, Any]] = {}

PROMPT_CACHE_LOCK = threading.Lock()
PROMPT_CACHE: Dict[str, Tuple[List[VoiceClonePromptItem], Dict[str, Any]]] = {}

VOICE_STORE: Optional[Path] = None
PROMPT_DIR: Optional[Path] = None
INDEX_PATH: Optional[Path] = None
BASE_DIR = Path(__file__).resolve().parent
TEST_PAGE_PATH = BASE_DIR / "voice_clone_stream_test.html"


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _apply_tf32(enabled: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _set_current_config(
    model_id: str,
    device: str,
    dtype: str,
    flash_attn: bool,
    tf32: bool,
    safe_mode: bool,
) -> None:
    global CURRENT_CONFIG
    CURRENT_CONFIG = {
        "model": model_id,
        "device": device,
        "dtype": dtype,
        "flash_attn": bool(flash_attn),
        "tf32": bool(tf32),
        "safe_mode": bool(safe_mode),
    }


def _clear_model() -> None:
    global CURRENT_TTS, CURRENT_MODEL_ID
    if CURRENT_TTS is None:
        return
    old = CURRENT_TTS
    CURRENT_TTS = None
    CURRENT_MODEL_ID = None
    del old
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def _init_voice_store(path: Path) -> None:
    global VOICE_STORE, PROMPT_DIR, INDEX_PATH, VOICE_INDEX
    VOICE_STORE = path
    PROMPT_DIR = VOICE_STORE / "prompts"
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH = VOICE_STORE / "voices.json"
    VOICE_INDEX = _load_voice_index(INDEX_PATH)


def _load_voice_index(index_path: Path) -> Dict[str, Dict[str, Any]]:
    if not index_path.exists():
        return {}
    try:
        payload = json.loads(index_path.read_text())
    except Exception:
        return {}
    voices = payload.get("voices", {})
    return voices if isinstance(voices, dict) else {}


def _torch_load_safe(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _deserialize_prompt_items(payload: Dict[str, Any]) -> Tuple[List[VoiceClonePromptItem], Dict[str, Any]]:
    items_raw = payload.get("items", None)
    if not isinstance(items_raw, list) or len(items_raw) == 0:
        raise ValueError("Empty or invalid prompt items.")
    items: List[VoiceClonePromptItem] = []
    for d in items_raw:
        if not isinstance(d, dict):
            raise ValueError("Invalid prompt item format.")
        ref_code = d.get("ref_code", None)
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        ref_spk = d.get("ref_spk_embedding", None)
        if ref_spk is None:
            raise ValueError("Missing ref_spk_embedding in prompt item.")
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                ref_text=d.get("ref_text", None),
            )
        )
    meta = payload.get("meta", {})
    return items, meta if isinstance(meta, dict) else {}


def _load_prompt_items(voice_id: str) -> Tuple[List[VoiceClonePromptItem], Dict[str, Any]]:
    with PROMPT_CACHE_LOCK:
        cached = PROMPT_CACHE.get(voice_id)
    if cached is not None:
        return cached

    with VOICE_INDEX_LOCK:
        meta = VOICE_INDEX.get(voice_id)
    if not meta:
        raise ValueError("Voice ID not found.")
    rel_path = meta.get("path")
    if not rel_path:
        raise ValueError("Voice path missing.")

    payload = _torch_load_safe(VOICE_STORE / rel_path)
    items, prompt_meta = _deserialize_prompt_items(payload)

    with PROMPT_CACHE_LOCK:
        PROMPT_CACHE[voice_id] = (items, prompt_meta)
    return items, prompt_meta


def _validate_prompt_items(items: List[VoiceClonePromptItem], tts: Qwen3TTSModel) -> None:
    vocab_size = int(tts.model.talker.config.vocab_size)
    num_groups = int(tts.model.talker.config.num_code_groups)
    for idx, item in enumerate(items):
        ref_code = item.ref_code
        if ref_code is None:
            continue
        if not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        if ref_code.dim() != 2 or ref_code.shape[1] != num_groups:
            raise ValueError(
                f"ref_code shape mismatch for prompt[{idx}]: "
                f"expected (*, {num_groups}), got {tuple(ref_code.shape)}."
            )
        if ref_code.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
            ref_code = ref_code.long()
        if ref_code.numel() > 0:
            min_id = int(ref_code.min().item())
            max_id = int(ref_code.max().item())
            if min_id < 0 or max_id >= vocab_size:
                raise ValueError(
                    f"ref_code out of range for model vocab_size={vocab_size}: min={min_id}, max={max_id}. "
                    "Recreate the voice prompt with the same model."
                )
        item.ref_code = ref_code


def _load_tts(model_id: str, device: str, dtype: torch.dtype, flash_attn: bool) -> Qwen3TTSModel:
    global CURRENT_TTS, CURRENT_MODEL_ID
    attn_impl = "flash_attention_2" if flash_attn else "eager"
    tts = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    if getattr(tts.model, "tts_model_type", None) != "base":
        raise ValueError("Only Base models support voice cloning.")
    CURRENT_TTS = tts
    CURRENT_MODEL_ID = model_id
    return tts


class StreamRequest(BaseModel):
    voice_id: str
    text: str
    language: str = "Auto"
    chunk_size: int = 8
    left_context_size: int = 25
    do_sample: Optional[bool] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    subtalker_dosample: Optional[bool] = None
    subtalker_top_k: Optional[int] = None
    subtalker_top_p: Optional[float] = None
    subtalker_temperature: Optional[float] = None
    max_new_tokens: Optional[int] = None


class ModelLoadRequest(BaseModel):
    model_id: str
    device: Optional[str] = None
    dtype: Optional[str] = None
    flash_attn: Optional[bool] = None
    tf32: Optional[bool] = None
    clear_cache: bool = True
    safe_mode: Optional[bool] = None


def build_app(args: argparse.Namespace) -> FastAPI:
    _init_voice_store(Path(args.voice_store))
    _apply_tf32(bool(args.tf32))
    with MODEL_LOCK:
        _load_tts(args.model, args.device, _dtype_from_str(args.dtype), args.flash_attn)
        _set_current_config(args.model, args.device, args.dtype, args.flash_attn, args.tf32, False)

    app = FastAPI()

    @app.get("/")
    def index():
        if not TEST_PAGE_PATH.exists():
            raise HTTPException(status_code=404, detail="Test page not found.")
        return FileResponse(TEST_PAGE_PATH, media_type="text/html")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": CURRENT_MODEL_ID,
            "device": CURRENT_CONFIG.get("device"),
            "dtype": CURRENT_CONFIG.get("dtype"),
            "flash_attn": CURRENT_CONFIG.get("flash_attn"),
            "tf32": CURRENT_CONFIG.get("tf32"),
            "safe_mode": CURRENT_CONFIG.get("safe_mode"),
        }

    @app.get("/models")
    def list_models():
        with VOICE_INDEX_LOCK:
            voices = list(VOICE_INDEX.values())
        models = sorted({meta.get("model_id") for meta in voices if meta.get("model_id")})
        return {"models": models}

    @app.get("/voices")
    def list_voices():
        with VOICE_INDEX_LOCK:
            voices = dict(VOICE_INDEX)
        return {"voices": voices}

    @app.post("/voice/stream")
    def voice_stream(payload: StreamRequest):
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Target text is required.")
        try:
            items, prompt_meta = _load_prompt_items(payload.voice_id)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

        prompt_model = prompt_meta.get("model_id")
        if prompt_model and prompt_model != CURRENT_MODEL_ID:
            raise HTTPException(status_code=400, detail=f"Voice created with {prompt_model}. Switch model to match.")
        tts = CURRENT_TTS
        if tts is None:
            raise HTTPException(status_code=500, detail="Model not loaded.")
        try:
            _validate_prompt_items(items, tts)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        gen_kwargs = {
            "do_sample": payload.do_sample,
            "top_k": payload.top_k,
            "top_p": payload.top_p,
            "temperature": payload.temperature,
            "repetition_penalty": payload.repetition_penalty,
            "subtalker_dosample": payload.subtalker_dosample,
            "subtalker_top_k": payload.subtalker_top_k,
            "subtalker_top_p": payload.subtalker_top_p,
            "subtalker_temperature": payload.subtalker_temperature,
            "max_new_tokens": payload.max_new_tokens,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        if CURRENT_CONFIG.get("safe_mode"):
            gen_kwargs.setdefault("subtalker_dosample", False)

        def _iter_events():
            header = {"sr": int(tts.model.speech_tokenizer.get_output_sample_rate()), "dtype": "int16"}
            yield f"event:meta\ndata:{json.dumps(header)}\n\n"
            chunk_idx = 0
            with MODEL_LOCK:
                for wav_chunk, sr in tts.generate_voice_clone_stream(
                    text=payload.text,
                    language=payload.language,
                    voice_clone_prompt=items,
                    chunk_size=payload.chunk_size,
                    left_context_size=payload.left_context_size,
                    **gen_kwargs,
                ):
                    pcm16 = (np.clip(wav_chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
                    b64 = base64.b64encode(pcm16.tobytes()).decode("ascii")
                    data = {"i": chunk_idx, "sr": sr, "audio": b64}
                    yield f"data:{json.dumps(data)}\n\n"
                    chunk_idx += 1
            yield "event:end\ndata:done\n\n"

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(_iter_events(), media_type="text/event-stream", headers=headers)

    @app.post("/model/load")
    def load_model(payload: ModelLoadRequest):
        model_id = (payload.model_id or "").strip()
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required.")
        if not MODEL_LOCK.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Model is busy. Wait for streaming to finish.")
        try:
            current_device = CURRENT_CONFIG.get("device") or args.device
            current_dtype = CURRENT_CONFIG.get("dtype") or args.dtype
            current_flash = (
                CURRENT_CONFIG.get("flash_attn") if CURRENT_CONFIG.get("flash_attn") is not None else args.flash_attn
            )
            current_tf32 = CURRENT_CONFIG.get("tf32") if CURRENT_CONFIG.get("tf32") is not None else args.tf32
            current_safe = CURRENT_CONFIG.get("safe_mode") if CURRENT_CONFIG.get("safe_mode") is not None else False

            device = payload.device or current_device
            dtype_str = payload.dtype or current_dtype
            flash_attn = payload.flash_attn if payload.flash_attn is not None else current_flash
            tf32 = payload.tf32 if payload.tf32 is not None else current_tf32
            safe_mode = payload.safe_mode if payload.safe_mode is not None else current_safe

            _clear_model()
            _apply_tf32(bool(tf32))
            _load_tts(model_id, device, _dtype_from_str(dtype_str), bool(flash_attn))
            _set_current_config(model_id, device, dtype_str, bool(flash_attn), bool(tf32), bool(safe_mode))
            if payload.clear_cache:
                with PROMPT_CACHE_LOCK:
                    PROMPT_CACHE.clear()
            return {"status": "ok", "model": CURRENT_MODEL_ID, "safe_mode": CURRENT_CONFIG.get("safe_mode")}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        finally:
            MODEL_LOCK.release()

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-clone-stream-service",
        description="Launch a streaming voice-clone service (SSE).",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=True,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Base model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device for device_map, e.g. cpu, cuda:0.")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for loading the model (default: float16).",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2 (default: enabled for streaming).",
    )
    parser.add_argument(
        "--tf32/--no-tf32",
        dest="tf32",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable TF32 matmul/cudnn on supported GPUs (default: enabled).",
    )
    parser.add_argument("--voice-store", default="voice_store", help="Directory to store voice prompts.")
    parser.add_argument("--ip", default="0.0.0.0", help="Server bind IP (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8002, help="Server port (default: 8002).")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    app = build_app(args)
    import uvicorn

    uvicorn.run(app, host=args.ip, port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
