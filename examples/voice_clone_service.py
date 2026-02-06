# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Gradio service for reusable voice-clone prompts.
"""

import argparse
import json
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem


DEFAULT_MODELS = [
    ("Base 1.7B", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    ("Base 0.6B", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
]

MODEL_LOCK = threading.Lock()
MODEL_CACHE: Dict[Tuple[str, str, str, bool], Qwen3TTSModel] = {}
CURRENT_TTS: Optional[Qwen3TTSModel] = None
CURRENT_MODEL_ID: Optional[str] = None

VOICE_INDEX_LOCK = threading.Lock()
VOICE_INDEX: Dict[str, Dict[str, Any]] = {}

PROMPT_CACHE_LOCK = threading.Lock()
PROMPT_CACHE: Dict[str, Tuple[List[VoiceClonePromptItem], Dict[str, Any]]] = {}

VOICE_STORE: Optional[Path] = None
PROMPT_DIR: Optional[Path] = None
INDEX_PATH: Optional[Path] = None


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _collect_gen_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    mapping = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m <= 1.0 + 1e-6:
            pass
        else:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    wav = np.asarray(wav, dtype=np.float32)
    return sr, wav


def _build_model_choices(initial_model: str) -> Tuple[List[str], Dict[str, str], str]:
    label_map = {label: model_id for label, model_id in DEFAULT_MODELS}
    choices = list(label_map.keys())
    initial_label = None
    for label, model_id in label_map.items():
        if model_id == initial_model:
            initial_label = label
            break
    if initial_label is None:
        custom_label = f"Custom ({initial_model})"
        choices.insert(0, custom_label)
        label_map[custom_label] = initial_model
        initial_label = custom_label
    return choices, label_map, initial_label


def _get_lang_choices(tts: Qwen3TTSModel) -> Tuple[List[str], Dict[str, str]]:
    supported = []
    if callable(getattr(tts.model, "get_supported_languages", None)):
        supported = tts.model.get_supported_languages()
    supported = [x for x in (supported or []) if x]
    if "Auto" not in supported:
        supported = ["Auto"] + supported
    choices, mapping = _build_choices_and_map(supported)
    if "Auto" not in mapping:
        mapping["Auto"] = "Auto"
        if "Auto" not in choices:
            choices.insert(0, "Auto")
    return choices, mapping


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


def _save_voice_index(index_path: Path, voices: Dict[str, Dict[str, Any]]) -> None:
    payload = {"voices": voices}
    tmp_path = index_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    tmp_path.replace(index_path)


def _add_voice_meta(voice_id: str, meta: Dict[str, Any]) -> None:
    with VOICE_INDEX_LOCK:
        VOICE_INDEX[voice_id] = meta
        _save_voice_index(INDEX_PATH, VOICE_INDEX)


def _build_voice_choices(model_id: Optional[str] = None) -> Tuple[List[str], Dict[str, str]]:
    with VOICE_INDEX_LOCK:
        voices = dict(VOICE_INDEX)
    items = list(voices.items())
    items.sort(key=lambda kv: kv[1].get("created_at", ""), reverse=True)
    choices: List[str] = []
    mapping: Dict[str, str] = {}
    for voice_id, meta in items:
        voice_model = meta.get("model_id")
        if model_id and voice_model != model_id:
            continue
        name = meta.get("name") or voice_id[:8]
        disp = f"{name} ({voice_id[:8]})"
        choices.append(disp)
        mapping[disp] = voice_id
    return choices, mapping


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
    hidden_size = int(tts.model.talker.config.hidden_size)
    for idx, item in enumerate(items):
        ref_code = item.ref_code
        if ref_code is not None:
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

        ref_spk = item.ref_spk_embedding
        if ref_spk is None:
            raise ValueError(f"Missing ref_spk_embedding in prompt[{idx}].")
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        if ref_spk.dim() != 1 or int(ref_spk.shape[0]) != hidden_size:
            raise ValueError(
                f"ref_spk_embedding size mismatch for prompt[{idx}]: expected {hidden_size}, got {tuple(ref_spk.shape)}. "
                "Recreate the voice prompt with the same model."
            )
        item.ref_spk_embedding = ref_spk


def _load_tts(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    flash_attn: bool,
    cache_models: bool,
) -> Qwen3TTSModel:
    global CURRENT_TTS, CURRENT_MODEL_ID
    key = (model_id, device, str(dtype), bool(flash_attn))
    if key in MODEL_CACHE:
        CURRENT_TTS = MODEL_CACHE[key]
        CURRENT_MODEL_ID = model_id
        return CURRENT_TTS

    if not cache_models:
        MODEL_CACHE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    attn_impl = "flash_attention_2" if flash_attn else None
    tts = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    if getattr(tts.model, "tts_model_type", None) != "base":
        raise ValueError("Only Base models support voice cloning.")

    MODEL_CACHE[key] = tts
    CURRENT_TTS = tts
    CURRENT_MODEL_ID = model_id
    return tts


def _get_current_tts() -> Qwen3TTSModel:
    if CURRENT_TTS is None:
        raise RuntimeError("Model not loaded.")
    return CURRENT_TTS


def build_app(args: argparse.Namespace) -> gr.Blocks:
    model_choices, model_label_map, initial_label = _build_model_choices(args.model)

    with MODEL_LOCK:
        tts = _load_tts(args.model, args.device, _dtype_from_str(args.dtype), args.flash_attn, args.cache_models)

    lang_choices, lang_map = _get_lang_choices(tts)
    voice_choices, voice_map = _build_voice_choices(args.model)

    gen_kwargs_default = _collect_gen_kwargs(args)

    def switch_model(
        model_label: str,
        cur_model: str,
        cur_lang_map: Dict[str, str],
        cur_voice_map: Dict[str, str],
    ):
        model_id = model_label_map.get(model_label, model_label)
        try:
            with MODEL_LOCK:
                tts_local = _load_tts(model_id, args.device, _dtype_from_str(args.dtype), args.flash_attn, args.cache_models)
            choices, mapping = _get_lang_choices(tts_local)
            value = "Auto" if "Auto" in choices else (choices[0] if choices else None)
            voice_choices, voice_map = _build_voice_choices(model_id)
            voice_value = voice_choices[0] if voice_choices else None
            status = f"Loaded {model_id}"
            return (
                status,
                gr.update(choices=choices, value=value),
                model_id,
                mapping,
                gr.update(choices=voice_choices, value=voice_value),
                voice_map,
            )
        except Exception as e:
            return (
                f"Failed to load model: {type(e).__name__}: {e}",
                gr.update(),
                cur_model,
                cur_lang_map,
                gr.update(),
                cur_voice_map,
            )

    def refresh_voices(cur_model: str):
        choices, mapping = _build_voice_choices(cur_model)
        value = choices[0] if choices else None
        return gr.update(choices=choices, value=value), mapping

    def save_voice(ref_audio, ref_text: str, use_xvec: bool, voice_name: str, cur_model: str):
        try:
            audio_tuple = _audio_to_tuple(ref_audio)
            if audio_tuple is None:
                return "Reference audio is required.", gr.update(), {}
            if (not use_xvec) and (not ref_text or not ref_text.strip()):
                return "Reference text is required when x-vector only is disabled.", gr.update(), {}

            tts_local = _get_current_tts()
            with MODEL_LOCK:
                items = tts_local.create_voice_clone_prompt(
                    ref_audio=audio_tuple,
                    ref_text=(ref_text.strip() if ref_text else None),
                    x_vector_only_mode=bool(use_xvec),
                )

            voice_id = uuid.uuid4().hex
            name = (voice_name or "").strip() or voice_id[:8]
            created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            rel_path = str(Path("prompts") / f"{voice_id}.pt")
            payload = {
                "items": [asdict(it) for it in items],
                "meta": {
                    "voice_id": voice_id,
                    "name": name,
                    "created_at": created_at,
                    "model_id": cur_model,
                },
            }
            torch.save(payload, VOICE_STORE / rel_path)

            meta = {
                "name": name,
                "path": rel_path,
                "created_at": created_at,
                "model_id": cur_model,
            }
            _add_voice_meta(voice_id, meta)

            with PROMPT_CACHE_LOCK:
                PROMPT_CACHE[voice_id] = (items, payload["meta"])

            choices, mapping = _build_voice_choices(cur_model)
            value = choices[0] if choices else None
            status = f"Saved voice '{name}' ({voice_id})."
            return status, gr.update(choices=choices, value=value), mapping
        except Exception as e:
            return f"{type(e).__name__}: {e}", gr.update(), {}

    def run_tts(
        voice_choice: str,
        text: str,
        lang_disp: str,
        voice_map_local: Dict[str, str],
        lang_map_local: Dict[str, str],
        cur_model: str,
    ):
        try:
            if not text or not text.strip():
                return None, "Target text is required."
            if not voice_choice:
                return None, "Please select a voice."
            voice_id = voice_map_local.get(voice_choice)
            if not voice_id:
                return None, "Unknown voice selection."

            items, prompt_meta = _load_prompt_items(voice_id)
            prompt_model = prompt_meta.get("model_id")
            if prompt_model and prompt_model != cur_model:
                return None, f"Voice created with {prompt_model}. Switch model to match."

            tts_local = _get_current_tts()
            try:
                _validate_prompt_items(items, tts_local)
            except ValueError as exc:
                return None, str(exc)

            language = lang_map_local.get(lang_disp, "Auto")
            with MODEL_LOCK:
                wavs, sr = tts_local.generate_voice_clone(
                    text=text.strip(),
                    language=language,
                    voice_clone_prompt=items,
                    **gen_kwargs_default,
                )
            return _wav_to_gradio_audio(wavs[0], sr), "Finished."
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    theme = gr.themes.Soft(font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"])
    css = ".gradio-container {max-width: none !important;}"

    with gr.Blocks(theme=theme, css=css) as demo:
        model_state = gr.State(args.model)
        lang_state = gr.State(lang_map)
        voice_map_state = gr.State(voice_map)
        gr.Markdown(
            """
# Qwen3-TTS Voice Clone Service
- Create reusable voice prompts from reference audio
- Generate speech from stored voices
"""
        )

        with gr.Row():
            model_dd = gr.Dropdown(
                label="Model",
                choices=model_choices,
                value=initial_label,
                interactive=True,
            )
            load_btn = gr.Button("Load Model", variant="primary")
            model_status = gr.Textbox(label="Model Status", value=f"Loaded {args.model}", interactive=False)

        with gr.Tabs():
            with gr.Tab("Voice Provider"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ref_audio = gr.Audio(label="Reference Audio", type="numpy")
                        ref_text = gr.Textbox(label="Reference Text", lines=2)
                        xvec_only = gr.Checkbox(label="Use x-vector only", value=False)
                        voice_name = gr.Textbox(label="Voice Name (optional)", lines=1)
                        save_btn = gr.Button("Create Voice", variant="primary")
                    with gr.Column(scale=3):
                        save_status = gr.Textbox(label="Status", lines=2)

            with gr.Tab("TTS"):
                with gr.Row():
                    with gr.Column(scale=2):
                        voice_dd = gr.Dropdown(
                            label="Voice",
                            choices=voice_choices,
                            value=(voice_choices[0] if voice_choices else None),
                            interactive=True,
                        )
                        refresh_btn = gr.Button("Refresh Voice List")
                        text_in = gr.Textbox(label="Target Text", lines=4)
                        lang_in = gr.Dropdown(
                            label="Language",
                            choices=lang_choices,
                            value=("Auto" if "Auto" in lang_choices else (lang_choices[0] if lang_choices else None)),
                            interactive=True,
                        )
                        gen_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=3):
                        audio_out = gr.Audio(label="Output Audio", type="numpy")
                        gen_status = gr.Textbox(label="Status", lines=2)

        load_btn.click(
            switch_model,
            inputs=[model_dd, model_state, lang_state, voice_map_state],
            outputs=[model_status, lang_in, model_state, lang_state, voice_dd, voice_map_state],
        )
        refresh_btn.click(
            refresh_voices,
            inputs=[model_state],
            outputs=[voice_dd, voice_map_state],
        )
        save_btn.click(
            save_voice,
            inputs=[ref_audio, ref_text, xvec_only, voice_name, model_state],
            outputs=[save_status, voice_dd, voice_map_state],
        )
        gen_btn.click(
            run_tts,
            inputs=[voice_dd, text_in, lang_in, voice_map_state, lang_state, model_state],
            outputs=[audio_out, gen_status],
        )

    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-clone-service",
        description="Launch a reusable voice-clone Gradio service.",
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
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Torch dtype for loading the model (default: bfloat16).",
    )
    parser.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2 (default: enabled).",
    )
    parser.add_argument("--cache-models", action="store_true", help="Keep multiple models in memory.")
    parser.add_argument("--voice-store", default="voice_store", help="Directory to store voice prompts.")
    parser.add_argument("--ip", default="0.0.0.0", help="Server bind IP (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000).")
    parser.add_argument("--concurrency", type=int, default=4, help="Gradio queue concurrency (default: 4).")

    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max new tokens for generation (optional).")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (optional).")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (optional).")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling (optional).")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty (optional).")
    parser.add_argument("--subtalker-top-k", type=int, default=None, help="Subtalker top-k (optional).")
    parser.add_argument("--subtalker-top-p", type=float, default=None, help="Subtalker top-p (optional).")
    parser.add_argument("--subtalker-temperature", type=float, default=None, help="Subtalker temperature (optional).")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _init_voice_store(Path(args.voice_store))

    demo = build_app(args)
    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(
        server_name=args.ip,
        server_port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
