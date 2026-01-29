# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import io
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from ..core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]


@dataclass
class VoiceClonePromptItem:
    """
    Container for one sample's voice-clone prompt information that can be fed to the model.

    Fields are aligned with `Qwen3TTSForConditionalGeneration.generate(..., voice_clone_prompt=...)`.
    """
    ref_code: Optional[torch.Tensor]                 # (T, Q) or (T,) depending on tokenizer 25Hz/12Hz
    ref_spk_embedding: torch.Tensor                  # (D,)
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


class Qwen3TTSModel:
    """
    A HuggingFace-style wrapper for Qwen3 TTS models (CustomVoice/VoiceDesign/Base) that provides:
      - from_pretrained() initialization via AutoModel/AutoProcessor
      - generation APIs for:
          * CustomVoice: generate_custom_voice()
          * VoiceDesign: generate_voice_design()
          * Base: generate_voice_clone() + create_voice_clone_prompt()
      - consistent output: (wavs: List[np.ndarray], sample_rate: int)

    Notes:
      - This wrapper expects the underlying model class to be `Qwen3TTSForConditionalGeneration`
      - Language / speaker validation is done via model methods:
          model.get_supported_languages(), model.get_supported_speakers()
    """

    def __init__(self, model: Qwen3TTSForConditionalGeneration, processor, generate_defaults: Optional[Dict[str, Any]] = None):
        self.model = model
        self.processor = processor
        self.generate_defaults = generate_defaults or {}

        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "Qwen3TTSModel":
        """
        Load a Qwen3 TTS model and its processor in HuggingFace `from_pretrained` style.

        This method:
          1) Loads config via AutoConfig (so your side can register model_type -> config/model).
          2) Loads the model via AutoModel.from_pretrained(...), forwarding `kwargs` unchanged.
          3) Loads the processor via AutoProcessor.from_pretrained(model_path).
          4) Loads optional `generate_config.json` from the model directory/repo snapshot if present.

        Args:
            pretrained_model_name_or_path (str):
                HuggingFace repo id or local directory of the model.
            **kwargs:
                Forwarded as-is into `AutoModel.from_pretrained(...)`.
                Typical examples: device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2".

        Returns:
            Qwen3TTSModel:
                Wrapper instance containing `model`, `processor`, and generation defaults.
        """
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, Qwen3TTSForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3TTSForConditionalGeneration. "
            )

        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True,)

        generate_defaults = model.generate_config
        return cls(model=model, processor=processor, generate_defaults=generate_defaults)

    def _supported_languages_set(self) -> Optional[set]:
        langs = getattr(self.model, "get_supported_languages", None)
        if callable(langs):
            v = langs()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _supported_speakers_set(self) -> Optional[set]:
        spks = getattr(self.model, "get_supported_speakers", None)
        if callable(spks):
            v = spks()
            if v is None:
                return None
            return set([str(x).lower() for x in v])
        return None

    def _validate_languages(self, languages: List[str]) -> None:
        """
        Validate that requested languages are supported by the model.

        Args:
            languages (List[str]): Language names for each sample.

        Raises:
            ValueError: If any language is not supported.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return

        bad = []
        for lang in languages:
            if lang is None:
                bad.append(lang)
                continue
            if str(lang).lower() not in supported:
                bad.append(lang)
        if bad:
            raise ValueError(f"Unsupported languages: {bad}. Supported: {sorted(supported)}")

    def _validate_speakers(self, speakers: List[Optional[str]]) -> None:
        """
        Validate that requested speakers are supported by the Instruct model.

        Args:
            speakers (List[Optional[str]]): Speaker names for each sample.

        Raises:
            ValueError: If any speaker is not supported.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return

        bad = []
        for spk in speakers:
            if spk is None or spk == "":
                continue
            if str(spk).lower() not in supported:
                bad.append(spk)
        if bad:
            raise ValueError(f"Unsupported speakers: {bad}. Supported: {sorted(supported)}")

    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True
        if ("/" not in s and "\\" not in s) and len(s) > 256:
            return True
        return False

    def _is_url(self, s: str) -> bool:
        try:
            u = urlparse(s)
            return u.scheme in ("http", "https") and bool(u.netloc)
        except Exception:
            return False

    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:
        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        if self._is_url(x):
            with urllib.request.urlopen(x) as resp:
                audio_bytes = resp.read()
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        elif self._is_probably_base64(x):
            wav_bytes = self._decode_base64_to_wav_bytes(x)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        else:
            audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - (np.ndarray, sr): waveform + sampling rate
          - list of the above

        Args:
            audios:
                Audio input(s).

        Returns:
            List[Tuple[np.ndarray, int]]:
                List of (float32 waveform, original sr).

        Raises:
            ValueError: If a numpy waveform is provided without sr.
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        for i, a in enumerate(out):
            if a[0].ndim > 1:
                a[0] = np.mean(a[0], axis=-1).astype(np.float32)
                out[i] = (a[0], a[1])
        return out

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _tokenize_texts(self, texts: List[str]) -> List[torch.Tensor]:
        input_ids = []
        for text in texts:
            input = self.processor(text=text, return_tensors="pt", padding=True)
            input_id = input["input_ids"].to(self.device)
            input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
            input_ids.append(input_id)
        return input_ids

    def _merge_generate_kwargs(
        self,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_dosample: Optional[bool] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Merge user-provided generation arguments with defaults from `generate_config.json`.

        Rule:
          - If the user explicitly passes a value (not None), use it.
          - Otherwise, use the value from generate_config.json if present.
          - Otherwise, fall back to the hard defaults.

        Args:
            do_sample, top_k, top_p, temperature, repetition_penalty,
            subtalker_dosample, subtalker_top_k, subtalker_top_p, subtalker_temperature, max_new_tokens:
                Common generation parameters.
            **kwargs:
                Other arguments forwarded to model.generate().

        Returns:
            Dict[str, Any]: Final kwargs to pass into model.generate().
        """
        hard_defaults = dict(
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            max_new_tokens=2048,
        )

        def pick(name: str, user_val: Any) -> Any:
            if user_val is not None:
                return user_val
            if name in self.generate_defaults:
                return self.generate_defaults[name]
            return hard_defaults[name]

        merged = dict(kwargs)
        merged.update(
            do_sample=pick("do_sample", do_sample),
            top_k=pick("top_k", top_k),
            top_p=pick("top_p", top_p),
            temperature=pick("temperature", temperature),
            repetition_penalty=pick("repetition_penalty", repetition_penalty),
            subtalker_dosample=pick("subtalker_dosample", subtalker_dosample),
            subtalker_top_k=pick("subtalker_top_k", subtalker_top_k),
            subtalker_top_p=pick("subtalker_top_p", subtalker_top_p),
            subtalker_temperature=pick("subtalker_temperature", subtalker_temperature),
            max_new_tokens=pick("max_new_tokens", max_new_tokens),
        )
        return merged

    # voice clone model
    @torch.inference_mode()
    def create_voice_clone_prompt(
        self,
        ref_audio: Union[AudioLike, List[AudioLike]],
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
    ) -> List[VoiceClonePromptItem]:
        """
        Build voice-clone prompt items from reference audio (and optionally reference text) using Base model.

        Modes:
          - x_vector_only_mode=True:
              Only speaker embedding is used to clone voice; ref_text/ref_code are ignored.
              This is mutually exclusive with ICL.
          - x_vector_only_mode=False:
              ICL mode is enabled automatically (icl_mode=True). In this case ref_text is required,
              because the model continues/conditions on the reference text + reference speech codes.

        Batch behavior:
          - ref_audio can be a single item or a list.
          - ref_text and x_vector_only_mode can be scalars or lists.
          - If any of them are lists with length > 1, lengths must match.

        Audio input:
          - str: local wav path / URL / base64
          - (np.ndarray, sr): waveform + sampling rate

        Args:
            ref_audio:
                Reference audio(s) used to extract:
                  - ref_code via `model.speech_tokenizer.encode(...)`
                  - ref_spk_embedding via `model.extract_speaker_embedding(...)` (resampled to 24k)
            ref_text:
                Reference transcript(s). Required when x_vector_only_mode=False (ICL mode).
            x_vector_only_mode:
                Whether to use speaker embedding only. If False, ICL mode will be used.

        Returns:
            List[VoiceClonePromptItem]:
                List of prompt items that can be converted into `voice_clone_prompt` dict.

        Raises:
            ValueError:
                - If x_vector_only_mode=False but ref_text is missing.
                - If batch lengths mismatch.
        """
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support create_voice_clone_prompt, Please check Model Card or Readme for more details."
            )
        
        ref_audio_list = self._ensure_list(ref_audio)
        ref_text_list = self._ensure_list(ref_text) if isinstance(ref_text, list) else ([ref_text] * len(ref_audio_list))
        xvec_list = self._ensure_list(x_vector_only_mode) if isinstance(x_vector_only_mode, list) else ([x_vector_only_mode] * len(ref_audio_list))

        if len(ref_text_list) != len(ref_audio_list) or len(xvec_list) != len(ref_audio_list):
            raise ValueError(
                f"Batch size mismatch: ref_audio={len(ref_audio_list)}, ref_text={len(ref_text_list)}, x_vector_only_mode={len(xvec_list)}"
            )

        normalized = self._normalize_audio_inputs(ref_audio_list)

        ref_wavs_for_code: List[np.ndarray] = []
        ref_sr_for_code: List[int] = []
        for wav, sr in normalized:
            ref_wavs_for_code.append(wav)
            ref_sr_for_code.append(sr)

        if len(set(ref_sr_for_code)) == 1:
            enc = self.model.speech_tokenizer.encode(ref_wavs_for_code, sr=ref_sr_for_code[0])
            ref_codes = enc.audio_codes
        else:
            ref_codes = []
            for wav, sr in normalized:
                ref_codes.append(self.model.speech_tokenizer.encode(wav, sr=sr).audio_codes[0])

        items: List[VoiceClonePromptItem] = []
        for i, ((wav, sr), code, rtext, xvec_only) in enumerate(zip(normalized, ref_codes, ref_text_list, xvec_list)):
            if not xvec_only:
                if rtext is None or rtext == "":
                    raise ValueError(f"ref_text is required when x_vector_only_mode=False (ICL mode). Bad index={i}")

            wav_resample = wav
            if sr != self.model.speaker_encoder_sample_rate:
                wav_resample = librosa.resample(y=wav_resample.astype(np.float32), 
                                           orig_sr=int(sr), 
                                           target_sr=self.model.speaker_encoder_sample_rate)

            spk_emb = self.model.extract_speaker_embedding(audio=wav_resample,
                                                           sr=self.model.speaker_encoder_sample_rate)

            items.append(
                VoiceClonePromptItem(
                    ref_code=None if xvec_only else code,
                    ref_spk_embedding=spk_emb,
                    x_vector_only_mode=bool(xvec_only),
                    icl_mode=bool(not xvec_only),
                    ref_text=rtext,
                )
            )
        return items

    def _prompt_items_to_voice_clone_prompt(self, items: List[VoiceClonePromptItem]) -> Dict[str, Any]:
        return dict(
            ref_code=[it.ref_code for it in items],
            ref_spk_embedding=[it.ref_spk_embedding for it in items],
            x_vector_only_mode=[it.x_vector_only_mode for it in items],
            icl_mode=[it.icl_mode for it in items],
        )

    # voice clone model
    @torch.no_grad()
    def generate_voice_clone(
        self,
        text: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        ref_audio: Optional[Union[AudioLike, List[AudioLike]]] = None,
        ref_text: Optional[Union[str, List[Optional[str]]]] = None,
        x_vector_only_mode: Union[bool, List[bool]] = False,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[VoiceClonePromptItem]]] = None,
        non_streaming_mode: bool = False,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Voice clone speech using the Base model.

        You can provide either:
          - (ref_audio, ref_text, x_vector_only_mode) and let this method build the prompt, OR
          - `VoiceClonePromptItem` returned by `create_voice_clone_prompt`, OR
          - a list of `VoiceClonePromptItem` returned by `create_voice_clone_prompt`.
        
        `ref_audio` Supported forms:
        - str: wav path / URL / base64 audio string
        - (np.ndarray, sr): waveform + sampling rate
        - list of the above

        Input flexibility:
          - text/language can be scalar or list.
          - prompt can be single or batch.
          - If batch mode (len(text)>1), lengths must match.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            ref_audio:
                Reference audio(s) for prompt building. Required if voice_clone_prompt is not provided.
            ref_text:
                Reference text(s) used for ICL mode (required when x_vector_only_mode=False).
            x_vector_only_mode:
                If True, only speaker embedding is used (ignores ref_text/ref_code).
                If False, ICL mode is used automatically.
            voice_clone_prompt:
                list[VoiceClonePromptItem] from `create_voice_clone_prompt`.
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`, 
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)

        Raises:
            ValueError:
                If batch sizes mismatch or required prompt inputs are missing.
        """
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_clone, Please check Model Card or Readme for more details."
            )
        
        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(texts) != len(languages):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}")

        self._validate_languages(languages)

        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
            prompt_items = self.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=x_vector_only_mode)
            if len(prompt_items) == 1 and len(texts) > 1:
                prompt_items = prompt_items * len(texts)
            if len(prompt_items) != len(texts):
                raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_texts_for_ids = [it.ref_text for it in prompt_items]
        else:
            if isinstance(voice_clone_prompt, list):
                prompt_items = voice_clone_prompt
                if len(prompt_items) == 1 and len(texts) > 1:
                    prompt_items = prompt_items * len(texts)
                if len(prompt_items) != len(texts):
                    raise ValueError(f"Batch size mismatch: prompt={len(prompt_items)}, text={len(texts)}")
                voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
                ref_texts_for_ids = [it.ref_text for it in prompt_items]
            else:
                voice_clone_prompt_dict = voice_clone_prompt
                ref_texts_for_ids = None

        input_texts = [self._build_assistant_text(t) for t in texts]
        input_ids = self._tokenize_texts(input_texts)

        ref_ids = None
        if ref_texts_for_ids is not None:
            ref_ids = []
            for i, rt in enumerate(ref_texts_for_ids):
                if rt is None or rt == "":
                    ref_ids.append(None)
                else:
                    ref_tok = self._tokenize_texts([self._build_ref_text(rt)])[0]
                    ref_ids.append(ref_tok)

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=voice_clone_prompt_dict,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        codes_for_decode = []
        for i, codes in enumerate(talker_codes_list):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
            else:
                codes_for_decode.append(codes)

        wavs_all, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_for_decode])

        wavs_out: List[np.ndarray] = []
        for i, wav in enumerate(wavs_all):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                ref_len = int(ref_code_list[i].shape[0])
                total_len = int(codes_for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wavs_out.append(wav[cut:])
            else:
                wavs_out.append(wav)

        return wavs_out, fs

    @torch.inference_mode()
    def generate_voice_clone_stream(
        self,
        text: str,
        language: Optional[str] = None,
        ref_audio: Optional[AudioLike] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
        voice_clone_prompt: Optional[Union[Dict[str, Any], List[VoiceClonePromptItem]]] = None,
        non_streaming_mode: bool = False,
        chunk_size: int = 8,
        left_context_size: int = 25,
        **kwargs,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """
        Stream voice-clone audio chunks (single sample only).

        This method yields (wav_chunk, sample_rate) tuples as soon as each chunk is decoded.
        """
        if self.model.tts_model_type != "base":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_clone_stream, Please check Model Card or Readme for more details."
            )
        if not isinstance(text, str):
            raise ValueError("Streaming mode only supports a single text string.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if left_context_size < 0:
            raise ValueError("left_context_size must be >= 0.")

        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
            prompt_items = self.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only_mode,
            )
            if len(prompt_items) != 1:
                raise ValueError("Streaming mode only supports a single prompt item.")
            voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(prompt_items)
            ref_text_for_ids = prompt_items[0].ref_text
        else:
            if isinstance(voice_clone_prompt, list):
                if len(voice_clone_prompt) != 1:
                    raise ValueError("Streaming mode only supports a single prompt item.")
                voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(voice_clone_prompt)
                ref_text_for_ids = voice_clone_prompt[0].ref_text
            else:
                voice_clone_prompt_dict = voice_clone_prompt
                ref_text_for_ids = None

        def _ensure_list_field(val):
            if val is None:
                return None
            return val if isinstance(val, list) else [val]

        voice_clone_prompt_dict = {
            "ref_code": _ensure_list_field(voice_clone_prompt_dict.get("ref_code")),
            "ref_spk_embedding": _ensure_list_field(voice_clone_prompt_dict.get("ref_spk_embedding")),
            "x_vector_only_mode": _ensure_list_field(voice_clone_prompt_dict.get("x_vector_only_mode", False)),
            "icl_mode": _ensure_list_field(voice_clone_prompt_dict.get("icl_mode", False)),
        }

        if language is None:
            language = "Auto"
        self._validate_languages([language])

        input_ids = self._tokenize_texts([self._build_assistant_text(text)])
        input_id = input_ids[0]

        ref_id = None
        if ref_text_for_ids:
            ref_id = self._tokenize_texts([self._build_ref_text(ref_text_for_ids)])[0]

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        if self.model.speech_tokenizer.get_model_type() != "qwen3_tts_tokenizer_12hz":
            raise ValueError("Streaming decode currently supports only 12Hz tokenizer models.")

        decoder = self.model.speech_tokenizer.model.decoder
        decoder_device = getattr(self.model.speech_tokenizer, "device", self.model.talker.device)
        decode_upsample = int(getattr(decoder, "total_upsample", 0)) or int(
            self.model.speech_tokenizer.get_decode_upsample_rate()
        )

        def _decode_codes(codes: torch.Tensor, drop_codes: int) -> Optional[np.ndarray]:
            if codes.numel() == 0:
                return None
            codes_t = codes.transpose(0, 1).unsqueeze(0).to(decoder_device)
            wav = decoder(codes_t).squeeze(0).squeeze(0)
            if drop_codes > 0:
                drop = drop_codes * decode_upsample
                if drop < wav.shape[-1]:
                    wav = wav[drop:]
                else:
                    wav = wav[:0]
            return wav.detach().to(torch.float32).cpu().numpy()

        suppress_token_ids = [
            i
            for i in range(self.model.config.talker_config.vocab_size - 1024, self.model.config.talker_config.vocab_size)
            if i not in (self.model.config.talker_config.codec_eos_token_id,)
        ]

        def _apply_repetition_penalty(logits: torch.Tensor, prev_tokens: List[int], penalty: float) -> torch.Tensor:
            if penalty is None or penalty == 1.0 or not prev_tokens:
                return logits
            for tok in set(prev_tokens):
                if logits[tok] < 0:
                    logits[tok] *= penalty
                else:
                    logits[tok] /= penalty
            return logits

        def _filter_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
            if top_k and top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_keep = values[-1]
                logits = torch.where(logits < min_keep, torch.tensor(-float("inf"), device=logits.device), logits)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum_probs = probs.cumsum(dim=-1)
                mask = cum_probs > top_p
                if mask.numel() > 0:
                    mask[0] = False
                sorted_logits[mask] = -float("inf")
                logits = torch.empty_like(logits).scatter_(0, sorted_indices, sorted_logits)
            return logits

        def _sample_next(logits: torch.Tensor, prev_tokens: List[int]) -> int:
            logits = logits.clone()
            if suppress_token_ids:
                logits[suppress_token_ids] = -float("inf")
            logits = _apply_repetition_penalty(logits, prev_tokens, float(gen_kwargs["repetition_penalty"]))
            temperature = float(gen_kwargs["temperature"]) if gen_kwargs.get("temperature") is not None else 1.0
            if temperature <= 0:
                temperature = 1.0
            logits = logits / temperature
            logits = _filter_top_k_top_p(
                logits,
                int(gen_kwargs["top_k"]) if gen_kwargs.get("top_k") is not None else 0,
                float(gen_kwargs["top_p"]) if gen_kwargs.get("top_p") is not None else 1.0,
            )
            if not bool(gen_kwargs.get("do_sample", True)):
                return int(torch.argmax(logits).item())
            probs = torch.softmax(logits.float(), dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0:
                return int(torch.argmax(logits).item())
            return int(torch.multinomial(probs, num_samples=1).item())

        voice_clone_spk_embeds = None
        if voice_clone_prompt_dict.get("ref_spk_embedding") is not None:
            voice_clone_spk_embeds = self.model.generate_speaker_prompt(voice_clone_prompt_dict)

        if voice_clone_spk_embeds is None:
            speaker_embed = None
        else:
            if voice_clone_prompt_dict["x_vector_only_mode"][0] or voice_clone_prompt_dict["icl_mode"][0]:
                speaker_embed = voice_clone_spk_embeds[0]
            else:
                speaker_embed = None

        if language.lower() == "auto":
            language_id = None
        else:
            language_id = self.model.config.talker_config.codec_language_id[language.lower()]

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self.model.talker.text_projection(
            self.model.talker.get_text_embeddings()(
                torch.tensor(
                    [[self.model.config.tts_bos_token_id, self.model.config.tts_eos_token_id, self.model.config.tts_pad_token_id]],
                    device=self.model.talker.device,
                    dtype=input_id.dtype,
                )
            )
        ).chunk(3, dim=1)

        if language_id is None:
            codec_prefill_list = [[
                self.model.config.talker_config.codec_nothink_id,
                self.model.config.talker_config.codec_think_bos_id,
                self.model.config.talker_config.codec_think_eos_id,
            ]]
        else:
            codec_prefill_list = [[
                self.model.config.talker_config.codec_think_id,
                self.model.config.talker_config.codec_think_bos_id,
                language_id,
                self.model.config.talker_config.codec_think_eos_id,
            ]]

        codec_input_embedding_0 = self.model.talker.get_input_embeddings()(
            torch.tensor(codec_prefill_list, device=self.model.talker.device, dtype=input_id.dtype)
        )
        codec_input_embedding_1 = self.model.talker.get_input_embeddings()(
            torch.tensor(
                [[self.model.config.talker_config.codec_pad_id, self.model.config.talker_config.codec_bos_id]],
                device=self.model.talker.device,
                dtype=input_id.dtype,
            )
        )
        if speaker_embed is None:
            codec_input_embedding = torch.cat([codec_input_embedding_0, codec_input_embedding_1], dim=1)
        else:
            codec_input_embedding = torch.cat(
                [codec_input_embedding_0, speaker_embed.view(1, 1, -1), codec_input_embedding_1], dim=1
            )

        talker_input_embed_role = self.model.talker.text_projection(
            self.model.talker.get_text_embeddings()(input_id[:, :3])
        )
        _talker_input_embed = torch.cat(
            (tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1), tts_bos_embed),
            dim=1,
        ) + codec_input_embedding[:, :-1]
        talker_input_embed = torch.cat((talker_input_embed_role, _talker_input_embed), dim=1)

        ref_code_list = voice_clone_prompt_dict.get("ref_code")
        ref_code_item = ref_code_list[0] if ref_code_list else None
        if ref_code_item is not None and voice_clone_prompt_dict["icl_mode"][0]:
            if ref_id is None:
                raise ValueError("ref_text is required for ICL mode in streaming generation.")
            icl_input_embed, trailing_text_hidden = self.model.generate_icl_prompt(
                text_id=input_id[:, 3:-5],
                ref_id=ref_id[:, 3:-2],
                ref_code=ref_code_item.to(self.model.talker.device),
                tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=non_streaming_mode,
            )
            talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
        else:
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    self.model.talker.text_projection(
                        self.model.talker.get_text_embeddings()(input_id[:, 3:4])
                    ) + codec_input_embedding[:, -1:],
                ],
                dim=1,
            )
            if non_streaming_mode:
                talker_input_embed = talker_input_embed[:, :-1]
                talker_input_embed = torch.cat(
                    [
                        talker_input_embed,
                        torch.cat(
                            (
                                self.model.talker.text_projection(
                                    self.model.talker.get_text_embeddings()(input_id[:, 3:-5])
                                ),
                                tts_eos_embed,
                            ),
                            dim=1,
                        )
                        + self.model.talker.get_input_embeddings()(
                            torch.tensor(
                                [[self.model.config.talker_config.codec_pad_id]] * (input_id[:, 3:-5].shape[1] + 1),
                                device=self.model.talker.device,
                                dtype=input_id.dtype,
                            )
                        ),
                        tts_pad_embed
                        + self.model.talker.get_input_embeddings()(
                            torch.tensor(
                                [[self.model.config.talker_config.codec_bos_id]],
                                device=self.model.talker.device,
                                dtype=input_id.dtype,
                            )
                        ),
                    ],
                    dim=1,
                )
                trailing_text_hidden = tts_pad_embed
            else:
                trailing_text_hidden = torch.cat(
                    (
                        self.model.talker.text_projection(
                            self.model.talker.get_text_embeddings()(input_id[:, 4:-5])
                        ),
                        tts_eos_embed,
                    ),
                    dim=1,
                )

        prefill = self.model.talker(
            inputs_embeds=talker_input_embed,
            attention_mask=None,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            subtalker_dosample=bool(gen_kwargs.get("subtalker_dosample", True)),
            subtalker_top_k=gen_kwargs.get("subtalker_top_k"),
            subtalker_top_p=gen_kwargs.get("subtalker_top_p"),
            subtalker_temperature=gen_kwargs.get("subtalker_temperature"),
        )

        next_token = _sample_next(prefill.logits[:, -1, :].squeeze(0), [])
        past_key_values = prefill.past_key_values
        past_hidden = prefill.past_hidden
        generation_step = prefill.generation_step

        ref_code = ref_code_item
        context_codes = None
        if ref_code is not None:
            if left_context_size > 0:
                context_codes = ref_code[-left_context_size:].to(decoder_device)
            else:
                context_codes = ref_code.to(decoder_device)

        pending_codes: List[torch.Tensor] = []
        prev_tokens: List[int] = []
        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 2048))
        eos_token_id = int(self.model.config.talker_config.codec_eos_token_id)
        sample_rate = int(self.model.speech_tokenizer.get_output_sample_rate())

        for _ in range(max_new_tokens):
            if next_token == eos_token_id:
                break

            output = self.model.talker(
                input_ids=torch.tensor([[next_token]], device=self.model.talker.device),
                attention_mask=None,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                past_hidden=past_hidden,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                generation_step=generation_step,
                subtalker_dosample=bool(gen_kwargs.get("subtalker_dosample", True)),
                subtalker_top_k=gen_kwargs.get("subtalker_top_k"),
                subtalker_top_p=gen_kwargs.get("subtalker_top_p"),
                subtalker_temperature=gen_kwargs.get("subtalker_temperature"),
            )

            codec_ids = output.hidden_states[1]
            if codec_ids is None:
                break
            pending_codes.append(codec_ids.squeeze(0))
            prev_tokens.append(int(next_token))

            if len(pending_codes) >= chunk_size:
                chunk = torch.stack(pending_codes[:chunk_size], dim=0)
                pending_codes = pending_codes[chunk_size:]
                if context_codes is not None and context_codes.numel() > 0:
                    decode_codes = torch.cat([context_codes, chunk], dim=0)
                    drop = int(context_codes.shape[0])
                else:
                    decode_codes = chunk
                    drop = 0
                wav_chunk = _decode_codes(decode_codes, drop)
                if wav_chunk is not None and wav_chunk.size > 0:
                    yield wav_chunk, sample_rate
                if left_context_size > 0:
                    context_codes = decode_codes[-left_context_size:]
                else:
                    context_codes = None

            past_key_values = output.past_key_values
            past_hidden = output.past_hidden
            generation_step = output.generation_step
            next_token = _sample_next(output.logits[:, -1, :].squeeze(0), prev_tokens)

        if pending_codes:
            chunk = torch.stack(pending_codes, dim=0)
            if context_codes is not None and context_codes.numel() > 0:
                decode_codes = torch.cat([context_codes, chunk], dim=0)
                drop = int(context_codes.shape[0])
            else:
                decode_codes = chunk
                drop = 0
            wav_chunk = _decode_codes(decode_codes, drop)
            if wav_chunk is not None and wav_chunk.size > 0:
                yield wav_chunk, sample_rate

    # voice design model
    @torch.no_grad()
    def generate_voice_design(
        self,
        text: Union[str, List[str]],
        instruct: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with the VoiceDesign model using natural-language style instructions.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            instruct:
                Instruction(s) describing desired voice/style. Empty string is allowed (treated as no instruction).
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`, 
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)
        """
        if self.model.tts_model_type != "voice_design":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_voice_design, Please check Model Card or Readme for more details."
            )
        
        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        instructs = self._ensure_list(instruct)

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        if not (len(texts) == len(languages) == len(instructs)):
            raise ValueError(f"Batch size mismatch: text={len(texts)}, language={len(languages)}, instruct={len(instructs)}")

        self._validate_languages(languages)

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        instruct_ids: List[Optional[torch.Tensor]] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs

    # custom voice model
    @torch.no_grad()
    def generate_custom_voice(
        self,
        text: Union[str, List[str]],
        speaker: Union[str, List[str]],
        language: Union[str, List[str]] = None,
        instruct: Optional[Union[str, List[str]]] = None,
        non_streaming_mode: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with the CustomVoice model using a predefined speaker id, optionally controlled by instruction text.

        Args:
            text:
                Text(s) to synthesize.
            language:
                Language(s) for each sample.
            speaker:
                Speaker name(s). Will be validated against `model.get_supported_speakers()` (case-insensitive).
            instruct:
                Optional instruction(s). If None, treated as empty (no instruction).
            non_streaming_mode:
                Using non-streaming text input, this option currently only simulates streaming text input when set to `false`, 
                rather than enabling true streaming input or streaming generation.
            do_sample:
                Whether to use sampling, recommended to be set to `true` for most use cases.
            top_k:
                Top-k sampling parameter.
            top_p:
                Top-p sampling parameter.
            temperature:
                Sampling temperature; higher => more random.
            repetition_penalty:
                Penalty to reduce repeated tokens/codes.
            subtalker_dosample:
                Sampling switch for the sub-talker (only valid for qwen3-tts-tokenizer-v2) if applicable.
            subtalker_top_k:
                Top-k for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_top_p:
                Top-p for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            subtalker_temperature:
                Temperature for sub-talker sampling (only valid for qwen3-tts-tokenizer-v2).
            max_new_tokens:
                Maximum number of new codec tokens to generate.
            **kwargs:
                Any other keyword arguments supported by HuggingFace Transformers `generate()` can be passed.
                They will be forwarded to the underlying `Qwen3TTSForConditionalGeneration.generate(...)`.

        Returns:
            Tuple[List[np.ndarray], int]:
                (wavs, sample_rate)

        Raises:
            ValueError:
                If any speaker/language is unsupported or batch sizes mismatch.
        """
        if self.model.tts_model_type != "custom_voice":
            raise ValueError(
                f"model with \ntokenizer_type: {self.model.tokenizer_type}\n"
                f"tts_model_size: {self.model.tts_model_size}\n"
                f"tts_model_type: {self.model.tts_model_type}\n"
                "does not support generate_custom_voice, Please check Model Card or Readme for more details."
            )

        texts = self._ensure_list(text)
        languages = self._ensure_list(language) if isinstance(language, list) else ([language] * len(texts) if language is not None else ["Auto"] * len(texts))
        speakers = self._ensure_list(speaker)
        if self.model.tts_model_size in "0b6": # for 0b6 model, instruct is not supported
            instruct = None
        instructs = self._ensure_list(instruct) if isinstance(instruct, list) else ([instruct] * len(texts) if instruct is not None else [""] * len(texts))

        if len(languages) == 1 and len(texts) > 1:
            languages = languages * len(texts)
        if len(speakers) == 1 and len(texts) > 1:
            speakers = speakers * len(texts)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)

        if not (len(texts) == len(languages) == len(speakers) == len(instructs)):
            raise ValueError(
                f"Batch size mismatch: text={len(texts)}, language={len(languages)}, speaker={len(speakers)}, instruct={len(instructs)}"
            )

        self._validate_languages(languages)
        self._validate_speakers(speakers)

        input_ids = self._tokenize_texts([self._build_assistant_text(t) for t in texts])

        instruct_ids: List[Optional[torch.Tensor]] = []
        for ins in instructs:
            if ins is None or ins == "":
                instruct_ids.append(None)
            else:
                instruct_ids.append(self._tokenize_texts([self._build_instruct_text(ins)])[0])

        gen_kwargs = self._merge_generate_kwargs(**kwargs)

        talker_codes_list, _ = self.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=languages,
            speakers=speakers,
            non_streaming_mode=non_streaming_mode,
            **gen_kwargs,
        )

        wavs, fs = self.model.speech_tokenizer.decode([{"audio_codes": c} for c in talker_codes_list])
        return wavs, fs


    def get_supported_speakers(self) -> Optional[List[str]]:
        """
        List supported speaker names for the current model.

        This is a convenience wrapper around `model.get_supported_speakers()`.
        If the underlying model does not expose speaker constraints (returns None),
        this method also returns None.

        Returns:
            Optional[List[str]]:
                - A sorted list of supported speaker names (lowercased), if available.
                - None if the model does not provide supported speakers.
        """
        supported = self._supported_speakers_set()
        if supported is None:
            return None
        return sorted(supported)


    def get_supported_languages(self) -> Optional[List[str]]:
        """
        List supported language names for the current model.

        This is a convenience wrapper around `model.get_supported_languages()`.
        If the underlying model does not expose language constraints (returns None),
        this method also returns None.

        Returns:
            Optional[List[str]]:
                - A sorted list of supported language names (lowercased), if available.
                - None if the model does not provide supported languages.
        """
        supported = self._supported_languages_set()
        if supported is None:
            return None
        return sorted(supported)
