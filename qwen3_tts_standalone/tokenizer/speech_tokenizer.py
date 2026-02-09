# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone Qwen3 TTS Tokenizer (12Hz) wrapper.

This provides a simplified interface for the 12Hz tokenizer that minimizes
transformers dependencies. The encoder still uses transformers (MimiModel),
but the decoder is fully standalone.
"""

import base64
import io
import os
import urllib.request
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence

from .config import SpeechTokenizerConfig
from .model import (
    SpeechTokenizerModel,
    SpeechTokenizerEncoderOutput,
    SpeechTokenizerDecoderOutput,
)


AudioInput = Union[
    str,  # wav path, or base64 string
    np.ndarray,  # 1-D float array
    List[str],
    List[np.ndarray],
]


class SpeechTokenizer:
    """
    Standalone wrapper for Qwen3 TTS 12Hz Tokenizer.

    This class provides encoding and decoding functionality with minimal
    transformers dependencies. The encoder uses transformers (MimiModel),
    but the decoder is fully standalone.
    
    Usage:
        tokenizer = Qwen3TTSTokenizerStandalone.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz")
        
        # Encode audio to codes
        encoded = tokenizer.encode("audio.wav")
        
        # Decode codes back to audio
        wavs, sr = tokenizer.decode(encoded)
    """

    def __init__(self):
        self.decoder_model = None
        self.encoder_model = None  # transformers-based
        self.feature_extractor = None
        self.config = None
        self.device = None
        self.dtype = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "Qwen3TTSSpeechTokenizer":
        """
        Load tokenizer from pretrained weights.

        Args:
            pretrained_model_name_or_path: HuggingFace repo id or local directory
            device: Target device (e.g., "cuda:0", "cpu")
            dtype: Target dtype (e.g., torch.float16, torch.bfloat16)
            **kwargs: Additional arguments

        Returns:
            Initialized tokenizer instance
        """
        try:
            from transformers import AutoFeatureExtractor, AutoConfig, AutoModel
        except ImportError:
            raise ImportError(
                "The speech tokenizer encoder requires the 'transformers' library. "
                "Install it with: pip install transformers"
            )
        
        # Import the encoder model from the original qwen_tts package
        # This is required for encoding audio to codes
        try:
            from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
                Qwen3TTSTokenizerV2Config,
            )
            from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
                Qwen3TTSTokenizerV2Model,
            )
        except ImportError:
            raise ImportError(
                "Could not import encoder model from qwen_tts. "
                "The speech tokenizer encoder is not yet fully standalone."
            )
        
        inst = cls()
        
        # Set device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        inst.device = device
        inst.dtype = dtype or torch.float32
        
        # Register with AutoConfig/AutoModel
        AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
        AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)
        
        # Load feature extractor (for preprocessing audio)
        inst.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        
        # Load the full original model for encoder
        original_model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        inst.encoder_model = original_model.encoder
        inst.encoder_model = inst.encoder_model.to(device).to(inst.dtype)
        
        # Load config and create standalone decoder
        inst.config = SpeechTokenizerConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Create standalone decoder model and load weights
        inst.decoder_model = SpeechTokenizerModel(inst.config)
        
        # Copy decoder weights from original model
        decoder_state_dict = original_model.decoder.state_dict()
        inst.decoder_model.decoder.load_state_dict(decoder_state_dict, strict=True)
        inst.decoder_model = inst.decoder_model.to(device).to(inst.dtype)
        
        return inst

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

    def load_audio(self, x: str, target_sr: int) -> np.ndarray:
        """Load audio from file path or base64 string."""
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

        if sr != target_sr:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

        return audio.astype(np.float32)

    def _normalize_audio_inputs(
        self,
        audios: AudioInput,
        sr: Optional[int],
    ) -> List[np.ndarray]:
        """Normalize audio inputs to list of numpy arrays."""
        target_sr = int(self.feature_extractor.sampling_rate)

        if isinstance(audios, (str, np.ndarray)):
            audios = [audios]

        if len(audios) == 0:
            return []

        if isinstance(audios[0], str):
            return [self.load_audio(x, target_sr=target_sr) for x in audios]

        if sr is None:
            raise ValueError("For numpy waveform input, you must provide `sr`.")

        out: List[np.ndarray] = []
        for a in audios:
            if not isinstance(a, np.ndarray):
                raise TypeError("Mixed input types are not supported.")
            if a.ndim > 1:
                a = np.mean(a, axis=-1)
            if int(sr) != target_sr:
                a = librosa.resample(y=a.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)
            out.append(a.astype(np.float32))
        return out

    def encode(
        self,
        audios: AudioInput,
        sr: Optional[int] = None,
        return_dict: bool = True,
    ) -> Union[SpeechTokenizerEncoderOutput, Tuple]:
        """
        Encode audio to discrete codes.

        Args:
            audios: Audio input (path, base64, numpy array, or list)
            sr: Sample rate for numpy input
            return_dict: Whether to return a dataclass

        Returns:
            Encoded audio codes
        """
        wavs = self._normalize_audio_inputs(audios, sr=sr)

        inputs = self.feature_extractor(
            raw_audio=wavs,
            sampling_rate=int(self.feature_extractor.sampling_rate),
            return_tensors="pt",
        )
        inputs = inputs.to(self.device).to(self.dtype)

        with torch.inference_mode():
            # Use the transformers-based encoder
            input_values = inputs["input_values"].squeeze(1).unsqueeze(1)
            encoded_frames = self.encoder_model.encode(input_values=input_values, return_dict=True)
            
            audio_codes = encoded_frames.audio_codes[:, :self.config.encoder_valid_num_quantizers]
            padding_mask = inputs["padding_mask"].squeeze(1)
            
            audio_codes = [
                code[..., :-(-mask.sum() // self.config.encode_downsample_rate)].transpose(0, 1) 
                for code, mask in zip(audio_codes, padding_mask)
            ]

        if not return_dict:
            return (audio_codes,)

        return SpeechTokenizerEncoderOutput(audio_codes=audio_codes)

    def decode(
        self,
        encoded,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Decode codes back to waveform.

        Args:
            encoded: Encoded output from encode() or dict with audio_codes

        Returns:
            Tuple of (list of waveforms, sample rate)
        """
        def _to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                return x
            x = np.asarray(x)
            t = torch.from_numpy(x)
            if dtype is not None:
                t = t.to(dtype)
            return t

        # Normalize encoded to audio_codes_list
        if hasattr(encoded, "audio_codes"):
            audio_codes_list = encoded.audio_codes
        elif isinstance(encoded, dict):
            audio_codes_list = encoded["audio_codes"]
        elif isinstance(encoded, list):
            audio_codes_list = [e["audio_codes"] for e in encoded]
        else:
            raise TypeError("`encoded` must be an encode output, a dict, or a list of dicts.")

        # Prepare codes tensor
        if isinstance(audio_codes_list, torch.Tensor):
            t = audio_codes_list
            if t.dim() == 2:
                t = t.unsqueeze(0)
            audio_codes_padded = t.to(self.device)
        else:
            audio_codes_list = [_to_tensor(c, dtype=torch.long) for c in audio_codes_list]
            audio_codes_padded = pad_sequence(audio_codes_list, batch_first=True, padding_value=0).to(self.device)

        with torch.inference_mode():
            dec = self.decoder_model.decode(audio_codes_padded, return_dict=True)
            wav_tensors = dec.audio_values

        wavs = [w.to(torch.float32).detach().cpu().numpy() for w in wav_tensors]
        return wavs, int(self.decoder_model.get_output_sample_rate())

    def get_model_type(self) -> str:
        """Get model type string."""
        return self.config.model_type

    def get_input_sample_rate(self) -> int:
        """Get input sample rate for encoding."""
        return int(self.config.input_sample_rate)

    def get_output_sample_rate(self) -> int:
        """Get output sample rate for decoded audio."""
        return int(self.config.output_sample_rate)

    def get_encode_downsample_rate(self) -> int:
        """Get encoder downsample rate."""
        return int(self.config.encode_downsample_rate)

    def get_decode_upsample_rate(self) -> int:
        """Get decoder upsample rate."""
        return int(self.config.decode_upsample_rate)


# Backward compatibility alias
Qwen3TTSSpeechTokenizer = SpeechTokenizer


__all__ = [
    "SpeechTokenizer",
    # Backward compatibility
    "Qwen3TTSSpeechTokenizer",
]
