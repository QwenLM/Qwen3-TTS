"""
Model manager singleton for Qwen3-TTS inference.

Provides lazy loading of the Qwen3-TTS model with thread-safe singleton pattern.
Configured for DGX Spark with FlashInfer attention backend.
"""

import io
import threading
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


class ModelManager:
    """Thread-safe singleton for managing Qwen3-TTS model lifecycle."""

    _instance: Optional["ModelManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._model: Optional[Qwen3TTSModel] = None
        self._model_lock: threading.Lock = threading.Lock()
        self._initialized = True

    @property
    def model_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None

    def load_model(self) -> Qwen3TTSModel:
        """
        Load the Qwen3-TTS model with FlashInfer backend.

        Uses lazy loading pattern - model is loaded on first call.
        Subsequent calls return the cached model instance.

        Returns:
            Qwen3TTSModel: The loaded TTS model wrapper.
        """
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = Qwen3TTSModel.from_pretrained(
                        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                        device_map="cuda:0",
                        dtype=torch.bfloat16,
                        attn_implementation="flashinfer",
                    )
        return self._model

    def get_model(self) -> Qwen3TTSModel:
        """
        Get the loaded model, loading it if necessary.

        Returns:
            Qwen3TTSModel: The loaded TTS model wrapper.
        """
        return self.load_model()

    def generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech audio from text using custom voice synthesis.

        Args:
            text: The text to synthesize.
            speaker: Speaker name (e.g., 'Vivian', 'Ryan').
            language: Language code ('Chinese', 'English', or 'Auto').
            instruct: Optional style instruction for synthesis.

        Returns:
            Tuple of (audio_array, sample_rate) where audio_array is a
            float32 numpy array of the waveform.
        """
        model = self.get_model()

        kwargs = {}
        if instruct is not None:
            kwargs["instruct"] = instruct

        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            **kwargs,
        )

        return wavs[0], sample_rate

    def audio_to_wav_bytes(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """
        Convert audio numpy array to WAV bytes.

        Args:
            audio: Float32 numpy array of audio waveform.
            sample_rate: Audio sample rate in Hz.

        Returns:
            Binary WAV data.
        """
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()

    def get_supported_speakers(self) -> List[str]:
        """
        Get list of supported speaker names.

        Returns:
            List of speaker name strings.
        """
        model = self.get_model()
        speakers = model.get_supported_speakers()
        if speakers is None:
            return []
        return list(speakers)

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.

        Returns:
            List of language code strings.
        """
        model = self.get_model()
        languages = model.get_supported_languages()
        if languages is None:
            return []
        return list(languages)


def get_model_manager() -> ModelManager:
    """
    Get the global ModelManager singleton instance.

    Returns:
        ModelManager: The singleton instance.
    """
    return ModelManager()
