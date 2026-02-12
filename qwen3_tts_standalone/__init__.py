# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS Standalone Implementation

This package provides a fully standalone implementation of Qwen3-TTS that
minimizes dependencies on the transformers library.

Main components:
- TTS: The main text-to-speech model
- Talker: Generates audio codec tokens from text embeddings
- CodePredictor: Predicts higher codebook layers
- SpeechTokenizer: Encodes/decodes audio to/from discrete tokens

Usage:
    from qwen3_tts_standalone import Qwen3TTSModel
    
    model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-0.5B")
    audio, sample_rate = model.generate("Hello world!", speaker="Chelsie")
"""

# Core models
from .tts import TTS
from .talker import Talker
from .code_predictor import CodePredictor
from .speaker_encoder import SpeakerEncoder

# Configuration (new names)
from .configuration import (
    TTSConfig,
    TalkerConfig,
    CodePredictorConfig,
    SpeakerEncoderConfig,
    BaseConfig,
)

# Tokenizer (new names)
from .tokenizer import (
    SpeechTokenizer,
    SpeechTokenizerConfig,
    SpeechTokenizerModel,
)

# Processor
from .processor import Processor

# High-level inference API
from .inference import Qwen3TTSModel

# Simplified voice cloner (ICL-only)
from .voice_cloner import VoiceCloner

# Base model
from .base_model import BaseModel

__all__ = [
    # Models
    "TTS",
    "Talker",
    "CodePredictor",
    "SpeakerEncoder",
    # Configuration
    "TTSConfig",
    "TalkerConfig",
    "CodePredictorConfig",
    "SpeakerEncoderConfig",
    "BaseConfig",
    # Tokenizer
    "SpeechTokenizer",
    "SpeechTokenizerConfig",
    "SpeechTokenizerModel",
    # Processor
    "Processor",
    # Inference
    "Qwen3TTSModel",
    # Simplified voice cloner
    "VoiceCloner",
    # Base
    "BaseModel",
]

__version__ = "0.1.0"
