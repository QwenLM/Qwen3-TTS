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

# Base model
from .base_model import BaseModel

# Backward compatibility aliases
Qwen3TTSSpeakerEncoderStandalone = SpeakerEncoder
Qwen3TTSConfigStandalone = TTSConfig
Qwen3TTSTalkerConfigStandalone = TalkerConfig
Qwen3TTSTalkerCodePredictorConfigStandalone = CodePredictorConfig
Qwen3TTSSpeakerEncoderConfigStandalone = SpeakerEncoderConfig
Qwen3TTSSpeechTokenizer = SpeechTokenizer
Qwen3TTSSpeechTokenizerConfig = SpeechTokenizerConfig
Qwen3TTSSpeechTokenizerModel = SpeechTokenizerModel
Qwen3TTSProcessor = Processor
Qwen3TTSModelStandalone = Qwen3TTSModel
StandalonePreTrainedModel = BaseModel


__all__ = [
    # Models (new names)
    "TTS",
    "Talker",
    "CodePredictor",
    "SpeakerEncoder",
    # Configuration (new names)
    "TTSConfig",
    "TalkerConfig",
    "CodePredictorConfig",
    "SpeakerEncoderConfig",
    "BaseConfig",
    # Tokenizer (new names)
    "SpeechTokenizer",
    "SpeechTokenizerConfig",
    "SpeechTokenizerModel",
    # Processor
    "Processor",
    # Inference
    "Qwen3TTSModel",
    # Base
    "BaseModel",
    # Backward compatibility aliases
    "Qwen3TTSSpeakerEncoderStandalone",
    "Qwen3TTSConfigStandalone",
    "Qwen3TTSTalkerConfigStandalone",
    "Qwen3TTSTalkerCodePredictorConfigStandalone",
    "Qwen3TTSSpeakerEncoderConfigStandalone",
    "Qwen3TTSSpeechTokenizer",
    "Qwen3TTSSpeechTokenizerConfig",
    "Qwen3TTSSpeechTokenizerModel",
    "Qwen3TTSProcessor",
    "Qwen3TTSModelStandalone",
    "StandalonePreTrainedModel",
]

__version__ = "0.1.0"
