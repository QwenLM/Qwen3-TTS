# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Speech tokenizer for Qwen3-TTS standalone.

This module provides the speech tokenizer (audio codec) for encoding
and decoding audio waveforms to/from discrete tokens.
"""

from .speech_tokenizer import SpeechTokenizer
from .config import SpeechTokenizerConfig, SpeechDecoderConfig, MimiEncoderConfig
from .model import SpeechTokenizerModel


__all__ = [
    "SpeechTokenizer",
    "SpeechTokenizerConfig",
    "SpeechDecoderConfig",
    "MimiEncoderConfig",
    "SpeechTokenizerModel",
]
