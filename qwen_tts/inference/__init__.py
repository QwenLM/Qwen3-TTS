# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3 TTS inference module."""

# Note: Imports are lazy to avoid circular import issues.
# Import the modules directly when needed.

__all__ = [
    "Qwen3TTSTokenizer",
    "Qwen3TTSTokenizerStandalone",
]


def __getattr__(name):
    """Lazy import to avoid circular imports."""
    if name == "Qwen3TTSTokenizer":
        from .qwen3_tts_tokenizer import Qwen3TTSTokenizer
        return Qwen3TTSTokenizer
    
    if name == "Qwen3TTSTokenizerStandalone":
        from .qwen3_tts_tokenizer_standalone import Qwen3TTSTokenizerStandalone
        return Qwen3TTSTokenizerStandalone
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
