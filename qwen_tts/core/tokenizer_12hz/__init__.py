# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3 TTS 12Hz Tokenizer module."""

# Note: Imports are lazy to avoid circular import issues.
# Import the modules directly when needed.

__all__ = [
    # Original (transformers-based)
    "Qwen3TTSTokenizerV2Config",
    "Qwen3TTSTokenizerV2DecoderConfig",
    "Qwen3TTSTokenizerV2Model",
    "Qwen3TTSTokenizerV2PreTrainedModel",
    # Standalone
    "Qwen3TTSTokenizerV2ConfigStandalone",
    "Qwen3TTSTokenizerV2DecoderConfigStandalone",
    "MimiEncoderConfigStandalone",
    "convert_to_standalone_config",
    "Qwen3TTSTokenizerV2ModelStandalone",
    "Qwen3TTSTokenizerV2DecoderStandalone",
    "Qwen3TTSTokenizerV2EncoderOutputStandalone",
    "Qwen3TTSTokenizerV2DecoderOutputStandalone",
]


def __getattr__(name):
    """Lazy import to avoid circular imports."""
    if name in ("Qwen3TTSTokenizerV2Config", "Qwen3TTSTokenizerV2DecoderConfig"):
        from .configuration_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Config,
            Qwen3TTSTokenizerV2DecoderConfig,
        )
        return locals()[name]
    
    if name in ("Qwen3TTSTokenizerV2Model", "Qwen3TTSTokenizerV2PreTrainedModel"):
        from .modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Model,
            Qwen3TTSTokenizerV2PreTrainedModel,
        )
        return locals()[name]
    
    if name in (
        "Qwen3TTSTokenizerV2ConfigStandalone",
        "Qwen3TTSTokenizerV2DecoderConfigStandalone",
        "MimiEncoderConfigStandalone",
        "convert_to_standalone_config",
    ):
        from .configuration_qwen3_tts_tokenizer_v2_standalone import (
            Qwen3TTSTokenizerV2ConfigStandalone,
            Qwen3TTSTokenizerV2DecoderConfigStandalone,
            MimiEncoderConfigStandalone,
            convert_to_standalone_config,
        )
        return locals()[name]
    
    if name in (
        "Qwen3TTSTokenizerV2ModelStandalone",
        "Qwen3TTSTokenizerV2DecoderStandalone",
        "Qwen3TTSTokenizerV2EncoderOutputStandalone",
        "Qwen3TTSTokenizerV2DecoderOutputStandalone",
    ):
        from .modeling_qwen3_tts_tokenizer_v2_standalone import (
            Qwen3TTSTokenizerV2ModelStandalone,
            Qwen3TTSTokenizerV2DecoderStandalone,
            Qwen3TTSTokenizerV2EncoderOutputStandalone,
            Qwen3TTSTokenizerV2DecoderOutputStandalone,
        )
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
