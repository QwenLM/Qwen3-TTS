# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone utilities for Qwen3-TTS models.

This package provides minimal replacements for various transformers utilities,
enabling the standalone models to work without transformers dependencies.
"""

from .activations import ACT2FN, GELUActivation, NewGELUActivation, SiLUActivation
from .attention import (
    ALL_ATTENTION_FUNCTIONS,
    flash_attention_forward,
    repeat_kv,
    sdpa_attention_forward,
)
from .cache import Cache, DynamicCache
from .generation import GenerateOutput, GenerationMixin
from .masking import create_causal_mask, create_sliding_window_causal_mask
from .outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from .rope import (
    ROPE_INIT_FUNCTIONS,
    _compute_default_rope_parameters,
    _compute_dynamic_rope_parameters,
    _compute_linear_rope_parameters,
    dynamic_rope_update,
)
from .utils import cached_file, can_return_tuple

__all__ = [
    # Activations
    "ACT2FN",
    "SiLUActivation",
    "GELUActivation",
    "NewGELUActivation",
    # Attention
    "ALL_ATTENTION_FUNCTIONS",
    "repeat_kv",
    "sdpa_attention_forward",
    "flash_attention_forward",
    # Cache
    "Cache",
    "DynamicCache",
    # Generation
    "GenerateOutput",
    "GenerationMixin",
    # Masking
    "create_causal_mask",
    "create_sliding_window_causal_mask",
    # Outputs
    "ModelOutput",
    "BaseModelOutputWithPast",
    "CausalLMOutputWithPast",
    # RoPE
    "ROPE_INIT_FUNCTIONS",
    "_compute_default_rope_parameters",
    "_compute_linear_rope_parameters",
    "_compute_dynamic_rope_parameters",
    "dynamic_rope_update",
    # Utils
    "can_return_tuple",
    "cached_file",
]
