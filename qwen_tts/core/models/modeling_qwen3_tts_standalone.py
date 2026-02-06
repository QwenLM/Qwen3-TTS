# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""
PyTorch Qwen3TTS standalone model.

This module re-exports all components from the reorganized standalone modules for
backward compatibility. New code should import directly from:
- speaker_encoder_standalone: Speaker encoder and mel spectrogram
- layers_standalone: Shared layers (RMSNorm, RoPE, Attention, Decoder layers)
- talker_base_standalone: Base Talker transformer model
- code_predictor_standalone: CodePredictor for higher codebook prediction
- talker_standalone: Talker for first codebook generation
- tts_standalone: TTS top-level model
"""

import logging

# Re-export from utils
from .utils import download_weights_from_hf_specific

# Re-export from speaker_encoder_standalone
from .speaker_encoder_standalone import (
    TimeDelayNetBlock,
    Res2NetBlock,
    SqueezeExcitationBlock,
    AttentiveStatisticsPooling,
    SqueezeExcitationRes2NetBlock,
    Qwen3TTSSpeakerEncoderStandalone,
    mel_spectrogram,
    dynamic_range_compression_torch,
)

# Re-export from layers_standalone
from .layers_standalone import (
    Qwen3TTSRMSNormStandalone,
    Qwen3TTSRotaryEmbeddingStandalone,
    Qwen3TTSTalkerRotaryEmbeddingStandalone,
    rotate_half,
    apply_rotary_pos_emb,
    apply_multimodal_rotary_pos_emb,
    eager_attention_forward,
    Qwen3TTSAttentionStandalone,
    Qwen3TTSTalkerAttentionStandalone,
    Qwen3TTSTalkerTextMLPStandalone,
    Qwen3TTSTalkerResizeMLPStandalone,
    Qwen3TTSDecoderLayerStandalone,
    Qwen3TTSTalkerDecoderLayerStandalone,
    ROPE_INIT_FUNCTIONS_EXTENDED,
)

# Re-export from talker_base_standalone
from .talker_base_standalone import (
    Qwen3TTSTalkerTextPreTrainedModelStandalone,
    Qwen3TTSTalkerModelStandalone,
)

# Re-export from base_model_standalone
from .base_model_standalone import StandalonePreTrainedModel

# Re-export from configuration
from .configuration_qwen3_tts_standalone import (
    Qwen3TTSConfigStandalone,
    Qwen3TTSSpeakerEncoderConfigStandalone,
    Qwen3TTSTalkerCodePredictorConfigStandalone,
    Qwen3TTSTalkerConfigStandalone,
)

# Re-export standalone utilities
from .standalone import (
    ACT2FN,
    ALL_ATTENTION_FUNCTIONS,
    BaseModelOutputWithPast,
    Cache,
    CausalLMOutputWithPast,
    DynamicCache,
    GenerateOutput,
    GenerationMixin,
    ModelOutput,
    ROPE_INIT_FUNCTIONS,
    cached_file,
    can_return_tuple,
    create_causal_mask,
    create_sliding_window_causal_mask,
    dynamic_rope_update,
    repeat_kv,
)

logger = logging.getLogger(__name__)


# Base pretrained model class for TTS
class Qwen3TTSPreTrainedModelStandalone(StandalonePreTrainedModel):
    """Base class for all Qwen3TTS standalone models."""
    
    config_class = Qwen3TTSConfigStandalone
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3TTSDecoderLayerStandalone"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False
    _supports_attention_backend = True

    def _init_weights(self, module):
        from torch import nn
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()


# Backward compatibility aliases for old class names
# These point to the new simplified implementations
from .tts_standalone import TTS as Qwen3TTSForConditionalGenerationStandalone
from .talker_standalone import Talker as Qwen3TTSTalkerForConditionalGenerationStandalone


__all__ = [
    # Speaker encoder
    "TimeDelayNetBlock",
    "Res2NetBlock",
    "SqueezeExcitationBlock",
    "AttentiveStatisticsPooling",
    "SqueezeExcitationRes2NetBlock",
    "Qwen3TTSSpeakerEncoderStandalone",
    "mel_spectrogram",
    "dynamic_range_compression_torch",
    # Layers
    "Qwen3TTSRMSNormStandalone",
    "Qwen3TTSRotaryEmbeddingStandalone",
    "Qwen3TTSTalkerRotaryEmbeddingStandalone",
    "rotate_half",
    "apply_rotary_pos_emb",
    "apply_multimodal_rotary_pos_emb",
    "eager_attention_forward",
    "Qwen3TTSAttentionStandalone",
    "Qwen3TTSTalkerAttentionStandalone",
    "Qwen3TTSTalkerTextMLPStandalone",
    "Qwen3TTSTalkerResizeMLPStandalone",
    "Qwen3TTSDecoderLayerStandalone",
    "Qwen3TTSTalkerDecoderLayerStandalone",
    # Talker base
    "Qwen3TTSTalkerTextPreTrainedModelStandalone",
    "Qwen3TTSTalkerModelStandalone",
    # Base classes
    "Qwen3TTSPreTrainedModelStandalone",
    "StandalonePreTrainedModel",
    # High-level models (backward compat aliases)
    "Qwen3TTSForConditionalGenerationStandalone",
    "Qwen3TTSTalkerForConditionalGenerationStandalone",
    # Utilities
    "download_weights_from_hf_specific",
    "ROPE_INIT_FUNCTIONS_EXTENDED",
]
