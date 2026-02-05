# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone Qwen3TTSTokenizerV2 (12Hz tokenizer) model.

This module provides transformers-free model implementations.
"""

import logging
import math
import os
from dataclasses import dataclass
from functools import wraps
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

from .configuration_qwen3_tts_tokenizer_v2_standalone import (
    Qwen3TTSTokenizerV2ConfigStandalone,
    Qwen3TTSTokenizerV2DecoderConfigStandalone,
)
from ..models.standalone import (
    ACT2FN,
    ALL_ATTENTION_FUNCTIONS,
    Cache,
    DynamicCache,
    ROPE_INIT_FUNCTIONS,
    create_causal_mask,
    create_sliding_window_causal_mask,
    dynamic_rope_update,
    repeat_kv,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Output Data Classes
# =============================================================================

@dataclass
class Qwen3TTSTokenizerV2EncoderOutputStandalone:
    """Output from the encoder."""
    audio_codes: List[torch.LongTensor] = None


@dataclass
class Qwen3TTSTokenizerV2DecoderOutputStandalone:
    """Output from the decoder."""
    audio_values: List[torch.FloatTensor] = None


@dataclass
class BaseModelOutputWithPastStandalone:
    """Base model output with past key values."""
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# =============================================================================
# RoPE Utilities
# =============================================================================

def _default_rope_init_fn(config, device=None):
    """Default RoPE initialization function."""
    base = config.rope_theta
    dim = config.head_dim
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    attention_scaling = 1.0
    return inv_freq, attention_scaling


ROPE_INIT_FUNCTIONS_EXTENDED = {**ROPE_INIT_FUNCTIONS, "default": _default_rope_init_fn}


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# Attention Utilities
# =============================================================================

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """Eager attention forward pass."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# =============================================================================
# Convolutional Layers
# =============================================================================

class Qwen3TTSTokenizerV2CausalConvNetStandalone(nn.Module):
    """Causal 1D convolution with proper padding."""
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        groups=1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride

    def _get_extra_padding_for_conv1d(self, hidden_state: torch.Tensor) -> int:
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def forward(self, hidden_state):
        extra_padding = self._get_extra_padding_for_conv1d(hidden_state)
        hidden_state = F.pad(hidden_state, (self.padding, int(extra_padding)), mode="constant", value=0)
        return self.conv(hidden_state).contiguous()


class Qwen3TTSTokenizerV2CausalTransConvNetStandalone(nn.Module):
    """Causal transposed convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        pad = kernel_size - stride
        self.left_pad = math.ceil(pad)
        self.right_pad = pad = self.left_pad

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = hidden_state[..., self.left_pad : hidden_state.shape[-1] - self.right_pad]
        return hidden_state.contiguous()


class Qwen3TTSTokenizerV2ConvNeXtBlockStandalone(nn.Module):
    """ConvNeXt block."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = Qwen3TTSTokenizerV2CausalConvNetStandalone(
            dim, dim, kernel_size=7, groups=dim, dilation=1,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, hidden_states):
        input = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = self.gamma * hidden_states
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = input + hidden_states
        return hidden_states


# =============================================================================
# Transformer Components
# =============================================================================

class Qwen3TTSTokenizerV2DecoderRotaryEmbeddingStandalone(nn.Module):
    """Rotary position embeddings."""
    
    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfigStandalone, device=None):
        super().__init__()
        self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS_EXTENDED[self.rope_type]
        
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3TTSTokenizerV2DecoderRMSNormStandalone(nn.Module):
    """RMS normalization."""
    
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3TTSTokenizerV2DecoderLayerScaleStandalone(nn.Module):
    """Layer scale for transformer blocks."""
    
    def __init__(self, config):
        super().__init__()
        channels = config.hidden_size
        initial_scale = config.layer_scale_initial_scale
        self.scale = nn.Parameter(torch.full((channels,), initial_scale, requires_grad=True))

    def forward(self, x: torch.Tensor):
        return self.scale * x


class Qwen3TTSTokenizerV2DecoderMlpStandalone(nn.Module):
    """MLP layer."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3TTSTokenizerV2DecoderAttentionStandalone(nn.Module):
    """Multi-head attention."""
    
    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfigStandalone, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.sliding_window = config.sliding_window

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if getattr(self.config, "_attn_implementation", "eager") != "eager":
            attn_impl = self.config._attn_implementation
            if attn_impl in ALL_ATTENTION_FUNCTIONS:
                attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3TTSTokenizerV2DecoderTransformerLayerStandalone(nn.Module):
    """Transformer decoder layer."""
    
    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfigStandalone, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3TTSTokenizerV2DecoderAttentionStandalone(config, layer_idx)
        self.mlp = Qwen3TTSTokenizerV2DecoderMlpStandalone(config)
        self.input_layernorm = Qwen3TTSTokenizerV2DecoderRMSNormStandalone(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3TTSTokenizerV2DecoderRMSNormStandalone(config.hidden_size, config.rms_norm_eps)
        self.self_attn_layer_scale = Qwen3TTSTokenizerV2DecoderLayerScaleStandalone(config)
        self.mlp_layer_scale = Qwen3TTSTokenizerV2DecoderLayerScaleStandalone(config)
        self.attention_type = "sliding_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + self.self_attn_layer_scale(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_layer_scale(hidden_states)

        return hidden_states


class Qwen3TTSTokenizerV2DecoderTransformerModelStandalone(nn.Module):
    """Transformer model for the decoder."""
    
    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfigStandalone):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3TTSTokenizerV2DecoderTransformerLayerStandalone(config, layer_idx) 
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3TTSTokenizerV2DecoderRMSNormStandalone(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3TTSTokenizerV2DecoderRotaryEmbeddingStandalone(config=config)
        self.has_sliding_layers = "sliding_attention" in config.layer_types
        self.window_size = config.sliding_window

        self.input_proj = nn.Linear(config.latent_dim, config.hidden_size)
        self.output_proj = nn.Linear(config.hidden_size, config.latent_dim)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPastStandalone:
        inputs_embeds = self.input_proj(inputs_embeds)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create causal masks
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        hidden_states = self.output_proj(hidden_states)
        
        return BaseModelOutputWithPastStandalone(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# =============================================================================
# Vocoder Components
# =============================================================================

class SnakeBetaStandalone(nn.Module):
    """Snake activation with beta parameter."""
    
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)
        self.no_div_by_zero = 0.000000001

    def forward(self, hidden_states):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )
        return hidden_states


class Qwen3TTSTokenizerV2DecoderDecoderResidualUnitStandalone(nn.Module):
    """Residual unit for the vocoder decoder."""
    
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        self.act1 = SnakeBetaStandalone(dim)
        self.conv1 = Qwen3TTSTokenizerV2CausalConvNetStandalone(dim, dim, kernel_size=7, dilation=dilation)
        self.act2 = SnakeBetaStandalone(dim)
        self.conv2 = Qwen3TTSTokenizerV2CausalConvNetStandalone(dim, dim, kernel_size=1)

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.act1(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.act2(hidden_state)
        hidden_state = self.conv2(hidden_state)
        return hidden_state + residual


class Qwen3TTSTokenizerV2DecoderDecoderBlockStandalone(nn.Module):
    """Decoder block for upsampling."""
    
    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfigStandalone, layer_idx):
        super().__init__()
        in_dim = config.decoder_dim // 2**layer_idx
        out_dim = config.decoder_dim // 2 ** (layer_idx + 1)
        upsample_rate = config.upsample_rates[layer_idx]

        block = [
            SnakeBetaStandalone(in_dim),
            Qwen3TTSTokenizerV2CausalTransConvNetStandalone(in_dim, out_dim, 2 * upsample_rate, upsample_rate),
        ]

        for dilation in (1, 3, 9):
            block.append(Qwen3TTSTokenizerV2DecoderDecoderResidualUnitStandalone(out_dim, dilation))

        self.block = nn.ModuleList(block)

    def forward(self, hidden):
        for block in self.block:
            hidden = block(hidden)
        return hidden


# =============================================================================
# Vector Quantization
# =============================================================================

class EuclideanCodebookStandalone(nn.Module):
    """Euclidean codebook for vector quantization."""
    
    def __init__(self, dim: int, codebook_size: int, epsilon: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon
        self.cluster_usage = nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = nn.Parameter(torch.zeros(codebook_size, dim))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]
        quantized = F.embedding(codes, embedding)
        return quantized


class VectorQuantizationStandalone(nn.Module):
    """Single-level vector quantization."""
    
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim

        requires_projection = codebook_dim != dim
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.epsilon = epsilon
        self._codebook = EuclideanCodebookStandalone(
            dim=codebook_dim, codebook_size=codebook_size, epsilon=epsilon
        )
        self.codebook_size = codebook_size

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self._codebook.decode(codes)
        quantized = self.project_out(quantized)
        quantized = quantized.transpose(1, 2)
        return quantized


class ResidualVectorQuantizationStandalone(nn.Module):
    """Residual vector quantization."""
    
    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantizationStandalone(**kwargs) for _ in range(num_quantizers)]
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = torch.zeros([1], device=codes.device)[0]
        for idx, layer_codes in enumerate(codes):
            layer = self.layers[idx]
            quantized = quantized + layer.decode(layer_codes)
        return quantized


class ResidualVectorQuantizerStandalone(nn.Module):
    """Residual vector quantizer with projections."""
    
    def __init__(
        self,
        dimension: int = 128,
        input_dimension: Optional[int] = None,
        output_dimension: Optional[int] = None,
        n_q: int = 8,
        bins: int = 1024,
        force_projection: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_q = n_q
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.bins = bins
        
        if self.input_dimension == self.dimension and not force_projection:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Conv1d(self.input_dimension, self.dimension, 1, bias=False)
        
        if self.output_dimension == self.dimension and not force_projection:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Conv1d(self.dimension, self.output_dimension, 1, bias=False)
        
        self.vq = ResidualVectorQuantizationStandalone(
            dim=self.dimension, codebook_size=self.bins, num_quantizers=self.n_q
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        quantized = self.output_proj(quantized)
        return quantized


class SplitResidualVectorQuantizerStandalone(nn.Module):
    """Split residual vector quantizer with semantic and acoustic parts."""
    
    def __init__(self, *, n_q: int = 8, n_q_semantic: int = 1, **kwargs):
        super().__init__()
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        
        self.rvq_first = ResidualVectorQuantizerStandalone(
            n_q=n_q_semantic, force_projection=True, **kwargs
        )
        self.rvq_rest = ResidualVectorQuantizerStandalone(
            n_q=n_q - n_q_semantic, force_projection=True, **kwargs
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.rvq_first.decode(codes[:, : self.n_q_semantic])
        if codes.shape[1] > self.n_q_semantic:
            quantized = quantized + self.rvq_rest.decode(codes[:, self.n_q_semantic :])
        return quantized


# =============================================================================
# Main Decoder
# =============================================================================

class Qwen3TTSTokenizerV2DecoderStandalone(nn.Module):
    """Main decoder for the 12Hz tokenizer."""
    
    def __init__(self, config: Qwen3TTSTokenizerV2DecoderConfigStandalone):
        super().__init__()
        self.config = config
        self.total_upsample = int(np.prod(list(config.upsample_rates) + list(config.upsampling_ratios)))
        
        self.pre_transformer = Qwen3TTSTokenizerV2DecoderTransformerModelStandalone(config)
        
        self.quantizer = SplitResidualVectorQuantizerStandalone(
            dimension=config.codebook_dim // 2,
            n_q=config.num_quantizers,
            n_q_semantic=1,
            bins=config.codebook_size,
            input_dimension=config.codebook_dim,
            output_dimension=config.codebook_dim,
        )

        self.pre_conv = Qwen3TTSTokenizerV2CausalConvNetStandalone(
            config.codebook_dim, config.latent_dim, kernel_size=3,
        )

        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList([
                    Qwen3TTSTokenizerV2CausalTransConvNetStandalone(config.latent_dim, config.latent_dim, factor, factor),
                    Qwen3TTSTokenizerV2ConvNeXtBlockStandalone(config.latent_dim),
                ])
            )
        self.upsample = nn.ModuleList(upsample)

        decoder = [Qwen3TTSTokenizerV2CausalConvNetStandalone(config.latent_dim, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3TTSTokenizerV2DecoderDecoderBlockStandalone(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [
            SnakeBetaStandalone(output_dim),
            Qwen3TTSTokenizerV2CausalConvNetStandalone(output_dim, 1, 7),
        ]
        self.decoder = nn.ModuleList(decoder)

    def forward(self, codes):
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layers of codes, got {codes.shape[1]}")

        hidden = self.quantizer.decode(codes)
        hidden = self.pre_conv(hidden).transpose(1, 2)

        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        
        return wav.clamp(min=-1, max=1)

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        """Decode in chunks for memory efficiency."""
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)


# =============================================================================
# Main Model
# =============================================================================

class Qwen3TTSTokenizerV2ModelStandalone(nn.Module):
    """
    Standalone 12Hz tokenizer model (decode only).
    
    Note: This standalone version only supports decoding. For encoding,
    use the transformers-based MimiModel directly.
    """
    
    def __init__(self, config: Qwen3TTSTokenizerV2ConfigStandalone):
        super().__init__()
        self.config = config
        
        self.encoder_valid_num_quantizers = config.encoder_valid_num_quantizers
        self.input_sample_rate = config.input_sample_rate
        self.output_sample_rate = config.output_sample_rate
        self.decode_upsample_rate = config.decode_upsample_rate
        self.encode_downsample_rate = config.encode_downsample_rate

        self.decoder = Qwen3TTSTokenizerV2DecoderStandalone(config.decoder_config)
    
    def get_model_type(self) -> str:
        return self.config.model_type
    
    def get_input_sample_rate(self) -> int:
        return self.input_sample_rate
    
    def get_output_sample_rate(self) -> int:
        return self.output_sample_rate
    
    def get_encode_downsample_rate(self) -> int:
        return self.encode_downsample_rate
    
    def get_decode_upsample_rate(self) -> int:
        return self.decode_upsample_rate

    def decode(
        self,
        audio_codes: torch.Tensor,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Qwen3TTSTokenizerV2DecoderOutputStandalone]:
        """
        Decode audio codes to waveform.
        
        Args:
            audio_codes: Tensor of shape (batch_size, codes_length, num_quantizers)
            return_dict: Whether to return a dataclass
            
        Returns:
            Decoded audio waveforms
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        audio_values = self.decoder.chunked_decode(audio_codes.transpose(1, 2)).squeeze(1)

        audio_lengths = (audio_codes[..., 0] > 0).sum(1) * self.decode_upsample_rate
        audio_values = [a[:l] for a, l in zip(audio_values, audio_lengths)]

        if not return_dict:
            return (audio_values,)

        return Qwen3TTSTokenizerV2DecoderOutputStandalone(audio_values=audio_values)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> "Qwen3TTSTokenizerV2ModelStandalone":
        """
        Load model from pretrained weights.
        
        Args:
            pretrained_model_name_or_path: Path or HuggingFace Hub repo ID
            device: Target device
            dtype: Target dtype
            
        Returns:
            Loaded model
        """
        from huggingface_hub import hf_hub_download, snapshot_download
        import json
        
        # Determine if local or HuggingFace Hub
        if os.path.isdir(pretrained_model_name_or_path):
            model_dir = pretrained_model_name_or_path
        else:
            model_dir = snapshot_download(
                pretrained_model_name_or_path,
                allow_patterns=["*.json", "*.safetensors", "*.bin"],
            )
        
        # Load config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        config = Qwen3TTSTokenizerV2ConfigStandalone.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        model._load_weights(model_dir)
        
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)
        
        return model
    
    def _load_weights(self, model_dir: str) -> None:
        """Load weights from directory."""
        import glob
        
        # Try safetensors first
        safetensor_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
        if safetensor_files:
            from safetensors.torch import load_file
            state_dict = {}
            for f in safetensor_files:
                state_dict.update(load_file(f))
        else:
            # Fall back to PyTorch format
            bin_files = glob.glob(os.path.join(model_dir, "*.bin"))
            state_dict = {}
            for f in bin_files:
                state_dict.update(torch.load(f, map_location="cpu"))
        
        # Filter to decoder weights only and remap keys
        decoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("decoder."):
                new_key = key[len("decoder."):]
                decoder_state_dict[new_key] = value
        
        # Load decoder weights
        missing, unexpected = self.decoder.load_state_dict(decoder_state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys in decoder: {missing[:10]}...")
        if unexpected:
            logger.warning(f"Unexpected keys in decoder: {unexpected[:10]}...")


__all__ = [
    "Qwen3TTSTokenizerV2ModelStandalone",
    "Qwen3TTSTokenizerV2DecoderStandalone",
    "Qwen3TTSTokenizerV2EncoderOutputStandalone",
    "Qwen3TTSTokenizerV2DecoderOutputStandalone",
]
