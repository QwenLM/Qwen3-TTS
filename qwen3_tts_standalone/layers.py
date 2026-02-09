# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shared layers for Qwen3-TTS standalone models.

This module contains the building blocks used by both the CodePredictor and Talker models:
- RMSNorm
- Rotary Embeddings (standard and multimodal)
- Attention layers
- Decoder layers
- MLP layers
"""

from typing import Any, Callable, Optional

import torch
from torch import nn, Tensor

from .configuration import CodePredictorConfig, TalkerConfig
from .utils import (
    ACT2FN,
    ALL_ATTENTION_FUNCTIONS,
    DynamicCache,
    ROPE_INIT_FUNCTIONS,
    dynamic_rope_update,
    repeat_kv,
)

# Type alias for configs that work with these layers
LayerConfig = CodePredictorConfig | TalkerConfig | Any

# Extended ROPE_INIT_FUNCTIONS with backward compatibility
ROPE_INIT_FUNCTIONS_EXTENDED: dict[str, Callable] = {
    **ROPE_INIT_FUNCTIONS,
    "default": ROPE_INIT_FUNCTIONS["default"],
}


class RMSNorm(nn.Module):
    """RMSNorm (Root Mean Square Layer Normalization)."""

    weight: nn.Parameter
    variance_epsilon: float

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (standard or multimodal)."""

    multimodal: bool
    rope_type: str
    max_seq_len_cached: int
    original_max_seq_len: int
    attention_scaling: float
    inv_freq: Tensor
    original_inv_freq: Tensor

    def __init__(
        self,
        config: LayerConfig,
        device: Optional[torch.device] = None,
        multimodal: bool = False,
    ) -> None:
        super().__init__()
        self.multimodal = multimodal
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type")
            )
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS_EXTENDED[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(
        self, x: Tensor, position_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Compute rotary position embeddings.
        
        Args:
            x: Input tensor, used for dtype/device info [batch, seq_len, hidden_size]
            position_ids: Position indices. 
                - Standard: [batch, seq_len]
                - Multimodal: [3, batch, seq_len]
        
        Returns:
            Tuple of (cos, sin) embeddings with same dtype as input
        """
        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )

        if self.multimodal:
            # Multimodal RoPE: expand inv_freq to shape (3, ...) for multimodal RoPE
            inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
                3, position_ids.shape[1], -1, 1
            )
            position_ids_expanded = position_ids[:, :, None, :].float()
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(2, 3)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
        else:
            # Standard RoPE
            inv_freq_expanded = (
                self.inv_freq[None, :, None]
                .float()
                .expand(position_ids.shape[0], -1, 1)
                .to(x.device)
            )
            position_ids_expanded = position_ids[:, None, :].float()
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def TalkerRotaryEmbedding(
    config: LayerConfig, device: Optional[torch.device] = None
) -> RotaryEmbedding:
    """Backwards compatibility: TalkerRotaryEmbedding uses multimodal mode by default."""
    return RotaryEmbedding(config, device=device, multimodal=True)


def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    position_ids: Optional[Tensor] = None,
    unsqueeze_dim: int = 1,
) -> tuple[Tensor, Tensor]:
    """
    Applies Rotary Position Embedding to query and key tensors.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine embeddings [batch, seq_len, head_dim]
        sin: Sine embeddings [batch, seq_len, head_dim]
        position_ids: Optional position indices (unused, for API compatibility)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    mrope_section: list[int],
    mrope_interleaved: bool = False,
    unsqueeze_dim: int = 1,
) -> tuple[Tensor, Tensor]:
    """
    Applies Multimodal Rotary Position Embedding to query and key tensors.
    
    Args:
        q: Query tensor [batch, heads, seq_len, head_dim]
        k: Key tensor [batch, heads, seq_len, head_dim]
        cos: Cosine embeddings [3, batch, seq_len, head_dim]
        sin: Sine embeddings [3, batch, seq_len, head_dim]
        mrope_section: List of section sizes for each modality
        mrope_interleaved: Whether to use interleaved rope pattern
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    if mrope_interleaved:

        def apply_interleaved_rope(x: Tensor, modality_num: int) -> Tensor:
            x_t = x[0].clone()
            index_ranges = []
            for i, n in enumerate(mrope_section[1:], 1):
                beg_idx = i
                end_idx = n * modality_num
                index_ranges.append((beg_idx, end_idx))
            for beg_idx, end_idx in index_ranges:
                x_t[..., beg_idx:end_idx:modality_num] = x[
                    beg_idx, ..., beg_idx:end_idx:modality_num
                ]
            return x_t

        dim = cos.shape[-1]
        modality_num = len(mrope_section)
        cos = torch.cat(
            [apply_interleaved_rope(cos[..., : dim // 2], modality_num)] * 2, dim=-1
        ).unsqueeze(unsqueeze_dim)
        sin = torch.cat(
            [apply_interleaved_rope(sin[..., : dim // 2], modality_num)] * 2, dim=-1
        ).unsqueeze(unsqueeze_dim)
    else:
        mrope_section = mrope_section * 2
        cos = torch.cat(
            [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
        ).unsqueeze(unsqueeze_dim)
        sin = torch.cat(
            [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
        ).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def eager_attention_forward(
    module: "Attention",
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[Tensor, Tensor]:
    """
    Eager attention implementation.
    
    Args:
        module: Attention module (for num_key_value_groups)
        query: Query tensor [batch, heads, seq_len, head_dim]
        key: Key tensor [batch, kv_heads, seq_len, head_dim]
        value: Value tensor [batch, kv_heads, seq_len, head_dim]
        attention_mask: Optional causal mask
        scaling: Attention scaling factor (typically 1/sqrt(head_dim))
        dropout: Dropout probability
        
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Attention(nn.Module):
    """Multi-headed attention (standard or multimodal RoPE)."""

    config: LayerConfig
    layer_idx: int
    use_multimodal_rope: bool
    head_dim: int
    num_key_value_groups: int
    scaling: float
    attention_dropout: float
    is_causal: bool
    sliding_window: Optional[int]
    rope_scaling: Optional[dict[str, Any]]

    def __init__(
        self,
        config: LayerConfig,
        layer_idx: int,
        use_multimodal_rope: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.use_multimodal_rope = use_multimodal_rope
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Sliding window: for standard attention, use layer_types; for multimodal, use config.sliding_window
        if use_multimodal_rope:
            self.sliding_window = getattr(config, "sliding_window", None)
            self.rope_scaling = config.rope_scaling
        else:
            # Standard attention: check layer_types if available
            layer_types = getattr(config, "layer_types", None)
            if layer_types and layer_types[layer_idx] == "sliding_attention":
                self.sliding_window = config.sliding_window
            else:
                self.sliding_window = None
            self.rope_scaling = None

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass for attention.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) from RotaryEmbedding
            attention_mask: Optional causal mask
            past_key_values: Optional KV cache
            cache_position: Position in cache for incremental decoding
            
        Returns:
            Tuple of (output, attention_weights)
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings

        if self.use_multimodal_rope:
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                self.rope_scaling["mrope_section"],
                self.rope_scaling["interleaved"],
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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
        return self.o_proj(attn_output), attn_weights


def TalkerAttention(config: LayerConfig, layer_idx: int) -> Attention:
    """Backwards compatibility: TalkerAttention uses multimodal RoPE by default."""
    return Attention(config, layer_idx, use_multimodal_rope=True)


class TalkerMLP(nn.Module):
    """SwiGLU MLP for the Talker/CodePredictor."""

    hidden_size: int
    intermediate_size: int

    def __init__(
        self,
        config: LayerConfig,
        intermediate_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else config.intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ResizeMLP(nn.Module):
    """MLP for resizing text embeddings to match codec embeddings."""

    def __init__(
        self,
        input_size: int,
        intermediate_size: int,
        output_size: int,
        act: str,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size, bias=bias)
        self.linear_fc2 = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = ACT2FN[act]

    def forward(self, hidden_state: Tensor) -> Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class DecoderLayer(nn.Module):
    """Transformer decoder layer (standard or multimodal RoPE)."""

    hidden_size: int

    def __init__(
        self,
        config: LayerConfig,
        layer_idx: int,
        use_multimodal_rope: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(
            config=config, layer_idx=layer_idx, use_multimodal_rope=use_multimodal_rope
        )
        self.mlp = TalkerMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[Tensor] = None,
        position_embeddings: Optional[tuple[Tensor, Tensor]] = None,
        **kwargs: Any,
    ) -> tuple[Tensor, ...]:
        """
        Forward pass for decoder layer.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional causal mask
            position_ids: Position indices (unused, for API compatibility)
            past_key_values: Optional KV cache
            output_attentions: Whether to return attention weights
            use_cache: Whether to use KV caching (unused, for API compatibility)
            cache_position: Position in cache for incremental decoding
            position_embeddings: Tuple of (cos, sin) from RotaryEmbedding
            
        Returns:
            Tuple of (hidden_states,) or (hidden_states, attention_weights)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs: tuple[Tensor, ...] = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


def TalkerDecoderLayer(config: LayerConfig, layer_idx: int) -> DecoderLayer:
    """Backwards compatibility: TalkerDecoderLayer uses multimodal RoPE by default."""
    return DecoderLayer(config, layer_idx, use_multimodal_rope=True)


__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "TalkerRotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "apply_multimodal_rotary_pos_emb",
    "eager_attention_forward",
    "Attention",
    "TalkerAttention",
    "TalkerMLP",
    "ResizeMLP",
    "DecoderLayer",
    "TalkerDecoderLayer",
    "ROPE_INIT_FUNCTIONS_EXTENDED",
]
