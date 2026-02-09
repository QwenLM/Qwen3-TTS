# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Talker base transformer model for Qwen3-TTS.

This module contains the base transformer decoder (TalkerModel)
used by the Talker for generating audio codec tokens.
"""

import logging
from typing import Any, Optional, Type

import torch
from torch import nn, Tensor

from .base_model import BaseModel
from .configuration import TalkerConfig
from .layers import (
    RMSNorm,
    DecoderLayer,
    RotaryEmbedding,
)
from .utils import (
    BaseModelOutputWithPast,
    DynamicCache,
    can_return_tuple,
    create_causal_mask,
    create_sliding_window_causal_mask,
)

logger = logging.getLogger(__name__)


class TalkerPreTrainedModel(BaseModel):
    """Base class for Talker text models."""

    config: TalkerConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True
    _no_split_modules: list[str] = []
    _skip_keys_device_placement: list[str] = ["past_key_values"]
    _supports_flash_attn: bool = True
    _supports_sdpa: bool = True
    _supports_flex_attn: bool = True
    _supports_cache_class: bool = True
    _supports_quantized_cache: bool = True
    _supports_static_cache: bool = False
    _supports_attention_backend: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for a module."""
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)


class TalkerModel(TalkerPreTrainedModel):
    """
    Transformer decoder model for generating audio codec tokens.

    This is the base model that generates the first codebook tokens.
    It uses multimodal RoPE for positional encoding.
    """

    config_class: Type[TalkerConfig] = TalkerConfig
    base_model_prefix: str = "talker.model"

    # Model components
    layers: nn.ModuleList
    norm: RMSNorm
    rotary_emb: RotaryEmbedding
    codec_embedding: nn.Embedding
    text_embedding: nn.Embedding
    gradient_checkpointing: bool
    padding_idx: Optional[int]
    vocab_size: int

    def __init__(self, config: TalkerConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList(
            [
                DecoderLayer(config, layer_idx, use_multimodal_rope=True)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config, multimodal=True)
        self.gradient_checkpointing = False
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.text_hidden_size)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """Return codec embedding layer."""
        return self.codec_embedding

    def get_text_embeddings(self) -> nn.Embedding:
        """Return text embedding layer."""
        return self.text_embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Set input embeddings."""
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[Tensor] = None,
        **flash_attn_kwargs: Any,
    ) -> BaseModelOutputWithPast:
        """
        Forward pass through the transformer.

        Args:
            input_ids: Token IDs [batch, seq_len] (mutually exclusive with inputs_embeds)
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position indices [3, batch, seq_len] for multimodal RoPE
            past_key_values: KV cache from previous forward passes
            inputs_embeds: Input embeddings [batch, seq_len, hidden_size]
            use_cache: Whether to return updated KV cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            cache_position: Position in cache for incremental decoding

        Returns:
            BaseModelOutputWithPast containing hidden states, cache, and optionally
            attention weights and all hidden states.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. "
                    "Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # Multimodal RoPE uses 3 position dimensions (temporal, height, width)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(
                3, inputs_embeds.shape[0], -1
            )
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        mask_function = (
            create_causal_mask
            if self.config.sliding_window is None
            else create_sliding_window_causal_mask
        )
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states: Optional[tuple[Tensor, ...]] = () if output_hidden_states else None
        all_self_attns: Optional[tuple[Tensor, ...]] = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


__all__ = [
    "TalkerPreTrainedModel",
    "TalkerModel",
]
