# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone masking utilities.

This module provides minimal replacements for transformers.masking_utils.
"""

from typing import Optional

import torch

from .cache import Cache


def create_causal_mask(
    config,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a standard causal attention mask.
    
    This is a simplified standalone replacement for transformers.masking_utils.create_causal_mask.
    
    Args:
        config: Model configuration.
        input_embeds: Input embeddings of shape (batch_size, seq_len, hidden_dim).
        attention_mask: Optional 2D attention mask for padding.
        cache_position: Tensor indicating current indices.
        past_key_values: Optional past key values cache.
        position_ids: Optional position IDs.
    
    Returns:
        4D causal attention mask or None if using SDPA with is_causal=True.
    """
    # For SDPA, we can return None and let is_causal=True handle it
    attn_impl = getattr(config, "_attn_implementation", "eager")
    if attn_impl == "sdpa" and attention_mask is None:
        return None
    
    batch_size, seq_len = input_embeds.shape[:2]
    dtype = input_embeds.dtype
    device = input_embeds.device
    
    # Determine the total key/value sequence length
    if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
        past_len = past_key_values.get_seq_length()
    else:
        past_len = 0
    
    kv_seq_len = past_len + seq_len
    
    # Create causal mask: positions can only attend to earlier positions
    # Shape: (1, 1, seq_len, kv_seq_len)
    causal_mask = torch.full(
        (1, 1, seq_len, kv_seq_len),
        fill_value=torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    
    # Fill in the causal pattern: each position can attend to itself and all previous positions
    if cache_position is not None:
        # Use cache_position to determine which positions are visible
        for i in range(seq_len):
            pos = cache_position[i].item() if cache_position.numel() > i else i
            causal_mask[0, 0, i, :pos + 1] = 0.0
    else:
        # Standard causal mask
        for i in range(seq_len):
            causal_mask[0, 0, i, :past_len + i + 1] = 0.0
    
    # Apply padding mask if provided
    if attention_mask is not None and attention_mask.ndim == 2:
        # Expand attention_mask from (batch, kv_seq_len) to (batch, 1, 1, kv_seq_len)
        expanded_mask = attention_mask[:, None, None, :].to(dtype=dtype)
        # Convert 0s to -inf, 1s to 0
        inverted_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
        # Combine with causal mask
        causal_mask = causal_mask.expand(batch_size, -1, -1, -1) + inverted_mask[:, :, :, :kv_seq_len]
    
    return causal_mask


def create_sliding_window_causal_mask(
    config,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> Optional[torch.Tensor]:
    """
    Create a sliding window causal attention mask.
    
    This is a simplified standalone replacement for transformers.masking_utils.create_sliding_window_causal_mask.
    
    Args:
        config: Model configuration with sliding_window attribute.
        input_embeds: Input embeddings of shape (batch_size, seq_len, hidden_dim).
        attention_mask: Optional 2D attention mask for padding.
        cache_position: Tensor indicating current indices.
        past_key_values: Optional past key values cache.
        position_ids: Optional position IDs.
    
    Returns:
        4D sliding window causal attention mask.
    """
    batch_size, seq_len = input_embeds.shape[:2]
    dtype = input_embeds.dtype
    device = input_embeds.device
    
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        # Fall back to regular causal mask
        return create_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids)
    
    # Determine the total key/value sequence length
    if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
        past_len = past_key_values.get_seq_length()
    else:
        past_len = 0
    
    kv_seq_len = past_len + seq_len
    
    # Create sliding window causal mask
    causal_mask = torch.full(
        (1, 1, seq_len, kv_seq_len),
        fill_value=torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    
    if cache_position is not None:
        for i in range(seq_len):
            pos = cache_position[i].item() if cache_position.numel() > i else i
            # Can attend to positions within sliding_window of current position
            start = max(0, pos - sliding_window + 1)
            causal_mask[0, 0, i, start:pos + 1] = 0.0
    else:
        for i in range(seq_len):
            current_pos = past_len + i
            start = max(0, current_pos - sliding_window + 1)
            causal_mask[0, 0, i, start:current_pos + 1] = 0.0
    
    # Apply padding mask if provided
    if attention_mask is not None and attention_mask.ndim == 2:
        expanded_mask = attention_mask[:, None, None, :].to(dtype=dtype)
        inverted_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
        causal_mask = causal_mask.expand(batch_size, -1, -1, -1) + inverted_mask[:, :, :, :kv_seq_len]
    
    return causal_mask
