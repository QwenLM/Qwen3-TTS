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
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    past_key_values: Optional[Cache],
) -> torch.Tensor:
    """
    Create a standard causal attention mask.
    
    Args:
        input_embeds: Input embeddings of shape (batch_size, seq_len, hidden_dim).
        attention_mask: Optional 2D attention mask for padding. [batch, seq_len]
        cache_position: Tensor indicating current indices.
        past_key_values: Optional past key values cache.
    
    Returns:
        4D causal attention mask.
    """
    batch_size, seq_len = input_embeds.shape[:2]
    dtype = input_embeds.dtype
    device = input_embeds.device
    
    # Determine the total key/value sequence length
    if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
        past_len = past_key_values.get_seq_length()
    else:
        past_len = 0
    
    kv_seq_len = past_len + seq_len
    
    # Create column indices for key/value positions: (1, kv_seq_len)
    col_idx = torch.arange(kv_seq_len, device=device).unsqueeze(0)
    
    # Create row positions: each query's absolute position
    if cache_position is not None:
        # cache_position gives the absolute position of each query token: (seq_len,) -> (seq_len, 1)
        row_positions = cache_position.unsqueeze(1)
    else:
        # Standard case: query i is at absolute position past_len + i
        row_positions = (past_len + torch.arange(seq_len, device=device)).unsqueeze(1)
    
    # A query can attend to key/value positions <= its own position
    # can_attend shape: (seq_len, kv_seq_len)
    can_attend = col_idx <= row_positions
    
    # Create mask: 0 where can attend, -inf where cannot
    min_val = torch.finfo(dtype).min
    causal_mask = torch.where(can_attend, torch.tensor(0.0, dtype=dtype, device=device), min_val)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, kv_seq_len)
    
    # Apply padding mask if provided
    if attention_mask is not None and attention_mask.ndim == 2:
        # Expand attention_mask from (batch, kv_seq_len) to (batch, 1, 1, kv_seq_len)
        expanded_mask = attention_mask[:, None, None, :].to(dtype=dtype)
        # Convert 0s to -inf, 1s to 0
        inverted_mask = (1.0 - expanded_mask) * min_val
        # Combine with causal mask
        causal_mask = causal_mask.expand(batch_size, -1, -1, -1) + inverted_mask[:, :, :, :kv_seq_len]
    
    return causal_mask


def create_sliding_window_causal_mask(
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    past_key_values: Optional[Cache],
    sliding_window: int,
) -> torch.Tensor:
    """
    Create a sliding window causal attention mask.
    
    Args:
        input_embeds: Input embeddings of shape (batch_size, seq_len, hidden_dim).
        attention_mask: Optional 2D attention mask for padding.
        cache_position: Tensor indicating current indices.
        past_key_values: Optional past key values cache.
        sliding_window: The sliding window size.
    
    Returns:
        4D sliding window causal attention mask.
    """
    batch_size, seq_len = input_embeds.shape[:2]
    dtype = input_embeds.dtype
    device = input_embeds.device
    
    # Determine the total key/value sequence length
    if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
        past_len = past_key_values.get_seq_length()
    else:
        past_len = 0
    
    kv_seq_len = past_len + seq_len
    
    # Create column indices for key/value positions: (1, kv_seq_len)
    col_idx = torch.arange(kv_seq_len, device=device).unsqueeze(0)
    
    # Create row positions: each query's absolute position
    if cache_position is not None:
        # cache_position gives the absolute position of each query token: (seq_len,) -> (seq_len, 1)
        row_positions = cache_position.unsqueeze(1)
    else:
        # Standard case: query i is at absolute position past_len + i
        row_positions = (past_len + torch.arange(seq_len, device=device)).unsqueeze(1)
    
    # Sliding window: can attend to positions in [pos - window + 1, pos]
    # Lower bound clamped to 0
    lower_bound = (row_positions - sliding_window + 1).clamp(min=0)
    
    # A query can attend to key/value positions within the window and <= its own position
    # can_attend shape: (seq_len, kv_seq_len)
    can_attend = (col_idx >= lower_bound) & (col_idx <= row_positions)
    
    # Create mask: 0 where can attend, -inf where cannot
    min_val = torch.finfo(dtype).min
    causal_mask = torch.where(can_attend, torch.tensor(0.0, dtype=dtype, device=device), min_val)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, kv_seq_len)
    
    # Apply padding mask if provided
    if attention_mask is not None and attention_mask.ndim == 2:
        expanded_mask = attention_mask[:, None, None, :].to(dtype=dtype)
        inverted_mask = (1.0 - expanded_mask) * min_val
        causal_mask = causal_mask.expand(batch_size, -1, -1, -1) + inverted_mask[:, :, :, :kv_seq_len]
    
    return causal_mask
