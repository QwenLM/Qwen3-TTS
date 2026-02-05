# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone attention functions.

This module provides minimal replacements for transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS.
"""

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for multi-query/grouped-query attention.
    
    This is equivalent to torch.repeat_interleave(hidden_states, n_rep, dim=1),
    but more memory-efficient using expand + reshape.
    
    Args:
        hidden_states: Key or value states of shape (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each head
    
    Returns:
        Repeated states of shape (batch, num_kv_heads * n_rep, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple:
    """
    Scaled Dot-Product Attention using PyTorch's F.scaled_dot_product_attention.
    
    Args:
        module: The attention module (used to get num_key_value_groups)
        query: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        key: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        value: Value tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        attention_mask: Optional attention mask
        dropout: Dropout probability
        scaling: Optional attention scaling factor
        is_causal: Whether to use causal masking
        **kwargs: Additional keyword arguments (ignored)
    
    Returns:
        Tuple of (attn_output, None) where attn_output has shape (batch, seq_len, num_heads, head_dim)
    """
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    
    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)
    
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    return attn_output, None


def flash_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple:
    """
    Flash Attention 2 forward pass.
    
    Note: This requires the flash_attn package to be installed.
    Falls back to SDPA if flash_attn is not available.
    
    Args:
        module: The attention module
        query: Query tensor
        key: Key tensor  
        value: Value tensor
        attention_mask: Optional attention mask (unused in Flash Attention)
        dropout: Dropout probability
        scaling: Optional attention scaling factor
        sliding_window: Optional sliding window size
        softcap: Optional softmax capping value
        **kwargs: Additional keyword arguments
    
    Returns:
        Tuple of (attn_output, None)
    """
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        # Fall back to SDPA if flash_attn is not installed
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, 
            dropout=dropout, scaling=scaling, **kwargs
        )
    
    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    
    # Handle dtype conversion
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = torch.bfloat16
        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)
    
    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)
    
    flash_kwargs = {
        "softmax_scale": scaling,
        "causal": is_causal,
    }
    if sliding_window is not None:
        flash_kwargs["window_size"] = (sliding_window, sliding_window)
    
    attn_output = flash_attn_func(
        query, key, value,
        dropout_p=dropout if module.training else 0.0,
        **flash_kwargs,
    )
    
    return attn_output, None


# Registry of attention implementations
ALL_ATTENTION_FUNCTIONS = {
    "sdpa": sdpa_attention_forward,
    "flash_attention_2": flash_attention_forward,
}
