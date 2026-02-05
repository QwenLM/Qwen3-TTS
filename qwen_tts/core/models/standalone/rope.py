# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone Rotary Position Embedding (RoPE) utilities.

This module provides minimal replacements for transformers.modeling_rope_utils.
"""

from functools import wraps

import torch


def _compute_default_rope_parameters(config, device=None, seq_len=None):
    """
    Computes the inverse frequencies according to the original RoPE implementation.
    
    Args:
        config: Model configuration with rope_theta, head_dim, etc.
        device: Device to create tensors on.
        seq_len: Current sequence length (unused for default RoPE).
    
    Returns:
        Tuple of (inv_freq, attention_factor).
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    
    attention_factor = 1.0  # Unused in default RoPE
    
    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


def _compute_linear_rope_parameters(config, device=None, seq_len=None):
    """Linear scaling of RoPE frequencies."""
    inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len)
    factor = config.rope_scaling.get("factor", 1.0)
    inv_freq = inv_freq / factor
    return inv_freq, attention_factor


def _compute_dynamic_rope_parameters(config, device=None, seq_len=None):
    """Dynamic NTK-aware scaling of RoPE frequencies."""
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    
    max_position_embeddings = config.max_position_embeddings
    factor = config.rope_scaling.get("factor", 1.0)
    
    attention_factor = 1.0
    
    if seq_len is not None and seq_len > max_position_embeddings:
        # Dynamic scaling
        base = base * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
    
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


# RoPE initialization functions registry
ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "linear": _compute_linear_rope_parameters,
    "dynamic": _compute_dynamic_rope_parameters,
}


def dynamic_rope_update(rope_forward):
    """
    Decorator to update RoPE parameters in the forward pass for dynamic RoPE types.
    
    Args:
        rope_forward: The forward pass of the RoPE implementation.
    
    Returns:
        The decorated forward pass.
    """
    def dynamic_frequency_update(self, position_ids, device):
        """Dynamic RoPE layers recompute inv_freq when growing beyond cached length."""
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        
        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len
    
    @wraps(rope_forward)
    def wrapper(self, x, position_ids):
        if "dynamic" in self.rope_type:
            dynamic_frequency_update(self, position_ids, device=x.device)
        return rope_forward(self, x, position_ids)
    
    return wrapper
