# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone cache utilities for KV caching during generation.

This module provides minimal replacements for transformers.cache_utils.
"""

from typing import Optional

import torch


class Cache:
    """
    Base class for KV cache.
    
    This is a minimal standalone replacement for transformers.cache_utils.Cache.
    """
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states."""
        raise NotImplementedError
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple:
        """Updates the cache with new key/value states and returns the full cached states."""
        raise NotImplementedError


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated.
    
    This is a minimal standalone replacement for transformers.cache_utils.DynamicCache.
    It stores key and value states for each layer as they are generated.
    """
    
    def __init__(self):
        super().__init__()
        self._seen_tokens = 0
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
    
    def __len__(self) -> int:
        """Returns the number of layers in the cache."""
        return len(self.key_cache)
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of cached states for a given layer."""
        if layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx].shape[-2]
        return 0
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple:
        """
        Updates the cache with new key/value states.
        
        Args:
            key_states: New key states of shape (batch, num_heads, seq_len, head_dim)
            value_states: New value states of shape (batch, num_heads, seq_len, head_dim)
            layer_idx: Index of the layer to update
            cache_kwargs: Additional kwargs (unused in this simple implementation)
        
        Returns:
            Tuple of (full_key_states, full_value_states) including cached values
        """
        # Initialize cache for new layers
        if layer_idx >= len(self.key_cache):
            # Extend cache to include this layer
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(None)
                self.value_cache.append(None)
        
        if self.key_cache[layer_idx] is None:
            # First time seeing this layer
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            # Concatenate with existing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        # Track tokens at layer 0
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def reset(self):
        """Resets the cache."""
        self._seen_tokens = 0
        self.key_cache = []
        self.value_cache = []
