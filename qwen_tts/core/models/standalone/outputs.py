# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone model output classes.

This module provides minimal replacements for transformers.modeling_outputs.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch

from .cache import Cache


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass.
    
    Has a `__getitem__` that allows indexing by integer or slice (like a tuple) 
    or strings (like a dictionary) that will ignore the `None` attributes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __post_init__(self):
        # Convert dataclass fields to OrderedDict entries
        class_fields = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        for key, value in class_fields.items():
            self[key] = value
    
    def __getitem__(self, k):
        if isinstance(k, str):
            return super().__getitem__(k)
        else:
            return self.to_tuple()[k]
    
    def to_tuple(self):
        """Convert to tuple, filtering out None values."""
        return tuple(v for v in self.values() if v is not None)
    
    def __iter__(self):
        """Iterate over non-None values."""
        return iter(v for v in self.values() if v is not None)


@dataclass
class BaseModelOutputWithPast(ModelOutput):
    """
    Base class for model outputs that may contain past key/values (for caching).
    """
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    
    def __post_init__(self):
        self["last_hidden_state"] = self.last_hidden_state
        self["past_key_values"] = self.past_key_values
        self["hidden_states"] = self.hidden_states
        self["attentions"] = self.attentions


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model outputs with past key/values.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    
    def __post_init__(self):
        self["loss"] = self.loss
        self["logits"] = self.logits
        self["past_key_values"] = self.past_key_values
        self["hidden_states"] = self.hidden_states
        self["attentions"] = self.attentions
