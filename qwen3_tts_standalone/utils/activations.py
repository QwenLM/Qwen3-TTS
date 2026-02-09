# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone activation functions.

This module provides minimal replacements for transformers.activations.ACT2FN.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SiLUActivation(nn.Module):
    """SiLU (Sigmoid Linear Unit) activation function, also known as Swish."""
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.silu(input)


class GELUActivation(nn.Module):
    """GELU activation function."""
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input)


class NewGELUActivation(nn.Module):
    """
    GELU activation function with tanh approximation (as used in GPT-2/BERT).
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(
            (2.0 / torch.pi) ** 0.5 * (input + 0.044715 * torch.pow(input, 3.0))
        ))


class _ACT2FN:
    """
    Activation function lookup that instantiates the appropriate activation module.
    
    This is a standalone replacement for transformers.activations.ACT2FN.
    """
    _activations = {
        "silu": SiLUActivation,
        "swish": nn.SiLU,
        "gelu": GELUActivation,
        "gelu_new": NewGELUActivation,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    
    def __getitem__(self, key: str) -> nn.Module:
        if key not in self._activations:
            raise KeyError(
                f"Activation function '{key}' not found. "
                f"Available activations: {list(self._activations.keys())}"
            )
        return self._activations[key]()


ACT2FN = _ACT2FN()
