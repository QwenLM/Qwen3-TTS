# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Shared fixtures and utilities for Qwen3TTS tests.
"""

import pytest
import torch
import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def copy_weights(src_module: torch.nn.Module, dst_module: torch.nn.Module):
    """Copy weights from source module to destination module."""
    src_state_dict = src_module.state_dict()
    dst_module.load_state_dict(src_state_dict)


@pytest.fixture
def seed():
    """Fixture to set seed before each test."""
    set_seed(42)
    return 42
