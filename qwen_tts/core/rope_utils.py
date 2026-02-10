# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

def _compute_default_rope_parameters(config, device, seq_len=None, layer_type=None):
    """
    Computes the inverse frequencies for the default RoPE implementation.
    This is added for compatibility with transformers >= 5.x which removed 'default' from ROPE_INIT_FUNCTIONS.
    """
    if hasattr(config, "standardize_rope_params"):
        config.standardize_rope_params()
    
    # Get rope_parameters from config if it exists, otherwise use attributes
    if hasattr(config, "rope_parameters"):
        rope_params = config.rope_parameters.get(layer_type) if layer_type is not None and isinstance(config.rope_parameters, dict) and layer_type in config.rope_parameters else config.rope_parameters
    else:
        rope_params = {}

    if isinstance(rope_params, dict):
        base = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000))
        partial_rotary_factor = rope_params.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0))
    else:
        base = getattr(config, "rope_theta", 10000)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        # Fallback for models that don't have head_dim explicitly
        hidden_size = getattr(config, "hidden_size", 4096)
        num_heads = getattr(config, "num_attention_heads", 32)
        head_dim = hidden_size // num_heads
    
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0

def patch_rope_init_functions():
    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
