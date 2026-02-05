
# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
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
"""
Standalone base model class that replaces transformers.PreTrainedModel.

This module provides:
- Weight loading from safetensors and pytorch files
- Device mapping support
- Generation mixin functionality
- HuggingFace Hub integration

The goal is to provide a drop-in replacement for PreTrainedModel that doesn't
require the transformers library for core functionality.
"""

from __future__ import annotations

import json
import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torch import nn
from transformers import GenerationConfig

from .configuration_qwen3_tts_standalone import BaseConfig

logger = logging.getLogger(__name__)

# Type variable for model classes
M = TypeVar("M", bound="StandalonePreTrainedModel")


def _load_safetensors_file(filepath: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load a safetensors file and return the state dict."""
    try:
        from safetensors.torch import load_file
        return load_file(filepath, device=device)
    except ImportError:
        raise ImportError(
            "safetensors is required for loading .safetensors files. "
            "Install it with: pip install safetensors"
        )


def _load_pytorch_file(filepath: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load a pytorch .bin file and return the state dict."""
    return torch.load(filepath, map_location=device, weights_only=True)


def _get_checkpoint_files(model_path: str) -> tuple[List[str], str]:
    """
    Find all checkpoint files in a model directory.
    
    Returns:
        Tuple of (list of file paths, file type: 'safetensors' or 'pytorch')
    """
    # Check for safetensors files first (preferred)
    safetensors_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if safetensors_files:
        return safetensors_files, "safetensors"
    
    # Fall back to pytorch files
    pytorch_files = sorted(glob(os.path.join(model_path, "*.bin")))
    if pytorch_files:
        # Filter out optimizer states
        pytorch_files = [f for f in pytorch_files if "optimizer" not in f.lower()]
        return pytorch_files, "pytorch"
    
    # Check for single model file
    single_safetensors = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_safetensors):
        return [single_safetensors], "safetensors"
    
    single_pytorch = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(single_pytorch):
        return [single_pytorch], "pytorch"
    
    return [], "unknown"


def _parse_device_map(
    device_map: Optional[Union[str, Dict[str, Any]]],
    model: nn.Module,
) -> Dict[str, torch.device]:
    """
    Parse device_map specification into a mapping of module names to devices.
    
    Supports:
    - None: Keep all modules on CPU
    - "auto": Place all modules on CUDA if available, else CPU
    - "cuda:0", "cpu", etc.: Place all modules on the specified device
    - Dict[str, str]: Custom mapping of module names to devices
    """
    if device_map is None:
        return {}
    
    if isinstance(device_map, str):
        if device_map == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = device_map
        return {"": torch.device(device)}
    
    # Dictionary mapping
    return {k: torch.device(v) for k, v in device_map.items()}


def _apply_device_map(model: nn.Module, device_map: Dict[str, torch.device]) -> None:
    """Apply device mapping to a model."""
    if not device_map:
        return
    
    # If there's a root device, move the whole model
    if "" in device_map:
        model.to(device_map[""])
        return
    
    # Otherwise, move specific modules
    for name, device in device_map.items():
        parts = name.split(".")
        module = model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                logger.warning(f"Module {name} not found in model")
                break
        else:
            module.to(device)


class StandaloneGenerationMixin:
    """
    Mixin class that provides generation capabilities for standalone models.
    
    This is a simplified version that works with our standalone base class.
    For more complex generation, models can still use transformers' GenerationMixin.
    """
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare model inputs for generation.
        
        This method should be overridden by subclasses for model-specific logic.
        """
        # If we have past_key_values, only use last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            # Don't pass inputs_embeds if we're using past
            inputs_embeds = None
        
        model_inputs = {}
        
        # Use inputs_embeds if provided (first generation step)
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids
        
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
        
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        
        return model_inputs
    
    def can_generate(self) -> bool:
        """Returns True if the model can generate sequences."""
        return True


class StandalonePreTrainedModel(nn.Module, StandaloneGenerationMixin):
    """
    Base class for standalone pretrained models.
    
    This class provides:
    - Weight loading from safetensors/pytorch files
    - HuggingFace Hub downloading
    - Device mapping
    - Basic generation support
    
    Subclasses should override:
    - config_class: The configuration class for this model
    - _init_weights: Weight initialization method
    """
    
    # Configuration class (should be overridden by subclasses)
    config_class: Type[BaseConfig] = BaseConfig
    
    # Base model prefix for loading nested weights
    base_model_prefix: str = "model"
    
    # Modules that should not be split across devices
    _no_split_modules: List[str] = []
    
    # Keys to skip during device placement
    _skip_keys_device_placement: Union[str, List[str]] = []
    
    # Feature flags
    supports_gradient_checkpointing: bool = True
    _supports_flash_attn: bool = True
    _supports_sdpa: bool = True
    _supports_cache_class: bool = True
    _supports_static_cache: bool = False
    _supports_attention_backend: bool = True
    
    # Input name for generation (can be overridden by subclasses)
    main_input_name: str = "input_ids"
    
    # Whether the model is stateful (for cache handling)
    _is_stateful: bool = False
    
    def __init__(self, config: BaseConfig, *args, **kwargs):
        super().__init__()
        self.config = config
        self._device = torch.device("cpu")
        self.gradient_checkpointing = False
        
        # Initialize generation config for GenerationMixin compatibility
        self.generation_config = GenerationConfig()
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32
    
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for a module.
        
        Override this method in subclasses for custom initialization.
        """
        std = getattr(self.config, "initializer_range", 0.02)
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def post_init(self) -> None:
        """
        Post-initialization hook.
        
        This method is called after the model is initialized and can be used
        for additional setup like weight initialization or tying weights.
        """
        self.apply(self._init_weights)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        """Enable gradient checkpointing for the model."""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing for the model."""
        self.gradient_checkpointing = False
    
    def get_input_embeddings(self) -> Optional[nn.Module]:
        """Get input embeddings module."""
        return None
    
    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set input embeddings module."""
        pass
    
    def get_output_embeddings(self) -> Optional[nn.Module]:
        """Get output embeddings module."""
        return None
    
    def set_output_embeddings(self, value: nn.Module) -> None:
        """Set output embeddings module."""
        pass
    
    def tie_weights(self) -> None:
        """
        Tie input and output embeddings if configured.
        """
        if getattr(self.config, "tie_word_embeddings", False):
            output_embeddings = self.get_output_embeddings()
            input_embeddings = self.get_input_embeddings()
            if output_embeddings is not None and input_embeddings is not None:
                output_embeddings.weight = input_embeddings.weight
    
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """Resize token embeddings."""
        old_embeddings = self.get_input_embeddings()
        if old_embeddings is None:
            raise ValueError("Model does not have input embeddings")
        
        old_num_tokens, embedding_dim = old_embeddings.weight.shape
        
        if new_num_tokens == old_num_tokens:
            return old_embeddings
        
        new_embeddings = nn.Embedding(new_num_tokens, embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        
        # Copy old embeddings
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        self.set_input_embeddings(new_embeddings)
        
        # Update config
        if hasattr(self.config, "vocab_size"):
            self.config.vocab_size = new_num_tokens
        
        return new_embeddings
    
    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        """
        Save model weights and configuration to a directory.
        
        Args:
            save_directory: Directory to save to.
            safe_serialization: Use safetensors format (recommended).
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save_pretrained(save_directory)
        
        # Save weights
        state_dict = self.state_dict()
        
        if safe_serialization:
            try:
                from safetensors.torch import save_file
                save_file(state_dict, save_directory / "model.safetensors")
            except ImportError:
                logger.warning("safetensors not available, falling back to pytorch format")
                torch.save(state_dict, save_directory / "pytorch_model.bin")
        else:
            torch.save(state_dict, save_directory / "pytorch_model.bin")
        
        logger.info(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(
        cls: Type[M],
        pretrained_model_name_or_path: Union[str, Path],
        *model_args,
        config: Optional[BaseConfig] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[str] = None,
        revision: str = "main",
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        dtype: Optional[torch.dtype] = None,
        use_safetensors: Optional[bool] = None,
        **kwargs,
    ) -> M:
        """
        Load a pretrained model from a local directory or HuggingFace Hub.
        
        Args:
            pretrained_model_name_or_path: Local path or HuggingFace repo id.
            config: Optional configuration object. If not provided, will be loaded.
            cache_dir: Directory to cache downloaded files.
            force_download: Force re-download even if cached.
            local_files_only: Only use local files, don't download.
            token: HuggingFace Hub token for private repos.
            revision: Git revision (branch, tag, commit) to use.
            device_map: Device placement specification.
            dtype: Data type for model parameters.
            use_safetensors: Whether to prefer safetensors format.
            **kwargs: Additional arguments (ignored for compatibility).
            
        Returns:
            Loaded model instance.
        """
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        
        # Determine if it's a local path or HuggingFace repo
        if os.path.isdir(pretrained_model_name_or_path):
            model_path = pretrained_model_name_or_path
        else:
            # Download from HuggingFace Hub
            model_path = snapshot_download(
                repo_id=pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
        
        # Load config if not provided
        if config is None:
            config = cls.config_class.from_pretrained(
                model_path,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
        
        # Store path in config
        config._name_or_path = pretrained_model_name_or_path
        
        # Filter out kwargs that shouldn't be passed to __init__
        # These are common kwargs used by transformers but not needed for model initialization
        init_kwargs = {k: v for k, v in kwargs.items() if k not in [
            "attn_implementation", "torch_dtype", "low_cpu_mem_usage",
            "offload_folder", "offload_state_dict", "output_loading_info",
        ]}
        
        # Create model instance
        model = cls(config, *model_args, **init_kwargs)
        
        # Find and load weights
        checkpoint_files, file_type = _get_checkpoint_files(model_path)
        
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {model_path}")
        
        # Determine load device (load to CPU first, then move)
        load_device = "cpu"
        
        # Load all checkpoint shards
        full_state_dict = {}
        for filepath in checkpoint_files:
            logger.info(f"Loading weights from {filepath}")
            
            if file_type == "safetensors" or filepath.endswith(".safetensors"):
                shard_state_dict = _load_safetensors_file(filepath, device=load_device)
            else:
                shard_state_dict = _load_pytorch_file(filepath, device=load_device)
            
            full_state_dict.update(shard_state_dict)
        
        # Load state dict into model
        missing_keys, unexpected_keys = model.load_state_dict(full_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys in state dict: {missing_keys[:10]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state dict: {unexpected_keys[:10]}...")
        
        # Apply dtype if specified
        if dtype is not None:
            model = model.to(dtype=dtype)
        
        # Apply device mapping
        if device_map is not None:
            parsed_device_map = _parse_device_map(device_map, model)
            _apply_device_map(model, parsed_device_map)
        
        # Set to eval mode
        model.eval()
        
        return model
    
    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Count the number of parameters in the model.
        
        Args:
            only_trainable: Only count trainable parameters.
            
        Returns:
            Number of parameters.
        """
        params = self.parameters()
        if only_trainable:
            params = filter(lambda p: p.requires_grad, params)
        return sum(p.numel() for p in params)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config.model_type})"


__all__ = [
    "StandalonePreTrainedModel",
    "StandaloneGenerationMixin",
    "BaseConfig",
]
