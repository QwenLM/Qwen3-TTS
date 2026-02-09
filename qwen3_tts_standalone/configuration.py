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
Configuration classes for Qwen3TTS standalone models.

These configuration classes are fully standalone and inherit from BaseConfig,
which provides serialization, deserialization, and HuggingFace Hub loading
without requiring the transformers library.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

# Type variable for config classes
T = TypeVar("T", bound="BaseConfig")

# Valid rope types supported by the model
VALID_ROPE_TYPES = {"default", "linear", "dynamic", "yarn", "longrope", "llama3"}

# Valid layer types for attention
VALID_LAYER_TYPES = {"full_attention", "sliding_attention"}


def layer_type_validation(layer_types: List[str]) -> None:
    """
    Validate that all layer types in the list are valid attention types.
    
    Args:
        layer_types: List of layer type strings to validate.
        
    Raises:
        ValueError: If any layer type is not valid.
    """
    if not layer_types:
        return
    
    invalid_types = set(layer_types) - VALID_LAYER_TYPES
    if invalid_types:
        raise ValueError(
            f"Invalid layer types: {invalid_types}. "
            f"Valid types are: {VALID_LAYER_TYPES}"
        )


def rope_config_validation(config: "BaseConfig") -> None:
    """
    Validate the rope_scaling configuration.
    
    Args:
        config: Configuration object with rope_scaling attribute.
        
    Raises:
        ValueError: If rope_scaling configuration is invalid.
    """
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return
    
    if not isinstance(rope_scaling, dict):
        raise ValueError(f"rope_scaling must be a dictionary, got {type(rope_scaling)}")
    
    rope_type = rope_scaling.get("rope_type", "default")
    if rope_type not in VALID_ROPE_TYPES:
        raise ValueError(
            f"Invalid rope_type: {rope_type}. "
            f"Valid types are: {VALID_ROPE_TYPES}"
        )
    
    # Validate factor for non-default rope types
    if rope_type != "default":
        factor = rope_scaling.get("factor")
        if factor is not None and factor <= 0:
            raise ValueError(f"rope_scaling factor must be positive, got {factor}")


class BaseConfig:
    """
    Base configuration class providing serialization/deserialization functionality.
    
    This class is designed to eventually replace transformers.PretrainedConfig for 
    standalone models, providing essential features like:
    - to_dict() / from_dict() conversion
    - JSON save/load functionality
    - Copy functionality
    - Compatibility with HuggingFace Hub loading
    
    All configuration classes should inherit from this base class.
    
    Note: This class includes some attributes that mirror PretrainedConfig to ease
    the transition away from transformers dependency.
    """
    
    # Configuration file name for saving/loading
    CONFIG_NAME = "config.json"
    
    # Model type identifier (should be overridden by subclasses)
    model_type: str = "base"
    
    # Attribute map for renaming (PretrainedConfig compatibility)
    attribute_map: Dict[str, str] = {}
    
    def __init__(self, **kwargs):
        """
        Initialize base configuration.
        
        Args:
            **kwargs: Additional keyword arguments stored as attributes.
        """
        # PretrainedConfig compatibility attributes
        self._name_or_path = kwargs.pop("_name_or_path", "")
        # Default to "sdpa" (scaled dot-product attention) if not specified
        self._attn_implementation = kwargs.pop("_attn_implementation", "sdpa")
        self._attn_implementation_internal = kwargs.pop("_attn_implementation_internal", "sdpa")
        # Experts implementation (for MoE models, None for standard models)
        self._experts_implementation = kwargs.pop("_experts_implementation", None)
        
        # Common config attributes (with defaults from PretrainedConfig)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.return_dict = kwargs.pop("return_dict", True)
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)
        
        # Token IDs
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        
        # Store any additional kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def name_or_path(self) -> str:
        """Get the model name or path."""
        return self._name_or_path
    
    @name_or_path.setter
    def name_or_path(self, value: str):
        """Set the model name or path."""
        self._name_or_path = str(value)
    
    @property
    def has_no_defaults_at_init(self) -> bool:
        """Whether the config has no defaults at initialization (for generation params check)."""
        return False
    
    def _get_generation_parameters(self) -> Dict[str, Any]:
        """
        Get generation parameters from the config.
        
        This method is used by GenerationMixin to check for generation-related
        parameters in the config. For standalone models, we return an empty dict
        as generation params should be in GenerationConfig, not the model config.
        
        Note: We explicitly return an empty dict to avoid false positives from
        common config attributes like output_hidden_states, output_attentions, etc.
        which are not meant to be generation parameters.
        """
        # Return empty dict - generation params should be in GenerationConfig
        return {}
    
    def get_text_config(self, decoder: bool = False) -> "BaseConfig":
        """
        Get the text config for the model.
        
        This method is used by GenerationMixin for cache preparation.
        For most models, this just returns self.
        
        Args:
            decoder: Whether to get the decoder config (for encoder-decoder models).
            
        Returns:
            The text configuration (usually self).
        """
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dictionary containing all configuration parameters.
        """
        output = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, BaseConfig):
                output[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                output[key] = [
                    v.to_dict() if isinstance(v, BaseConfig) else v
                    for v in value
                ]
            else:
                output[key] = value
        
        # Add model_type from class attribute
        output["model_type"] = self.model_type
        
        return output
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any], **kwargs) -> T:
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dict values.
            
        Returns:
            Configuration instance.
        """
        # Make a copy to avoid modifying the original
        config_dict = copy.deepcopy(config_dict)
        
        # Remove model_type as it's a class attribute
        config_dict.pop("model_type", None)
        
        # Override with any provided kwargs
        config_dict.update(kwargs)
        
        return cls(**config_dict)
    
    def to_json_string(self, indent: int = 2) -> str:
        """
        Convert configuration to a JSON string.
        
        Args:
            indent: JSON indentation level.
            
        Returns:
            JSON string representation of the configuration.
        """
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)
    
    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """
        Save configuration to a directory.
        
        Args:
            save_directory: Directory path to save the configuration.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        config_path = save_directory / self.CONFIG_NAME
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(self.to_json_string())
        
        logger.info(f"Configuration saved to {config_path}")
    
    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_model_name_or_path: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[str] = None,
        revision: str = "main",
        **kwargs
    ) -> T:
        """
        Load configuration from a local directory, file, or HuggingFace Hub.
        
        Args:
            pretrained_model_name_or_path: Local path or HuggingFace repo id.
            cache_dir: Directory to cache downloaded files.
            force_download: Force re-download even if cached.
            local_files_only: Only use local files, don't download.
            token: HuggingFace Hub token for private repos.
            revision: Git revision (branch, tag, commit) to use.
            **kwargs: Additional keyword arguments to override loaded values.
            
        Returns:
            Configuration instance.
        """
        path = Path(pretrained_model_name_or_path)
        
        # Check if it's a local path
        if path.is_dir():
            config_path = path / cls.CONFIG_NAME
        elif path.is_file():
            config_path = path
        else:
            # Try to download from HuggingFace Hub
            try:
                from huggingface_hub import hf_hub_download
                config_path = hf_hub_download(
                    repo_id=str(pretrained_model_name_or_path),
                    filename=cls.CONFIG_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                )
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required to download from HuggingFace Hub. "
                    "Install it with: pip install huggingface_hub"
                )
            except Exception as e:
                raise FileNotFoundError(
                    f"Configuration file not found locally or on HuggingFace Hub: "
                    f"{pretrained_model_name_or_path}. Error: {e}"
                )
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        # Store the path in the config
        config_dict["_name_or_path"] = str(pretrained_model_name_or_path)
        
        return cls.from_dict(config_dict, **kwargs)
    
    def copy(self: T) -> T:
        """
        Create a deep copy of the configuration.
        
        Returns:
            A new configuration instance with copied values.
        """
        return self.__class__.from_dict(self.to_dict())
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from a dictionary.
        
        Args:
            config_dict: Dictionary with values to update.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_json_string()})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseConfig):
            return False
        return self.to_dict() == other.to_dict()


class SpeakerEncoderConfig(BaseConfig):
    r"""
    Configuration class for Qwen3TTS Speaker Encoder (ECAPA-TDNN based).
    
    This configuration stores parameters for the speaker encoder model that extracts
    speaker embeddings from mel-spectrograms.
    
    Args:
        mel_dim (`int`, *optional*, defaults to 128):
            The dimension of the input mel-spectrogram.
        enc_dim (`int`, *optional*, defaults to 1024):
            The dimension of the final speaker embedding.
        enc_channels (`list[int]`, *optional*, defaults to `[512, 512, 512, 512, 1536]`):
            Output channels for each TDNN/SERes2Net layer in the encoder.
        enc_kernel_sizes (`list[int]`, *optional*, defaults to `[5, 3, 3, 3, 1]`):
            Kernel sizes for each layer in the encoder.
        enc_dilations (`list[int]`, *optional*, defaults to `[1, 2, 3, 4, 1]`):
            Dilations for each layer in the encoder.
        enc_attention_channels (`int`, *optional*, defaults to 128):
            Number of attention channels in the AttentiveStatisticsPooling layer.
        enc_res2net_scale (`int`, *optional*, defaults to 8):
            Scale of the Res2NetBlock in the encoder.
        enc_se_channels (`int`, *optional*, defaults to 128):
            Number of channels in the squeeze part of the SqueezeExcitationBlock.
        sample_rate (`int`, *optional*, defaults to 24000):
            Audio sample rate in Hz.
    """
    
    model_type = "qwen3_tts_speaker_encoder"
    
    def __init__(
        self,
        mel_dim: int = 128,
        enc_dim: int = 1024,
        enc_channels: Optional[List[int]] = None,
        enc_kernel_sizes: Optional[List[int]] = None,
        enc_dilations: Optional[List[int]] = None,
        enc_attention_channels: int = 128,
        enc_res2net_scale: int = 8,
        enc_se_channels: int = 128,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mel_dim = mel_dim
        self.enc_dim = enc_dim
        self.enc_channels = enc_channels if enc_channels is not None else [512, 512, 512, 512, 1536]
        self.enc_kernel_sizes = enc_kernel_sizes if enc_kernel_sizes is not None else [5, 3, 3, 3, 1]
        self.enc_dilations = enc_dilations if enc_dilations is not None else [1, 2, 3, 4, 1]
        self.enc_attention_channels = enc_attention_channels
        self.enc_res2net_scale = enc_res2net_scale
        self.enc_se_channels = enc_se_channels
        self.sample_rate = sample_rate


class CodePredictorConfig(BaseConfig):
    r"""
    Configuration class for the Code Predictor model.
    
    This transformer-based model predicts audio codec codes from hidden representations.
    
    Args:
        vocab_size (`int`, *optional*, defaults to 2048):
            Vocabulary size of the code predictor.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 5):
            Number of hidden layers in the Transformer.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads for Grouped Query Attention.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The activation function.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            Maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMS normalization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use KV cache.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            RoPE scaling configuration dictionary.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention projections.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window size.
        max_window_layers (`int`, *optional*, defaults to 28):
            Number of layers using full attention.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout ratio.
        num_code_groups (`int`, *optional*, defaults to 32):
            Number of code groups for the codec.
    """
    
    model_type = "qwen3_tts_talker_code_predictor"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        vocab_size: int = 2048,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 16,
        num_key_value_heads: Optional[int] = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        attention_bias: bool = False,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 28,
        layer_types: Optional[List[str]] = None,
        attention_dropout: float = 0.0,
        num_code_groups: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers
        
        # For backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.num_code_groups = num_code_groups
        
        # Handle rope_scaling
        self.rope_scaling = rope_scaling
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)
        
        # Handle layer_types
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)


class TalkerConfig(BaseConfig):
    r"""
    Configuration class for the Qwen3TTS Talker model.
    
    This is the main TTS decoder model that generates audio codec codes from text.
    
    Args:
        code_predictor_config (`dict` or `CodePredictorConfig`, *optional*):
            Configuration for the code predictor sub-model.
        vocab_size (`int`, *optional*, defaults to 3072):
            Vocabulary size of the talker model.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 20):
            Number of hidden layers.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            Number of key-value heads for GQA.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation function.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            Maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Weight initialization standard deviation.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            RMS normalization epsilon.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use KV cache.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            RoPE base period.
        rope_scaling (`Dict`, *optional*):
            RoPE scaling configuration.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use attention bias.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window size.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout ratio.
        num_code_groups (`int`, *optional*, defaults to 32):
            Number of codec code groups.
        text_hidden_size (`int`, *optional*, defaults to 2048):
            Text encoder hidden size.
        codec_eos_token_id (`int`, *optional*, defaults to 4198):
            Codec end-of-sequence token ID.
        codec_think_id (`int`, *optional*, defaults to 4202):
            Codec think token ID.
        codec_nothink_id (`int`, *optional*, defaults to 4203):
            Codec no-think token ID.
        codec_think_bos_id (`int`, *optional*, defaults to 4204):
            Codec think begin-of-sequence token ID.
        codec_think_eos_id (`int`, *optional*, defaults to 4205):
            Codec think end-of-sequence token ID.
        codec_pad_id (`int`, *optional*, defaults to 4196):
            Codec padding token ID.
        codec_bos_id (`int`, *optional*, defaults to 4197):
            Codec begin-of-sequence token ID.
        spk_id (`int`, *optional*):
            Speaker ID.
        spk_is_dialect (`bool`, *optional*):
            Whether speaker uses dialect.
        codec_language_id (`int`, *optional*):
            Language ID for the codec.
    """
    
    model_type = "qwen3_tts_talker"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        code_predictor_config: Optional[Union[Dict[str, Any], CodePredictorConfig]] = None,
        vocab_size: int = 3072,
        hidden_size: int = 1024,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        attention_bias: bool = False,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        attention_dropout: float = 0.0,
        num_code_groups: int = 32,
        text_hidden_size: int = 2048,
        codec_eos_token_id: int = 4198,
        codec_think_id: int = 4202,
        codec_nothink_id: int = 4203,
        codec_think_bos_id: int = 4204,
        codec_think_eos_id: int = 4205,
        codec_pad_id: int = 4196,
        codec_bos_id: int = 4197,
        spk_id: Optional[int] = None,
        spk_is_dialect: Optional[bool] = None,
        codec_language_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.num_code_groups = num_code_groups
        self.text_hidden_size = text_hidden_size
        
        # Codec token IDs
        self.codec_eos_token_id = codec_eos_token_id
        self.codec_think_id = codec_think_id
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
        self.codec_language_id = codec_language_id
        self.spk_id = spk_id
        self.spk_is_dialect = spk_is_dialect
        
        # Handle rope_scaling
        self.rope_scaling = rope_scaling
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        
        # Handle code_predictor_config
        if code_predictor_config is None:
            self.code_predictor_config = CodePredictorConfig()
            logger.info("code_predictor_config is None. Initializing with default values.")
        elif isinstance(code_predictor_config, CodePredictorConfig):
            self.code_predictor_config = code_predictor_config
        else:
            self.code_predictor_config = CodePredictorConfig(**code_predictor_config)


class TTSConfig(BaseConfig):
    """
    Main configuration class for Qwen3TTS model.
    
    This configuration combines the talker and speaker encoder configurations
    for the complete TTS system.
    
    Args:
        talker_config (`dict` or `TalkerConfig`, *optional*):
            Configuration for the talker model.
        speaker_encoder_config (`dict` or `SpeakerEncoderConfig`, *optional*):
            Configuration for the speaker encoder.
        tokenizer_type (`str`, *optional*):
            Type of tokenizer to use.
        tts_model_size (`str`, *optional*):
            Size of the TTS model (e.g., "0.5B").
        tts_model_type (`str`, *optional*):
            Type of TTS model (e.g., "base").
        im_start_token_id (`int`, *optional*, defaults to 151644):
            Image start token ID.
        im_end_token_id (`int`, *optional*, defaults to 151645):
            Image end token ID.
        tts_pad_token_id (`int`, *optional*, defaults to 151671):
            TTS padding token ID.
        tts_bos_token_id (`int`, *optional*, defaults to 151672):
            TTS begin-of-sequence token ID.
        tts_eos_token_id (`int`, *optional*, defaults to 151673):
            TTS end-of-sequence token ID.
    """
    
    model_type = "qwen3_tts"
    
    def __init__(
        self,
        talker_config: Optional[Union[Dict[str, Any], TalkerConfig]] = None,
        speaker_encoder_config: Optional[Union[Dict[str, Any], SpeakerEncoderConfig]] = None,
        tokenizer_type: Optional[str] = None,
        tts_model_size: Optional[str] = None,
        tts_model_type: Optional[str] = None,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        tts_pad_token_id: int = 151671,
        tts_bos_token_id: int = 151672,
        tts_eos_token_id: int = 151673,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Handle talker_config
        if talker_config is None:
            self.talker_config = TalkerConfig()
            logger.info("talker_config is None. Initializing with default values.")
        elif isinstance(talker_config, TalkerConfig):
            self.talker_config = talker_config
        else:
            self.talker_config = TalkerConfig(**talker_config)
        
        # Handle speaker_encoder_config
        if speaker_encoder_config is None:
            self.speaker_encoder_config = SpeakerEncoderConfig()
            logger.info("speaker_encoder_config is None. Initializing with default values.")
        elif isinstance(speaker_encoder_config, SpeakerEncoderConfig):
            self.speaker_encoder_config = speaker_encoder_config
        else:
            self.speaker_encoder_config = SpeakerEncoderConfig(**speaker_encoder_config)
        
        self.tokenizer_type = tokenizer_type
        self.tts_model_size = tts_model_size
        self.tts_model_type = tts_model_type
        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id


# =============================================================================
# Conversion functions: Transform old (transformers-based) configs to new ones
# =============================================================================

def convert_speaker_encoder_config(
    old_config: "Qwen3TTSSpeakerEncoderConfig"  # noqa: F821
) -> SpeakerEncoderConfig:
    """
    Convert a transformers-based Qwen3TTSSpeakerEncoderConfig to standalone version.
    
    Args:
        old_config: Original transformers-based configuration.
        
    Returns:
        Standalone configuration instance.
    """
    return SpeakerEncoderConfig(
        mel_dim=old_config.mel_dim,
        enc_dim=old_config.enc_dim,
        enc_channels=old_config.enc_channels,
        enc_kernel_sizes=old_config.enc_kernel_sizes,
        enc_dilations=old_config.enc_dilations,
        enc_attention_channels=old_config.enc_attention_channels,
        enc_res2net_scale=old_config.enc_res2net_scale,
        enc_se_channels=old_config.enc_se_channels,
        sample_rate=old_config.sample_rate,
    )


def convert_code_predictor_config(
    old_config: "Qwen3TTSTalkerCodePredictorConfig"  # noqa: F821
) -> CodePredictorConfig:
    """
    Convert a transformers-based Qwen3TTSTalkerCodePredictorConfig to standalone version.
    
    Args:
        old_config: Original transformers-based configuration.
        
    Returns:
        Standalone configuration instance.
    """
    return CodePredictorConfig(
        vocab_size=old_config.vocab_size,
        hidden_size=old_config.hidden_size,
        intermediate_size=old_config.intermediate_size,
        num_hidden_layers=old_config.num_hidden_layers,
        num_attention_heads=old_config.num_attention_heads,
        num_key_value_heads=old_config.num_key_value_heads,
        head_dim=old_config.head_dim,
        hidden_act=old_config.hidden_act,
        max_position_embeddings=old_config.max_position_embeddings,
        initializer_range=old_config.initializer_range,
        rms_norm_eps=old_config.rms_norm_eps,
        use_cache=old_config.use_cache,
        tie_word_embeddings=getattr(old_config, "tie_word_embeddings", False),
        rope_theta=old_config.rope_theta,
        rope_scaling=old_config.rope_scaling,
        attention_bias=old_config.attention_bias,
        use_sliding_window=old_config.use_sliding_window,
        sliding_window=old_config.sliding_window if old_config.use_sliding_window else 4096,
        max_window_layers=old_config.max_window_layers,
        layer_types=old_config.layer_types,
        attention_dropout=old_config.attention_dropout,
        num_code_groups=old_config.num_code_groups,
    )


def convert_talker_config(
    old_config: "Qwen3TTSTalkerConfig"  # noqa: F821
) -> TalkerConfig:
    """
    Convert a transformers-based Qwen3TTSTalkerConfig to standalone version.
    
    Args:
        old_config: Original transformers-based configuration.
        
    Returns:
        Standalone configuration instance.
    """
    # Convert nested code_predictor_config
    code_predictor_config = None
    if hasattr(old_config, "code_predictor_config") and old_config.code_predictor_config is not None:
        code_predictor_config = convert_code_predictor_config(old_config.code_predictor_config)
    
    return TalkerConfig(
        code_predictor_config=code_predictor_config,
        vocab_size=old_config.vocab_size,
        hidden_size=old_config.hidden_size,
        intermediate_size=old_config.intermediate_size,
        num_hidden_layers=old_config.num_hidden_layers,
        num_attention_heads=old_config.num_attention_heads,
        num_key_value_heads=old_config.num_key_value_heads,
        hidden_act=old_config.hidden_act,
        max_position_embeddings=old_config.max_position_embeddings,
        initializer_range=old_config.initializer_range,
        rms_norm_eps=old_config.rms_norm_eps,
        use_cache=old_config.use_cache,
        tie_word_embeddings=getattr(old_config, "tie_word_embeddings", False),
        rope_theta=old_config.rope_theta,
        rope_scaling=old_config.rope_scaling,
        attention_bias=old_config.attention_bias,
        use_sliding_window=old_config.use_sliding_window,
        sliding_window=old_config.sliding_window if old_config.use_sliding_window else 4096,
        attention_dropout=old_config.attention_dropout,
        num_code_groups=old_config.num_code_groups,
        text_hidden_size=old_config.text_hidden_size,
        codec_eos_token_id=old_config.codec_eos_token_id,
        codec_think_id=old_config.codec_think_id,
        codec_nothink_id=old_config.codec_nothink_id,
        codec_think_bos_id=old_config.codec_think_bos_id,
        codec_think_eos_id=old_config.codec_think_eos_id,
        codec_pad_id=old_config.codec_pad_id,
        codec_bos_id=old_config.codec_bos_id,
        spk_id=old_config.spk_id,
        spk_is_dialect=old_config.spk_is_dialect,
        codec_language_id=old_config.codec_language_id,
    )


def convert_tts_config(
    old_config: "Qwen3TTSConfig"  # noqa: F821
) -> TTSConfig:
    """
    Convert a transformers-based Qwen3TTSConfig to standalone version.
    
    Args:
        old_config: Original transformers-based configuration.
        
    Returns:
        Standalone configuration instance.
    """
    # Convert nested configs
    talker_config = None
    if hasattr(old_config, "talker_config") and old_config.talker_config is not None:
        talker_config = convert_talker_config(old_config.talker_config)
    
    speaker_encoder_config = None
    if hasattr(old_config, "speaker_encoder_config") and old_config.speaker_encoder_config is not None:
        speaker_encoder_config = convert_speaker_encoder_config(old_config.speaker_encoder_config)
    
    return TTSConfig(
        talker_config=talker_config,
        speaker_encoder_config=speaker_encoder_config,
        tokenizer_type=old_config.tokenizer_type,
        tts_model_size=old_config.tts_model_size,
        tts_model_type=old_config.tts_model_type,
        im_start_token_id=old_config.im_start_token_id,
        im_end_token_id=old_config.im_end_token_id,
        tts_pad_token_id=old_config.tts_pad_token_id,
        tts_bos_token_id=old_config.tts_bos_token_id,
        tts_eos_token_id=old_config.tts_eos_token_id,
    )


# Backward compatibility aliases
Qwen3TTSConfigStandalone = TTSConfig
Qwen3TTSTalkerConfigStandalone = TalkerConfig
Qwen3TTSSpeakerEncoderConfigStandalone = SpeakerEncoderConfig
Qwen3TTSTalkerCodePredictorConfigStandalone = CodePredictorConfig


__all__ = [
    # Base class
    "BaseConfig",
    # Configuration classes (new names)
    "TTSConfig",
    "TalkerConfig",
    "SpeakerEncoderConfig",
    "CodePredictorConfig",
    # Backward compatibility aliases
    "Qwen3TTSConfigStandalone",
    "Qwen3TTSTalkerConfigStandalone",
    "Qwen3TTSSpeakerEncoderConfigStandalone",
    "Qwen3TTSTalkerCodePredictorConfigStandalone",
    # Validation functions
    "layer_type_validation",
    "rope_config_validation",
    "VALID_ROPE_TYPES",
    "VALID_LAYER_TYPES",
    # Conversion functions
    "convert_tts_config",
    "convert_talker_config",
    "convert_speaker_encoder_config",
    "convert_code_predictor_config",
]
