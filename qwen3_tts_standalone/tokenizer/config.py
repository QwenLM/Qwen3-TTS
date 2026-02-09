# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone configuration for Qwen3TTSTokenizerV2 (12Hz tokenizer).

This module provides transformers-free configuration classes.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class SpeechDecoderConfig:
    """
    Configuration for the 12Hz tokenizer decoder.
    
    This is a standalone replacement for the transformers-based config.
    """
    codebook_size: int = 2048
    codebook_dim: int = 256
    hidden_size: int = 1024
    latent_dim: int = 1024
    max_position_embeddings: int = 8000
    rope_theta: float = 10000.0
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: Optional[int] = None  # If None, computed in __post_init__
    attention_bias: bool = False
    sliding_window: int = 72
    intermediate_size: int = 3072
    hidden_act: str = "silu"
    layer_scale_initial_scale: float = 0.01
    rms_norm_eps: float = 1e-5
    num_hidden_layers: int = 8
    num_quantizers: int = 16
    upsample_rates: Tuple[int, ...] = (8, 5, 4, 3)
    upsampling_ratios: Tuple[int, ...] = (2, 2)
    decoder_dim: int = 1536
    attention_dropout: float = 0.0
    _attn_implementation: str = "sdpa"
    
    def __post_init__(self):
        """Compute head_dim if not provided."""
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
    
    @property
    def layer_types(self) -> List[str]:
        """All layers in code2wav should be sliding attention."""
        return ["sliding_attention"] * self.num_hidden_layers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        d = asdict(self)
        # Convert tuples to lists for JSON serialization
        d["upsample_rates"] = list(d["upsample_rates"])
        d["upsampling_ratios"] = list(d["upsampling_ratios"])
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpeechDecoderConfig":
        """Create config from dictionary."""
        # Convert lists back to tuples
        if "upsample_rates" in d and isinstance(d["upsample_rates"], list):
            d["upsample_rates"] = tuple(d["upsample_rates"])
        if "upsampling_ratios" in d and isinstance(d["upsampling_ratios"], list):
            d["upsampling_ratios"] = tuple(d["upsampling_ratios"])
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class MimiEncoderConfig:
    """
    Minimal configuration for the Mimi encoder.
    
    This contains only the fields needed for the standalone encoder.
    """
    audio_channels: int = 1
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 2048
    hidden_act: str = "gelu"
    max_position_embeddings: int = 8000
    initializer_range: float = 0.02
    use_cache: bool = True
    sampling_rate: int = 24000
    frame_rate: float = 12.5
    encodec_frame_rate: int = 75
    num_codebooks: int = 32
    codebook_size: int = 2048
    codebook_dim: int = 256
    use_conv_shortcut: bool = False
    vector_quantization_hidden_dimension: int = 256
    upsample_groups: int = 512
    num_filters: int = 64
    residual_kernel_size: int = 3
    compress: int = 2
    dilation_growth_rate: int = 2
    upsampling_ratios: Tuple[int, ...] = (8, 6, 5, 4)
    norm_type: str = "weight_norm"
    kernel_size: int = 7
    last_kernel_size: int = 3
    use_causal_conv: bool = True
    trim_right_ratio: float = 1.0
    pad_mode: str = "constant"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        d = asdict(self)
        if isinstance(d.get("upsampling_ratios"), tuple):
            d["upsampling_ratios"] = list(d["upsampling_ratios"])
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MimiEncoderConfig":
        """Create config from dictionary."""
        if "upsampling_ratios" in d and isinstance(d["upsampling_ratios"], list):
            d["upsampling_ratios"] = tuple(d["upsampling_ratios"])
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class SpeechTokenizerConfig:
    """
    Main configuration for the 12Hz tokenizer.
    
    This is a standalone replacement for the transformers-based config.
    """
    model_type: str = "qwen3_tts_tokenizer_12hz"
    encoder_config: Optional[Union[Dict, MimiEncoderConfig]] = None
    decoder_config: Optional[Union[Dict, SpeechDecoderConfig]] = None
    encoder_valid_num_quantizers: int = 16
    input_sample_rate: int = 24000
    output_sample_rate: int = 24000
    decode_upsample_rate: int = 1920
    encode_downsample_rate: int = 1920
    return_dict: bool = True
    
    def __post_init__(self):
        """Initialize nested configs if needed."""
        if self.encoder_config is None:
            self.encoder_config = MimiEncoderConfig()
        elif isinstance(self.encoder_config, dict):
            self.encoder_config = MimiEncoderConfig.from_dict(self.encoder_config)
        
        if self.decoder_config is None:
            self.decoder_config = SpeechDecoderConfig()
        elif isinstance(self.decoder_config, dict):
            self.decoder_config = SpeechDecoderConfig.from_dict(self.decoder_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_type": self.model_type,
            "encoder_config": self.encoder_config.to_dict() if self.encoder_config else None,
            "decoder_config": self.decoder_config.to_dict() if self.decoder_config else None,
            "encoder_valid_num_quantizers": self.encoder_valid_num_quantizers,
            "input_sample_rate": self.input_sample_rate,
            "output_sample_rate": self.output_sample_rate,
            "decode_upsample_rate": self.decode_upsample_rate,
            "encode_downsample_rate": self.encode_downsample_rate,
            "return_dict": self.return_dict,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpeechTokenizerConfig":
        """Create config from dictionary."""
        encoder_config = d.get("encoder_config")
        decoder_config = d.get("decoder_config")
        
        return cls(
            model_type=d.get("model_type", "qwen3_tts_tokenizer_12hz"),
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            encoder_valid_num_quantizers=d.get("encoder_valid_num_quantizers", 16),
            input_sample_rate=d.get("input_sample_rate", 24000),
            output_sample_rate=d.get("output_sample_rate", 24000),
            decode_upsample_rate=d.get("decode_upsample_rate", 1920),
            encode_downsample_rate=d.get("encode_downsample_rate", 1920),
            return_dict=d.get("return_dict", True),
        )
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save configuration to directory."""
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "SpeechTokenizerConfig":
        """Load configuration from directory or HuggingFace Hub."""
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        else:
            # Download from HuggingFace Hub
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="config.json",
            )
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


def convert_to_standalone_config(
    original_config,
) -> SpeechTokenizerConfig:
    """
    Convert original transformers-based config to standalone config.
    
    Args:
        original_config: A Qwen3TTSTokenizerV2Config instance
        
    Returns:
        SpeechTokenizerConfig
    """
    # Extract encoder config
    encoder_dict = {}
    if hasattr(original_config, "encoder_config") and original_config.encoder_config:
        enc = original_config.encoder_config
        encoder_dict = {
            "audio_channels": getattr(enc, "audio_channels", 1),
            "hidden_size": getattr(enc, "hidden_size", 512),
            "num_hidden_layers": getattr(enc, "num_hidden_layers", 8),
            "num_attention_heads": getattr(enc, "num_attention_heads", 8),
            "num_key_value_heads": getattr(enc, "num_key_value_heads", 8),
            "head_dim": getattr(enc, "head_dim", 64),
            "intermediate_size": getattr(enc, "intermediate_size", 2048),
            "hidden_act": getattr(enc, "hidden_act", "gelu"),
            "sampling_rate": getattr(enc, "sampling_rate", 24000),
            "frame_rate": getattr(enc, "frame_rate", 12.5),
            "num_codebooks": getattr(enc, "num_codebooks", 32),
            "codebook_size": getattr(enc, "codebook_size", 2048),
            "codebook_dim": getattr(enc, "codebook_dim", 256),
        }
    
    # Extract decoder config
    decoder_dict = {}
    if hasattr(original_config, "decoder_config") and original_config.decoder_config:
        dec = original_config.decoder_config
        decoder_dict = {
            "codebook_size": getattr(dec, "codebook_size", 2048),
            "codebook_dim": getattr(dec, "codebook_dim", 256),
            "hidden_size": getattr(dec, "hidden_size", 1024),
            "latent_dim": getattr(dec, "latent_dim", 1024),
            "max_position_embeddings": getattr(dec, "max_position_embeddings", 8000),
            "rope_theta": getattr(dec, "rope_theta", 10000.0),
            "num_attention_heads": getattr(dec, "num_attention_heads", 16),
            "num_key_value_heads": getattr(dec, "num_key_value_heads", 16),
            "attention_bias": getattr(dec, "attention_bias", False),
            "sliding_window": getattr(dec, "sliding_window", 72),
            "intermediate_size": getattr(dec, "intermediate_size", 3072),
            "hidden_act": getattr(dec, "hidden_act", "silu"),
            "layer_scale_initial_scale": getattr(dec, "layer_scale_initial_scale", 0.01),
            "rms_norm_eps": getattr(dec, "rms_norm_eps", 1e-5),
            "num_hidden_layers": getattr(dec, "num_hidden_layers", 8),
            "num_quantizers": getattr(dec, "num_quantizers", 16),
            "upsample_rates": tuple(getattr(dec, "upsample_rates", (8, 5, 4, 3))),
            "upsampling_ratios": tuple(getattr(dec, "upsampling_ratios", (2, 2))),
            "decoder_dim": getattr(dec, "decoder_dim", 1536),
            "attention_dropout": getattr(dec, "attention_dropout", 0.0),
        }
    
    return SpeechTokenizerConfig(
        model_type=getattr(original_config, "model_type", "qwen3_tts_tokenizer_12hz"),
        encoder_config=encoder_dict,
        decoder_config=decoder_dict,
        encoder_valid_num_quantizers=getattr(original_config, "encoder_valid_num_quantizers", 16),
        input_sample_rate=getattr(original_config, "input_sample_rate", 24000),
        output_sample_rate=getattr(original_config, "output_sample_rate", 24000),
        decode_upsample_rate=getattr(original_config, "decode_upsample_rate", 1920),
        encode_downsample_rate=getattr(original_config, "encode_downsample_rate", 1920),
    )


__all__ = [
    "SpeechTokenizerConfig",
    "SpeechDecoderConfig",
    "MimiEncoderConfig",
    "convert_to_standalone_config",
]
