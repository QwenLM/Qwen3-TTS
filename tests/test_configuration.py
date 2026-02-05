# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the standalone configuration classes.

These tests verify that:
1. Standalone configs work without transformers dependency
2. Serialization/deserialization works correctly
3. Conversion from transformers-based configs works
4. Validation functions work correctly
"""

import json
import tempfile
from pathlib import Path

import pytest

# Standalone configurations (transformers-free)
from qwen_tts.core.models.configuration_qwen3_tts_standalone import (
    BaseConfig,
    Qwen3TTSConfigStandalone,
    Qwen3TTSTalkerConfigStandalone,
    Qwen3TTSSpeakerEncoderConfigStandalone,
    Qwen3TTSTalkerCodePredictorConfigStandalone,
    layer_type_validation,
    rope_config_validation,
    convert_tts_config,
    convert_talker_config,
    convert_speaker_encoder_config,
    convert_code_predictor_config,
)

# Original configurations (for conversion tests)
from qwen_tts.core.models.configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSTalkerConfig,
    Qwen3TTSSpeakerEncoderConfig,
    Qwen3TTSTalkerCodePredictorConfig,
)


class TestBaseConfig:
    """Test the BaseConfig class functionality."""

    def test_basic_instantiation(self):
        """Test that BaseConfig can be instantiated with kwargs."""
        config = BaseConfig(foo="bar", num=42)
        assert config.foo == "bar"
        assert config.num == 42

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = BaseConfig(foo="bar", num=42)
        d = config.to_dict()
        
        assert isinstance(d, dict)
        assert d["foo"] == "bar"
        assert d["num"] == 42
        assert d["model_type"] == "base"

    def test_from_dict(self):
        """Test construction from dictionary."""
        d = {"foo": "bar", "num": 42, "model_type": "base"}
        config = BaseConfig.from_dict(d)
        
        assert config.foo == "bar"
        assert config.num == 42

    def test_to_json_string(self):
        """Test JSON string serialization."""
        config = BaseConfig(foo="bar", num=42)
        json_str = config.to_json_string()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["foo"] == "bar"
        assert parsed["num"] == 42

    def test_copy(self):
        """Test deep copy functionality."""
        config = BaseConfig(foo="bar", nested={"a": 1})
        copied = config.copy()
        
        assert copied.foo == config.foo
        assert copied.nested == config.nested
        
        # Modify original, copied should be unchanged
        config.nested["a"] = 999
        assert copied.nested["a"] == 1

    def test_update(self):
        """Test update from dictionary."""
        config = BaseConfig(foo="bar", num=42)
        config.update({"foo": "baz", "num": 100})
        
        assert config.foo == "baz"
        assert config.num == 100

    def test_save_and_load_pretrained(self):
        """Test saving and loading configuration."""
        config = BaseConfig(foo="bar", num=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            config.save_pretrained(tmpdir)
            
            # Verify file exists
            config_path = Path(tmpdir) / "config.json"
            assert config_path.exists()
            
            # Load
            loaded = BaseConfig.from_pretrained(tmpdir)
            
            assert loaded.foo == "bar"
            assert loaded.num == 42

    def test_equality(self):
        """Test configuration equality comparison."""
        config1 = BaseConfig(foo="bar", num=42)
        config2 = BaseConfig(foo="bar", num=42)
        config3 = BaseConfig(foo="different", num=42)
        
        assert config1 == config2
        assert config1 != config3

    def test_pretrained_config_compatibility_attributes(self):
        """Test that BaseConfig has PretrainedConfig compatibility attributes."""
        config = BaseConfig(foo="bar")
        
        # Check default values for compatibility attributes
        assert config._name_or_path == ""
        # _attn_implementation defaults to "sdpa" for transformers compatibility
        assert config._attn_implementation == "sdpa"
        assert config._attn_implementation_internal == "sdpa"
        assert config.output_hidden_states is False
        assert config.output_attentions is False
        assert config.return_dict is True
        assert config.is_encoder_decoder is False
        assert config.is_decoder is False
        assert config.pad_token_id is None
        assert config.bos_token_id is None
        assert config.eos_token_id is None

    def test_name_or_path_property(self):
        """Test the name_or_path property."""
        config = BaseConfig()
        
        # Test getter
        assert config.name_or_path == ""
        
        # Test setter
        config.name_or_path = "some/path"
        assert config.name_or_path == "some/path"
        assert config._name_or_path == "some/path"

    def test_from_pretrained_stores_path(self):
        """Test that from_pretrained stores the path in _name_or_path."""
        config = BaseConfig(foo="bar")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded = BaseConfig.from_pretrained(tmpdir)
            
            # The path should be stored
            assert loaded._name_or_path == tmpdir

    def test_attribute_map_exists(self):
        """Test that attribute_map class attribute exists."""
        assert hasattr(BaseConfig, "attribute_map")
        assert isinstance(BaseConfig.attribute_map, dict)


class TestSpeakerEncoderConfig:
    """Test Qwen3TTSSpeakerEncoderConfigStandalone."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = Qwen3TTSSpeakerEncoderConfigStandalone()
        
        assert config.mel_dim == 128
        assert config.enc_dim == 1024
        assert config.enc_channels == [512, 512, 512, 512, 1536]
        assert config.enc_kernel_sizes == [5, 3, 3, 3, 1]
        assert config.enc_dilations == [1, 2, 3, 4, 1]
        assert config.enc_attention_channels == 128
        assert config.enc_res2net_scale == 8
        assert config.enc_se_channels == 128
        assert config.sample_rate == 24000

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = Qwen3TTSSpeakerEncoderConfigStandalone(
            mel_dim=80,
            enc_dim=256,
            enc_channels=[256, 256, 512],
            sample_rate=16000,
        )
        
        assert config.mel_dim == 80
        assert config.enc_dim == 256
        assert config.enc_channels == [256, 256, 512]
        assert config.sample_rate == 16000

    def test_serialization_roundtrip(self):
        """Test that config survives serialization roundtrip."""
        original = Qwen3TTSSpeakerEncoderConfigStandalone(
            mel_dim=80,
            enc_dim=256,
        )
        
        d = original.to_dict()
        restored = Qwen3TTSSpeakerEncoderConfigStandalone.from_dict(d)
        
        assert restored.mel_dim == original.mel_dim
        assert restored.enc_dim == original.enc_dim
        assert restored.model_type == "qwen3_tts_speaker_encoder"


class TestCodePredictorConfig:
    """Test Qwen3TTSTalkerCodePredictorConfigStandalone."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = Qwen3TTSTalkerCodePredictorConfigStandalone()
        
        assert config.vocab_size == 2048
        assert config.hidden_size == 1024
        assert config.intermediate_size == 3072
        assert config.num_hidden_layers == 5
        assert config.num_attention_heads == 16
        assert config.num_key_value_heads == 8
        assert config.head_dim == 128
        assert config.hidden_act == "silu"
        assert config.num_code_groups == 32

    def test_layer_types_auto_generation(self):
        """Test that layer_types is auto-generated when not provided."""
        config = Qwen3TTSTalkerCodePredictorConfigStandalone(
            num_hidden_layers=5,
            use_sliding_window=False,
        )
        
        assert config.layer_types == ["full_attention"] * 5

    def test_layer_types_with_sliding_window(self):
        """Test layer_types generation with sliding window."""
        config = Qwen3TTSTalkerCodePredictorConfigStandalone(
            num_hidden_layers=32,
            use_sliding_window=True,
            max_window_layers=28,
        )
        
        # First 28 should be full_attention, rest sliding_attention
        assert config.layer_types[:28] == ["full_attention"] * 28
        assert config.layer_types[28:] == ["sliding_attention"] * 4

    def test_rope_scaling_validation(self):
        """Test rope_scaling validation."""
        # Valid rope_scaling
        config = Qwen3TTSTalkerCodePredictorConfigStandalone(
            rope_scaling={"rope_type": "linear", "factor": 2.0}
        )
        assert config.rope_scaling["rope_type"] == "linear"
        
        # Invalid rope_type should raise
        with pytest.raises(ValueError, match="Invalid rope_type"):
            Qwen3TTSTalkerCodePredictorConfigStandalone(
                rope_scaling={"rope_type": "invalid_type"}
            )

    def test_serialization_roundtrip(self):
        """Test that config survives serialization roundtrip."""
        original = Qwen3TTSTalkerCodePredictorConfigStandalone(
            vocab_size=4096,
            hidden_size=2048,
            rope_scaling={"rope_type": "linear", "factor": 2.0},
        )
        
        d = original.to_dict()
        restored = Qwen3TTSTalkerCodePredictorConfigStandalone.from_dict(d)
        
        assert restored.vocab_size == original.vocab_size
        assert restored.hidden_size == original.hidden_size
        assert restored.rope_scaling == original.rope_scaling


class TestTalkerConfig:
    """Test Qwen3TTSTalkerConfigStandalone."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = Qwen3TTSTalkerConfigStandalone()
        
        assert config.vocab_size == 3072
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 20
        assert config.text_hidden_size == 2048
        assert config.codec_eos_token_id == 4198

    def test_nested_code_predictor_config(self):
        """Test that code_predictor_config is properly nested."""
        config = Qwen3TTSTalkerConfigStandalone()
        
        assert isinstance(config.code_predictor_config, Qwen3TTSTalkerCodePredictorConfigStandalone)
        assert config.code_predictor_config.vocab_size == 2048

    def test_code_predictor_config_from_dict(self):
        """Test that code_predictor_config can be provided as dict."""
        config = Qwen3TTSTalkerConfigStandalone(
            code_predictor_config={"vocab_size": 4096, "hidden_size": 512}
        )
        
        assert config.code_predictor_config.vocab_size == 4096
        assert config.code_predictor_config.hidden_size == 512

    def test_code_predictor_config_as_instance(self):
        """Test that code_predictor_config can be provided as instance."""
        cp_config = Qwen3TTSTalkerCodePredictorConfigStandalone(vocab_size=8192)
        config = Qwen3TTSTalkerConfigStandalone(code_predictor_config=cp_config)
        
        assert config.code_predictor_config.vocab_size == 8192

    def test_serialization_roundtrip_with_nested(self):
        """Test that nested config survives serialization roundtrip."""
        original = Qwen3TTSTalkerConfigStandalone(
            vocab_size=4096,
            code_predictor_config={"vocab_size": 8192, "hidden_size": 2048}
        )
        
        d = original.to_dict()
        restored = Qwen3TTSTalkerConfigStandalone.from_dict(d)
        
        assert restored.vocab_size == original.vocab_size
        assert restored.code_predictor_config.vocab_size == 8192
        assert restored.code_predictor_config.hidden_size == 2048


class TestMainConfig:
    """Test Qwen3TTSConfigStandalone."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = Qwen3TTSConfigStandalone()
        
        assert config.im_start_token_id == 151644
        assert config.im_end_token_id == 151645
        assert config.tts_pad_token_id == 151671
        assert config.tts_bos_token_id == 151672
        assert config.tts_eos_token_id == 151673

    def test_nested_configs(self):
        """Test that nested configs are properly initialized."""
        config = Qwen3TTSConfigStandalone()
        
        assert isinstance(config.talker_config, Qwen3TTSTalkerConfigStandalone)
        assert isinstance(config.speaker_encoder_config, Qwen3TTSSpeakerEncoderConfigStandalone)
        assert isinstance(
            config.talker_config.code_predictor_config,
            Qwen3TTSTalkerCodePredictorConfigStandalone
        )

    def test_nested_configs_from_dict(self):
        """Test that nested configs can be provided as dicts."""
        config = Qwen3TTSConfigStandalone(
            talker_config={"vocab_size": 4096, "hidden_size": 2048},
            speaker_encoder_config={"mel_dim": 80, "enc_dim": 256},
        )
        
        assert config.talker_config.vocab_size == 4096
        assert config.talker_config.hidden_size == 2048
        assert config.speaker_encoder_config.mel_dim == 80
        assert config.speaker_encoder_config.enc_dim == 256

    def test_full_serialization_roundtrip(self):
        """Test full config with all nested configs survives roundtrip."""
        original = Qwen3TTSConfigStandalone(
            talker_config={
                "vocab_size": 4096,
                "code_predictor_config": {"vocab_size": 8192}
            },
            speaker_encoder_config={"mel_dim": 80},
            tokenizer_type="qwen3-tts-tokenizer-v2",
            tts_model_size="0.5B",
        )
        
        # Test to_dict
        d = original.to_dict()
        assert d["talker_config"]["vocab_size"] == 4096
        assert d["talker_config"]["code_predictor_config"]["vocab_size"] == 8192
        assert d["speaker_encoder_config"]["mel_dim"] == 80
        
        # Test roundtrip
        restored = Qwen3TTSConfigStandalone.from_dict(d)
        assert restored.talker_config.vocab_size == 4096
        assert restored.talker_config.code_predictor_config.vocab_size == 8192
        assert restored.speaker_encoder_config.mel_dim == 80
        assert restored.tokenizer_type == "qwen3-tts-tokenizer-v2"
        assert restored.tts_model_size == "0.5B"

    def test_save_and_load_full_config(self):
        """Test saving and loading full config to/from file."""
        original = Qwen3TTSConfigStandalone(
            talker_config={"vocab_size": 4096},
            speaker_encoder_config={"enc_dim": 512},
            tokenizer_type="test-tokenizer",
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original.save_pretrained(tmpdir)
            
            loaded = Qwen3TTSConfigStandalone.from_pretrained(tmpdir)
            
            assert loaded.talker_config.vocab_size == 4096
            assert loaded.speaker_encoder_config.enc_dim == 512
            assert loaded.tokenizer_type == "test-tokenizer"


class TestValidationFunctions:
    """Test validation utility functions."""

    def test_layer_type_validation_valid(self):
        """Test that valid layer types pass validation."""
        # Should not raise
        layer_type_validation(["full_attention", "full_attention"])
        layer_type_validation(["sliding_attention", "sliding_attention"])
        layer_type_validation(["full_attention", "sliding_attention"])

    def test_layer_type_validation_invalid(self):
        """Test that invalid layer types raise error."""
        with pytest.raises(ValueError, match="Invalid layer types"):
            layer_type_validation(["full_attention", "invalid_type"])

    def test_layer_type_validation_empty(self):
        """Test that empty list passes validation."""
        layer_type_validation([])

    def test_rope_config_validation_none(self):
        """Test that None rope_scaling passes validation."""
        class MockConfig:
            rope_scaling = None
        
        # Should not raise
        rope_config_validation(MockConfig())

    def test_rope_config_validation_valid(self):
        """Test that valid rope_scaling passes validation."""
        class MockConfig:
            rope_scaling = {"rope_type": "linear", "factor": 2.0}
        
        # Should not raise
        rope_config_validation(MockConfig())

    def test_rope_config_validation_default_type(self):
        """Test that default rope_type is inferred."""
        class MockConfig:
            rope_scaling = {}  # No rope_type specified
        
        # Should not raise - defaults to "default"
        rope_config_validation(MockConfig())

    def test_rope_config_validation_invalid_type(self):
        """Test that invalid rope_type raises error."""
        class MockConfig:
            rope_scaling = {"rope_type": "invalid"}
        
        with pytest.raises(ValueError, match="Invalid rope_type"):
            rope_config_validation(MockConfig())

    def test_rope_config_validation_invalid_factor(self):
        """Test that negative factor raises error."""
        class MockConfig:
            rope_scaling = {"rope_type": "linear", "factor": -1.0}
        
        with pytest.raises(ValueError, match="factor must be positive"):
            rope_config_validation(MockConfig())


class TestConfigConversion:
    """Test conversion functions from transformers-based configs."""

    def test_convert_speaker_encoder_config(self):
        """Test converting speaker encoder config."""
        old_config = Qwen3TTSSpeakerEncoderConfig(
            mel_dim=80,
            enc_dim=256,
            enc_channels=[256, 512, 512],
            sample_rate=16000,
        )
        
        new_config = convert_speaker_encoder_config(old_config)
        
        assert isinstance(new_config, Qwen3TTSSpeakerEncoderConfigStandalone)
        assert new_config.mel_dim == 80
        assert new_config.enc_dim == 256
        assert new_config.enc_channels == [256, 512, 512]
        assert new_config.sample_rate == 16000

    def test_convert_code_predictor_config(self):
        """Test converting code predictor config."""
        old_config = Qwen3TTSTalkerCodePredictorConfig(
            vocab_size=4096,
            hidden_size=2048,
            num_hidden_layers=8,
        )
        
        new_config = convert_code_predictor_config(old_config)
        
        assert isinstance(new_config, Qwen3TTSTalkerCodePredictorConfigStandalone)
        assert new_config.vocab_size == 4096
        assert new_config.hidden_size == 2048
        assert new_config.num_hidden_layers == 8

    def test_convert_talker_config(self):
        """Test converting talker config with nested code predictor."""
        old_config = Qwen3TTSTalkerConfig(
            vocab_size=6144,
            hidden_size=2048,
            code_predictor_config={"vocab_size": 8192},
        )
        
        new_config = convert_talker_config(old_config)
        
        assert isinstance(new_config, Qwen3TTSTalkerConfigStandalone)
        assert new_config.vocab_size == 6144
        assert new_config.hidden_size == 2048
        assert isinstance(new_config.code_predictor_config, Qwen3TTSTalkerCodePredictorConfigStandalone)
        assert new_config.code_predictor_config.vocab_size == 8192

    def test_convert_tts_config(self):
        """Test converting full TTS config with all nested configs."""
        old_config = Qwen3TTSConfig(
            talker_config={"vocab_size": 6144},
            speaker_encoder_config={"mel_dim": 80},
            tokenizer_type="qwen3-tts-tokenizer-v2",
            tts_model_size="0.5B",
        )
        
        new_config = convert_tts_config(old_config)
        
        assert isinstance(new_config, Qwen3TTSConfigStandalone)
        assert isinstance(new_config.talker_config, Qwen3TTSTalkerConfigStandalone)
        assert isinstance(new_config.speaker_encoder_config, Qwen3TTSSpeakerEncoderConfigStandalone)
        assert new_config.talker_config.vocab_size == 6144
        assert new_config.speaker_encoder_config.mel_dim == 80
        assert new_config.tokenizer_type == "qwen3-tts-tokenizer-v2"
        assert new_config.tts_model_size == "0.5B"

    def test_converted_config_serialization(self):
        """Test that converted config can be serialized and restored."""
        old_config = Qwen3TTSConfig(
            talker_config={"vocab_size": 6144},
            speaker_encoder_config={"mel_dim": 80},
        )
        
        new_config = convert_tts_config(old_config)
        
        # Serialize and restore
        with tempfile.TemporaryDirectory() as tmpdir:
            new_config.save_pretrained(tmpdir)
            loaded = Qwen3TTSConfigStandalone.from_pretrained(tmpdir)
            
            assert loaded.talker_config.vocab_size == 6144
            assert loaded.speaker_encoder_config.mel_dim == 80


class TestConfigEquivalenceWithOriginal:
    """Test that standalone configs maintain equivalence with original configs."""

    def test_speaker_encoder_config_equivalence(self):
        """Test that standalone and original configs have same values."""
        params = {
            "mel_dim": 128,
            "enc_dim": 256,
            "enc_channels": [512, 512, 512],
            "sample_rate": 16000,
        }
        
        original = Qwen3TTSSpeakerEncoderConfig(**params)
        standalone = Qwen3TTSSpeakerEncoderConfigStandalone(**params)
        
        assert original.mel_dim == standalone.mel_dim
        assert original.enc_dim == standalone.enc_dim
        assert original.enc_channels == standalone.enc_channels
        assert original.sample_rate == standalone.sample_rate

    def test_code_predictor_config_equivalence(self):
        """Test code predictor config equivalence."""
        params = {
            "vocab_size": 4096,
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
        }
        
        original = Qwen3TTSTalkerCodePredictorConfig(**params)
        standalone = Qwen3TTSTalkerCodePredictorConfigStandalone(**params)
        
        assert original.vocab_size == standalone.vocab_size
        assert original.hidden_size == standalone.hidden_size
        assert original.num_hidden_layers == standalone.num_hidden_layers
        assert original.num_attention_heads == standalone.num_attention_heads

    def test_talker_config_equivalence(self):
        """Test talker config equivalence."""
        params = {
            "vocab_size": 6144,
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "text_hidden_size": 4096,
        }
        
        original = Qwen3TTSTalkerConfig(**params)
        standalone = Qwen3TTSTalkerConfigStandalone(**params)
        
        assert original.vocab_size == standalone.vocab_size
        assert original.hidden_size == standalone.hidden_size
        assert original.num_hidden_layers == standalone.num_hidden_layers
        assert original.text_hidden_size == standalone.text_hidden_size

    def test_main_config_equivalence(self):
        """Test main config equivalence."""
        params = {
            "tokenizer_type": "qwen3-tts-tokenizer-v2",
            "tts_model_size": "0.5B",
            "im_start_token_id": 100,
            "tts_eos_token_id": 200,
        }
        
        original = Qwen3TTSConfig(**params)
        standalone = Qwen3TTSConfigStandalone(**params)
        
        assert original.tokenizer_type == standalone.tokenizer_type
        assert original.tts_model_size == standalone.tts_model_size
        assert original.im_start_token_id == standalone.im_start_token_id
        assert original.tts_eos_token_id == standalone.tts_eos_token_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
