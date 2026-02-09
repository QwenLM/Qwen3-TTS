# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the qwen3_tts_standalone package.

These tests verify that the standalone package can be imported and used
independently of the original qwen_tts package.
"""

import pytest
import torch


class TestStandalonePackageImports:
    """Test that all expected modules can be imported from the standalone package."""
    
    def test_import_main_module(self):
        """Test that the main module can be imported."""
        import qwen3_tts_standalone
        assert hasattr(qwen3_tts_standalone, "__version__")
    
    def test_import_tts(self):
        """Test that TTS can be imported."""
        from qwen3_tts_standalone import TTS
        assert TTS is not None
    
    def test_import_talker(self):
        """Test that Talker can be imported."""
        from qwen3_tts_standalone import Talker
        assert Talker is not None
    
    def test_import_code_predictor(self):
        """Test that CodePredictor can be imported."""
        from qwen3_tts_standalone import CodePredictor
        assert CodePredictor is not None
    
    def test_import_configurations(self):
        """Test that configuration classes can be imported."""
        from qwen3_tts_standalone import (
            TTSConfig,
            TalkerConfig,
            CodePredictorConfig,
            SpeakerEncoderConfig,
        )
        assert all([
            TTSConfig,
            TalkerConfig,
            CodePredictorConfig,
            SpeakerEncoderConfig,
        ])
    
    def test_import_processor(self):
        """Test that processor can be imported."""
        from qwen3_tts_standalone import Processor
        assert Processor is not None
    
    def test_import_utils(self):
        """Test that utils can be imported."""
        from qwen3_tts_standalone.utils import (
            DynamicCache,
            sample_top_k_top_p,
            create_causal_mask,
        )
        assert all([DynamicCache, sample_top_k_top_p, create_causal_mask])


class TestStandaloneCodePredictor:
    """Test CodePredictor from the standalone package."""
    
    @pytest.fixture
    def small_config(self):
        """Create a small config for testing."""
        from qwen3_tts_standalone import CodePredictorConfig
        return CodePredictorConfig(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=128,
            rms_norm_eps=1e-5,
            max_position_embeddings=256,
            num_code_groups=4,
        )
    
    def test_instantiation(self, small_config):
        """Test that CodePredictor can be instantiated."""
        from qwen3_tts_standalone import CodePredictor
        model = CodePredictor(
            config=small_config,
            embedding_dim=64,
        )
        assert model is not None
    
    def test_generate(self, small_config):
        """Test that CodePredictor can generate."""
        from qwen3_tts_standalone import CodePredictor
        model = CodePredictor(
            config=small_config,
            embedding_dim=64,
        )
        
        batch_size = 2
        seq_len = 3
        # inputs_embeds: [batch, seq, hidden]
        inputs_embeds = torch.randn(batch_size, seq_len, 64)
        
        output = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=3,  # num_code_groups - 1
            do_sample=False,
        )
        
        # Should predict num_code_groups - 1 tokens
        assert output.sequences.shape == (batch_size, 3)


class TestStandaloneTalker:
    """Test Talker from the standalone package."""
    
    @pytest.fixture
    def small_talker_config(self):
        """Create a small Talker config for testing."""
        from qwen3_tts_standalone import (
            TalkerConfig,
            CodePredictorConfig,
        )
        code_predictor_config = CodePredictorConfig(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=128,
            rms_norm_eps=1e-5,
            max_position_embeddings=256,
            num_code_groups=4,
        )
        # head_dim = hidden_size / num_attention_heads = 64 / 4 = 16
        # mrope_section must sum to head_dim / 2 = 8
        rope_scaling = {
            "rope_type": "default",
            "mrope_section": [2, 3, 3],  # sum = 8 = head_dim / 2
            "interleaved": False,
        }
        return TalkerConfig(
            vocab_size=256,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=128,
            rms_norm_eps=1e-5,
            max_position_embeddings=256,
            num_code_groups=4,
            code_predictor_config=code_predictor_config,
            codec_eos_token_id=255,
            rope_scaling=rope_scaling,
            text_vocab_size=256,
            text_hidden_size=64,
        )
    
    def test_instantiation(self, small_talker_config):
        """Test that Talker can be instantiated."""
        from qwen3_tts_standalone import Talker
        model = Talker(small_talker_config)
        assert model is not None
    
    def test_generate(self, small_talker_config):
        """Test that Talker can generate."""
        from qwen3_tts_standalone import Talker
        model = Talker(small_talker_config)
        
        batch_size = 2
        seq_len = 10
        inputs_embeds = torch.randn(batch_size, seq_len, 64)
        
        output = model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=5,
            do_sample=False,
        )
        
        assert output.sequences.shape[0] == batch_size
        assert output.sequences.shape[1] <= 5


class TestStandaloneDynamicCache:
    """Test DynamicCache from the standalone package."""
    
    def test_instantiation(self):
        """Test that DynamicCache can be instantiated."""
        from qwen3_tts_standalone.utils import DynamicCache
        cache = DynamicCache()
        assert cache is not None
    
    def test_update_and_get(self):
        """Test cache update and retrieval."""
        from qwen3_tts_standalone.utils import DynamicCache
        cache = DynamicCache()
        
        key = torch.randn(2, 4, 10, 16)
        value = torch.randn(2, 4, 10, 16)
        
        # Update returns the updated key/value
        returned_key, returned_value = cache.update(key, value, layer_idx=0)
        
        # Should be the same since this is the first update
        assert torch.allclose(returned_key, key)
        assert torch.allclose(returned_value, value)
        
        # Also accessible via cache lists
        assert torch.allclose(cache.key_cache[0], key)
        assert torch.allclose(cache.value_cache[0], value)
