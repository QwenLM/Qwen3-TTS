# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for standalone utility modules.

These tests verify the standalone implementations work correctly
without requiring the transformers library.
"""

import pytest
import torch
from torch import nn

from tests.conftest import set_seed


class TestCache:
    """Tests for standalone Cache and DynamicCache classes."""

    def test_dynamic_cache_initialization(self):
        """Test DynamicCache initializes correctly."""
        from qwen3_tts_standalone.utils import DynamicCache
        
        cache = DynamicCache()
        assert len(cache) == 0
        assert cache.get_seq_length() == 0

    def test_dynamic_cache_update_single_layer(self):
        """Test updating cache for a single layer."""
        from qwen3_tts_standalone.utils import DynamicCache
        
        cache = DynamicCache()
        
        # Create key/value tensors: (batch, num_heads, seq_len, head_dim)
        key = torch.randn(2, 4, 10, 64)
        value = torch.randn(2, 4, 10, 64)
        
        # Update cache
        cached_key, cached_value = cache.update(key, value, layer_idx=0)
        
        assert len(cache) == 1
        assert cache.get_seq_length(0) == 10
        assert torch.equal(cached_key, key)
        assert torch.equal(cached_value, value)

    def test_dynamic_cache_incremental_update(self):
        """Test incremental updates to cache."""
        from qwen3_tts_standalone.utils import DynamicCache
        
        cache = DynamicCache()
        
        # First update: prefill with 10 tokens
        key1 = torch.randn(2, 4, 10, 64)
        value1 = torch.randn(2, 4, 10, 64)
        cache.update(key1, value1, layer_idx=0)
        
        assert cache.get_seq_length(0) == 10
        
        # Second update: add 1 token
        key2 = torch.randn(2, 4, 1, 64)
        value2 = torch.randn(2, 4, 1, 64)
        cached_key, cached_value = cache.update(key2, value2, layer_idx=0)
        
        assert cache.get_seq_length(0) == 11
        assert cached_key.shape == (2, 4, 11, 64)
        assert cached_value.shape == (2, 4, 11, 64)
        
        # Verify concatenation is correct
        assert torch.equal(cached_key[:, :, :10, :], key1)
        assert torch.equal(cached_key[:, :, 10:, :], key2)

    def test_dynamic_cache_multiple_layers(self):
        """Test cache with multiple layers."""
        from qwen3_tts_standalone.utils import DynamicCache
        
        cache = DynamicCache()
        
        # Update layer 0
        key0 = torch.randn(2, 4, 10, 64)
        value0 = torch.randn(2, 4, 10, 64)
        cache.update(key0, value0, layer_idx=0)
        
        # Update layer 1
        key1 = torch.randn(2, 4, 10, 64)
        value1 = torch.randn(2, 4, 10, 64)
        cache.update(key1, value1, layer_idx=1)
        
        assert len(cache) == 2
        assert cache.get_seq_length(0) == 10
        assert cache.get_seq_length(1) == 10

    def test_dynamic_cache_reset(self):
        """Test cache reset functionality."""
        from qwen3_tts_standalone.utils import DynamicCache
        
        cache = DynamicCache()
        
        key = torch.randn(2, 4, 10, 64)
        value = torch.randn(2, 4, 10, 64)
        cache.update(key, value, layer_idx=0)
        
        assert len(cache) == 1
        
        cache.reset()
        
        assert len(cache) == 0
        assert cache.get_seq_length() == 0


class TestActivations:
    """Tests for standalone activation functions."""

    def test_act2fn_silu(self):
        """Test SiLU activation lookup."""
        from qwen3_tts_standalone.utils import ACT2FN
        
        activation = ACT2FN["silu"]
        x = torch.randn(2, 10)
        output = activation(x)
        
        expected = torch.nn.functional.silu(x)
        assert torch.allclose(output, expected)

    def test_act2fn_gelu(self):
        """Test GELU activation lookup."""
        from qwen3_tts_standalone.utils import ACT2FN
        
        activation = ACT2FN["gelu"]
        x = torch.randn(2, 10)
        output = activation(x)
        
        expected = torch.nn.functional.gelu(x)
        assert torch.allclose(output, expected)

    def test_act2fn_invalid_key(self):
        """Test that invalid activation name raises KeyError."""
        from qwen3_tts_standalone.utils import ACT2FN
        
        with pytest.raises(KeyError):
            _ = ACT2FN["invalid_activation"]

    def test_gelu_new_activation(self):
        """Test NewGELUActivation (tanh approximation)."""
        from qwen3_tts_standalone.utils import NewGELUActivation
        
        activation = NewGELUActivation()
        x = torch.tensor([0.0, 1.0, -1.0, 2.0])
        output = activation(x)
        
        # Should produce values close to GELU
        expected_gelu = torch.nn.functional.gelu(x)
        assert torch.allclose(output, expected_gelu, atol=0.1)


class TestRepeatKV:
    """Tests for repeat_kv function."""

    def test_repeat_kv_no_repeat(self):
        """Test repeat_kv with n_rep=1 (no repeat needed)."""
        from qwen3_tts_standalone.utils import repeat_kv
        
        x = torch.randn(2, 4, 10, 64)
        output = repeat_kv(x, n_rep=1)
        
        assert torch.equal(output, x)

    def test_repeat_kv_double(self):
        """Test repeat_kv with n_rep=2."""
        from qwen3_tts_standalone.utils import repeat_kv
        
        x = torch.randn(2, 4, 10, 64)  # 4 kv heads
        output = repeat_kv(x, n_rep=2)  # expand to 8 heads
        
        assert output.shape == (2, 8, 10, 64)
        # Check that heads are properly repeated
        assert torch.equal(output[:, 0, :, :], x[:, 0, :, :])
        assert torch.equal(output[:, 1, :, :], x[:, 0, :, :])
        assert torch.equal(output[:, 2, :, :], x[:, 1, :, :])
        assert torch.equal(output[:, 3, :, :], x[:, 1, :, :])


class TestModelOutputs:
    """Tests for standalone ModelOutput classes."""

    def test_base_model_output_with_past(self):
        """Test BaseModelOutputWithPast dataclass."""
        from qwen3_tts_standalone.utils import BaseModelOutputWithPast
        
        hidden_states = torch.randn(2, 10, 256)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        
        assert torch.equal(output.last_hidden_state, hidden_states)
        assert output["last_hidden_state"] is not None
        assert output.past_key_values is None

    def test_causal_lm_output_with_past(self):
        """Test CausalLMOutputWithPast dataclass."""
        from qwen3_tts_standalone.utils import CausalLMOutputWithPast
        
        logits = torch.randn(2, 10, 1000)
        output = CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        
        assert torch.equal(output.logits, logits)
        assert output["logits"] is not None
        assert output.loss is None

    def test_model_output_to_tuple(self):
        """Test ModelOutput to_tuple method filters None values."""
        from qwen3_tts_standalone.utils import BaseModelOutputWithPast
        
        hidden_states = torch.randn(2, 10, 256)
        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
        
        # to_tuple should only include non-None values
        result = output.to_tuple()
        assert len(result) == 1
        assert torch.equal(result[0], hidden_states)


class TestRoPE:
    """Tests for standalone RoPE utilities."""

    def test_rope_init_functions_registry(self):
        """Test ROPE_INIT_FUNCTIONS contains expected types."""
        from qwen3_tts_standalone.utils import ROPE_INIT_FUNCTIONS
        
        assert "default" in ROPE_INIT_FUNCTIONS
        assert "linear" in ROPE_INIT_FUNCTIONS
        assert "dynamic" in ROPE_INIT_FUNCTIONS

    def test_compute_default_rope_parameters(self):
        """Test default RoPE parameter computation."""
        from qwen3_tts_standalone.utils import ROPE_INIT_FUNCTIONS
        
        class MockConfig:
            rope_theta = 10000.0
            hidden_size = 256
            num_attention_heads = 4
            head_dim = 64
        
        config = MockConfig()
        inv_freq, attention_factor = ROPE_INIT_FUNCTIONS["default"](config, device="cpu")
        
        assert inv_freq.shape == (32,)  # head_dim // 2
        assert attention_factor == 1.0


class TestMasking:
    """Tests for standalone masking utilities."""

    def test_create_causal_mask_sdpa_returns_none(self):
        """Test create_causal_mask returns None for SDPA when no attention mask."""
        from qwen3_tts_standalone.utils import create_causal_mask
        
        class MockConfig:
            _attn_implementation = "sdpa"
        
        input_embeds = torch.randn(2, 10, 256)
        cache_position = torch.arange(10)
        
        mask = create_causal_mask(
            config=MockConfig(),
            input_embeds=input_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
        )
        
        assert mask is None

    def test_create_causal_mask_eager(self):
        """Test create_causal_mask creates proper mask for eager attention."""
        from qwen3_tts_standalone.utils import create_causal_mask
        
        class MockConfig:
            _attn_implementation = "eager"
        
        input_embeds = torch.randn(2, 5, 256)
        cache_position = torch.arange(5)
        
        mask = create_causal_mask(
            config=MockConfig(),
            input_embeds=input_embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
        )
        
        assert mask is not None
        assert mask.shape == (1, 1, 5, 5)
        
        # Check causal pattern: position i can attend to positions 0..i
        # Non-masked positions should be 0, masked should be -inf
        for i in range(5):
            # Can attend to positions <= i (should be 0)
            assert (mask[0, 0, i, :i+1] == 0).all()
            # Cannot attend to positions > i (should be -inf)
            if i < 4:
                assert (mask[0, 0, i, i+1:] < -1e30).all()


class TestAttentionFunctions:
    """Tests for standalone attention functions."""

    def test_all_attention_functions_registry(self):
        """Test ALL_ATTENTION_FUNCTIONS contains expected implementations."""
        from qwen3_tts_standalone.utils import ALL_ATTENTION_FUNCTIONS
        
        assert "sdpa" in ALL_ATTENTION_FUNCTIONS
        assert "flash_attention_2" in ALL_ATTENTION_FUNCTIONS

    def test_sdpa_attention_forward_basic(self):
        """Test SDPA attention forward pass."""
        from qwen3_tts_standalone.utils import sdpa_attention_forward
        
        class MockModule(nn.Module):
            is_causal = True
        
        module = MockModule()
        batch, heads, seq_len, head_dim = 2, 4, 10, 64
        
        query = torch.randn(batch, heads, seq_len, head_dim)
        key = torch.randn(batch, heads, seq_len, head_dim)
        value = torch.randn(batch, heads, seq_len, head_dim)
        
        output, _ = sdpa_attention_forward(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=None,
        )
        
        # Output should be (batch, seq_len, heads, head_dim) after transpose
        assert output.shape == (batch, seq_len, heads, head_dim)


class TestGenerationMixin:
    """Tests for standalone GenerationMixin."""

    def test_generate_output_dataclass(self):
        """Test GenerateOutput dataclass."""
        from qwen3_tts_standalone.utils import GenerateOutput
        
        sequences = torch.randint(0, 1000, (2, 20))
        output = GenerateOutput(sequences=sequences)
        
        assert torch.equal(output.sequences, sequences)
        assert output.hidden_states is None
        assert output.attentions is None

    def test_generation_mixin_update_kwargs(self):
        """Test _update_model_kwargs_for_generation method."""
        from qwen3_tts_standalone.utils import GenerationMixin, CausalLMOutputWithPast, DynamicCache
        
        class MockModel(GenerationMixin):
            pass
        
        model = MockModel()
        
        cache = DynamicCache()
        outputs = CausalLMOutputWithPast(
            logits=torch.randn(2, 1, 1000),
            past_key_values=cache,
        )
        
        model_kwargs = {
            "past_key_values": None,
            "cache_position": torch.tensor([5]),
        }
        
        updated_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1
        )
        
        assert updated_kwargs["past_key_values"] is cache
        assert updated_kwargs["cache_position"].item() == 6


class TestCacheEquivalence:
    """Test standalone cache produces same results as transformers cache."""

    def test_dynamic_cache_equivalence(self):
        """Test standalone DynamicCache behaves like transformers DynamicCache."""
        from qwen3_tts_standalone.utils import DynamicCache as StandaloneDynamicCache
        from transformers.cache_utils import DynamicCache as TransformersDynamicCache
        
        set_seed(42)
        
        # Create both caches
        standalone_cache = StandaloneDynamicCache()
        transformers_cache = TransformersDynamicCache()
        
        # Test data
        key = torch.randn(2, 4, 10, 64)
        value = torch.randn(2, 4, 10, 64)
        
        # Update both caches
        s_key, s_value = standalone_cache.update(key.clone(), value.clone(), layer_idx=0)
        t_key, t_value = transformers_cache.update(key.clone(), value.clone(), layer_idx=0)
        
        # Should produce identical results
        assert torch.equal(s_key, t_key)
        assert torch.equal(s_value, t_value)
        assert standalone_cache.get_seq_length(0) == transformers_cache.get_seq_length(0)
        
        # Test incremental update
        key2 = torch.randn(2, 4, 1, 64)
        value2 = torch.randn(2, 4, 1, 64)
        
        s_key2, s_value2 = standalone_cache.update(key2.clone(), value2.clone(), layer_idx=0)
        t_key2, t_value2 = transformers_cache.update(key2.clone(), value2.clone(), layer_idx=0)
        
        assert torch.equal(s_key2, t_key2)
        assert torch.equal(s_value2, t_value2)
        assert standalone_cache.get_seq_length(0) == transformers_cache.get_seq_length(0)
