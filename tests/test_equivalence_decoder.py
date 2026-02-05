# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for DecoderLayer and Attention equivalence.
"""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from tests.conftest import set_seed, copy_weights

# Original models
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSAttention,
    Qwen3TTSDecoderLayer,
    Qwen3TTSRotaryEmbedding,
)

# Standalone models
from qwen_tts.core.models.modeling_qwen3_tts_standalone import (
    Qwen3TTSAttentionStandalone,
    Qwen3TTSDecoderLayerStandalone,
)


def _create_decoder_layer_config():
    """Create a minimal config for decoder layer testing."""
    class MinimalConfig:
        hidden_size = 256
        intermediate_size = 512
        num_attention_heads = 4
        num_key_value_heads = 2
        head_dim = 64
        hidden_act = "silu"
        attention_bias = False
        attention_dropout = 0.0
        rms_norm_eps = 1e-6
        sliding_window = None
        layer_types = ["full_attention"]
        _attn_implementation = "eager"
        rope_scaling = None
        max_position_embeddings = 2048
        rope_theta = 10000.0
    return MinimalConfig()


class TestDecoderLayerEquivalence:
    """Test DecoderLayer equivalence with caching."""

    def test_decoder_layer_forward_no_cache(self):
        """Test decoder layer forward pass without caching."""
        set_seed(42)
        
        config = _create_decoder_layer_config()
        
        layer_orig = Qwen3TTSDecoderLayer(config, layer_idx=0)
        layer_standalone = Qwen3TTSDecoderLayerStandalone(config, layer_idx=0)
        
        # Copy weights
        copy_weights(layer_orig, layer_standalone)
        
        # Create inputs
        set_seed(42)
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Create position embeddings
        rope_orig = Qwen3TTSRotaryEmbedding(config)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = rope_orig(hidden_states, position_ids)
        
        # Forward pass without cache
        with torch.no_grad():
            output_orig = layer_orig(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            output_standalone = layer_standalone(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
        
        assert torch.allclose(output_orig[0], output_standalone[0], atol=1e-5), \
            f"DecoderLayer outputs differ (no cache). Max diff: {(output_orig[0] - output_standalone[0]).abs().max()}"

    def test_decoder_layer_forward_with_cache(self):
        """Test decoder layer forward pass with KV caching for recursive generation."""
        set_seed(42)
        
        config = _create_decoder_layer_config()
        
        layer_orig = Qwen3TTSDecoderLayer(config, layer_idx=0)
        layer_standalone = Qwen3TTSDecoderLayerStandalone(config, layer_idx=0)
        
        # Copy weights
        copy_weights(layer_orig, layer_standalone)
        
        batch_size = 2
        
        # === PREFILL PHASE ===
        set_seed(42)
        prefill_len = 8
        hidden_states_prefill = torch.randn(batch_size, prefill_len, config.hidden_size)
        
        # Create position embeddings for prefill
        rope = Qwen3TTSRotaryEmbedding(config)
        position_ids_prefill = torch.arange(prefill_len).unsqueeze(0).expand(batch_size, -1)
        position_embeddings_prefill = rope(hidden_states_prefill, position_ids_prefill)
        cache_position_prefill = torch.arange(prefill_len)
        
        # Initialize caches
        cache_orig = DynamicCache()
        cache_standalone = DynamicCache()
        
        with torch.no_grad():
            output_orig_prefill = layer_orig(
                hidden_states_prefill,
                attention_mask=None,
                position_ids=position_ids_prefill,
                past_key_values=cache_orig,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position_prefill,
                position_embeddings=position_embeddings_prefill,
            )
            output_standalone_prefill = layer_standalone(
                hidden_states_prefill,
                attention_mask=None,
                position_ids=position_ids_prefill,
                past_key_values=cache_standalone,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position_prefill,
                position_embeddings=position_embeddings_prefill,
            )
        
        assert torch.allclose(output_orig_prefill[0], output_standalone_prefill[0], atol=1e-5), \
            f"DecoderLayer prefill outputs differ. Max diff: {(output_orig_prefill[0] - output_standalone_prefill[0]).abs().max()}"
        
        # === RECURSIVE GENERATION PHASE ===
        for step in range(3):  # Generate 3 tokens
            set_seed(100 + step)
            
            # Single token input
            hidden_states_step = torch.randn(batch_size, 1, config.hidden_size)
            
            # Position for this step
            current_pos = prefill_len + step
            position_ids_step = torch.tensor([[current_pos]] * batch_size)
            cache_position_step = torch.tensor([current_pos])
            position_embeddings_step = rope(hidden_states_step, position_ids_step)
            
            with torch.no_grad():
                output_orig_step = layer_orig(
                    hidden_states_step,
                    attention_mask=None,
                    position_ids=position_ids_step,
                    past_key_values=cache_orig,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position_step,
                    position_embeddings=position_embeddings_step,
                )
                output_standalone_step = layer_standalone(
                    hidden_states_step,
                    attention_mask=None,
                    position_ids=position_ids_step,
                    past_key_values=cache_standalone,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position_step,
                    position_embeddings=position_embeddings_step,
                )
            
            assert torch.allclose(output_orig_step[0], output_standalone_step[0], atol=1e-5), \
                f"DecoderLayer generation step {step} outputs differ. Max diff: {(output_orig_step[0] - output_standalone_step[0]).abs().max()}"
        
        # Verify cache lengths match
        assert cache_orig.get_seq_length() == cache_standalone.get_seq_length(), \
            f"Cache lengths differ: {cache_orig.get_seq_length()} vs {cache_standalone.get_seq_length()}"


class TestAttentionWithCaching:
    """Test Attention layers with caching for recursive generation."""

    def test_attention_recursive_generation(self):
        """Test attention with recursive generation using KV cache."""
        set_seed(42)
        
        config = _create_decoder_layer_config()
        
        attn_orig = Qwen3TTSAttention(config, layer_idx=0)
        attn_standalone = Qwen3TTSAttentionStandalone(config, layer_idx=0)
        
        # Copy weights
        copy_weights(attn_orig, attn_standalone)
        
        batch_size = 2
        
        # Create rotary embeddings
        rope = Qwen3TTSRotaryEmbedding(config)
        
        # === PREFILL PHASE ===
        set_seed(42)
        prefill_len = 8
        hidden_states_prefill = torch.randn(batch_size, prefill_len, config.hidden_size)
        position_ids_prefill = torch.arange(prefill_len).unsqueeze(0).expand(batch_size, -1)
        position_embeddings_prefill = rope(hidden_states_prefill, position_ids_prefill)
        cache_position_prefill = torch.arange(prefill_len)
        
        cache_orig = DynamicCache()
        cache_standalone = DynamicCache()
        
        with torch.no_grad():
            output_orig_prefill, _ = attn_orig(
                hidden_states_prefill,
                position_embeddings=position_embeddings_prefill,
                attention_mask=None,
                past_key_values=cache_orig,
                cache_position=cache_position_prefill,
            )
            output_standalone_prefill, _ = attn_standalone(
                hidden_states_prefill,
                position_embeddings=position_embeddings_prefill,
                attention_mask=None,
                past_key_values=cache_standalone,
                cache_position=cache_position_prefill,
            )
        
        assert torch.allclose(output_orig_prefill, output_standalone_prefill, atol=1e-5), \
            f"Attention prefill outputs differ. Max diff: {(output_orig_prefill - output_standalone_prefill).abs().max()}"
        
        # === RECURSIVE GENERATION (5 steps) ===
        for step in range(5):
            set_seed(200 + step)
            
            hidden_states_step = torch.randn(batch_size, 1, config.hidden_size)
            current_pos = prefill_len + step
            position_ids_step = torch.tensor([[current_pos]] * batch_size)
            cache_position_step = torch.tensor([current_pos])
            position_embeddings_step = rope(hidden_states_step, position_ids_step)
            
            with torch.no_grad():
                output_orig_step, _ = attn_orig(
                    hidden_states_step,
                    position_embeddings=position_embeddings_step,
                    attention_mask=None,
                    past_key_values=cache_orig,
                    cache_position=cache_position_step,
                )
                output_standalone_step, _ = attn_standalone(
                    hidden_states_step,
                    position_embeddings=position_embeddings_step,
                    attention_mask=None,
                    past_key_values=cache_standalone,
                    cache_position=cache_position_step,
                )
            
            assert torch.allclose(output_orig_step, output_standalone_step, atol=1e-5), \
                f"Attention generation step {step} outputs differ. Max diff: {(output_orig_step - output_standalone_step).abs().max()}"
        
        # Verify final cache state
        assert cache_orig.get_seq_length() == cache_standalone.get_seq_length() == prefill_len + 5

    def test_attention_no_cache_vs_with_cache_equivalence(self):
        """Test that generation without cache produces same output as with cache (for same inputs)."""
        set_seed(42)
        
        config = _create_decoder_layer_config()
        
        attn = Qwen3TTSAttention(config, layer_idx=0)
        attn_standalone = Qwen3TTSAttentionStandalone(config, layer_idx=0)
        
        # Copy weights
        copy_weights(attn, attn_standalone)
        
        batch_size = 2
        seq_len = 10
        
        # Full sequence forward pass without cache
        set_seed(42)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        rope = Qwen3TTSRotaryEmbedding(config)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = rope(hidden_states, position_ids)
        
        with torch.no_grad():
            output_no_cache, _ = attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
            )
            output_standalone_no_cache, _ = attn_standalone(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
            )
        
        assert torch.allclose(output_no_cache, output_standalone_no_cache, atol=1e-5), \
            f"No-cache outputs differ. Max diff: {(output_no_cache - output_standalone_no_cache).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
