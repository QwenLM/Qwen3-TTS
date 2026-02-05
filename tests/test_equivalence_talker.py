# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for TalkerDecoderLayer and TalkerAttention equivalence with multimodal rope.
"""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from tests.conftest import set_seed, copy_weights

# Original models
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSTalkerAttention,
    Qwen3TTSTalkerDecoderLayer,
    Qwen3TTSTalkerRotaryEmbedding,
)

# Standalone models
from qwen_tts.core.models.modeling_qwen3_tts_standalone import (
    Qwen3TTSTalkerAttentionStandalone,
    Qwen3TTSTalkerDecoderLayerStandalone,
)


def _create_talker_config():
    """Create a minimal config for talker decoder layer testing."""
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
        _attn_implementation = "eager"
        # mrope_section must sum to head_dim/2 because apply_multimodal_rotary_pos_emb doubles it
        rope_scaling = {
            "rope_type": "default",
            "mrope_section": [8, 12, 12],  # sum=32, after *2 = 64 = head_dim
            "interleaved": False,
        }
        max_position_embeddings = 2048
        rope_theta = 10000.0
    return MinimalConfig()


class TestTalkerDecoderLayerEquivalence:
    """Test TalkerDecoderLayer equivalence with caching."""

    def test_talker_decoder_layer_forward_no_cache(self):
        """Test talker decoder layer forward pass without caching."""
        set_seed(42)
        
        config = _create_talker_config()
        
        layer_orig = Qwen3TTSTalkerDecoderLayer(config, layer_idx=0)
        layer_standalone = Qwen3TTSTalkerDecoderLayerStandalone(config, layer_idx=0)
        
        # Copy weights
        copy_weights(layer_orig, layer_standalone)
        
        # Create inputs
        set_seed(42)
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Create position embeddings for talker (3D rope)
        rope_orig = Qwen3TTSTalkerRotaryEmbedding(config)
        # Position IDs for talker are 3D: (3, batch_size, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
        position_embeddings = rope_orig(hidden_states, position_ids)
        
        # Forward pass without cache
        with torch.no_grad():
            output_orig = layer_orig(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids[0],  # 2D for layer
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            output_standalone = layer_standalone(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids[0],
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
        
        assert torch.allclose(output_orig[0], output_standalone[0], atol=1e-5), \
            f"TalkerDecoderLayer outputs differ (no cache). Max diff: {(output_orig[0] - output_standalone[0]).abs().max()}"

    def test_talker_decoder_layer_forward_with_cache(self):
        """Test talker decoder layer forward pass with KV caching for recursive generation."""
        set_seed(42)
        
        config = _create_talker_config()
        
        layer_orig = Qwen3TTSTalkerDecoderLayer(config, layer_idx=0)
        layer_standalone = Qwen3TTSTalkerDecoderLayerStandalone(config, layer_idx=0)
        
        # Copy weights
        copy_weights(layer_orig, layer_standalone)
        
        batch_size = 2
        
        # === PREFILL PHASE ===
        set_seed(42)
        prefill_len = 8
        hidden_states_prefill = torch.randn(batch_size, prefill_len, config.hidden_size)
        
        # Create position embeddings for prefill
        rope = Qwen3TTSTalkerRotaryEmbedding(config)
        position_ids_prefill = torch.arange(prefill_len).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
        position_embeddings_prefill = rope(hidden_states_prefill, position_ids_prefill)
        cache_position_prefill = torch.arange(prefill_len)
        
        # Initialize caches
        cache_orig = DynamicCache()
        cache_standalone = DynamicCache()
        
        with torch.no_grad():
            output_orig_prefill = layer_orig(
                hidden_states_prefill,
                attention_mask=None,
                position_ids=position_ids_prefill[0],
                past_key_values=cache_orig,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position_prefill,
                position_embeddings=position_embeddings_prefill,
            )
            output_standalone_prefill = layer_standalone(
                hidden_states_prefill,
                attention_mask=None,
                position_ids=position_ids_prefill[0],
                past_key_values=cache_standalone,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position_prefill,
                position_embeddings=position_embeddings_prefill,
            )
        
        assert torch.allclose(output_orig_prefill[0], output_standalone_prefill[0], atol=1e-5), \
            f"TalkerDecoderLayer prefill outputs differ. Max diff: {(output_orig_prefill[0] - output_standalone_prefill[0]).abs().max()}"
        
        # === RECURSIVE GENERATION PHASE ===
        for step in range(3):  # Generate 3 tokens
            set_seed(100 + step)
            
            # Single token input
            hidden_states_step = torch.randn(batch_size, 1, config.hidden_size)
            
            # Position for this step
            current_pos = prefill_len + step
            position_ids_step = torch.tensor([[[current_pos]] * batch_size] * 3)  # (3, batch, 1)
            cache_position_step = torch.tensor([current_pos])
            position_embeddings_step = rope(hidden_states_step, position_ids_step)
            
            with torch.no_grad():
                output_orig_step = layer_orig(
                    hidden_states_step,
                    attention_mask=None,
                    position_ids=position_ids_step[0],
                    past_key_values=cache_orig,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position_step,
                    position_embeddings=position_embeddings_step,
                )
                output_standalone_step = layer_standalone(
                    hidden_states_step,
                    attention_mask=None,
                    position_ids=position_ids_step[0],
                    past_key_values=cache_standalone,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position_step,
                    position_embeddings=position_embeddings_step,
                )
            
            assert torch.allclose(output_orig_step[0], output_standalone_step[0], atol=1e-5), \
                f"TalkerDecoderLayer generation step {step} outputs differ. Max diff: {(output_orig_step[0] - output_standalone_step[0]).abs().max()}"
        
        # Verify cache lengths match
        assert cache_orig.get_seq_length() == cache_standalone.get_seq_length(), \
            f"Cache lengths differ: {cache_orig.get_seq_length()} vs {cache_standalone.get_seq_length()}"


class TestTalkerAttentionWithCaching:
    """Test TalkerAttention layers with multimodal rope and caching."""

    def test_talker_attention_recursive_generation(self):
        """Test talker attention with recursive generation using KV cache."""
        set_seed(42)
        
        config = _create_talker_config()
        
        attn_orig = Qwen3TTSTalkerAttention(config, layer_idx=0)
        attn_standalone = Qwen3TTSTalkerAttentionStandalone(config, layer_idx=0)
        
        # Copy weights
        copy_weights(attn_orig, attn_standalone)
        
        batch_size = 2
        
        # Create talker rotary embeddings (3D)
        rope = Qwen3TTSTalkerRotaryEmbedding(config)
        
        # === PREFILL PHASE ===
        set_seed(42)
        prefill_len = 8
        hidden_states_prefill = torch.randn(batch_size, prefill_len, config.hidden_size)
        # 3D position ids for talker
        position_ids_prefill = torch.arange(prefill_len).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
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
            f"TalkerAttention prefill outputs differ. Max diff: {(output_orig_prefill - output_standalone_prefill).abs().max()}"
        
        # === RECURSIVE GENERATION (5 steps) ===
        for step in range(5):
            set_seed(300 + step)
            
            hidden_states_step = torch.randn(batch_size, 1, config.hidden_size)
            current_pos = prefill_len + step
            position_ids_step = torch.tensor([[[current_pos]] * batch_size] * 3)
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
                f"TalkerAttention generation step {step} outputs differ. Max diff: {(output_orig_step - output_standalone_step).abs().max()}"


class TestTalkerCacheConsistency:
    """Test that original and standalone talker layers produce consistent results with caching."""

    def test_talker_cached_generation_equivalence(self):
        """
        Test that original and standalone talker layers produce identical outputs during
        cached generation with multimodal rope.
        """
        set_seed(42)
        
        config = _create_talker_config()
        
        # Create original and standalone layers
        layer_orig = Qwen3TTSTalkerDecoderLayer(config, layer_idx=0)
        layer_standalone = Qwen3TTSTalkerDecoderLayerStandalone(config, layer_idx=0)
        copy_weights(layer_orig, layer_standalone)
        
        batch_size = 2
        total_len = 10
        prefill_len = 5
        
        # Generate full input sequence
        set_seed(42)
        full_hidden_states = torch.randn(batch_size, total_len, config.hidden_size)
        
        rope = Qwen3TTSTalkerRotaryEmbedding(config)
        
        # Initialize caches
        cache_orig = DynamicCache()
        cache_standalone = DynamicCache()
        
        # === PREFILL PHASE ===
        hidden_states_prefill = full_hidden_states[:, :prefill_len, :]
        # 3D position IDs for talker
        position_ids_prefill = torch.arange(prefill_len).unsqueeze(0).unsqueeze(0).expand(3, batch_size, -1)
        position_embeddings_prefill = rope(hidden_states_prefill, position_ids_prefill)
        cache_position_prefill = torch.arange(prefill_len)
        
        with torch.no_grad():
            output_orig_prefill = layer_orig(
                hidden_states_prefill,
                attention_mask=None,
                position_ids=position_ids_prefill[0],
                past_key_values=cache_orig,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position_prefill,
                position_embeddings=position_embeddings_prefill,
            )
            output_standalone_prefill = layer_standalone(
                hidden_states_prefill,
                attention_mask=None,
                position_ids=position_ids_prefill[0],
                past_key_values=cache_standalone,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position_prefill,
                position_embeddings=position_embeddings_prefill,
            )
        
        assert torch.allclose(output_orig_prefill[0], output_standalone_prefill[0], atol=1e-5), \
            f"Talker prefill outputs differ. Max diff: {(output_orig_prefill[0] - output_standalone_prefill[0]).abs().max()}"
        
        # === INCREMENTAL GENERATION PHASE ===
        for step in range(prefill_len, total_len):
            hidden_states_step = full_hidden_states[:, step:step+1, :]
            position_ids_step = torch.tensor([[[step]] * batch_size] * 3)  # (3, batch, 1)
            cache_position_step = torch.tensor([step])
            position_embeddings_step = rope(hidden_states_step, position_ids_step)
            
            with torch.no_grad():
                output_orig_step = layer_orig(
                    hidden_states_step,
                    attention_mask=None,
                    position_ids=position_ids_step[0],
                    past_key_values=cache_orig,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position_step,
                    position_embeddings=position_embeddings_step,
                )
                output_standalone_step = layer_standalone(
                    hidden_states_step,
                    attention_mask=None,
                    position_ids=position_ids_step[0],
                    past_key_values=cache_standalone,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position_step,
                    position_embeddings=position_embeddings_step,
                )
            
            assert torch.allclose(output_orig_step[0], output_standalone_step[0], atol=1e-5), \
                f"Talker generation step {step - prefill_len} outputs differ. Max diff: {(output_orig_step[0] - output_standalone_step[0]).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
