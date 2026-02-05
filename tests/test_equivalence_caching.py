# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for cache consistency between original and standalone models.
"""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from tests.conftest import set_seed, copy_weights

# Original models
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSDecoderLayer,
    Qwen3TTSRotaryEmbedding,
)

# Standalone models
from qwen_tts.core.models.modeling_qwen3_tts_standalone import (
    Qwen3TTSDecoderLayerStandalone,
)


class TestCacheConsistency:
    """Test that original and standalone produce consistent results with caching."""

    def test_cached_generation_equivalence(self):
        """
        Test that original and standalone layers produce identical outputs during
        cached generation (prefill + incremental steps).
        """
        set_seed(42)
        
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
        
        config = MinimalConfig()
        
        # Create original and standalone layers
        layer_orig = Qwen3TTSDecoderLayer(config, layer_idx=0)
        layer_standalone = Qwen3TTSDecoderLayerStandalone(config, layer_idx=0)
        copy_weights(layer_orig, layer_standalone)
        
        batch_size = 2
        total_len = 12
        prefill_len = 6
        
        # Generate full input sequence
        set_seed(42)
        full_hidden_states = torch.randn(batch_size, total_len, config.hidden_size)
        
        rope = Qwen3TTSRotaryEmbedding(config)
        
        # Initialize caches
        cache_orig = DynamicCache()
        cache_standalone = DynamicCache()
        
        # === PREFILL PHASE ===
        hidden_states_prefill = full_hidden_states[:, :prefill_len, :]
        position_ids_prefill = torch.arange(prefill_len).unsqueeze(0).expand(batch_size, -1)
        position_embeddings_prefill = rope(hidden_states_prefill, position_ids_prefill)
        cache_position_prefill = torch.arange(prefill_len)
        
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
            f"Prefill outputs differ. Max diff: {(output_orig_prefill[0] - output_standalone_prefill[0]).abs().max()}"
        
        # === INCREMENTAL GENERATION PHASE ===
        for step in range(prefill_len, total_len):
            hidden_states_step = full_hidden_states[:, step:step+1, :]
            position_ids_step = torch.tensor([[step]] * batch_size)
            cache_position_step = torch.tensor([step])
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
                f"Generation step {step - prefill_len} outputs differ. Max diff: {(output_orig_step[0] - output_standalone_step[0]).abs().max()}"
        
        # Verify cache lengths match
        assert cache_orig.get_seq_length() == cache_standalone.get_seq_length() == total_len, \
            f"Cache lengths differ: orig={cache_orig.get_seq_length()}, standalone={cache_standalone.get_seq_length()}, expected={total_len}"

    def test_long_sequence_caching(self):
        """Test caching with longer sequences."""
        set_seed(42)
        
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
            max_position_embeddings = 4096
            rope_theta = 10000.0
        
        config = MinimalConfig()
        
        layer_orig = Qwen3TTSDecoderLayer(config, layer_idx=0)
        layer_standalone = Qwen3TTSDecoderLayerStandalone(config, layer_idx=0)
        copy_weights(layer_orig, layer_standalone)
        
        batch_size = 1
        prefill_len = 50
        gen_steps = 20
        
        set_seed(42)
        hidden_states = torch.randn(batch_size, prefill_len + gen_steps, config.hidden_size)
        
        rope = Qwen3TTSRotaryEmbedding(config)
        
        cache_orig = DynamicCache()
        cache_standalone = DynamicCache()
        
        # Prefill
        hidden_prefill = hidden_states[:, :prefill_len, :]
        position_ids = torch.arange(prefill_len).unsqueeze(0)
        position_embeddings = rope(hidden_prefill, position_ids)
        
        with torch.no_grad():
            out_orig = layer_orig(
                hidden_prefill,
                position_ids=position_ids,
                past_key_values=cache_orig,
                use_cache=True,
                cache_position=torch.arange(prefill_len),
                position_embeddings=position_embeddings,
            )
            out_standalone = layer_standalone(
                hidden_prefill,
                position_ids=position_ids,
                past_key_values=cache_standalone,
                use_cache=True,
                cache_position=torch.arange(prefill_len),
                position_embeddings=position_embeddings,
            )
        
        assert torch.allclose(out_orig[0], out_standalone[0], atol=1e-5)
        
        # Generate tokens one by one
        for i in range(gen_steps):
            pos = prefill_len + i
            hidden_step = hidden_states[:, pos:pos+1, :]
            position_ids = torch.tensor([[pos]])
            position_embeddings = rope(hidden_step, position_ids)
            
            with torch.no_grad():
                out_orig = layer_orig(
                    hidden_step,
                    position_ids=position_ids,
                    past_key_values=cache_orig,
                    use_cache=True,
                    cache_position=torch.tensor([pos]),
                    position_embeddings=position_embeddings,
                )
                out_standalone = layer_standalone(
                    hidden_step,
                    position_ids=position_ids,
                    past_key_values=cache_standalone,
                    use_cache=True,
                    cache_position=torch.tensor([pos]),
                    position_embeddings=position_embeddings,
                )
            
            assert torch.allclose(out_orig[0], out_standalone[0], atol=1e-5), \
                f"Step {i} outputs differ"
        
        # Final check
        assert cache_orig.get_seq_length() == cache_standalone.get_seq_length() == prefill_len + gen_steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
