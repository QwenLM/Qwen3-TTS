# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for basic component equivalence: RMSNorm, MLP, RotaryEmbedding.
"""

import pytest
import torch

from tests.conftest import set_seed, copy_weights

# Original models
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSRMSNorm,
    Qwen3TTSRotaryEmbedding,
    Qwen3TTSTalkerTextMLP,
)

# Standalone models
from qwen_tts.core.models.modeling_qwen3_tts_standalone import (
    Qwen3TTSRMSNormStandalone,
    Qwen3TTSRotaryEmbeddingStandalone,
    Qwen3TTSTalkerTextMLPStandalone,
)


class TestRMSNormEquivalence:
    """Test RMSNorm layer equivalence."""

    def test_rmsnorm_forward_equivalence(self):
        """Test that RMSNorm layers produce identical outputs."""
        set_seed(42)
        
        hidden_size = 256
        eps = 1e-6
        
        norm_orig = Qwen3TTSRMSNorm(hidden_size, eps=eps)
        norm_standalone = Qwen3TTSRMSNormStandalone(hidden_size, eps=eps)
        
        # Copy weights
        copy_weights(norm_orig, norm_standalone)
        
        # Test input
        set_seed(42)
        x = torch.randn(2, 10, hidden_size)
        
        # Forward pass
        with torch.no_grad():
            output_orig = norm_orig(x)
            output_standalone = norm_standalone(x)
        
        assert torch.allclose(output_orig, output_standalone, atol=1e-6), \
            f"RMSNorm outputs differ. Max diff: {(output_orig - output_standalone).abs().max()}"

    def test_rmsnorm_different_sizes(self):
        """Test RMSNorm with different hidden sizes."""
        for hidden_size in [128, 512, 1024]:
            set_seed(42)
            
            norm_orig = Qwen3TTSRMSNorm(hidden_size)
            norm_standalone = Qwen3TTSRMSNormStandalone(hidden_size)
            copy_weights(norm_orig, norm_standalone)
            
            x = torch.randn(2, 10, hidden_size)
            
            with torch.no_grad():
                output_orig = norm_orig(x)
                output_standalone = norm_standalone(x)
            
            assert torch.allclose(output_orig, output_standalone, atol=1e-6)


class TestMLPEquivalence:
    """Test MLP layer equivalence."""

    def test_talker_text_mlp_forward_equivalence(self):
        """Test that TalkerTextMLP layers produce identical outputs."""
        set_seed(42)
        
        # Create minimal config
        class MinimalConfig:
            hidden_size = 256
            intermediate_size = 512
            hidden_act = "silu"
        
        config = MinimalConfig()
        
        mlp_orig = Qwen3TTSTalkerTextMLP(config)
        mlp_standalone = Qwen3TTSTalkerTextMLPStandalone(config)
        
        # Copy weights
        copy_weights(mlp_orig, mlp_standalone)
        
        # Test input
        set_seed(42)
        x = torch.randn(2, 10, 256)
        
        # Forward pass
        with torch.no_grad():
            output_orig = mlp_orig(x)
            output_standalone = mlp_standalone(x)
        
        assert torch.allclose(output_orig, output_standalone, atol=1e-6), \
            f"TalkerTextMLP outputs differ. Max diff: {(output_orig - output_standalone).abs().max()}"


class TestRotaryEmbeddingEquivalence:
    """Test RotaryEmbedding equivalence."""

    def test_rotary_embedding_forward_equivalence(self):
        """Test that RotaryEmbedding produces identical outputs."""
        set_seed(42)
        
        # Create minimal config for rotary embedding
        class MinimalConfig:
            rope_scaling = None
            max_position_embeddings = 2048
            head_dim = 64
            rope_theta = 10000.0
        
        config = MinimalConfig()
        
        rope_orig = Qwen3TTSRotaryEmbedding(config)
        rope_standalone = Qwen3TTSRotaryEmbeddingStandalone(config)
        
        # Test input
        set_seed(42)
        x = torch.randn(2, 10, 64)
        position_ids = torch.arange(10).unsqueeze(0).expand(2, -1)
        
        # Forward pass
        with torch.no_grad():
            cos_orig, sin_orig = rope_orig(x, position_ids)
            cos_standalone, sin_standalone = rope_standalone(x, position_ids)
        
        assert torch.allclose(cos_orig, cos_standalone, atol=1e-6), \
            f"RotaryEmbedding cos outputs differ. Max diff: {(cos_orig - cos_standalone).abs().max()}"
        assert torch.allclose(sin_orig, sin_standalone, atol=1e-6), \
            f"RotaryEmbedding sin outputs differ. Max diff: {(sin_orig - sin_standalone).abs().max()}"

    def test_rotary_embedding_different_positions(self):
        """Test RotaryEmbedding with different position ranges."""
        class MinimalConfig:
            rope_scaling = None
            max_position_embeddings = 4096
            head_dim = 128
            rope_theta = 10000.0
        
        config = MinimalConfig()
        
        rope_orig = Qwen3TTSRotaryEmbedding(config)
        rope_standalone = Qwen3TTSRotaryEmbeddingStandalone(config)
        
        # Test with offset positions
        x = torch.randn(2, 5, 128)
        position_ids = torch.arange(100, 105).unsqueeze(0).expand(2, -1)
        
        with torch.no_grad():
            cos_orig, sin_orig = rope_orig(x, position_ids)
            cos_standalone, sin_standalone = rope_standalone(x, position_ids)
        
        assert torch.allclose(cos_orig, cos_standalone, atol=1e-6)
        assert torch.allclose(sin_orig, sin_standalone, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
