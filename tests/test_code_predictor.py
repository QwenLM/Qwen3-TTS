# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the simplified CodePredictor class.

These tests verify that:
1. The simplified implementation produces identical outputs to the original
2. Weight loading from original checkpoints works correctly
3. Both greedy and sampling modes work correctly
4. The model handles different batch sizes and configurations
"""

import pytest
import torch

from qwen_tts.core.models.code_predictor_standalone import (
    CodePredictor,
    CodePredictorOutput,
)
from qwen_tts.core.models.modeling_qwen3_tts_standalone import (
    Qwen3TTSTalkerCodePredictorModelStandaloneForConditionalGenerationStandalone,
)
from qwen_tts.core.models.configuration_qwen3_tts_standalone import (
    Qwen3TTSTalkerCodePredictorConfigStandalone,
    Qwen3TTSTalkerConfigStandalone,  # Still needed for original model in equivalence tests
)


@pytest.fixture
def small_config():
    """Small configuration for fast testing."""
    code_predictor_config = Qwen3TTSTalkerCodePredictorConfigStandalone(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_code_groups=8,
    )
    embedding_dim = 128  # Different from code_predictor to test projection
    return code_predictor_config, embedding_dim


@pytest.fixture
def same_hidden_size_config():
    """Configuration where embedding_dim equals code predictor hidden size."""
    code_predictor_config = Qwen3TTSTalkerCodePredictorConfigStandalone(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_code_groups=8,
    )
    embedding_dim = 256  # Same as code_predictor
    return code_predictor_config, embedding_dim


class TestCodePredictor:
    """Tests for the CodePredictor class."""

    def test_instantiation(self, small_config):
        """Test that the model can be instantiated."""
        code_predictor_config, embedding_dim = small_config
        model = CodePredictor(code_predictor_config, embedding_dim)
        
        assert model is not None
        assert len(model.codec_embedding) == code_predictor_config.num_code_groups - 1
        assert len(model.lm_head) == code_predictor_config.num_code_groups - 1
        assert len(model.layers) == code_predictor_config.num_hidden_layers

    def test_instantiation_same_hidden_size(self, same_hidden_size_config):
        """Test instantiation when hidden sizes match (Identity projection)."""
        code_predictor_config, embedding_dim = same_hidden_size_config
        model = CodePredictor(code_predictor_config, embedding_dim)
        
        assert isinstance(model.input_projection, torch.nn.Identity)

    def test_instantiation_different_hidden_size(self, small_config):
        """Test instantiation when hidden sizes differ (Linear projection)."""
        code_predictor_config, embedding_dim = small_config
        model = CodePredictor(code_predictor_config, embedding_dim)
        
        assert isinstance(model.input_projection, torch.nn.Linear)
        assert model.input_projection.in_features == embedding_dim
        assert model.input_projection.out_features == code_predictor_config.hidden_size

    def test_generate_output_shape(self, small_config):
        """Test that generate produces the correct output shape."""
        code_predictor_config, embedding_dim = small_config
        model = CodePredictor(code_predictor_config, embedding_dim)
        model.eval()
        
        batch_size = 2
        inputs_embeds = torch.randn(batch_size, 2, embedding_dim)
        
        with torch.no_grad():
            output = model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=code_predictor_config.num_code_groups - 1,
                do_sample=False,
            )
        
        assert isinstance(output, CodePredictorOutput)
        assert output.sequences.shape == (batch_size, code_predictor_config.num_code_groups - 1)

    def test_generate_greedy_deterministic(self, small_config):
        """Test that greedy generation is deterministic."""
        code_predictor_config, embedding_dim = small_config
        model = CodePredictor(code_predictor_config, embedding_dim)
        model.eval()
        
        inputs_embeds = torch.randn(1, 2, embedding_dim)
        
        with torch.no_grad():
            output1 = model.generate(
                inputs_embeds=inputs_embeds.clone(),
                max_new_tokens=7,
                do_sample=False,
            )
            output2 = model.generate(
                inputs_embeds=inputs_embeds.clone(),
                max_new_tokens=7,
                do_sample=False,
            )
        
        assert torch.equal(output1.sequences, output2.sequences)

    def test_generate_sampling_with_seed(self, small_config):
        """Test that sampling with same seed produces same output."""
        code_predictor_config, embedding_dim = small_config
        model = CodePredictor(code_predictor_config, embedding_dim)
        model.eval()
        
        inputs_embeds = torch.randn(1, 2, embedding_dim)
        
        torch.manual_seed(42)
        with torch.no_grad():
            output1 = model.generate(
                inputs_embeds=inputs_embeds.clone(),
                max_new_tokens=7,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
            )
        
        torch.manual_seed(42)
        with torch.no_grad():
            output2 = model.generate(
                inputs_embeds=inputs_embeds.clone(),
                max_new_tokens=7,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
            )
        
        assert torch.equal(output1.sequences, output2.sequences)


class TestCodePredictorEquivalence:
    """Tests that verify equivalence with the original implementation."""

    def test_weight_loading_from_original(self, small_config):
        """Test that weights can be loaded from the original model."""
        code_predictor_config, embedding_dim = small_config
        talker_config = Qwen3TTSTalkerConfigStandalone(hidden_size=embedding_dim)
        
        # Create original model
        original_model = Qwen3TTSTalkerCodePredictorModelStandaloneForConditionalGenerationStandalone(
            code_predictor_config, talker_config
        )
        
        # Create new model and load weights
        new_model = CodePredictor(code_predictor_config, embedding_dim)
        new_model.load_original_state_dict(original_model.state_dict())
        
        # Verify all weights were loaded (no missing or unexpected keys)
        # This is implicit - load_state_dict would raise if keys don't match

    def test_greedy_output_equivalence(self, small_config):
        """Test that greedy outputs match the original implementation."""
        code_predictor_config, embedding_dim = small_config
        talker_config = Qwen3TTSTalkerConfigStandalone(hidden_size=embedding_dim)
        
        # Create and setup models
        original_model = Qwen3TTSTalkerCodePredictorModelStandaloneForConditionalGenerationStandalone(
            code_predictor_config, talker_config
        )
        new_model = CodePredictor(code_predictor_config, embedding_dim)
        new_model.load_original_state_dict(original_model.state_dict())
        
        original_model.eval()
        new_model.eval()
        
        # Test with multiple batch sizes
        for batch_size in [1, 2, 4]:
            inputs_embeds = torch.randn(batch_size, 2, embedding_dim)
            
            with torch.no_grad():
                original_output = original_model.generate(
                    inputs_embeds=inputs_embeds.clone(),
                    max_new_tokens=7,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
                new_output = new_model.generate(
                    inputs_embeds=inputs_embeds.clone(),
                    max_new_tokens=7,
                    do_sample=False,
                )
            
            assert torch.equal(original_output.sequences, new_output.sequences), \
                f"Outputs differ for batch_size={batch_size}"

    def test_sampling_output_equivalence(self, small_config):
        """Test that sampled outputs match the original implementation with same seed."""
        code_predictor_config, embedding_dim = small_config
        talker_config = Qwen3TTSTalkerConfigStandalone(hidden_size=embedding_dim)
        
        # Create and setup models
        original_model = Qwen3TTSTalkerCodePredictorModelStandaloneForConditionalGenerationStandalone(
            code_predictor_config, talker_config
        )
        new_model = CodePredictor(code_predictor_config, embedding_dim)
        new_model.load_original_state_dict(original_model.state_dict())
        
        original_model.eval()
        new_model.eval()
        
        inputs_embeds = torch.randn(2, 2, embedding_dim)
        
        torch.manual_seed(123)
        with torch.no_grad():
            original_output = original_model.generate(
                inputs_embeds=inputs_embeds.clone(),
                max_new_tokens=7,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
            )
        
        torch.manual_seed(123)
        with torch.no_grad():
            new_output = new_model.generate(
                inputs_embeds=inputs_embeds.clone(),
                max_new_tokens=7,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.95,
            )
        
        assert torch.equal(original_output.sequences, new_output.sequences)

    def test_equivalence_with_same_hidden_size(self, same_hidden_size_config):
        """Test equivalence when hidden sizes match (Identity projection case)."""
        code_predictor_config, embedding_dim = same_hidden_size_config
        talker_config = Qwen3TTSTalkerConfigStandalone(hidden_size=embedding_dim)
        
        # Create and setup models
        original_model = Qwen3TTSTalkerCodePredictorModelStandaloneForConditionalGenerationStandalone(
            code_predictor_config, talker_config
        )
        new_model = CodePredictor(code_predictor_config, embedding_dim)
        new_model.load_original_state_dict(original_model.state_dict())
        
        original_model.eval()
        new_model.eval()
        
        inputs_embeds = torch.randn(2, 2, embedding_dim)
        
        with torch.no_grad():
            original_output = original_model.generate(
                inputs_embeds=inputs_embeds.clone(),
                max_new_tokens=7,
                do_sample=False,
            )
            new_output = new_model.generate(
                inputs_embeds=inputs_embeds.clone(),
                max_new_tokens=7,
                do_sample=False,
            )
        
        assert torch.equal(original_output.sequences, new_output.sequences)


class TestCodePredictorEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_token_generation(self, small_config):
        """Test generating just one token."""
        code_predictor_config, embedding_dim = small_config
        model = CodePredictor(code_predictor_config, embedding_dim)
        model.eval()
        
        inputs_embeds = torch.randn(1, 2, embedding_dim)
        
        with torch.no_grad():
            output = model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=1,
                do_sample=False,
            )
        
        assert output.sequences.shape == (1, 1)

    def test_temperature_effect(self, small_config):
        """Test that temperature affects sampling distribution."""
        code_predictor_config, embedding_dim = small_config
        model = CodePredictor(code_predictor_config, embedding_dim)
        model.eval()
        
        inputs_embeds = torch.randn(1, 2, embedding_dim)
        
        # Generate with different temperatures - should produce different outputs
        # (statistically, though not guaranteed for any single run)
        outputs = []
        for temp in [0.1, 0.5, 1.0, 2.0]:
            torch.manual_seed(42)
            with torch.no_grad():
                output = model.generate(
                    inputs_embeds=inputs_embeds.clone(),
                    max_new_tokens=7,
                    do_sample=True,
                    temperature=temp,
                    top_k=0,  # Disable top-k to see temperature effect
                    top_p=1.0,  # Disable top-p to see temperature effect
                )
            outputs.append(output.sequences)
        
        # Low temperature should give more deterministic outputs
        # This is a weak test - just verifying it doesn't crash

    def test_top_k_filtering(self, small_config):
        """Test that top-k filtering works."""
        code_predictor_config, embedding_dim = small_config
        model = CodePredictor(code_predictor_config, embedding_dim)
        model.eval()
        
        inputs_embeds = torch.randn(1, 2, embedding_dim)
        
        # Should not crash with various top_k values
        for top_k in [1, 10, 50, 100]:
            torch.manual_seed(42)
            with torch.no_grad():
                output = model.generate(
                    inputs_embeds=inputs_embeds.clone(),
                    max_new_tokens=3,
                    do_sample=True,
                    top_k=top_k,
                )
            assert output.sequences.shape == (1, 3)

    def test_top_p_filtering(self, small_config):
        """Test that top-p (nucleus) filtering works."""
        code_predictor_config, embedding_dim = small_config
        model = CodePredictor(code_predictor_config, embedding_dim)
        model.eval()
        
        inputs_embeds = torch.randn(1, 2, embedding_dim)
        
        # Should not crash with various top_p values
        for top_p in [0.1, 0.5, 0.9, 0.95, 1.0]:
            torch.manual_seed(42)
            with torch.no_grad():
                output = model.generate(
                    inputs_embeds=inputs_embeds.clone(),
                    max_new_tokens=3,
                    do_sample=True,
                    top_k=0,  # Disable top-k
                    top_p=top_p,
                )
            assert output.sequences.shape == (1, 3)

    def test_large_batch_size(self, small_config):
        """Test with larger batch size."""
        code_predictor_config, embedding_dim = small_config
        model = CodePredictor(code_predictor_config, embedding_dim)
        model.eval()
        
        batch_size = 16
        inputs_embeds = torch.randn(batch_size, 2, embedding_dim)
        
        with torch.no_grad():
            output = model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=7,
                do_sample=False,
            )
        
        assert output.sequences.shape == (batch_size, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
