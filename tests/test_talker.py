# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the refactored Talker class.
"""

import pytest
import torch

from tests.conftest import set_seed

from qwen3_tts_standalone import (
    TalkerConfig,
    CodePredictorConfig,
    Talker,
)
from qwen3_tts_standalone.talker import TalkerOutput


@pytest.fixture
def small_talker_config():
    """Create a minimal Talker config for testing."""
    # head_dim = hidden_size / num_attention_heads = 64 / 4 = 16
    # mrope_section must sum to head_dim / 2 = 8
    rope_scaling = {
        "rope_type": "default",
        "mrope_section": [2, 3, 3],  # sum = 8 = head_dim / 2
        "interleaved": False,
    }
    
    code_predictor_config = CodePredictorConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
    )
    
    talker_config = TalkerConfig(
        vocab_size=256,
        hidden_size=64,
        text_hidden_size=64,
        text_vocab_size=1000,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        num_code_groups=4,
        code_predictor_config=code_predictor_config,
        rope_scaling=rope_scaling,
        # Override EOS token to be within vocab_size
        codec_eos_token_id=255,
    )
    return talker_config


class TestTalkerInstantiation:
    """Test Talker class instantiation."""

    def test_instantiation(self, small_talker_config):
        """Test that Talker can be instantiated."""
        talker = Talker(small_talker_config)
        
        assert talker.config == small_talker_config
        assert talker.vocab_size == small_talker_config.vocab_size
        assert talker.num_code_groups == small_talker_config.num_code_groups
        assert talker.model is not None
        assert talker.text_projection is not None
        assert talker.codec_head is not None
        assert talker.code_predictor is not None

    def test_get_embeddings(self, small_talker_config):
        """Test that embedding accessors work."""
        talker = Talker(small_talker_config)
        
        input_embeddings = talker.get_input_embeddings()
        text_embeddings = talker.get_text_embeddings()
        
        assert input_embeddings is not None
        assert text_embeddings is not None


class TestTalkerForward:
    """Test Talker forward pass."""

    def test_forward_prefill(self, small_talker_config):
        """Test forward pass during prefill."""
        set_seed(42)
        talker = Talker(small_talker_config)
        talker.eval()
        
        batch_size = 2
        seq_len = 10
        hidden_size = small_talker_config.hidden_size
        
        inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            output = talker.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                is_prefill=True,
            )
        
        assert isinstance(output, TalkerOutput)
        assert output.logits.shape == (batch_size, seq_len, small_talker_config.vocab_size)
        assert output.hidden_state.shape == (batch_size, 1, hidden_size)
        assert output.codec_ids is None  # No codec IDs during prefill
        assert output.past_key_values is not None

    def test_forward_generation_step(self, small_talker_config):
        """Test forward pass during generation."""
        set_seed(42)
        talker = Talker(small_talker_config)
        talker.eval()
        
        batch_size = 2
        hidden_size = small_talker_config.hidden_size
        
        # Simulate prefill first
        inputs_embeds = torch.randn(batch_size, 10, hidden_size)
        attention_mask = torch.ones(batch_size, 10)
        
        with torch.no_grad():
            prefill_output = talker.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                is_prefill=True,
            )
            
            # Now do a generation step
            last_token = torch.randint(0, small_talker_config.vocab_size, (batch_size, 1))
            step_embeds = torch.zeros(batch_size, 1, hidden_size)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1)], dim=1)
            cache_position = torch.tensor([prefill_output.past_key_values.get_seq_length()])
            
            gen_output = talker.forward(
                inputs_embeds=step_embeds,
                attention_mask=attention_mask,
                past_key_values=prefill_output.past_key_values,
                past_hidden=prefill_output.hidden_state,
                cache_position=cache_position,
                is_prefill=False,
                last_predicted_token=last_token,
            )
        
        assert isinstance(gen_output, TalkerOutput)
        assert gen_output.logits.shape == (batch_size, 1, small_talker_config.vocab_size)
        assert gen_output.hidden_state.shape == (batch_size, 1, hidden_size)
        assert gen_output.codec_ids is not None
        assert gen_output.codec_ids.shape == (batch_size, small_talker_config.num_code_groups)


class TestTalkerGenerate:
    """Test Talker generate method."""

    def test_generate_basic(self, small_talker_config):
        """Test basic generation."""
        set_seed(42)
        talker = Talker(small_talker_config)
        talker.eval()
        
        batch_size = 2
        seq_len = 10
        hidden_size = small_talker_config.hidden_size
        
        inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        trailing_text_hidden = torch.randn(batch_size, 5, hidden_size)
        tts_pad_embed = torch.randn(1, 1, hidden_size)
        
        with torch.no_grad():
            output = talker.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                max_new_tokens=5,
                do_sample=False,
            )
        
        assert output.sequences.shape[0] == batch_size
        assert output.sequences.shape[1] <= 5  # May stop early due to EOS
        assert len(output.all_codec_ids) <= 4  # codec_ids are created after first token

    def test_generate_deterministic(self, small_talker_config):
        """Test that greedy generation is deterministic."""
        set_seed(42)
        talker = Talker(small_talker_config)
        talker.eval()
        
        batch_size = 1
        seq_len = 10
        hidden_size = small_talker_config.hidden_size
        
        inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        trailing_text_hidden = torch.randn(batch_size, 5, hidden_size)
        tts_pad_embed = torch.randn(1, 1, hidden_size)
        
        with torch.no_grad():
            # Run twice with same input (all greedy for determinism)
            output1 = talker.generate(
                inputs_embeds=inputs_embeds.clone(),
                attention_mask=attention_mask.clone(),
                trailing_text_hidden=trailing_text_hidden.clone(),
                tts_pad_embed=tts_pad_embed.clone(),
                max_new_tokens=5,
                do_sample=False,
                subtalker_dosample=False,  # Also make code predictor greedy
            )
            
            # Reset rope deltas for fresh generation
            talker.rope_deltas = None
            
            output2 = talker.generate(
                inputs_embeds=inputs_embeds.clone(),
                attention_mask=attention_mask.clone(),
                trailing_text_hidden=trailing_text_hidden.clone(),
                tts_pad_embed=tts_pad_embed.clone(),
                max_new_tokens=5,
                do_sample=False,
                subtalker_dosample=False,
            )
        
        assert torch.equal(output1.sequences, output2.sequences)

    def test_generate_with_sampling(self, small_talker_config):
        """Test generation with sampling produces output."""
        set_seed(42)
        talker = Talker(small_talker_config)
        talker.eval()
        
        batch_size = 2
        seq_len = 10
        hidden_size = small_talker_config.hidden_size
        
        inputs_embeds = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)
        trailing_text_hidden = torch.randn(batch_size, 5, hidden_size)
        tts_pad_embed = torch.randn(1, 1, hidden_size)
        
        with torch.no_grad():
            output = talker.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                max_new_tokens=5,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=1.0,
            )
        
        assert output.sequences.shape[0] == batch_size
        assert output.sequences.shape[1] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
