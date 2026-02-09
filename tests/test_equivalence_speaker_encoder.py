# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for SpeakerEncoder model equivalence.
"""

import pytest
import torch

from tests.conftest import set_seed, copy_weights

# Original models
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder

# Standalone models
from qwen3_tts_standalone.configuration import Qwen3TTSSpeakerEncoderConfigStandalone
from qwen3_tts_standalone.speaker_encoder import Qwen3TTSSpeakerEncoderStandalone


class TestSpeakerEncoderEquivalence:
    """Test SpeakerEncoder equivalence."""

    def test_speaker_encoder_forward_equivalence(self):
        """Test that SpeakerEncoder models produce identical outputs."""
        set_seed(42)
        
        config_orig = Qwen3TTSSpeakerEncoderConfig(
            mel_dim=128,
            enc_dim=192,
            enc_channels=[512, 512, 512, 512, 1536],
            enc_kernel_sizes=[5, 3, 3, 3, 1],
            enc_dilations=[1, 2, 3, 4, 1],
            enc_attention_channels=128,
            enc_res2net_scale=8,
            enc_se_channels=128,
        )
        config_standalone = Qwen3TTSSpeakerEncoderConfigStandalone(
            mel_dim=128,
            enc_dim=192,
            enc_channels=[512, 512, 512, 512, 1536],
            enc_kernel_sizes=[5, 3, 3, 3, 1],
            enc_dilations=[1, 2, 3, 4, 1],
            enc_attention_channels=128,
            enc_res2net_scale=8,
            enc_se_channels=128,
        )
        
        encoder_orig = Qwen3TTSSpeakerEncoder(config_orig)
        encoder_standalone = Qwen3TTSSpeakerEncoderStandalone(config_standalone)
        
        # Copy weights
        copy_weights(encoder_orig, encoder_standalone)
        
        # Test input (batch, time, mel_dim)
        set_seed(42)
        x = torch.randn(2, 100, 128)
        
        # Forward pass
        with torch.no_grad():
            output_orig = encoder_orig(x)
            output_standalone = encoder_standalone(x)
        
        assert torch.allclose(output_orig, output_standalone, atol=1e-5), \
            f"SpeakerEncoder outputs differ. Max diff: {(output_orig - output_standalone).abs().max()}"

    def test_speaker_encoder_different_lengths(self):
        """Test SpeakerEncoder with different input lengths."""
        set_seed(42)
        
        config_orig = Qwen3TTSSpeakerEncoderConfig(mel_dim=128, enc_dim=256)
        config_standalone = Qwen3TTSSpeakerEncoderConfigStandalone(mel_dim=128, enc_dim=256)
        
        encoder_orig = Qwen3TTSSpeakerEncoder(config_orig)
        encoder_standalone = Qwen3TTSSpeakerEncoderStandalone(config_standalone)
        copy_weights(encoder_orig, encoder_standalone)
        
        for seq_len in [50, 100, 200]:
            set_seed(42)
            x = torch.randn(2, seq_len, 128)
            
            with torch.no_grad():
                output_orig = encoder_orig(x)
                output_standalone = encoder_standalone(x)
            
            assert torch.allclose(output_orig, output_standalone, atol=1e-5), \
                f"SpeakerEncoder outputs differ for seq_len={seq_len}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
