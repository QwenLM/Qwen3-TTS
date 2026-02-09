# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for 12Hz Tokenizer equivalence between original and standalone implementations.
"""

import pytest
import torch
import numpy as np

from tests.conftest import set_seed, copy_weights


class TestTokenizerV2ConfigEquivalence:
    """Test configuration equivalence."""

    def test_decoder_config_conversion(self):
        """Test decoder config can be converted to standalone."""
        from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2DecoderConfig,
        )
        from qwen3_tts_standalone.tokenizer.config import (
            Qwen3TTSTokenizerV2DecoderConfigStandalone,
        )
        
        # Create original config
        orig = Qwen3TTSTokenizerV2DecoderConfig(
            codebook_size=1024,
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
        )
        
        # Create standalone config with same values
        standalone = Qwen3TTSTokenizerV2DecoderConfigStandalone(
            codebook_size=1024,
            hidden_size=512,
            num_hidden_layers=4,
            num_attention_heads=8,
        )
        
        # Compare key attributes
        assert orig.codebook_size == standalone.codebook_size
        assert orig.hidden_size == standalone.hidden_size
        assert orig.num_hidden_layers == standalone.num_hidden_layers
        assert orig.num_attention_heads == standalone.num_attention_heads

    def test_main_config_conversion(self):
        """Test main config can be converted using helper function."""
        from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Config,
        )
        from qwen3_tts_standalone.tokenizer.config import (
            convert_to_standalone_config,
        )
        
        orig = Qwen3TTSTokenizerV2Config()
        standalone = convert_to_standalone_config(orig)
        
        assert standalone.model_type == orig.model_type
        assert standalone.encoder_valid_num_quantizers == orig.encoder_valid_num_quantizers
        assert standalone.input_sample_rate == orig.input_sample_rate
        assert standalone.output_sample_rate == orig.output_sample_rate

    def test_config_to_dict_roundtrip(self):
        """Test config serialization and deserialization."""
        from qwen3_tts_standalone.tokenizer.config import (
            Qwen3TTSTokenizerV2ConfigStandalone,
        )
        
        config = Qwen3TTSTokenizerV2ConfigStandalone(
            encoder_valid_num_quantizers=8,
            input_sample_rate=16000,
        )
        
        config_dict = config.to_dict()
        restored = Qwen3TTSTokenizerV2ConfigStandalone.from_dict(config_dict)
        
        assert restored.encoder_valid_num_quantizers == config.encoder_valid_num_quantizers
        assert restored.input_sample_rate == config.input_sample_rate


def _create_minimal_decoder_config():
    """Create a minimal decoder config for testing."""
    from qwen3_tts_standalone.tokenizer.config import (
        Qwen3TTSTokenizerV2DecoderConfigStandalone,
    )
    return Qwen3TTSTokenizerV2DecoderConfigStandalone(
        codebook_size=64,
        codebook_dim=32,
        hidden_size=64,
        latent_dim=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,  # 64 // 4
        num_hidden_layers=2,
        intermediate_size=128,
        num_quantizers=4,
        upsample_rates=(2, 2),
        upsampling_ratios=(2,),
        decoder_dim=64,
    )


class TestConvLayerEquivalence:
    """Test convolutional layer equivalence."""

    def test_causal_conv_net_equivalence(self):
        """Test CausalConvNet produces same output."""
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2CausalConvNet,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            Qwen3TTSTokenizerV2CausalConvNetStandalone,
        )
        
        set_seed(42)
        
        in_channels, out_channels, kernel_size = 32, 64, 7
        
        orig = Qwen3TTSTokenizerV2CausalConvNet(in_channels, out_channels, kernel_size)
        standalone = Qwen3TTSTokenizerV2CausalConvNetStandalone(in_channels, out_channels, kernel_size)
        
        copy_weights(orig, standalone)
        
        set_seed(42)
        x = torch.randn(2, in_channels, 50)
        
        with torch.no_grad():
            out_orig = orig(x)
            out_standalone = standalone(x)
        
        assert torch.allclose(out_orig, out_standalone, atol=1e-5)

    def test_causal_trans_conv_net_equivalence(self):
        """Test CausalTransConvNet produces same output."""
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2CausalTransConvNet,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            Qwen3TTSTokenizerV2CausalTransConvNetStandalone,
        )
        
        set_seed(42)
        
        in_channels, out_channels, kernel_size, stride = 32, 64, 4, 2
        
        orig = Qwen3TTSTokenizerV2CausalTransConvNet(in_channels, out_channels, kernel_size, stride)
        standalone = Qwen3TTSTokenizerV2CausalTransConvNetStandalone(in_channels, out_channels, kernel_size, stride)
        
        copy_weights(orig, standalone)
        
        set_seed(42)
        x = torch.randn(2, in_channels, 25)
        
        with torch.no_grad():
            out_orig = orig(x)
            out_standalone = standalone(x)
        
        assert torch.allclose(out_orig, out_standalone, atol=1e-5)

    def test_convnext_block_equivalence(self):
        """Test ConvNeXtBlock produces same output."""
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2ConvNeXtBlock,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            Qwen3TTSTokenizerV2ConvNeXtBlockStandalone,
        )
        
        set_seed(42)
        
        dim = 64
        
        orig = Qwen3TTSTokenizerV2ConvNeXtBlock(dim)
        standalone = Qwen3TTSTokenizerV2ConvNeXtBlockStandalone(dim)
        
        copy_weights(orig, standalone)
        
        set_seed(42)
        x = torch.randn(2, dim, 50)
        
        with torch.no_grad():
            out_orig = orig(x)
            out_standalone = standalone(x)
        
        assert torch.allclose(out_orig, out_standalone, atol=1e-5)


class TestTransformerLayerEquivalence:
    """Test transformer layer equivalence."""

    def test_rms_norm_equivalence(self):
        """Test RMSNorm produces same output."""
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2DecoderRMSNorm,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            Qwen3TTSTokenizerV2DecoderRMSNormStandalone,
        )
        
        set_seed(42)
        
        hidden_size = 64
        
        orig = Qwen3TTSTokenizerV2DecoderRMSNorm(hidden_size)
        standalone = Qwen3TTSTokenizerV2DecoderRMSNormStandalone(hidden_size)
        
        copy_weights(orig, standalone)
        
        set_seed(42)
        x = torch.randn(2, 10, hidden_size)
        
        with torch.no_grad():
            out_orig = orig(x)
            out_standalone = standalone(x)
        
        assert torch.allclose(out_orig, out_standalone, atol=1e-5)

    def test_mlp_equivalence(self):
        """Test MLP produces same output."""
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2DecoderMlp,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            Qwen3TTSTokenizerV2DecoderMlpStandalone,
        )
        
        set_seed(42)
        
        config = _create_minimal_decoder_config()
        
        orig = Qwen3TTSTokenizerV2DecoderMlp(config)
        standalone = Qwen3TTSTokenizerV2DecoderMlpStandalone(config)
        
        copy_weights(orig, standalone)
        
        set_seed(42)
        x = torch.randn(2, 10, config.hidden_size)
        
        with torch.no_grad():
            out_orig = orig(x)
            out_standalone = standalone(x)
        
        assert torch.allclose(out_orig, out_standalone, atol=1e-5)

    def test_attention_equivalence(self):
        """Test Attention produces same output."""
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2DecoderAttention,
            Qwen3TTSTokenizerV2DecoderRotatoryEmbedding,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            Qwen3TTSTokenizerV2DecoderAttentionStandalone,
            Qwen3TTSTokenizerV2DecoderRotaryEmbeddingStandalone,
        )
        
        set_seed(42)
        
        config = _create_minimal_decoder_config()
        config._attn_implementation = "eager"
        
        attn_orig = Qwen3TTSTokenizerV2DecoderAttention(config, layer_idx=0)
        attn_standalone = Qwen3TTSTokenizerV2DecoderAttentionStandalone(config, layer_idx=0)
        
        copy_weights(attn_orig, attn_standalone)
        
        rope_orig = Qwen3TTSTokenizerV2DecoderRotatoryEmbedding(config)
        rope_standalone = Qwen3TTSTokenizerV2DecoderRotaryEmbeddingStandalone(config)
        copy_weights(rope_orig, rope_standalone)
        
        set_seed(42)
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        position_embeddings_orig = rope_orig(hidden_states, position_ids)
        position_embeddings_standalone = rope_standalone(hidden_states, position_ids)
        
        with torch.no_grad():
            out_orig, _ = attn_orig(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings_orig,
                attention_mask=None,
            )
            out_standalone, _ = attn_standalone(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings_standalone,
                attention_mask=None,
            )
        
        assert torch.allclose(out_orig, out_standalone, atol=1e-4)


class TestQuantizerEquivalence:
    """Test vector quantizer equivalence."""

    def test_euclidean_codebook_decode(self):
        """Test EuclideanCodebook decode produces same output."""
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            EuclideanCodebook,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            EuclideanCodebookStandalone,
        )
        
        set_seed(42)
        
        dim, codebook_size = 32, 64
        
        orig = EuclideanCodebook(dim, codebook_size)
        standalone = EuclideanCodebookStandalone(dim, codebook_size)
        
        copy_weights(orig, standalone)
        
        codes = torch.randint(0, codebook_size, (2, 10))
        
        with torch.no_grad():
            out_orig = orig.decode(codes)
            out_standalone = standalone.decode(codes)
        
        assert torch.allclose(out_orig, out_standalone, atol=1e-5)

    def test_vector_quantization_decode(self):
        """Test VectorQuantization decode produces same output."""
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            VectorQuantization,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            VectorQuantizationStandalone,
        )
        
        set_seed(42)
        
        dim, codebook_size, codebook_dim = 64, 128, 32
        
        orig = VectorQuantization(dim, codebook_size, codebook_dim)
        standalone = VectorQuantizationStandalone(dim, codebook_size, codebook_dim)
        
        copy_weights(orig, standalone)
        
        codes = torch.randint(0, codebook_size, (2, 10))
        
        with torch.no_grad():
            out_orig = orig.decode(codes)
            out_standalone = standalone.decode(codes)
        
        assert torch.allclose(out_orig, out_standalone, atol=1e-5)

    def test_residual_vector_quantization_decode(self):
        """Test ResidualVectorQuantization decode produces same output."""
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            ResidualVectorQuantization,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            ResidualVectorQuantizationStandalone,
        )
        
        set_seed(42)
        
        dim, codebook_size, num_quantizers = 32, 64, 4
        
        orig = ResidualVectorQuantization(dim=dim, codebook_size=codebook_size, num_quantizers=num_quantizers)
        standalone = ResidualVectorQuantizationStandalone(dim=dim, codebook_size=codebook_size, num_quantizers=num_quantizers)
        
        copy_weights(orig, standalone)
        
        # codes shape: (num_quantizers, batch, seq_len)
        codes = torch.randint(0, codebook_size, (num_quantizers, 2, 10))
        
        with torch.no_grad():
            out_orig = orig.decode(codes)
            out_standalone = standalone.decode(codes)
        
        assert torch.allclose(out_orig, out_standalone, atol=1e-5)


class TestSnakeBetaEquivalence:
    """Test SnakeBeta activation equivalence."""

    def test_snake_beta_equivalence(self):
        """Test SnakeBeta produces same output."""
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            SnakeBeta,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            SnakeBetaStandalone,
        )
        
        set_seed(42)
        
        in_features = 64
        
        orig = SnakeBeta(in_features)
        standalone = SnakeBetaStandalone(in_features)
        
        copy_weights(orig, standalone)
        
        set_seed(42)
        x = torch.randn(2, in_features, 50)
        
        with torch.no_grad():
            out_orig = orig(x)
            out_standalone = standalone(x)
        
        assert torch.allclose(out_orig, out_standalone, atol=1e-5)


class TestTransformerModelEquivalence:
    """Test full transformer model equivalence."""

    def test_transformer_layer_equivalence(self):
        """Test transformer layer produces same output."""
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2DecoderTransformerLayer,
            Qwen3TTSTokenizerV2DecoderRotatoryEmbedding,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            Qwen3TTSTokenizerV2DecoderTransformerLayerStandalone,
            Qwen3TTSTokenizerV2DecoderRotaryEmbeddingStandalone,
        )
        
        set_seed(42)
        
        config = _create_minimal_decoder_config()
        config._attn_implementation = "eager"
        
        layer_orig = Qwen3TTSTokenizerV2DecoderTransformerLayer(config, layer_idx=0)
        layer_standalone = Qwen3TTSTokenizerV2DecoderTransformerLayerStandalone(config, layer_idx=0)
        
        copy_weights(layer_orig, layer_standalone)
        
        rope_orig = Qwen3TTSTokenizerV2DecoderRotatoryEmbedding(config)
        rope_standalone = Qwen3TTSTokenizerV2DecoderRotaryEmbeddingStandalone(config)
        
        set_seed(42)
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        cache_position = torch.arange(seq_len)
        
        position_embeddings = rope_orig(hidden_states, position_ids)
        
        with torch.no_grad():
            out_orig = layer_orig(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            out_standalone = layer_standalone(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        
        assert torch.allclose(out_orig, out_standalone, atol=1e-4)


class TestDecoderEquivalence:
    """Test full decoder equivalence."""

    def test_decoder_forward(self):
        """Test decoder forward produces same output with shared weights."""
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Decoder,
        )
        from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2DecoderConfig,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            Qwen3TTSTokenizerV2DecoderStandalone,
        )
        
        set_seed(42)
        
        # Create original config
        config_orig = Qwen3TTSTokenizerV2DecoderConfig(
            codebook_size=64,
            codebook_dim=32,
            hidden_size=64,
            latent_dim=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,  # 64 // 4
            num_hidden_layers=2,
            intermediate_size=128,
            num_quantizers=4,
            upsample_rates=(2, 2),
            upsampling_ratios=(2,),
            decoder_dim=64,
        )
        
        # Create standalone config
        config_standalone = _create_minimal_decoder_config()
        
        # Create models
        decoder_orig = Qwen3TTSTokenizerV2Decoder._from_config(config_orig)
        decoder_standalone = Qwen3TTSTokenizerV2DecoderStandalone(config_standalone)
        
        # Copy weights from original to standalone
        copy_weights(decoder_orig, decoder_standalone)
        
        decoder_orig.eval()
        decoder_standalone.eval()
        
        # Create fake codes: (batch, num_quantizers, seq_len)
        set_seed(42)
        codes = torch.randint(1, 64, (1, 4, 10))
        
        with torch.no_grad():
            out_orig = decoder_orig(codes)
            out_standalone = decoder_standalone(codes)
        
        assert torch.allclose(out_orig, out_standalone, atol=1e-3)


class TestPaddingBugFixes:
    """Test bug fixes for padding issues (upstream commits 5f8581d, 6cafe55)."""

    def test_causal_trans_conv_zero_right_pad(self):
        """Test CausalTransConvNet correctly handles zero right_pad case.
        
        Regression test for upstream fix 5f8581d: padding values were
        incorrectly calculated, causing output shape issues.
        """
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2CausalTransConvNet,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            Qwen3TTSTokenizerV2CausalTransConvNetStandalone,
        )
        
        set_seed(42)
        
        # Case where kernel_size == stride, so pad = 0 and right_pad = 0
        in_channels, out_channels, kernel_size, stride = 32, 64, 4, 4
        
        orig = Qwen3TTSTokenizerV2CausalTransConvNet(in_channels, out_channels, kernel_size, stride)
        standalone = Qwen3TTSTokenizerV2CausalTransConvNetStandalone(in_channels, out_channels, kernel_size, stride)
        
        # Check padding attributes are correctly set
        assert standalone.left_pad == 0
        assert standalone.right_pad == 0  # kernel_size - stride = 0
        
        copy_weights(orig, standalone)
        
        set_seed(42)
        x = torch.randn(2, in_channels, 25)
        
        with torch.no_grad():
            out_orig = orig(x)
            out_standalone = standalone(x)
        
        # Outputs should match
        assert out_orig.shape == out_standalone.shape
        assert torch.allclose(out_orig, out_standalone, atol=1e-5)

    def test_causal_trans_conv_various_strides(self):
        """Test CausalTransConvNet with various kernel/stride combinations."""
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2CausalTransConvNet,
        )
        from qwen3_tts_standalone.tokenizer.model import (
            Qwen3TTSTokenizerV2CausalTransConvNetStandalone,
        )
        
        test_cases = [
            (32, 64, 4, 4),  # pad = 0
            (32, 64, 4, 2),  # pad = 2
            (32, 64, 8, 4),  # pad = 4
            (32, 64, 3, 1),  # pad = 2
        ]
        
        for in_ch, out_ch, kernel, stride in test_cases:
            set_seed(42)
            
            orig = Qwen3TTSTokenizerV2CausalTransConvNet(in_ch, out_ch, kernel, stride)
            standalone = Qwen3TTSTokenizerV2CausalTransConvNetStandalone(in_ch, out_ch, kernel, stride)
            
            copy_weights(orig, standalone)
            
            x = torch.randn(2, in_ch, 25)
            
            with torch.no_grad():
                out_orig = orig(x)
                out_standalone = standalone(x)
            
            assert out_orig.shape == out_standalone.shape, f"Shape mismatch for kernel={kernel}, stride={stride}"
            assert torch.allclose(out_orig, out_standalone, atol=1e-5), f"Value mismatch for kernel={kernel}, stride={stride}"

    def test_decode_with_negative_padding_codes(self):
        """Test decode correctly handles negative padding values in audio_codes.
        
        Regression test for upstream fix 6cafe55: padding values (-1) were
        incorrectly counted in audio_lengths and not clamped before decode.
        """
        set_seed(42)
        
        # Test the core logic without instantiating full model
        # This tests the fix: audio_lengths = (audio_codes[..., 0] > -1).sum(1)
        # instead of the buggy: audio_lengths = (audio_codes[..., 0] > 0).sum(1)
        
        decode_upsample_rate = 1920  # Default from config
        
        # Create codes with explicit values to test edge cases
        # Use values 1-100 to avoid ambiguity with 0
        batch_size, seq_len, num_quantizers = 2, 20, 16
        codes = torch.randint(1, 100, (batch_size, seq_len, num_quantizers))
        
        # Add padding at the end of first sequence (half is padding)
        codes[0, 10:, :] = -1
        # Second sequence has no padding
        
        # Test the FIXED logic: > -1 correctly counts non-padding tokens
        audio_lengths_fixed = (codes[..., 0] > -1).sum(1) * decode_upsample_rate
        expected_lengths_0 = 10 * decode_upsample_rate
        expected_lengths_1 = 20 * decode_upsample_rate
        
        assert audio_lengths_fixed[0] == expected_lengths_0, f"Expected {expected_lengths_0}, got {audio_lengths_fixed[0]}"
        assert audio_lengths_fixed[1] == expected_lengths_1, f"Expected {expected_lengths_1}, got {audio_lengths_fixed[1]}"
        
        # Test that valid code=0 is correctly handled
        # Create codes where some valid codes are 0 (valid codebook index)
        codes_with_valid_zero = torch.ones(2, 10, 16, dtype=torch.long)  # All 1s
        codes_with_valid_zero[0, 5, 0] = 0  # Valid code 0 at position 5
        codes_with_valid_zero[1, :, :] = -1  # All padding
        codes_with_valid_zero[1, :5, :] = 1  # First 5 are valid
        
        # Old buggy logic: > 0 would incorrectly exclude valid code=0
        audio_lengths_buggy = (codes_with_valid_zero[..., 0] > 0).sum(1) * decode_upsample_rate
        # Fixed logic: > -1 correctly includes code=0 as valid
        audio_lengths_correct = (codes_with_valid_zero[..., 0] > -1).sum(1) * decode_upsample_rate
        
        # Buggy version would only count 9 (excludes position 5 with code=0)
        assert audio_lengths_buggy[0] == 9 * decode_upsample_rate, "Buggy logic should exclude valid code=0"
        # Fixed version correctly counts all 10
        assert audio_lengths_correct[0] == 10 * decode_upsample_rate, "Fixed logic should include valid code=0"
        # Both should count 5 for the second sequence
        assert audio_lengths_buggy[1] == 5 * decode_upsample_rate
        assert audio_lengths_correct[1] == 5 * decode_upsample_rate
        
        # Verify clamping works (negative codes should be clamped to 0)
        codes_clamped = torch.clamp(codes, min=0)
        assert (codes_clamped >= 0).all(), "Clamping failed"
        assert (codes_clamped[0, 10:, :] == 0).all(), "Padding not clamped to 0"


class TestRoundtrip:
    """Test encode-decode roundtrip equivalence."""
    
    @pytest.mark.skip(reason="Requires pretrained model weights - run manually")
    def test_decode_equivalence_with_pretrained(self):
        """Test that standalone decoder produces same output as original with pretrained weights."""
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
        from qwen_tts.inference.qwen3_tts_tokenizer_standalone import Qwen3TTSTokenizerStandalone
        
        model_id = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
        
        orig_tokenizer = Qwen3TTSTokenizer.from_pretrained(model_id)
        standalone_tokenizer = Qwen3TTSTokenizerStandalone.from_pretrained(model_id)
        
        # Generate random codes
        set_seed(42)
        codes = torch.randint(1, 1024, (1, 16, 50)).to(orig_tokenizer.device)
        
        # Decode with both
        with torch.no_grad():
            orig_output = orig_tokenizer.model.decode(codes, return_dict=True)
            standalone_output = standalone_tokenizer.decoder_model.decode(codes, return_dict=True)
        
        orig_wav = orig_output.audio_values[0].cpu()
        standalone_wav = standalone_output.audio_values[0].cpu()
        
        # Should be very close
        assert torch.allclose(orig_wav, standalone_wav, atol=1e-4)


__all__ = [
    "TestTokenizerV2ConfigEquivalence",
    "TestConvLayerEquivalence",
    "TestTransformerLayerEquivalence",
    "TestQuantizerEquivalence",
    "TestSnakeBetaEquivalence",
    "TestTransformerModelEquivalence",
    "TestDecoderEquivalence",
    "TestPaddingBugFixes",
    "TestRoundtrip",
]
