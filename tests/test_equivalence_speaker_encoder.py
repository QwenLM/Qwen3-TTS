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
from qwen_tts.core.models.configuration_qwen3_tts import (
    Qwen3TTSSpeakerEncoderConfig,
)
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSSpeakerEncoder,
    TimeDelayNetBlock as TimeDelayNetBlockOrig,
    Res2NetBlock as Res2NetBlockOrig,
    SqueezeExcitationBlock as SqueezeExcitationBlockOrig,
    SqueezeExcitationRes2NetBlock as SqueezeExcitationRes2NetBlockOrig,
    AttentiveStatisticsPooling as AttentiveStatisticsPoolingOrig,
    mel_spectrogram as mel_spectrogram_orig,
    dynamic_range_compression_torch as dynamic_range_compression_orig,
)

# Standalone models
from qwen3_tts_standalone.configuration import (
    SpeakerEncoderConfig as SpeakerEncoderConfigStandalone,
)
from qwen3_tts_standalone.speaker_encoder import (
    SpeakerEncoder as SpeakerEncoderStandalone,
    TimeDelayNetBlock as TimeDelayNetBlockStandalone,
    Res2NetBlock as Res2NetBlockStandalone,
    SqueezeExcitationBlock as SqueezeExcitationBlockStandalone,
    SqueezeExcitationRes2NetBlock as SqueezeExcitationRes2NetBlockStandalone,
    AttentiveStatisticsPooling as AttentiveStatisticsPoolingStandalone,
    mel_spectrogram as mel_spectrogram_standalone,
    dynamic_range_compression_torch as dynamic_range_compression_standalone,
)


class TestTimeDelayNetBlockEquivalence:
    """Test TimeDelayNetBlock component equivalence."""

    def test_tdnn_forward_equivalence(self):
        """Test that TimeDelayNetBlock produces identical outputs."""
        set_seed(42)

        in_channels, out_channels = 128, 256
        kernel_size, dilation = 5, 2

        block_orig = TimeDelayNetBlockOrig(
            in_channels, out_channels, kernel_size, dilation
        )
        block_standalone = TimeDelayNetBlockStandalone(
            in_channels, out_channels, kernel_size, dilation
        )

        copy_weights(block_orig, block_standalone)

        set_seed(42)
        x = torch.randn(2, in_channels, 100)

        with torch.no_grad():
            output_orig = block_orig(x)
            output_standalone = block_standalone(x)

        diff = (output_orig - output_standalone).abs().max()
        assert torch.allclose(output_orig, output_standalone, atol=1e-6), \
            f"TimeDelayNetBlock outputs differ. Max diff: {diff}"

    def test_tdnn_different_configs(self):
        """Test TimeDelayNetBlock with different configurations."""
        configs = [
            (64, 128, 3, 1),
            (128, 256, 5, 2),
            (256, 512, 7, 4),
        ]

        for in_ch, out_ch, kernel, dilation in configs:
            set_seed(42)

            block_orig = TimeDelayNetBlockOrig(in_ch, out_ch, kernel, dilation)
            block_standalone = TimeDelayNetBlockStandalone(
                in_ch, out_ch, kernel, dilation
            )
            copy_weights(block_orig, block_standalone)

            x = torch.randn(2, in_ch, 50)

            with torch.no_grad():
                output_orig = block_orig(x)
                output_standalone = block_standalone(x)

            assert torch.allclose(output_orig, output_standalone, atol=1e-6), \
                f"TDNN differs for ({in_ch}, {out_ch}, {kernel}, {dilation})"


class TestRes2NetBlockEquivalence:
    """Test Res2NetBlock component equivalence."""

    def test_res2net_forward_equivalence(self):
        """Test that Res2NetBlock produces identical outputs."""
        set_seed(42)

        in_channels, out_channels = 512, 512
        scale, kernel_size, dilation = 8, 3, 2

        block_orig = Res2NetBlockOrig(
            in_channels, out_channels, scale, kernel_size, dilation
        )
        block_standalone = Res2NetBlockStandalone(
            in_channels, out_channels, scale, kernel_size, dilation
        )

        copy_weights(block_orig, block_standalone)

        set_seed(42)
        x = torch.randn(2, in_channels, 100)

        with torch.no_grad():
            output_orig = block_orig(x)
            output_standalone = block_standalone(x)

        diff = (output_orig - output_standalone).abs().max()
        assert torch.allclose(output_orig, output_standalone, atol=1e-6), \
            f"Res2NetBlock outputs differ. Max diff: {diff}"

    def test_res2net_different_scales(self):
        """Test Res2NetBlock with different scale values."""
        for scale in [4, 8, 16]:
            set_seed(42)

            channels = 512
            block_orig = Res2NetBlockOrig(channels, channels, scale, 3, 1)
            block_standalone = Res2NetBlockStandalone(
                channels, channels, scale, 3, 1
            )
            copy_weights(block_orig, block_standalone)

            x = torch.randn(2, channels, 50)

            with torch.no_grad():
                output_orig = block_orig(x)
                output_standalone = block_standalone(x)

            assert torch.allclose(output_orig, output_standalone, atol=1e-6), \
                f"Res2NetBlock outputs differ for scale={scale}"


class TestSqueezeExcitationBlockEquivalence:
    """Test SqueezeExcitationBlock component equivalence."""

    def test_se_forward_equivalence(self):
        """Test that SqueezeExcitationBlock produces identical outputs."""
        set_seed(42)

        in_channels, se_channels, out_channels = 512, 128, 512

        block_orig = SqueezeExcitationBlockOrig(
            in_channels, se_channels, out_channels
        )
        block_standalone = SqueezeExcitationBlockStandalone(
            in_channels, se_channels, out_channels
        )

        copy_weights(block_orig, block_standalone)

        set_seed(42)
        x = torch.randn(2, in_channels, 100)

        with torch.no_grad():
            output_orig = block_orig(x)
            output_standalone = block_standalone(x)

        diff = (output_orig - output_standalone).abs().max()
        assert torch.allclose(output_orig, output_standalone, atol=1e-6), \
            f"SqueezeExcitationBlock outputs differ. Max diff: {diff}"


class TestSqueezeExcitationRes2NetBlockEquivalence:
    """Test SqueezeExcitationRes2NetBlock component equivalence."""

    def test_se_res2net_forward_equivalence(self):
        """Test that SqueezeExcitationRes2NetBlock produces identical outputs."""
        set_seed(42)

        in_channels, out_channels = 512, 512
        res2net_scale, se_channels = 8, 128
        kernel_size, dilation = 3, 2

        block_orig = SqueezeExcitationRes2NetBlockOrig(
            in_channels, out_channels, res2net_scale,
            se_channels, kernel_size, dilation
        )
        block_standalone = SqueezeExcitationRes2NetBlockStandalone(
            in_channels, out_channels, res2net_scale,
            se_channels, kernel_size, dilation
        )

        copy_weights(block_orig, block_standalone)

        set_seed(42)
        x = torch.randn(2, in_channels, 100)

        with torch.no_grad():
            output_orig = block_orig(x)
            output_standalone = block_standalone(x)

        diff = (output_orig - output_standalone).abs().max()
        assert torch.allclose(output_orig, output_standalone, atol=1e-6), \
            f"SqueezeExcitationRes2NetBlock outputs differ. Max diff: {diff}"


class TestAttentiveStatisticsPoolingEquivalence:
    """Test AttentiveStatisticsPooling component equivalence."""

    def test_asp_forward_equivalence(self):
        """Test that AttentiveStatisticsPooling produces identical outputs."""
        set_seed(42)

        channels, attention_channels = 1536, 128

        pool_orig = AttentiveStatisticsPoolingOrig(channels, attention_channels)
        pool_standalone = AttentiveStatisticsPoolingStandalone(
            channels, attention_channels
        )

        copy_weights(pool_orig, pool_standalone)

        set_seed(42)
        x = torch.randn(2, channels, 100)

        with torch.no_grad():
            output_orig = pool_orig(x)
            output_standalone = pool_standalone(x)

        diff = (output_orig - output_standalone).abs().max()
        assert torch.allclose(output_orig, output_standalone, atol=1e-6), \
            f"AttentiveStatisticsPooling outputs differ. Max diff: {diff}"

    def test_asp_different_lengths(self):
        """Test AttentiveStatisticsPooling with different sequence lengths."""
        set_seed(42)

        channels, attention_channels = 512, 64

        pool_orig = AttentiveStatisticsPoolingOrig(channels, attention_channels)
        pool_standalone = AttentiveStatisticsPoolingStandalone(
            channels, attention_channels
        )
        copy_weights(pool_orig, pool_standalone)

        for seq_len in [25, 50, 100, 200]:
            set_seed(42)
            x = torch.randn(2, channels, seq_len)

            with torch.no_grad():
                output_orig = pool_orig(x)
                output_standalone = pool_standalone(x)

            assert torch.allclose(output_orig, output_standalone, atol=1e-5), \
                f"ASP outputs differ for seq_len={seq_len}"


class TestMelSpectrogramEquivalence:
    """Test mel_spectrogram function equivalence."""

    def test_dynamic_range_compression_equivalence(self):
        """Test dynamic_range_compression_torch produces identical outputs."""
        set_seed(42)

        # Positive values like mel spectrogram
        x = torch.abs(torch.randn(2, 128, 100))

        output_orig = dynamic_range_compression_orig(x)
        output_standalone = dynamic_range_compression_standalone(x)

        diff = (output_orig - output_standalone).abs().max()
        assert torch.allclose(output_orig, output_standalone, atol=1e-6), \
            f"dynamic_range_compression outputs differ. Max diff: {diff}"

    def test_mel_spectrogram_equivalence(self):
        """Test that mel_spectrogram produces identical outputs."""
        set_seed(42)

        # Parameters matching typical speaker encoder config
        n_fft = 2048
        num_mels = 128
        sampling_rate = 24000
        hop_size = 300
        win_size = 1200
        fmin = 0
        fmax = 8000

        # Generate random audio (typical 1-2 seconds at 24kHz)
        audio = torch.randn(2, 48000) * 0.5  # Keep values in reasonable range

        output_orig = mel_spectrogram_orig(
            audio, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax
        )
        output_standalone = mel_spectrogram_standalone(
            audio, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax
        )

        diff = (output_orig - output_standalone).abs().max()
        assert torch.allclose(output_orig, output_standalone, atol=1e-5), \
            f"mel_spectrogram outputs differ. Max diff: {diff}"

    def test_mel_spectrogram_different_lengths(self):
        """Test mel_spectrogram with different audio lengths."""
        n_fft = 2048
        num_mels = 128
        sampling_rate = 24000
        hop_size = 300
        win_size = 1200
        fmin = 0
        fmax = 8000

        for audio_len in [24000, 48000, 72000]:
            set_seed(42)
            audio = torch.randn(1, audio_len) * 0.5

            output_orig = mel_spectrogram_orig(
                audio, n_fft, num_mels, sampling_rate,
                hop_size, win_size, fmin, fmax
            )
            output_standalone = mel_spectrogram_standalone(
                audio, n_fft, num_mels, sampling_rate,
                hop_size, win_size, fmin, fmax
            )

            assert torch.allclose(output_orig, output_standalone, atol=1e-5), \
                f"mel_spectrogram outputs differ for audio_len={audio_len}"


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
        config_standalone = SpeakerEncoderConfigStandalone(
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
        encoder_standalone = SpeakerEncoderStandalone(config_standalone)

        # Copy weights
        copy_weights(encoder_orig, encoder_standalone)

        # Test input (batch, time, mel_dim)
        set_seed(42)
        x = torch.randn(2, 100, 128)

        # Forward pass
        with torch.no_grad():
            output_orig = encoder_orig(x)
            output_standalone = encoder_standalone(x)

        diff = (output_orig - output_standalone).abs().max()
        assert torch.allclose(output_orig, output_standalone, atol=1e-5), \
            f"SpeakerEncoder outputs differ. Max diff: {diff}"

    def test_speaker_encoder_different_lengths(self):
        """Test SpeakerEncoder with different input lengths."""
        set_seed(42)

        config_orig = Qwen3TTSSpeakerEncoderConfig(mel_dim=128, enc_dim=256)
        config_standalone = SpeakerEncoderConfigStandalone(
            mel_dim=128, enc_dim=256
        )

        encoder_orig = Qwen3TTSSpeakerEncoder(config_orig)
        encoder_standalone = SpeakerEncoderStandalone(config_standalone)
        copy_weights(encoder_orig, encoder_standalone)

        for seq_len in [50, 100, 200]:
            set_seed(42)
            x = torch.randn(2, seq_len, 128)

            with torch.no_grad():
                output_orig = encoder_orig(x)
                output_standalone = encoder_standalone(x)

            assert torch.allclose(output_orig, output_standalone, atol=1e-5), \
                f"SpeakerEncoder outputs differ for seq_len={seq_len}"

    def test_speaker_encoder_different_batch_sizes(self):
        """Test SpeakerEncoder with different batch sizes."""
        set_seed(42)

        config_orig = Qwen3TTSSpeakerEncoderConfig(mel_dim=128, enc_dim=256)
        config_standalone = SpeakerEncoderConfigStandalone(
            mel_dim=128, enc_dim=256
        )

        encoder_orig = Qwen3TTSSpeakerEncoder(config_orig)
        encoder_standalone = SpeakerEncoderStandalone(config_standalone)
        copy_weights(encoder_orig, encoder_standalone)

        for batch_size in [1, 2, 4, 8]:
            set_seed(42)
            x = torch.randn(batch_size, 100, 128)

            with torch.no_grad():
                output_orig = encoder_orig(x)
                output_standalone = encoder_standalone(x)

            assert torch.allclose(output_orig, output_standalone, atol=1e-5), \
                f"SpeakerEncoder outputs differ for batch_size={batch_size}"

    def test_speaker_encoder_different_configs(self):
        """Test SpeakerEncoder with different configurations."""
        configs = [
            {"mel_dim": 80, "enc_dim": 192},
            {"mel_dim": 128, "enc_dim": 512},
            {"mel_dim": 128, "enc_dim": 1024, "enc_attention_channels": 256},
        ]

        for config_kwargs in configs:
            set_seed(42)

            config_orig = Qwen3TTSSpeakerEncoderConfig(**config_kwargs)
            config_standalone = SpeakerEncoderConfigStandalone(**config_kwargs)

            encoder_orig = Qwen3TTSSpeakerEncoder(config_orig)
            encoder_standalone = SpeakerEncoderStandalone(config_standalone)
            copy_weights(encoder_orig, encoder_standalone)

            x = torch.randn(2, 100, config_kwargs["mel_dim"])

            with torch.no_grad():
                output_orig = encoder_orig(x)
                output_standalone = encoder_standalone(x)

            assert torch.allclose(output_orig, output_standalone, atol=1e-5), \
                f"SpeakerEncoder outputs differ for config {config_kwargs}"

    def test_speaker_encoder_gradient_equivalence(self):
        """Test that gradients are equivalent during backpropagation."""
        set_seed(42)

        config_orig = Qwen3TTSSpeakerEncoderConfig(mel_dim=128, enc_dim=192)
        config_standalone = SpeakerEncoderConfigStandalone(
            mel_dim=128, enc_dim=192
        )

        encoder_orig = Qwen3TTSSpeakerEncoder(config_orig)
        encoder_standalone = SpeakerEncoderStandalone(config_standalone)
        copy_weights(encoder_orig, encoder_standalone)

        encoder_orig.train()
        encoder_standalone.train()

        set_seed(42)
        x = torch.randn(2, 100, 128, requires_grad=True)
        x_standalone = x.clone().detach().requires_grad_(True)

        # Forward pass
        output_orig = encoder_orig(x)
        output_standalone = encoder_standalone(x_standalone)

        # Backward pass with same loss
        loss_orig = output_orig.sum()
        loss_standalone = output_standalone.sum()

        loss_orig.backward()
        loss_standalone.backward()

        # Check gradients
        diff = (x.grad - x_standalone.grad).abs().max()
        assert torch.allclose(x.grad, x_standalone.grad, atol=1e-5), \
            f"SpeakerEncoder input gradients differ. Max diff: {diff}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_speaker_encoder_cuda_equivalence(self):
        """Test SpeakerEncoder equivalence on CUDA."""
        set_seed(42)

        config_orig = Qwen3TTSSpeakerEncoderConfig(mel_dim=128, enc_dim=256)
        config_standalone = SpeakerEncoderConfigStandalone(
            mel_dim=128, enc_dim=256
        )

        encoder_orig = Qwen3TTSSpeakerEncoder(config_orig).cuda()
        encoder_standalone = SpeakerEncoderStandalone(config_standalone).cuda()
        copy_weights(encoder_orig, encoder_standalone)

        set_seed(42)
        x = torch.randn(2, 100, 128).cuda()

        with torch.no_grad():
            output_orig = encoder_orig(x)
            output_standalone = encoder_standalone(x)

        diff = (output_orig - output_standalone).abs().max()
        assert torch.allclose(output_orig, output_standalone, atol=1e-5), \
            f"SpeakerEncoder CUDA outputs differ. Max diff: {diff}"

    def test_speaker_encoder_float16_equivalence(self):
        """Test SpeakerEncoder equivalence with float16 dtype."""
        set_seed(42)

        config_orig = Qwen3TTSSpeakerEncoderConfig(mel_dim=128, enc_dim=256)
        config_standalone = SpeakerEncoderConfigStandalone(
            mel_dim=128, enc_dim=256
        )

        encoder_orig = Qwen3TTSSpeakerEncoder(config_orig).half()
        encoder_standalone = SpeakerEncoderStandalone(config_standalone).half()
        copy_weights(encoder_orig, encoder_standalone)

        set_seed(42)
        x = torch.randn(2, 100, 128).half()

        with torch.no_grad():
            output_orig = encoder_orig(x)
            output_standalone = encoder_standalone(x)

        # Use larger tolerance for float16
        diff = (output_orig - output_standalone).abs().max()
        assert torch.allclose(output_orig, output_standalone, atol=1e-3), \
            f"SpeakerEncoder float16 outputs differ. Max diff: {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
