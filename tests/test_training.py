# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for training/finetuning functions in the standalone implementation.

These tests verify:
1. forward_sub_talker_finetune produces correct output shapes
2. Gradient flow through the training path
3. Overfitting on synthetic data (loss decreases)
4. Equivalence with the original implementation (when transformers is available)
5. Overfitting on real test_data (requires pretrained model + GPU)
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import set_seed

from qwen3_tts_standalone import CodePredictor, CodePredictorConfig, Talker, TalkerConfig
from qwen3_tts_standalone.code_predictor import (
    CodePredictorFinetuneOutput,
    _causal_lm_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_code_predictor_config():
    """Small CodePredictor config for fast testing."""
    return CodePredictorConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_code_groups=4,
    )


@pytest.fixture
def small_talker_config(small_code_predictor_config):
    """Small Talker config for fast testing."""
    return TalkerConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        text_hidden_size=48,
        text_vocab_size=128,
        num_code_groups=4,
        code_predictor_config=small_code_predictor_config,
    )


@pytest.fixture
def small_talker(small_talker_config):
    """Create a small Talker model."""
    set_seed(42)
    return Talker(small_talker_config)


# ---------------------------------------------------------------------------
# Tests for _causal_lm_loss
# ---------------------------------------------------------------------------

class TestCausalLMLoss:
    """Test the standalone ForCausalLMLoss implementation."""

    def test_basic_loss_computation(self):
        """Test basic loss computation with known values."""
        set_seed(42)
        vocab_size = 10
        batch_size = 2
        seq_len = 5

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = _causal_lm_loss(logits, labels, vocab_size)

        assert loss.ndim == 0  # scalar
        assert loss.item() > 0  # cross-entropy is always positive
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_loss_with_shift(self):
        """Test that the loss applies the standard next-token shift."""
        vocab_size = 10
        batch_size = 1
        seq_len = 3

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = _causal_lm_loss(logits, labels, vocab_size)

        # Manual computation with shift
        shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
        shift_labels = labels[..., 1:].contiguous().view(-1)
        expected_loss = F.cross_entropy(shift_logits.float(), shift_labels)

        assert torch.allclose(loss, expected_loss, atol=1e-6)

    def test_loss_ignores_minus_100(self):
        """Test that labels with -100 are ignored."""
        vocab_size = 10
        batch_size = 1
        seq_len = 5

        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        # Only one valid label after shift
        labels[0, 2] = 3

        loss = _causal_lm_loss(logits, labels, vocab_size)
        assert not torch.isnan(loss)

    def test_loss_gradient_flows(self):
        """Test that gradients flow through the loss."""
        vocab_size = 10
        logits = torch.randn(2, 5, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (2, 5))

        loss = _causal_lm_loss(logits, labels, vocab_size)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.norm().item() > 0


# ---------------------------------------------------------------------------
# Tests for CodePredictor.forward_finetune
# ---------------------------------------------------------------------------

class TestCodePredictorForwardFinetune:
    """Test CodePredictor.forward_finetune()."""

    def test_output_shape(self, small_code_predictor_config):
        """Test that forward_finetune produces correct output shapes."""
        set_seed(42)
        embedding_dim = 32  # same as hidden_size
        model = CodePredictor(small_code_predictor_config, embedding_dim)
        model.eval()

        batch_size = 3
        num_code_groups = small_code_predictor_config.num_code_groups

        inputs_embeds = torch.randn(batch_size, num_code_groups, embedding_dim)
        labels = torch.randint(
            0, small_code_predictor_config.vocab_size, (batch_size, num_code_groups - 1)
        )

        with torch.no_grad():
            output = model.forward_finetune(inputs_embeds, labels)

        assert isinstance(output, CodePredictorFinetuneOutput)
        assert output.logits.shape == (
            batch_size,
            num_code_groups - 1,
            small_code_predictor_config.vocab_size,
        )
        assert output.loss is not None
        assert output.loss.ndim == 0

    def test_output_without_labels(self, small_code_predictor_config):
        """Test forward_finetune without labels (loss should be None)."""
        set_seed(42)
        embedding_dim = 32
        model = CodePredictor(small_code_predictor_config, embedding_dim)
        model.eval()

        inputs_embeds = torch.randn(2, small_code_predictor_config.num_code_groups, embedding_dim)

        with torch.no_grad():
            output = model.forward_finetune(inputs_embeds)

        assert output.logits.shape[0] == 2
        assert output.loss is None

    def test_deterministic(self, small_code_predictor_config):
        """Test that forward_finetune is deterministic."""
        set_seed(42)
        embedding_dim = 32
        model = CodePredictor(small_code_predictor_config, embedding_dim)
        model.eval()

        inputs_embeds = torch.randn(2, small_code_predictor_config.num_code_groups, embedding_dim)
        labels = torch.randint(
            0, small_code_predictor_config.vocab_size,
            (2, small_code_predictor_config.num_code_groups - 1),
        )

        with torch.no_grad():
            out1 = model.forward_finetune(inputs_embeds.clone(), labels.clone())
            out2 = model.forward_finetune(inputs_embeds.clone(), labels.clone())

        assert torch.allclose(out1.logits, out2.logits, atol=1e-6)
        assert torch.allclose(out1.loss, out2.loss, atol=1e-6)

    def test_with_projection(self, small_code_predictor_config):
        """Test forward_finetune when input_projection is Linear (not Identity)."""
        set_seed(42)
        embedding_dim = 64  # different from hidden_size=32
        model = CodePredictor(small_code_predictor_config, embedding_dim)
        model.eval()

        assert isinstance(model.input_projection, torch.nn.Linear)

        inputs_embeds = torch.randn(
            2, small_code_predictor_config.num_code_groups, embedding_dim
        )
        labels = torch.randint(
            0, small_code_predictor_config.vocab_size,
            (2, small_code_predictor_config.num_code_groups - 1),
        )

        with torch.no_grad():
            output = model.forward_finetune(inputs_embeds, labels)

        assert output.logits.shape == (
            2,
            small_code_predictor_config.num_code_groups - 1,
            small_code_predictor_config.vocab_size,
        )
        assert output.loss is not None

    def test_gradient_flow(self, small_code_predictor_config):
        """Test that gradients flow back through forward_finetune."""
        set_seed(42)
        embedding_dim = 32
        model = CodePredictor(small_code_predictor_config, embedding_dim)
        model.train()

        inputs_embeds = torch.randn(
            2, small_code_predictor_config.num_code_groups, embedding_dim,
            requires_grad=True,
        )
        labels = torch.randint(
            0, small_code_predictor_config.vocab_size,
            (2, small_code_predictor_config.num_code_groups - 1),
        )

        output = model.forward_finetune(inputs_embeds, labels)
        output.loss.backward()

        # Check gradients on input
        assert inputs_embeds.grad is not None
        assert inputs_embeds.grad.norm().item() > 0

        # Check gradients on model parameters that should be used
        # Note: codec_embedding weights are NOT used in forward_finetune
        # (they are used by the Talker to build inputs_embeds externally)
        params_with_grad = set()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                params_with_grad.add(name)

        # Core transformer layers and lm_heads should have gradients
        assert any("layers" in n for n in params_with_grad), \
            "No gradients in transformer layers"
        assert any("lm_head" in n for n in params_with_grad), \
            "No gradients in lm_head"
        assert any("norm" in n for n in params_with_grad), \
            "No gradients in norm layer"


# ---------------------------------------------------------------------------
# Tests for Talker.forward_sub_talker_finetune
# ---------------------------------------------------------------------------

class TestTalkerForwardSubTalkerFinetune:
    """Test Talker.forward_sub_talker_finetune()."""

    def test_output_shape(self, small_talker):
        """Test that forward_sub_talker_finetune produces correct shapes."""
        small_talker.eval()
        N = 5
        num_code_groups = small_talker.num_code_groups
        hidden_size = small_talker.config.hidden_size

        codec_ids = torch.randint(0, 64, (N, num_code_groups))
        hidden_states = torch.randn(N, hidden_size)

        with torch.no_grad():
            logits, loss = small_talker.forward_sub_talker_finetune(
                codec_ids, hidden_states
            )

        assert logits.shape == (N, num_code_groups - 1, 64)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_assertions(self, small_talker):
        """Test that assertions catch invalid inputs."""
        hidden_size = small_talker.config.hidden_size
        num_code_groups = small_talker.num_code_groups

        # Wrong codec_ids dimensions
        with pytest.raises(AssertionError):
            small_talker.forward_sub_talker_finetune(
                torch.randint(0, 64, (5,)),  # 1D instead of 2D
                torch.randn(5, hidden_size),
            )

        # Wrong hidden_states dimensions
        with pytest.raises(AssertionError):
            small_talker.forward_sub_talker_finetune(
                torch.randint(0, 64, (5, num_code_groups)),
                torch.randn(5,),  # 1D instead of 2D
            )

        # Mismatched batch sizes
        with pytest.raises(AssertionError):
            small_talker.forward_sub_talker_finetune(
                torch.randint(0, 64, (5, num_code_groups)),
                torch.randn(3, hidden_size),  # different batch size
            )

        # Wrong hidden size
        with pytest.raises(AssertionError):
            small_talker.forward_sub_talker_finetune(
                torch.randint(0, 64, (5, num_code_groups)),
                torch.randn(5, hidden_size + 1),  # wrong hidden size
            )

        # Wrong num_code_groups
        with pytest.raises(AssertionError):
            small_talker.forward_sub_talker_finetune(
                torch.randint(0, 64, (5, num_code_groups + 1)),  # wrong groups
                torch.randn(5, hidden_size),
            )

    def test_gradient_flow(self, small_talker):
        """Test gradient flow through forward_sub_talker_finetune."""
        small_talker.train()
        N = 4
        num_code_groups = small_talker.num_code_groups
        hidden_size = small_talker.config.hidden_size

        codec_ids = torch.randint(0, 64, (N, num_code_groups))
        hidden_states = torch.randn(N, hidden_size, requires_grad=True)

        logits, loss = small_talker.forward_sub_talker_finetune(
            codec_ids, hidden_states
        )
        loss.backward()

        # Gradients should flow to hidden_states
        assert hidden_states.grad is not None
        assert hidden_states.grad.norm().item() > 0

        # Gradients should flow to code predictor parameters
        has_grad = False
        for name, param in small_talker.code_predictor.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.norm().item() > 0:
                    has_grad = True
                    break
        assert has_grad, "No gradients in code_predictor parameters"

    def test_deterministic(self, small_talker):
        """Test that forward_sub_talker_finetune is deterministic."""
        small_talker.eval()
        N = 3
        num_code_groups = small_talker.num_code_groups
        hidden_size = small_talker.config.hidden_size

        set_seed(42)
        codec_ids = torch.randint(0, 64, (N, num_code_groups))
        hidden_states = torch.randn(N, hidden_size)

        with torch.no_grad():
            logits1, loss1 = small_talker.forward_sub_talker_finetune(
                codec_ids.clone(), hidden_states.clone()
            )
            logits2, loss2 = small_talker.forward_sub_talker_finetune(
                codec_ids.clone(), hidden_states.clone()
            )

        assert torch.allclose(logits1, logits2, atol=1e-6)
        assert torch.allclose(loss1, loss2, atol=1e-6)

    def test_different_batch_sizes(self, small_talker):
        """Test with various batch sizes."""
        small_talker.eval()
        num_code_groups = small_talker.num_code_groups
        hidden_size = small_talker.config.hidden_size

        for N in [1, 2, 8, 16]:
            codec_ids = torch.randint(0, 64, (N, num_code_groups))
            hidden_states = torch.randn(N, hidden_size)

            with torch.no_grad():
                logits, loss = small_talker.forward_sub_talker_finetune(
                    codec_ids, hidden_states
                )

            assert logits.shape == (N, num_code_groups - 1, 64)
            assert loss.ndim == 0


# ---------------------------------------------------------------------------
# Overfitting test (synthetic data)
# ---------------------------------------------------------------------------

class TestOverfittingSynthetic:
    """Test that the model can overfit on synthetic data."""

    def test_sub_talker_overfitting(self, small_talker_config):
        """Test that forward_sub_talker_finetune loss decreases during training."""
        set_seed(42)
        talker = Talker(small_talker_config)
        talker.train()

        num_code_groups = small_talker_config.num_code_groups
        hidden_size = small_talker_config.hidden_size

        # Create a small fixed dataset
        N = 8
        codec_ids = torch.randint(0, small_talker_config.vocab_size, (N, num_code_groups))
        hidden_states = torch.randn(N, hidden_size)

        # Only optimize code predictor parameters
        optimizer = torch.optim.Adam(talker.code_predictor.parameters(), lr=1e-3)

        losses = []
        for step in range(50):
            optimizer.zero_grad()
            _, loss = talker.forward_sub_talker_finetune(
                codec_ids, hidden_states.detach()
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease significantly
        initial_loss = sum(losses[:5]) / 5
        final_loss = sum(losses[-5:]) / 5
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )
        # Require at least 20% reduction
        assert final_loss < initial_loss * 0.8, (
            f"Loss decrease too small: initial={initial_loss:.4f}, final={final_loss:.4f}, "
            f"ratio={final_loss / initial_loss:.4f}"
        )

    def test_code_predictor_overfitting(self, small_code_predictor_config):
        """Test that CodePredictor.forward_finetune loss decreases during training."""
        set_seed(42)
        embedding_dim = 32
        model = CodePredictor(small_code_predictor_config, embedding_dim)
        model.train()

        num_code_groups = small_code_predictor_config.num_code_groups

        # Create fixed dataset
        batch_size = 8
        inputs_embeds = torch.randn(batch_size, num_code_groups, embedding_dim)
        labels = torch.randint(
            0, small_code_predictor_config.vocab_size,
            (batch_size, num_code_groups - 1),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for step in range(50):
            optimizer.zero_grad()
            output = model.forward_finetune(inputs_embeds, labels)
            output.loss.backward()
            optimizer.step()
            losses.append(output.loss.item())

        initial_loss = sum(losses[:5]) / 5
        final_loss = sum(losses[-5:]) / 5
        assert final_loss < initial_loss * 0.8, (
            f"Loss decrease too small: initial={initial_loss:.4f}, final={final_loss:.4f}"
        )


# ---------------------------------------------------------------------------
# Equivalence test with original implementation
# (requires transformers + qwen_tts package)
# ---------------------------------------------------------------------------

def _has_transformers():
    """Check if transformers and qwen_tts are available."""
    try:
        import transformers
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSTalkerForConditionalGeneration,
        )
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _has_transformers(),
    reason="Requires transformers and qwen_tts packages",
)
class TestEquivalenceWithOriginal:
    """Test equivalence between standalone and original forward_sub_talker_finetune."""

    def test_forward_sub_talker_finetune_equivalence(self):
        """
        Test that standalone and original forward_sub_talker_finetune
        produce identical outputs when given the same inputs and weights.
        """
        from qwen_tts.core.models.configuration_qwen3_tts import (
            Qwen3TTSTalkerConfig,
            Qwen3TTSTalkerCodePredictorConfig,
        )
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSTalkerForConditionalGeneration,
        )

        set_seed(42)

        # Create matching configs
        code_pred_config_orig = Qwen3TTSTalkerCodePredictorConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_code_groups=4,
        )

        talker_config_orig = Qwen3TTSTalkerConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            text_hidden_size=48,
            text_vocab_size=128,
            num_code_groups=4,
            code_predictor_config=code_pred_config_orig.to_dict(),
        )

        code_pred_config_standalone = CodePredictorConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_code_groups=4,
        )

        talker_config_standalone = TalkerConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            text_hidden_size=48,
            text_vocab_size=128,
            num_code_groups=4,
            code_predictor_config=code_pred_config_standalone,
        )

        # Create both models
        set_seed(42)
        orig_talker = Qwen3TTSTalkerForConditionalGeneration(talker_config_orig)

        standalone_talker = Talker(talker_config_standalone)

        # Copy weights from original to standalone
        standalone_talker.load_original_state_dict(orig_talker.state_dict())

        orig_talker.eval()
        standalone_talker.eval()

        # Create test inputs
        set_seed(123)
        N = 5
        codec_ids = torch.randint(0, 64, (N, 4))
        hidden_states = torch.randn(N, 32)

        # Run both
        with torch.no_grad():
            orig_logits, orig_loss = orig_talker.forward_sub_talker_finetune(
                codec_ids, hidden_states
            )
            standalone_logits, standalone_loss = (
                standalone_talker.forward_sub_talker_finetune(
                    codec_ids, hidden_states
                )
            )

        # Compare outputs
        assert torch.allclose(orig_logits, standalone_logits, atol=1e-5), (
            f"Logits differ. Max diff: "
            f"{(orig_logits - standalone_logits).abs().max().item()}"
        )
        assert torch.allclose(orig_loss, standalone_loss, atol=1e-5), (
            f"Losses differ: original={orig_loss.item():.6f}, "
            f"standalone={standalone_loss.item():.6f}"
        )


# ---------------------------------------------------------------------------
# Overfitting test with real test_data (requires GPU + pretrained model)
# ---------------------------------------------------------------------------

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
PROMPT_WAV = TEST_DATA_DIR / "prompt.wav"
TRANSCRIPTION_TXT = TEST_DATA_DIR / "transcription.txt"
CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA GPU",
)
@pytest.mark.skipif(
    not PROMPT_WAV.exists(),
    reason="test_data/prompt.wav not found",
)
class TestOverfittingRealData:
    """
    Test overfitting on real test_data/prompt.wav.

    This test loads a pretrained model, encodes the test audio to codec tokens,
    and verifies that forward_sub_talker_finetune loss decreases when training
    on this single example.
    """

    def test_sub_talker_overfitting_real_data(self):
        """Overfit the sub-talker on a single real audio sample."""
        import librosa
        import numpy as np

        from qwen3_tts_standalone import Qwen3TTSModel

        # Load the model
        model_wrapper = Qwen3TTSModel.from_pretrained(
            CHECKPOINT,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        tts = model_wrapper.model
        processor = model_wrapper.processor

        # Load and encode audio
        audio, sr = librosa.load(str(PROMPT_WAV), sr=None, mono=True)
        enc = tts.speech_tokenizer.encode([audio], sr=sr)
        audio_codes = enc.audio_codes[0]  # [T, 16]

        # Read transcription
        transcription = TRANSCRIPTION_TXT.read_text().strip()

        # Build text IDs
        text = f"<|im_start|>assistant\n{transcription}<|im_end|>\n<|im_start|>assistant\n"
        text_input = processor(text=text, return_tensors="pt", padding=True)
        text_ids = text_input["input_ids"].to(tts.device)

        # Get speaker embedding
        from qwen3_tts_standalone.speaker_encoder import mel_spectrogram

        audio_24k = librosa.resample(
            audio.astype(np.float32), orig_sr=sr, target_sr=24000
        )
        mels = mel_spectrogram(
            torch.from_numpy(audio_24k).unsqueeze(0),
            n_fft=1024, num_mels=128, sampling_rate=24000,
            hop_size=256, win_size=1024, fmin=0, fmax=12000,
        ).transpose(1, 2)
        speaker_embedding = tts.speaker_encoder(
            mels.to(tts.device).to(tts.dtype)
        ).detach()

        # Build input embeddings (matching sft_12hz.py collate_fn logic)
        codec_ids_tensor = audio_codes.to(tts.device).long()
        T_codec = codec_ids_tensor.shape[0]
        text_ids_trimmed = text_ids[:, :-5]
        T_text = text_ids_trimmed.shape[1]

        total_len = T_text + T_codec + 8
        input_ids = torch.zeros((1, total_len, 2), dtype=torch.long, device=tts.device)
        codec_ids_full = torch.zeros(
            (1, total_len, tts.talker.num_code_groups), dtype=torch.long, device=tts.device
        )
        text_emb_mask = torch.zeros((1, total_len), dtype=torch.bool, device=tts.device)
        codec_emb_mask = torch.zeros((1, total_len), dtype=torch.bool, device=tts.device)
        codec_mask = torch.zeros((1, total_len), dtype=torch.bool, device=tts.device)
        attention_mask = torch.zeros((1, total_len), dtype=torch.long, device=tts.device)
        codec_0_labels = torch.full(
            (1, total_len), -100, dtype=torch.long, device=tts.device
        )

        tc = tts.config.talker_config

        # Text channel
        input_ids[0, :3, 0] = text_ids_trimmed[0, :3]
        input_ids[0, 3:7, 0] = tts.config.tts_pad_token_id
        input_ids[0, 7, 0] = tts.config.tts_bos_token_id
        input_ids[0, 8 : 8 + T_text - 3, 0] = text_ids_trimmed[0, 3:]
        input_ids[0, 8 + T_text - 3, 0] = tts.config.tts_eos_token_id
        input_ids[0, 8 + T_text - 2 : 8 + T_text + T_codec, 0] = tts.config.tts_pad_token_id
        text_emb_mask[0, : 8 + T_text + T_codec] = True

        # Codec channel
        input_ids[0, 3:8, 1] = torch.tensor(
            [
                tc.codec_nothink_id,
                tc.codec_think_bos_id,
                tc.codec_think_eos_id,
                0,  # speaker embedding slot
                tc.codec_pad_id,
            ],
            device=tts.device,
        )
        input_ids[0, 8 : 8 + T_text - 3, 1] = tc.codec_pad_id
        input_ids[0, 8 + T_text - 3, 1] = tc.codec_pad_id
        input_ids[0, 8 + T_text - 2, 1] = tc.codec_bos_id
        input_ids[0, 8 + T_text - 1 : 8 + T_text - 1 + T_codec, 1] = codec_ids_tensor[:, 0]
        input_ids[0, 8 + T_text - 1 + T_codec, 1] = tc.codec_eos_token_id

        codec_0_labels[0, 8 + T_text - 1 : 8 + T_text - 1 + T_codec] = codec_ids_tensor[:, 0]
        codec_0_labels[0, 8 + T_text - 1 + T_codec] = tc.codec_eos_token_id

        codec_ids_full[0, 8 + T_text - 1 : 8 + T_text - 1 + T_codec, :] = codec_ids_tensor

        codec_emb_mask[0, 3 : 8 + T_text + T_codec] = True
        codec_emb_mask[0, 6] = False  # speaker embedding slot

        codec_mask[0, 8 + T_text - 1 : 8 + T_text - 1 + T_codec] = True
        attention_mask[0, : 8 + T_text + T_codec] = 1

        # Compute embeddings
        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]

        input_text_emb = tts.talker.text_embedding(input_text_ids) * text_emb_mask.unsqueeze(-1)
        input_codec_emb = tts.talker.codec_embedding(input_codec_ids) * codec_emb_mask.unsqueeze(-1)
        input_codec_emb[:, 6, :] = speaker_embedding

        input_embeddings = input_text_emb + input_codec_emb

        for i in range(1, tts.talker.num_code_groups):
            codec_i_emb = tts.talker.code_predictor.codec_embedding[i - 1](
                codec_ids_full[:, :, i]
            )
            codec_i_emb = codec_i_emb * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_i_emb

        # Run main talker forward (without labels, just to get hidden states)
        talker_out = tts.talker.forward(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            use_cache=False,
        )
        hidden_states_all = talker_out.hidden_state

        # Extract hidden states at codec positions
        talker_hidden_states = hidden_states_all[codec_mask[:, 1:]]
        talker_codec_ids = codec_ids_full[codec_mask]

        # Run sub-talker finetune
        optimizer = torch.optim.Adam(tts.talker.code_predictor.parameters(), lr=1e-4)

        losses = []
        for step in range(20):
            optimizer.zero_grad()
            _, loss = tts.talker.forward_sub_talker_finetune(
                talker_codec_ids, talker_hidden_states.detach()
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        initial_loss = sum(losses[:3]) / 3
        final_loss = sum(losses[-3:]) / 3
        assert final_loss < initial_loss, (
            f"Loss did not decrease on real data: "
            f"initial={initial_loss:.4f}, final={final_loss:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
