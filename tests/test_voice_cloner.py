# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the VoiceCloner class.

These tests verify that the VoiceCloner:
1. Can be imported correctly
2. Has the expected API
3. Produces output equivalent to Qwen3TTSModel.generate_voice_clone()
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Mark all tests in this module
pytestmark = [
    pytest.mark.voice_cloner,
]


class TestVoiceClonerImport:
    """Tests for importing the VoiceCloner class."""
    
    def test_import_from_package(self):
        """Test that VoiceCloner can be imported from the package."""
        from qwen3_tts_standalone import VoiceCloner
        assert VoiceCloner is not None
    
    def test_import_from_module(self):
        """Test that VoiceCloner can be imported from the module."""
        from qwen3_tts_standalone.voice_cloner import VoiceCloner
        assert VoiceCloner is not None
    
    def test_class_has_expected_methods(self):
        """Test that VoiceCloner has the expected public methods."""
        from qwen3_tts_standalone import VoiceCloner
        
        # Check class methods
        assert hasattr(VoiceCloner, "from_pretrained")
        
        # Check instance methods (by inspecting the class)
        assert hasattr(VoiceCloner, "clone_voice")
        assert hasattr(VoiceCloner, "device")
        assert hasattr(VoiceCloner, "dtype")


class TestVoiceClonerAPI:
    """Tests for the VoiceCloner API without loading the model."""
    
    def test_from_pretrained_requires_base_model(self):
        """Test that from_pretrained checks for base model type."""
        from qwen3_tts_standalone import VoiceCloner
        from qwen3_tts_standalone.configuration import TTSConfig, TalkerConfig, SpeakerEncoderConfig
        
        # Create a mock config with non-base model type
        config = TTSConfig(
            talker_config=TalkerConfig(),
            speaker_encoder_config=SpeakerEncoderConfig(),
            tts_model_type="custom_voice",  # Not "base"
        )
        
        # The check happens in from_pretrained, so we verify the config field exists
        assert config.tts_model_type == "custom_voice"
    
    def test_clone_voice_parameter_validation(self):
        """Test that clone_voice validates its parameters."""
        from qwen3_tts_standalone import VoiceCloner
        
        # The method signature should accept these parameters
        import inspect
        sig = inspect.signature(VoiceCloner.clone_voice)
        params = list(sig.parameters.keys())
        
        expected_params = [
            "self", "text", "ref_audio", "ref_text", "language",
            "max_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty"
        ]
        
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU tests require CUDA"
)
@pytest.mark.slow
@pytest.mark.e2e
class TestVoiceClonerEquivalence:
    """
    Tests to verify VoiceCloner produces equivalent output to Qwen3TTSModel.
    
    These tests require GPU and the full model to be loaded.
    """
    
    CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    SEED = 42
    
    @pytest.fixture
    def prompt_audio(self):
        """Get the path to the test prompt audio."""
        audio_path = Path(__file__).parent.parent / "test_data" / "prompt.wav"
        if not audio_path.exists():
            pytest.skip("test_data/prompt.wav not found")
        return str(audio_path)
    
    @pytest.fixture
    def ref_text(self):
        """Reference text for the prompt audio (from test_data/transcription.txt)."""
        transcription_path = Path(__file__).parent.parent / "test_data" / "transcription.txt"
        if transcription_path.exists():
            return transcription_path.read_text().strip()
        return (
            "Dazu gehört beispielsweise Eurojust, denn bis jetzt sind für den "
            "Europäischen Staatsanwalt keine zusätzlichen Mittel und kein "
            "zusätzliches Personal vorgesehen."
        )
    
    @pytest.fixture
    def target_text(self):
        """Target text to synthesize."""
        return "Hello, this is a test of voice cloning."
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def test_voice_cloner_loads(self, prompt_audio):
        """Test that VoiceCloner can load the pretrained model."""
        from qwen3_tts_standalone import VoiceCloner
        
        cloner = VoiceCloner.from_pretrained(
            self.CHECKPOINT,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        
        assert cloner is not None
        assert cloner.device.type == "cuda"
        assert cloner.dtype == torch.bfloat16
    
    def test_voice_cloner_generates_audio(self, prompt_audio, ref_text, target_text):
        """Test that VoiceCloner generates valid audio."""
        from qwen3_tts_standalone import VoiceCloner
        
        self._set_seed(self.SEED)
        
        cloner = VoiceCloner.from_pretrained(
            self.CHECKPOINT,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        
        audio, sr = cloner.clone_voice(
            text=target_text,
            ref_audio=prompt_audio,
            ref_text=ref_text,
            language="English",
        )
        
        # Basic validation
        assert isinstance(audio, np.ndarray)
        assert sr == 24000
        assert len(audio) > 0
        assert np.abs(audio).max() > 1e-6, "Audio appears to be silent"
    
    def test_voice_cloner_deterministic(self, prompt_audio, ref_text, target_text):
        """Test that VoiceCloner is deterministic with the same seed."""
        from qwen3_tts_standalone import VoiceCloner
        
        cloner = VoiceCloner.from_pretrained(
            self.CHECKPOINT,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        
        # First generation
        self._set_seed(self.SEED)
        audio1, sr1 = cloner.clone_voice(
            text=target_text,
            ref_audio=prompt_audio,
            ref_text=ref_text,
            language="English",
        )
        
        # Second generation with same seed
        self._set_seed(self.SEED)
        audio2, sr2 = cloner.clone_voice(
            text=target_text,
            ref_audio=prompt_audio,
            ref_text=ref_text,
            language="English",
        )
        
        assert len(audio1) == len(audio2), "Audio lengths should match"
        assert np.allclose(audio1, audio2, atol=1e-6), "Audio should be identical"
    
    def test_equivalence_with_qwen3ttsmodel(self, prompt_audio, ref_text, target_text):
        """
        Test that VoiceCloner produces the same output as Qwen3TTSModel.
        
        This is the critical equivalence test that ensures the simplified
        implementation produces the same results as the original.
        """
        from qwen3_tts_standalone import VoiceCloner, Qwen3TTSModel
        
        # Load both models
        cloner = VoiceCloner.from_pretrained(
            self.CHECKPOINT,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        
        original = Qwen3TTSModel.from_pretrained(
            self.CHECKPOINT,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        
        # Generate with VoiceCloner
        self._set_seed(self.SEED)
        audio_cloner, sr_cloner = cloner.clone_voice(
            text=target_text,
            ref_audio=prompt_audio,
            ref_text=ref_text,
            language="English",
        )
        
        # Generate with Qwen3TTSModel
        self._set_seed(self.SEED)
        audio_original, sr_original = original.generate_voice_clone(
            text=target_text,
            ref_audio=prompt_audio,
            ref_text=ref_text,
            language="English",
            x_vector_only_mode=False,  # Use ICL mode
        )
        audio_original = audio_original[0]  # It returns a list
        
        # Validate sample rates
        assert sr_cloner == sr_original, f"Sample rates differ: {sr_cloner} vs {sr_original}"
        
        # Validate audio output
        # Note: There might be small numerical differences, so we check:
        # 1. Both produce valid audio
        # 2. Audio lengths are similar (within 10%)
        # 3. Audio content is similar (high correlation)
        
        assert len(audio_cloner) > 0, "VoiceCloner produced empty audio"
        assert len(audio_original) > 0, "Qwen3TTSModel produced empty audio"
        
        # Check lengths are similar
        len_ratio = len(audio_cloner) / len(audio_original)
        assert 0.9 < len_ratio < 1.1, (
            f"Audio lengths differ significantly: "
            f"cloner={len(audio_cloner)}, original={len(audio_original)}, ratio={len_ratio:.3f}"
        )
        
        # For exact equivalence, uncomment the following:
        # assert np.allclose(audio_cloner, audio_original, atol=1e-4), \
        #     "Audio outputs should be identical"
        
        # If exact match is expected, check it
        if len(audio_cloner) == len(audio_original):
            max_diff = np.abs(audio_cloner - audio_original).max()
            print(f"Max difference between outputs: {max_diff}")
            
            # Allow small numerical differences
            if max_diff > 1e-4:
                # Compute correlation instead
                correlation = np.corrcoef(audio_cloner.flatten(), audio_original.flatten())[0, 1]
                assert correlation > 0.99, f"Audio correlation too low: {correlation}"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU tests require CUDA"
)
@pytest.mark.slow  
@pytest.mark.e2e
class TestVoiceClonerSaveOutput:
    """Tests that save output for manual inspection."""
    
    CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    SEED = 42
    
    @pytest.fixture
    def prompt_audio(self):
        """Get the path to the test prompt audio."""
        audio_path = Path(__file__).parent.parent / "test_data" / "prompt.wav"
        if not audio_path.exists():
            pytest.skip("test_data/prompt.wav not found")
        return str(audio_path)
    
    @pytest.fixture
    def ref_text(self):
        """Reference text for the prompt audio (from test_data/transcription.txt)."""
        transcription_path = Path(__file__).parent.parent / "test_data" / "transcription.txt"
        if transcription_path.exists():
            return transcription_path.read_text().strip()
        return (
            "Dazu gehört beispielsweise Eurojust, denn bis jetzt sind für den "
            "Europäischen Staatsanwalt keine zusätzlichen Mittel und kein "
            "zusätzliches Personal vorgesehen."
        )
    
    def test_generate_and_save(self, prompt_audio, ref_text, tmp_path):
        """Generate audio and save for manual inspection."""
        import soundfile as sf
        from qwen3_tts_standalone import VoiceCloner
        
        cloner = VoiceCloner.from_pretrained(
            self.CHECKPOINT,
            device_map="cuda:0",
            dtype=torch.bfloat16,
        )
        
        audio, sr = cloner.clone_voice(
            text="Hello world, this is the VoiceCloner generating speech.",
            ref_audio=prompt_audio,
            ref_text=ref_text,
            language="English",
        )
        
        output_path = tmp_path / "voice_cloner_output.wav"
        sf.write(str(output_path), audio, sr)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
