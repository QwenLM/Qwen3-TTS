# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simplified Voice Cloner for Qwen3-TTS using In-Context Learning (ICL).

This module provides a standalone, easy-to-understand implementation of voice
cloning using the ICL approach. It combines the model loading and inference
logic into a single class.

Usage:
    from qwen3_tts_standalone import VoiceCloner
    
    cloner = VoiceCloner.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    audio, sr = cloner.clone_voice(
        text="Hello, this is a test.",
        ref_audio="reference.wav",
        ref_text="This is the reference transcript.",
        language="English",
    )
"""

import io
import json
import os
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .base_model import BaseModel
from .configuration import TTSConfig
from .processor import Processor
from .speaker_encoder import SpeakerEncoder, mel_spectrogram
from .talker import Talker
from .tokenizer import SpeechTokenizer
from .utils import cached_file, download_weights_from_hf


class VoiceCloner(nn.Module):
    """
    A simplified voice cloning class using In-Context Learning (ICL).
    
    This class provides a clean, focused API for voice cloning that:
    - Loads a pretrained Qwen3-TTS Base model
    - Extracts speaker embeddings from reference audio
    - Generates speech that mimics the reference voice
    
    The ICL approach works by:
    1. Encoding the reference audio into codec tokens
    2. Extracting a speaker embedding (x-vector) from the reference
    3. Using both as context when generating new speech
    """
    
    def __init__(
        self,
        config: TTSConfig,
        talker: Talker,
        speaker_encoder: SpeakerEncoder,
        speech_tokenizer: SpeechTokenizer,
        processor: Processor,
        generate_config: dict,
    ):
        """
        Initialize the VoiceCloner.
        
        Note: Use `VoiceCloner.from_pretrained()` instead of this constructor.
        
        Args:
            config: TTS configuration
            talker: The talker model for generating codec tokens
            speaker_encoder: Model for extracting speaker embeddings
            speech_tokenizer: Audio codec for encoding/decoding audio
            processor: Text tokenizer
            generate_config: Generation parameters
        """
        super().__init__()
        self.config = config
        self.talker = talker
        self.speaker_encoder = speaker_encoder
        self.speech_tokenizer = speech_tokenizer
        self.processor = processor
        self.generate_config = generate_config
        
        # Store useful config values
        self._speaker_encoder_sample_rate = config.speaker_encoder_config.sample_rate
    
    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the model parameters."""
        return next(self.parameters()).dtype
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device_map: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs,
    ) -> "VoiceCloner":
        """
        Load a pretrained VoiceCloner from HuggingFace Hub or local path.
        
        Args:
            pretrained_model_name_or_path: HuggingFace repo id or local directory
            device_map: Device to load the model on (e.g., "cuda:0", "cpu")
            dtype: Data type for model weights (e.g., torch.bfloat16)
            cache_dir: Directory to cache downloaded files
            token: HuggingFace token for private repos
            
        Returns:
            A VoiceCloner instance ready for voice cloning
            
        Example:
            >>> cloner = VoiceCloner.from_pretrained(
            ...     "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            ...     device_map="cuda:0",
            ...     dtype=torch.bfloat16,
            ... )
        """
        # Load configuration
        config = cls._load_config(pretrained_model_name_or_path, cache_dir, token)
        
        # Verify this is a base model (required for ICL voice cloning)
        if config.tts_model_type != "base":
            raise ValueError(
                f"VoiceCloner requires a 'base' model type for ICL voice cloning. "
                f"Got: {config.tts_model_type}. "
                f"Please use Qwen/Qwen3-TTS-12Hz-1.7B-Base or similar."
            )
        
        # Create component models
        talker = Talker(config.talker_config)
        speaker_encoder = SpeakerEncoder(config.speaker_encoder_config)
        
        # Load model weights
        cls._load_weights(
            pretrained_model_name_or_path,
            talker,
            speaker_encoder,
            config,
            cache_dir,
            token,
            device_map,
            dtype,
        )
        
        # Load speech tokenizer
        speech_tokenizer = cls._load_speech_tokenizer(
            pretrained_model_name_or_path, cache_dir, token
        )
        
        # Load processor (text tokenizer)
        processor = Processor.from_pretrained(pretrained_model_name_or_path)
        
        # Load generation config
        generate_config = cls._load_generate_config(
            pretrained_model_name_or_path, cache_dir, token
        )
        
        # Create instance
        model = cls(
            config=config,
            talker=talker,
            speaker_encoder=speaker_encoder,
            speech_tokenizer=speech_tokenizer,
            processor=processor,
            generate_config=generate_config,
        )
        
        # Move to device and set dtype
        if device_map is not None:
            model = model.to(device_map)
        if dtype is not None:
            model = model.to(dtype=dtype)
        
        model.eval()
        return model
    
    @classmethod
    def _load_config(
        cls, path: str, cache_dir: Optional[str], token: Optional[str]
    ) -> TTSConfig:
        """Load configuration from local path or HuggingFace Hub."""
        if os.path.isdir(path):
            config_path = os.path.join(path, "config.json")
        else:
            config_path = hf_hub_download(
                repo_id=path, filename="config.json", cache_dir=cache_dir, token=token
            )
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        return TTSConfig.from_dict(config_dict)
    
    @classmethod
    def _load_weights(
        cls,
        path: str,
        talker: Talker,
        speaker_encoder: SpeakerEncoder,
        config: TTSConfig,
        cache_dir: Optional[str],
        token: Optional[str],
        device_map: Optional[str],
        dtype: Optional[torch.dtype],
    ):
        """Load model weights from checkpoint files."""
        from glob import glob
        from huggingface_hub import snapshot_download
        
        # Get local path
        if os.path.isdir(path):
            model_path = path
        else:
            model_path = snapshot_download(
                repo_id=path, cache_dir=cache_dir, token=token
            )
        
        # Find checkpoint files
        safetensors_files = sorted(glob(os.path.join(model_path, "*.safetensors")))
        
        if safetensors_files:
            from safetensors.torch import load_file
            state_dict = {}
            for f in safetensors_files:
                state_dict.update(load_file(f, device="cpu"))
        else:
            pytorch_files = sorted(glob(os.path.join(model_path, "*.bin")))
            pytorch_files = [f for f in pytorch_files if "optimizer" not in f.lower()]
            state_dict = {}
            for f in pytorch_files:
                state_dict.update(torch.load(f, map_location="cpu", weights_only=True))
        
        # Remap and load talker weights
        talker_state_dict = cls._extract_and_remap_talker_weights(state_dict)
        talker.load_state_dict(talker_state_dict, strict=False)
        
        # Load speaker encoder weights
        speaker_encoder_state_dict = {
            k.replace("speaker_encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("speaker_encoder.")
        }
        speaker_encoder.load_state_dict(speaker_encoder_state_dict, strict=False)
    
    @classmethod
    def _extract_and_remap_talker_weights(cls, state_dict: dict) -> dict:
        """Extract and remap talker weights from full state dict."""
        new_state_dict = {}
        
        for key, value in state_dict.items():
            if not key.startswith("talker."):
                continue
            
            new_key = key.replace("talker.", "")
            
            # Remap code_predictor keys
            if "code_predictor.model." in new_key:
                new_key = new_key.replace("code_predictor.model.", "code_predictor.")
            
            # Rename projection layer
            if "small_to_mtp_projection" in new_key:
                new_key = new_key.replace("small_to_mtp_projection", "input_projection")
            
            new_state_dict[new_key] = value
        
        return new_state_dict
    
    @classmethod
    def _load_speech_tokenizer(
        cls, path: str, cache_dir: Optional[str], token: Optional[str]
    ) -> SpeechTokenizer:
        """Load the speech tokenizer (audio codec)."""
        # Ensure speech tokenizer files are downloaded
        if not os.path.isdir(path):
            download_weights_from_hf(
                path, cache_dir=cache_dir, allow_patterns=["speech_tokenizer/*"]
            )
        
        tokenizer_config_path = cached_file(
            path, "speech_tokenizer/config.json", cache_dir=cache_dir, token=token
        )
        tokenizer_dir = os.path.dirname(tokenizer_config_path)
        return SpeechTokenizer.from_pretrained(tokenizer_dir)
    
    @classmethod
    def _load_generate_config(
        cls, path: str, cache_dir: Optional[str], token: Optional[str]
    ) -> dict:
        """Load generation configuration."""
        config_path = cached_file(
            path, "generation_config.json", cache_dir=cache_dir, token=token
        )
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio from file path, URL, or base64 string."""
        if audio_path.startswith(("http://", "https://")):
            import urllib.request
            with urllib.request.urlopen(audio_path) as resp:
                audio_bytes = resp.read()
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f, dtype="float32")
        else:
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Ensure mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        
        return audio.astype(np.float32), int(sr)
    
    @torch.inference_mode()
    def _extract_speaker_embedding(
        self, audio: np.ndarray, sr: int
    ) -> torch.Tensor:
        """
        Extract speaker embedding from audio using the speaker encoder.
        
        Args:
            audio: Audio waveform as numpy array (float32)
            sr: Sample rate of the audio
            
        Returns:
            Speaker embedding tensor
        """
        # Resample to speaker encoder's expected sample rate (24kHz)
        if sr != self._speaker_encoder_sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self._speaker_encoder_sample_rate,
            )
            sr = self._speaker_encoder_sample_rate
        
        # Compute mel spectrogram
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        mels = mel_spectrogram(
            audio_tensor,
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        
        # Extract embedding
        mels = mels.to(self.device).to(self.dtype)
        speaker_embedding = self.speaker_encoder(mels)[0]
        
        return speaker_embedding
    
    @torch.inference_mode()
    def _encode_reference_audio(
        self, audio: np.ndarray, sr: int
    ) -> torch.Tensor:
        """
        Encode reference audio into codec tokens.
        
        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate
            
        Returns:
            Codec tokens tensor [seq_len, num_codebooks]
        """
        result = self.speech_tokenizer.encode([audio], sr=sr)
        return result.audio_codes[0]  # [seq_len, num_codebooks] or [seq_len]
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text into input IDs."""
        formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        result = self.processor(text=formatted, return_tensors="pt", padding=True)
        return result["input_ids"].to(self.device)
    
    def _tokenize_ref_text(self, text: str) -> torch.Tensor:
        """Tokenize reference text into input IDs."""
        formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n"
        result = self.processor(text=formatted, return_tensors="pt", padding=True)
        return result["input_ids"].to(self.device)
    
    def _build_icl_prompt(
        self,
        text_ids: torch.Tensor,
        ref_ids: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the in-context learning prompt for voice cloning.
        
        This combines:
        - Reference text embeddings
        - Target text embeddings  
        - Reference audio codec embeddings
        
        The model learns the voice characteristics from the reference and
        applies them when generating speech for the target text.
        
        Args:
            text_ids: Target text token IDs
            ref_ids: Reference text token IDs
            ref_code: Reference audio codec tokens
            tts_pad_embed: Padding embedding
            tts_eos_embed: End-of-sequence embedding
            non_streaming_mode: If True, uses non-streaming prompt format.
                If False (default), uses streaming format.
        
        Returns:
            Tuple of (icl_input_embed, trailing_text_hidden)
        """
        # Combine ref and target text, add EOS 
        combined_text_ids = torch.cat([ref_ids, text_ids], dim=-1)
        text_embed = self.talker.text_projection(
            self.talker.text_embedding(combined_text_ids)
        )
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1) # shape [1, t_combined_text + 1, hidden_size]
        
        # Build codec embeddings from reference audio
        codec_embed_list = []
        num_codebooks = self.talker.num_code_groups
        
        for i in range(num_codebooks):
            if i == 0:
                # First codebook uses talker's main codec embedding
                codec_embed_list.append(
                    self.talker.codec_embedding(ref_code[:, :1])
                )
            else:
                # Higher codebooks use code predictor's embeddings
                codec_embed_list.append(
                    self.talker.code_predictor.codec_embedding[i - 1](
                        ref_code[:, i : i + 1]
                    )
                )
        
        # Sum all codebook embeddings
        codec_embed = torch.cat(codec_embed_list, dim=1).sum(1).unsqueeze(0) # shape [1, t_audio, hidden_size]
        
        # Add codec BOS token
        codec_bos = self.talker.codec_embedding(
            torch.tensor(
                [[self.config.talker_config.codec_bos_id]],
                device=self.device,
                dtype=text_ids.dtype,
            )
        ) # shape [1, 1, hidden_size]
        codec_embed = torch.cat([codec_bos, codec_embed], dim=1) # shape [1, t_audio + 1, hidden_size]
        
        # Compute lengths
        text_len = text_embed.shape[1]
        codec_len = codec_embed.shape[1]
        
        if non_streaming_mode:
            # Non-streaming: text + pad codec, then codec + pad text
            # [Text, ..., Text, Pad, ..., Pad]
            # [Pad, ..., Pad, Audio, ..., Audio]
            # both are summed -> icl_input
            text_with_pad = text_embed + self.talker.codec_embedding(
                torch.tensor(
                    [[self.config.talker_config.codec_pad_id] * text_len],
                    device=self.device,
                    dtype=text_ids.dtype,
                )
            ) # shape [1, t_text, hidden_size] Here t_text is something new
            codec_with_pad = codec_embed + tts_pad_embed # shape [1, t_audio, hidden_size] Here t_audio is something new
            icl_input = torch.cat([text_with_pad, codec_with_pad], dim=1) # shape [1, t_text + t_audio, hidden_size] 
            return icl_input, tts_pad_embed
        else:
            # Streaming mode: interleave text and codec
            # [Audio, Audio, Audio, ..., Audio]
            # [Text, Text, Text, Pad, ..., Pad]
            # both are summed -> icl_input
            if text_len > codec_len:
                # Text is longer: use codec_len of text, rest is trailing
                icl_input = text_embed[:, :codec_len] + codec_embed
                trailing_text = text_embed[:, codec_len:]
            else:
                # Codec is longer or equal: pad text with tts_pad_embed
                text_padded = torch.cat(
                    [text_embed] + [tts_pad_embed] * (codec_len - text_len),
                    dim=1,
                ) # shape [1, t_audio, hidden_size]
                icl_input = text_padded + codec_embed
                trailing_text = tts_pad_embed # shape [1, 1, hidden_size]
            return icl_input, trailing_text
    
    def _get_special_embeddings(
        self, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get TTS special token embeddings (BOS, EOS, PAD).
        
        These are just learned embedding stored added to the text embedding.
        """
        special_ids = torch.tensor(
            [[
                self.config.tts_bos_token_id,
                self.config.tts_eos_token_id,
                self.config.tts_pad_token_id,
            ]],
            device=self.device,
            dtype=dtype,
        )
        # note that the text embeddings are projected with a small 2 layers MLP
        special_embeds = self.talker.text_projection(
            self.talker.text_embedding(special_ids)
        )
        bos, eos, pad = special_embeds.chunk(3, dim=1)
        return bos, eos, pad
    
    def _build_codec_prefill(
        self,
        language_id: Optional[int],
        speaker_embed: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build codec prefill embeddings with language and speaker info.
        
        This build the prefill embeddings of shape [1, 6 or 7, hidden_size]
        For the size 6 we have
        [embedd_nothink, embedd_think_bos, embedd_think_eos, embedd_speaker, embedd_pad, embedd_bos]
        For the size 7 we have
        [embedd_think, embedd_think_bos, embedd_language, embedd_think_eos, embedd_speaker, embedd_pad, embedd_bos]
        """
        tc = self.config.talker_config
        
        if language_id is None:
            prefill_list = [[tc.codec_nothink_id, tc.codec_think_bos_id, tc.codec_think_eos_id]]
        else:
            prefill_list = [[tc.codec_think_id, tc.codec_think_bos_id, language_id, tc.codec_think_eos_id]]
        
        codec_embed_0 = self.talker.codec_embedding(
            torch.tensor(prefill_list, device=self.device, dtype=dtype)
        ) # Shape [1, 3 or 4, hidden_size]
        codec_embed_1 = self.talker.codec_embedding(
            torch.tensor(
                [[tc.codec_pad_id, tc.codec_bos_id]],
                device=self.device,
                dtype=dtype,
            )
        ) # Shape [1, 2, hidden_size]
        
        return torch.cat([
            codec_embed_0,
            speaker_embed.view(1, 1, -1), # this add one tokens
            codec_embed_1,
        ], dim=1) # Shape [1, 6 or 7, hidden_size]
    
    def _get_language_id(self, language: str) -> Optional[int]:
        """Get language ID from language name."""
        language_lower = language.lower()
        
        if language_lower == "auto":
            return None
        
        lang_map = self.config.talker_config.codec_language_id
        if lang_map and language_lower in lang_map:
            return lang_map[language_lower]
        
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported: {list(lang_map.keys()) if lang_map else ['auto']}"
        )
    
    @torch.inference_mode()
    def clone_voice(
        self,
        text: str,
        ref_audio: Union[str, np.ndarray, Tuple[np.ndarray, int]],
        ref_text: str,
        language: str = "auto",
        # Generation parameters
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        non_streaming_mode: bool = False,
    ) -> Tuple[np.ndarray, int]:
        """
        Clone a voice and generate speech for the given text.
        
        This method:
        1. Extracts speaker characteristics from the reference audio
        2. Encodes the reference audio into codec tokens
        3. Uses ICL to generate speech that sounds like the reference voice
        
        Args:
            text: The text to synthesize
            ref_audio: Reference audio - can be:
                - str: Path to audio file or URL
                - np.ndarray: Audio waveform (requires ref_sr in tuple)
                - Tuple[np.ndarray, int]: (waveform, sample_rate)
            ref_text: Transcript of the reference audio (required for ICL)
            language: Target language (default: "auto")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeated tokens
            non_streaming_mode: If True, uses non-streaming text input mode.
                Default False simulates streaming text input (recommended).
            
        Returns:
            Tuple of (audio_waveform, sample_rate)
            
        Example:
            >>> cloner = VoiceCloner.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
            >>> audio, sr = cloner.clone_voice(
            ...     text="Hello, how are you today?",
            ...     ref_audio="speaker_sample.wav",
            ...     ref_text="This is a sample of my voice.",
            ...     language="English",
            ... )
            >>> sf.write("output.wav", audio, sr)
        """
        # Load and process reference audio
        if isinstance(ref_audio, str):
            ref_wav, ref_sr = self._load_audio(ref_audio)
        elif isinstance(ref_audio, tuple):
            ref_wav, ref_sr = ref_audio
            ref_wav = ref_wav.astype(np.float32)
        else:
            raise ValueError(
                "ref_audio must be a path string or (waveform, sample_rate) tuple"
            )
        
        # Extract speaker embedding
        speaker_embed = self._extract_speaker_embedding(ref_wav, ref_sr) # Shape [hidden_size]
        
        # Encode reference audio into codec tokens
        ref_code = self._encode_reference_audio(ref_wav, ref_sr) # Shape [t_audio, num_codebooks]
        ref_code = ref_code.to(self.device)
        
        # Tokenize texts
        text_ids = self._tokenize_text(text) # Shape [1, t_text]
        ref_ids = self._tokenize_ref_text(ref_text) # Shape [1, t_ref_text]
        
        # Get language ID
        language_id = self._get_language_id(language) # int
        
        # Get special embeddings
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._get_special_embeddings(
            text_ids.dtype
        ) # Shapes [1, 1, hidden_size], [1, 1, hidden_size], [1, 1, hidden_size]
        
        # Build codec prefill with speaker embedding
        codec_input = self._build_codec_prefill(language_id, speaker_embed, text_ids.dtype)
        # Shape [1, 6 or 7, hidden_size]
        # [embedd_think, embedd_think_bos, embedd_language (optional), embedd_think_eos, embedd_speaker, embedd_pad, embedd_bos]
        
        # Build role embedding (first 3 tokens: <|im_start|>, assistant, \n)
        # This is the ChatML-style role prefix for the assistant turn
        role_embed = self.talker.text_projection(
            self.talker.text_embedding(text_ids[:, :3])
        )  # Shape [1, 3, hidden_size]
        
        # Build initial input: tts_pad * (N-2) + tts_bos, then add codec embeddings
        # The -2 accounts for: 1) tts_bos added separately, 2) codec_bos excluded via [:, :-1]
        # This aligns text embeddings with codec embeddings for element-wise addition:
        #   text:  [pad, pad, pad, pad, bos]  (5 elements for size 6 codec_input)
        #   codec: [nothink/think, think_bos, (lang), think_eos, speaker, pad] (excludes final bos)
        prefill_embed = torch.cat(
            (
                tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1),
                tts_bos_embed,
            ),
            dim=1,
        ) + codec_input[:, :-1] # Shape [1, 5 or 6, hidden_size]
        
        talker_input = torch.cat((role_embed, prefill_embed), dim=1) # shape [1, 8 or 9, hidden_size]
        
        # Build ICL prompt
        icl_input, trailing_text = self._build_icl_prompt(
            text_ids=text_ids[:, 3:-5],  # Remove special tokens
            ref_ids=ref_ids[:, 3:-2],    # Remove special tokens
            ref_code=ref_code,
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=non_streaming_mode,
        ) # shape [1, t_text + t_audio, hidden_size] (or something like that...)
        
        talker_input = torch.cat([talker_input, icl_input], dim=1)
        
        # Create attention mask
        batch_size = talker_input.shape[0]
        seq_len = talker_input.shape[1]
        attention_mask = torch.ones((batch_size, seq_len), device=self.device)
        
        # Build suppress tokens list (will have 0 probability of being generated)
        eos_token_id = self.config.talker_config.codec_eos_token_id
        suppress_tokens = [
            i for i in range(
                self.config.talker_config.vocab_size - 1024,
                self.config.talker_config.vocab_size,
            )
            if i != eos_token_id
        ]
        
        # Generate codec tokens
        talker_result = self.talker.generate(
            inputs_embeds=talker_input,
            attention_mask=attention_mask,
            trailing_text_hidden=trailing_text,
            tts_pad_embed=tts_pad_embed,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            suppress_tokens=suppress_tokens,
            subtalker_dosample=True,
            subtalker_top_k=top_k,
            subtalker_top_p=top_p,
            subtalker_temperature=temperature,
            output_hidden_states=False,
        )
        
        # Stack codec tokens
        if talker_result.all_codec_ids:
            talker_codes = torch.stack(talker_result.all_codec_ids, dim=1)
        else:
            talker_codes = torch.zeros(
                (batch_size, 0, self.talker.num_code_groups),
                device=self.device,
                dtype=torch.long,
            )
        
        # Find EOS and truncate
        first_codebook = talker_codes[:, :, 0] if talker_codes.shape[1] > 0 else talker_codes
        is_stop = first_codebook == eos_token_id
        stop_idx = torch.argmax(is_stop.int(), dim=1)
        has_stop = is_stop.any(dim=1)
        eff_len = torch.where(has_stop, stop_idx, talker_codes.shape[1])
        
        # Get codes for this sample
        codes = talker_codes[0, :eff_len[0]]
        
        # Prepend reference codes and decode
        full_codes = torch.cat([ref_code, codes], dim=0)
        wavs, sr = self.speech_tokenizer.decode([{"audio_codes": full_codes}])
        
        # Trim reference audio from output
        ref_len = ref_code.shape[0]
        total_len = full_codes.shape[0]
        cut_samples = int(ref_len / max(total_len, 1) * wavs[0].shape[0])
        
        output_audio = wavs[0][cut_samples:]
        
        return output_audio, sr


__all__ = ["VoiceCloner"]
