# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simplified TTS model for Qwen3-TTS.

This is a refactored version of Qwen3TTSForConditionalGenerationStandalone
with improved readability and explicit generation logic.
"""

import json
import os
from typing import Optional

import torch
import torch.nn as nn

from .configuration_qwen3_tts_standalone import Qwen3TTSConfigStandalone
from .speaker_encoder_standalone import (
    Qwen3TTSSpeakerEncoderStandalone,
    mel_spectrogram,
)
from .standalone import cached_file
from .base_model_standalone import StandalonePreTrainedModel
from .utils import download_weights_from_hf_specific
from ...inference.qwen3_tts_tokenizer_standalone import Qwen3TTSTokenizerStandalone
from .talker_standalone import Talker


class TTS(StandalonePreTrainedModel):
    """
    Main TTS model that orchestrates text-to-speech generation.
    
    The TTS model:
    1. Accepts text input (as token IDs)
    2. Optionally uses voice cloning prompts
    3. Uses the Talker to generate codec tokens
    4. The codec tokens can then be decoded to audio using the speech tokenizer
    
    This is a simplified version that makes the generation flow explicit.
    """
    
    config_class = Qwen3TTSConfigStandalone
    
    def __init__(self, config: Qwen3TTSConfigStandalone):
        super().__init__(config)
        self.config = config
        
        # Core talker model
        self.talker = Talker(config.talker_config)
        
        # Speaker encoder (only for base model)
        if config.tts_model_type == "base":
            self.speaker_encoder = Qwen3TTSSpeakerEncoderStandalone(
                config.speaker_encoder_config
            )
        else:
            self.speaker_encoder = None
        
        # Speech tokenizer (loaded separately via from_pretrained)
        self.speech_tokenizer = None
        self.generate_config = None
        
        # Model info
        self.supported_speakers = set(config.talker_config.spk_id.keys())
        self.supported_languages = {"auto"}
        for language_id in config.talker_config.codec_language_id.keys():
            if "dialect" not in language_id:
                self.supported_languages.add(language_id)
        
        self.speaker_encoder_sample_rate = config.speaker_encoder_config.sample_rate
        self.tokenizer_type = config.tokenizer_type
        self.tts_model_size = config.tts_model_size
        self.tts_model_type = config.tts_model_type
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
    
    def load_state_dict(self, state_dict: dict, strict: bool = True, assign: bool = False):
        """
        Load state dict with key remapping for the refactored structure.
        
        The original model has different key names that need to be mapped:
        - talker.code_predictor.model.* -> talker.code_predictor.*
        - talker.code_predictor.model.codec_embedding.* -> talker.code_predictor.codec_embedding.*
        """
        remapped_state_dict = self._remap_state_dict_keys(state_dict)
        return super().load_state_dict(remapped_state_dict, strict=strict, assign=assign)
    
    def _remap_state_dict_keys(self, state_dict: dict) -> dict:
        """
        Remap state dict keys from original model structure to refactored structure.
        
        Key mappings:
        - talker.code_predictor.model.layers.* -> talker.code_predictor.layers.*
        - talker.code_predictor.model.norm.* -> talker.code_predictor.norm.*
        - talker.code_predictor.model.rotary_emb.* -> talker.code_predictor.rotary_emb.*
        - talker.code_predictor.model.codec_embedding.* -> talker.code_predictor.codec_embedding.*
        - talker.code_predictor.model.small_to_mtp_projection.* -> talker.code_predictor.input_projection.*
        - talker.code_predictor.small_to_mtp_projection.* -> talker.code_predictor.input_projection.*
        """
        new_state_dict = {}
        
        for key, value in state_dict.items():
            new_key = key
            
            # Remap code_predictor keys
            if "talker.code_predictor.model." in key:
                # Remove the extra "model." prefix
                new_key = key.replace("talker.code_predictor.model.", "talker.code_predictor.")
            
            # Rename small_to_mtp_projection to input_projection (handle both with and without model. prefix)
            if "small_to_mtp_projection" in new_key:
                new_key = new_key.replace("small_to_mtp_projection", "input_projection")
            
            new_state_dict[new_key] = value
        
        return new_state_dict
    
    def load_speech_tokenizer(self, speech_tokenizer):
        """Load the speech tokenizer for audio encoding/decoding."""
        self.speech_tokenizer = speech_tokenizer
    
    def load_generate_config(self, generate_config):
        """Load generation config from JSON."""
        self.generate_config = generate_config
    
    def get_supported_speakers(self):
        """Get set of supported speaker names."""
        return self.supported_speakers
    
    def get_supported_languages(self):
        """Get set of supported languages."""
        return self.supported_languages
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        use_safetensors=None,
        device_map=None,
        dtype=None,
        **kwargs,
    ):
        """
        Load pretrained TTS model including speech tokenizer and generation config.
        """
        # Load the base model using parent class
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            device_map=device_map,
            dtype=dtype,
            **kwargs,
        )
        
        # Download speech tokenizer if loading from Hub
        if not local_files_only and not os.path.isdir(pretrained_model_name_or_path):
            download_weights_from_hf_specific(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                allow_patterns=["speech_tokenizer/*"],
                revision=revision,
            )
        
        # Load speech tokenizer
        speech_tokenizer_path = cached_file(
            pretrained_model_name_or_path,
            "speech_tokenizer/config.json",
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        if speech_tokenizer_path is None:
            raise ValueError(
                f"Speech tokenizer not found at {pretrained_model_name_or_path}/speech_tokenizer/"
            )
        
        speech_tokenizer_dir = os.path.dirname(speech_tokenizer_path)
        speech_tokenizer = Qwen3TTSTokenizerStandalone.from_pretrained(speech_tokenizer_dir)
        model.load_speech_tokenizer(speech_tokenizer)

        # Load generation config
        generate_config_path = cached_file(
            pretrained_model_name_or_path,
            "generation_config.json",
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
        )
        with open(generate_config_path, "r", encoding="utf-8") as f:
            generate_config = json.load(f)
        model.load_generate_config(generate_config)

        return model
    
    @torch.inference_mode()
    def extract_speaker_embedding(self, audio, sr):
        """
        Extract speaker embedding from audio.
        
        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate (must be 24000)
            
        Returns:
            Speaker embedding tensor
        """
        assert sr == 24000, "Only support 24kHz audio"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0), 
            n_fft=1024, 
            num_mels=128, 
            sampling_rate=24000,
            hop_size=256, 
            win_size=1024, 
            fmin=0, 
            fmax=12000
        ).transpose(1, 2)
        speaker_embedding = self.speaker_encoder(mels.to(self.device).to(self.dtype))[0]
        return speaker_embedding
    
    @torch.inference_mode()
    def generate_speaker_prompt(self, voice_clone_prompt: dict):
        """Extract speaker embeddings from voice clone prompt."""
        voice_clone_spk_embeds = []
        for index in range(len(voice_clone_prompt['ref_spk_embedding'])):
            ref_spk_embedding = voice_clone_prompt["ref_spk_embedding"][index]
            ref_spk_embedding = ref_spk_embedding.to(self.talker.device).to(self.talker.dtype)
            voice_clone_spk_embeds.append(ref_spk_embedding)
        return voice_clone_spk_embeds

    def _build_icl_prompt(
        self,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ):
        """Build in-context learning prompt for voice cloning."""
        # Text embed: (ref id + text id + eos) [1, T1, D]
        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1))
        )
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
        
        # Codec embed: (codec bos + codec) [1, T2, D]
        codec_embed = []
        for i in range(self.talker.num_code_groups):
            if i == 0:
                codec_embed.append(self.talker.get_input_embeddings()(ref_code[:, :1]))
            else:
                codec_embed.append(
                    self.talker.code_predictor.codec_embedding[i-1](ref_code[:, i:i+1])
                )
        codec_embed = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        
        codec_bos = self.talker.get_input_embeddings()(
            torch.tensor(
                [[self.config.talker_config.codec_bos_id]],
                device=self.talker.device,
                dtype=text_id.dtype,
            )
        )
        codec_embed = torch.cat([codec_bos, codec_embed], dim=1)
        
        # Compute lens
        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]
        
        if non_streaming_mode:
            icl_input_embed = text_embed + self.talker.get_input_embeddings()(
                torch.tensor(
                    [[self.config.talker_config.codec_pad_id] * text_lens],
                    device=self.talker.device,
                    dtype=text_id.dtype,
                )
            )
            icl_input_embed = torch.cat([icl_input_embed, codec_embed + tts_pad_embed], dim=1)
            return icl_input_embed, tts_pad_embed
        else:
            if text_lens > codec_lens:
                return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
            else:
                text_embed = torch.cat(
                    [text_embed] + [tts_pad_embed] * (codec_lens - text_lens), 
                    dim=1
                )
                return text_embed + codec_embed, tts_pad_embed

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[list[torch.Tensor]] = None,
        instruct_ids: Optional[list[torch.Tensor]] = None,
        ref_ids: Optional[list[torch.Tensor]] = None,
        voice_clone_prompt: dict = None,
        languages: list[str] = None,
        speakers: list[str] = None,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        **kwargs,
    ):
        """
        Generate speech codec tokens from text.
        
        Args:
            input_ids: List of input token tensors (one per batch item)
            instruct_ids: Optional instruction token tensors
            ref_ids: Optional reference text token tensors (for ICL mode)
            voice_clone_prompt: Voice cloning prompt dict with ref_code, ref_spk_embedding, etc.
            languages: List of language codes
            speakers: List of speaker names (for custom voice model)
            non_streaming_mode: Whether to use non-streaming mode
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            top_k, top_p, temperature: Sampling parameters
            subtalker_*: Sampling parameters for code predictor
            eos_token_id: End of sequence token ID
            repetition_penalty: Repetition penalty
            
        Returns:
            Tuple of (talker_codes_list, talker_hidden_states_list)
        """
        if eos_token_id is None:
            eos_token_id = self.config.talker_config.codec_eos_token_id
        
        # Build suppress tokens list
        suppress_tokens = [
            i for i in range(
                self.config.talker_config.vocab_size - 1024,
                self.config.talker_config.vocab_size
            )
            if i != eos_token_id
        ]
        
        # Prepare voice clone speaker embeddings
        voice_clone_spk_embeds = None
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)
        
        # Build input embeddings for each item in batch
        talker_input_embeds = [[] for _ in range(len(input_ids))]
        trailing_text_hiddens = []
        
        if speakers is None:
            speakers = [None] * len(input_ids)
        
        # Process instruction embeddings
        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    talker_input_embeds[index].append(
                        self.talker.text_projection(
                            self.talker.get_text_embeddings()(instruct_id)
                        )
                    )
        
        # Process each input
        for index, (input_id, language, speaker) in enumerate(
            zip(input_ids, languages, speakers)
        ):
            # Get speaker embedding
            speaker_embed = self._get_speaker_embed(
                speaker, voice_clone_spk_embeds, voice_clone_prompt, index, input_id.dtype
            )
            
            # Get language ID
            language_id = self._get_language_id(language, speaker)
            
            # Get special embeddings
            tts_bos_embed, tts_eos_embed, tts_pad_embed = self._get_special_embeddings(
                input_id.dtype
            )
            
            # Build codec prefill embeddings
            codec_input_embedding = self._build_codec_prefill(
                language_id, speaker_embed, input_id.dtype
            )
            
            # Build role embedding
            role_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(input_id[:, :3])
            )
            
            # Build initial input embedding
            prefill_embed = torch.cat(
                (
                    tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
                    tts_bos_embed,
                ),
                dim=1,
            ) + codec_input_embedding[:, :-1]
            
            talker_input_embed = torch.cat((role_embed, prefill_embed), dim=1)
            
            # Handle voice cloning / ICL mode
            if (voice_clone_prompt is not None 
                and voice_clone_prompt.get("ref_code") is not None 
                and voice_clone_prompt["icl_mode"][index]):
                icl_input_embed, trailing_text_hidden = self._build_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[index][:, 3:-2],
                    ref_code=voice_clone_prompt["ref_code"][index].to(self.talker.device),
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
            else:
                # Add first text token
                first_text_embed = self.talker.text_projection(
                    self.talker.get_text_embeddings()(input_id[:, 3:4])
                ) + codec_input_embedding[:, -1:]
                talker_input_embed = torch.cat([talker_input_embed, first_text_embed], dim=1)
                
                if non_streaming_mode:
                    talker_input_embed = talker_input_embed[:, :-1]
                    remaining_text = self.talker.text_projection(
                        self.talker.get_text_embeddings()(input_id[:, 3:-5])
                    )
                    remaining_text = torch.cat((remaining_text, tts_eos_embed), dim=1)
                    remaining_text = remaining_text + self.talker.get_input_embeddings()(
                        torch.tensor(
                            [[self.config.talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)],
                            device=self.talker.device,
                            dtype=input_id.dtype,
                        )
                    )
                    bos_embed = tts_pad_embed + self.talker.get_input_embeddings()(
                        torch.tensor(
                            [[self.config.talker_config.codec_bos_id]],
                            device=self.talker.device,
                            dtype=input_id.dtype,
                        )
                    )
                    talker_input_embed = torch.cat(
                        [talker_input_embed, remaining_text, bos_embed], 
                        dim=1
                    )
                    trailing_text_hidden = tts_pad_embed
                else:
                    remaining_text = self.talker.text_projection(
                        self.talker.get_text_embeddings()(input_id[:, 4:-5])
                    )
                    trailing_text_hidden = torch.cat((remaining_text, tts_eos_embed), dim=1)
            
            talker_input_embeds[index].append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)
        
        # Concatenate all embeddings for each batch item
        for index, talker_input_embed in enumerate(talker_input_embeds):
            talker_input_embeds[index] = torch.cat(
                [item for item in talker_input_embed if item is not None], 
                dim=1
            )
        
        # Pad and create attention mask (left padding)
        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
        sequences = [t.squeeze(0) for t in talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(
            sequences_reversed,
            batch_first=True,
            padding_value=0.0
        )
        talker_input_embeds = padded_reversed.flip(dims=[1])
        
        # Generate attention mask
        batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
        indices = torch.arange(max_len).expand(batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)
        
        # Pad trailing text hiddens
        pad_embedding_vector = tts_pad_embed.squeeze()
        sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
        trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
        padded_hiddens = torch.nn.utils.rnn.pad_sequence(
            sequences_to_pad,
            batch_first=True,
            padding_value=0.0
        )
        arange_tensor = torch.arange(
            max(trailing_text_original_lengths), 
            device=padded_hiddens.device
        ).expand(len(trailing_text_original_lengths), -1)
        lengths_tensor = torch.tensor(
            trailing_text_original_lengths, 
            device=padded_hiddens.device
        ).unsqueeze(1)
        padding_mask = arange_tensor >= lengths_tensor
        padded_hiddens[padding_mask] = pad_embedding_vector
        trailing_text_hiddens = padded_hiddens
        
        # Generate using talker
        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embeds,
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            suppress_tokens=suppress_tokens,
            subtalker_dosample=subtalker_dosample,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
            output_hidden_states=True,
        )
        
        # Process results
        # Stack all codec IDs: shape [batch, seq_len, num_code_groups]
        if talker_result.all_codec_ids:
            talker_codes = torch.stack(talker_result.all_codec_ids, dim=1)
        else:
            # Handle case with no generation steps
            talker_codes = torch.zeros(
                (batch_size, 0, self.talker.num_code_groups),
                device=self.device,
                dtype=torch.long,
            )
        
        # Get hidden states
        if talker_result.hidden_states:
            talker_hidden_states = torch.cat(
                [h[0] for h in talker_result.hidden_states if h[0] is not None],
                dim=1
            )[:, :-1]
        else:
            talker_hidden_states = torch.zeros(
                (batch_size, 0, self.config.talker_config.hidden_size),
                device=self.device,
                dtype=self.dtype,
            )
        
        # Find EOS positions and truncate
        first_codebook = talker_codes[:, :, 0] if talker_codes.shape[1] > 0 else talker_codes
        is_stop_token = (first_codebook == eos_token_id)
        stop_indices = torch.argmax(is_stop_token.int(), dim=1)
        has_stop_token = is_stop_token.any(dim=1)
        effective_lengths = torch.where(has_stop_token, stop_indices, talker_codes.shape[1])
        
        # Create per-sample lists
        talker_codes_list = [
            talker_codes[i, :length] for i, length in enumerate(effective_lengths)
        ]
        talker_hidden_states_list = [
            talker_hidden_states[i, :length] for i, length in enumerate(effective_lengths)
        ]
        
        return talker_codes_list, talker_hidden_states_list
    
    def _get_speaker_embed(
        self, 
        speaker: Optional[str], 
        voice_clone_spk_embeds: Optional[list],
        voice_clone_prompt: Optional[dict],
        index: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Get speaker embedding from either voice clone or speaker name."""
        if voice_clone_spk_embeds is None:
            if speaker == "" or speaker is None:
                return None
            if speaker.lower() not in self.config.talker_config.spk_id:
                raise NotImplementedError(f"Speaker {speaker} not implemented")
            spk_id = self.config.talker_config.spk_id[speaker.lower()]
            return self.talker.get_input_embeddings()(
                torch.tensor(spk_id, device=self.talker.device, dtype=dtype)
            )
        else:
            if (voice_clone_prompt["x_vector_only_mode"][index] 
                or voice_clone_prompt["icl_mode"][index]):
                return voice_clone_spk_embeds[index]
            return None
    
    def _get_language_id(
        self, 
        language: str, 
        speaker: Optional[str]
    ) -> Optional[int]:
        """Get language ID from language string."""
        if language is None:
            raise ValueError("Language must be specified")
        
        if language.lower() == "auto":
            language_id = None
        elif language.lower() not in self.config.talker_config.codec_language_id:
            raise NotImplementedError(f"Language {language} not implemented")
        else:
            language_id = self.config.talker_config.codec_language_id[language.lower()]
        
        # Handle dialect
        if (language.lower() in ["chinese", "auto"] 
            and speaker not in ("", None)
            and self.config.talker_config.spk_is_dialect.get(speaker.lower())):
            dialect = self.config.talker_config.spk_is_dialect[speaker.lower()]
            language_id = self.config.talker_config.codec_language_id[dialect]
        
        return language_id
    
    def _get_special_embeddings(self, dtype: torch.dtype):
        """Get TTS BOS, EOS, and PAD embeddings."""
        special_ids = torch.tensor(
            [[
                self.config.tts_bos_token_id,
                self.config.tts_eos_token_id,
                self.config.tts_pad_token_id,
            ]],
            device=self.talker.device,
            dtype=dtype,
        )
        special_embeds = self.talker.text_projection(
            self.talker.get_text_embeddings()(special_ids)
        )
        return special_embeds.chunk(3, dim=1)  # (bos, eos, pad)
    
    def _build_codec_prefill(
        self,
        language_id: Optional[int],
        speaker_embed: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build codec prefill embeddings with language and speaker info."""
        tc = self.config.talker_config
        
        if language_id is None:
            codec_prefill_list = [[
                tc.codec_nothink_id,
                tc.codec_think_bos_id,
                tc.codec_think_eos_id,
            ]]
        else:
            codec_prefill_list = [[
                tc.codec_think_id,
                tc.codec_think_bos_id,
                language_id,
                tc.codec_think_eos_id,
            ]]
        
        codec_embed_0 = self.talker.get_input_embeddings()(
            torch.tensor(codec_prefill_list, device=self.talker.device, dtype=dtype)
        )
        codec_embed_1 = self.talker.get_input_embeddings()(
            torch.tensor(
                [[tc.codec_pad_id, tc.codec_bos_id]],
                device=self.talker.device,
                dtype=dtype,
            )
        )
        
        if speaker_embed is None:
            return torch.cat([codec_embed_0, codec_embed_1], dim=1)
        else:
            return torch.cat([
                codec_embed_0,
                speaker_embed.view(1, 1, -1),
                codec_embed_1,
            ], dim=1)
