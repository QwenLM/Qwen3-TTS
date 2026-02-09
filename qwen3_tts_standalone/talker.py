# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simplified Talker model for Qwen3-TTS.

This is a refactored version of Qwen3TTSTalkerForConditionalGenerationStandalone
with improved readability and explicit generation logic.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .code_predictor import CodePredictor
from .configuration import TalkerConfig
from .transformer import TalkerModel
from .layers import ResizeMLP
from .utils import (
    DynamicCache,
    sample_top_k_top_p,
)


@dataclass
class TalkerOutput:
    """Output from a single Talker forward pass."""
    logits: torch.Tensor  # [batch_size, 1, vocab_size]
    hidden_state: torch.Tensor  # [batch_size, 1, hidden_size] - for code predictor
    codec_ids: Optional[torch.Tensor]  # [batch_size, num_code_groups] - all codebook IDs
    past_key_values: Optional[DynamicCache]


@dataclass
class TalkerGenerateOutput:
    """Output from Talker.generate()."""
    sequences: torch.Tensor  # [batch_size, seq_len] - first codebook tokens
    all_codec_ids: list  # List of [batch_size, num_code_groups] tensors per step
    hidden_states: Optional[tuple]  # Hidden states per step if requested


class Talker(nn.Module):
    """
    Talker model that generates audio codec tokens from text embeddings.
    
    The Talker:
    1. Takes text embeddings (from a text encoder)
    2. Generates the first codebook tokens autoregressively
    3. Uses a CodePredictor to generate higher codebook tokens for each step
    
    This is a simplified version that makes the generation loop explicit
    rather than using GenerationMixin.
    """
    
    def __init__(self, config: TalkerConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_code_groups = config.num_code_groups
        
        # Core transformer model
        self.model = TalkerModel(config)
        
        # Text projection: maps text hidden size to talker hidden size
        self.text_projection = ResizeMLP(
            config.text_hidden_size,
            config.text_hidden_size,
            config.hidden_size,
            config.hidden_act,
            bias=True,
        )
        
        # Codec head: predicts first codebook token
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Code predictor: generates higher codebook tokens
        self.code_predictor = CodePredictor(
            config=config.code_predictor_config,
            embedding_dim=config.hidden_size,
        )
        
        # For rope position tracking
        self.rope_deltas = None
    
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
    
    def get_input_embeddings(self):
        """Return codec embedding layer."""
        return self.model.codec_embedding
    
    def get_text_embeddings(self):
        """Return text embedding layer."""
        return self.model.text_embedding
    
    def _compute_rope_position_ids(
        self,
        attention_mask: torch.Tensor,
        cache_position: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        seq_length: int = 1,
    ) -> torch.Tensor:
        """Compute 3D rope position IDs for the talker model."""
        if attention_mask is not None:
            if cache_position is None or cache_position[0] == 0 or self.rope_deltas is None:
                # Initial position computation
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids = attention_mask.float().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                rope_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
                self.rope_deltas = rope_deltas - delta0
                return position_ids
            else:
                # Incremental position computation
                delta = cache_position[0] + self.rope_deltas
                position_ids = torch.arange(seq_length, device=attention_mask.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                return position_ids
        else:
            return None
    
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        past_hidden: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        # Code predictor sampling params
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        # Generation tracking
        is_prefill: bool = False,
        last_predicted_token: Optional[torch.Tensor] = None,
        # Text embeddings for generation
        trailing_text_hidden: Optional[torch.Tensor] = None,
        tts_pad_embed: Optional[torch.Tensor] = None,
        generation_step: int = 0,
    ) -> TalkerOutput:
        """
        Single forward pass of the Talker.
        
        Args:
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            past_key_values: KV cache from previous steps
            past_hidden: Hidden state from previous step [batch_size, 1, hidden_size]
            cache_position: Current position in cache
            use_cache: Whether to use KV caching
            subtalker_*: Sampling parameters for the code predictor
            is_prefill: Whether this is the prefill phase
            last_predicted_token: The first codebook token from the previous step
            trailing_text_hidden: Text embeddings to add during generation
            tts_pad_embed: Padding embedding when text is exhausted
            generation_step: Current generation step (for text indexing)
            
        Returns:
            TalkerOutput with logits, hidden states, codec IDs, and updated cache
        """
        batch_size = inputs_embeds.shape[0]
        seq_length = inputs_embeds.shape[1]
        
        # Compute position IDs
        position_ids = self._compute_rope_position_ids(
            attention_mask, cache_position, batch_size, seq_length
        )
        
        # During generation (not prefill), predict higher codebooks and compute input embeddings
        codec_ids = None
        if not is_prefill and past_hidden is not None and last_predicted_token is not None:
            # Get embedding of the last predicted first-codebook token
            last_id_hidden = self.get_input_embeddings()(last_predicted_token)
            
            # Use code predictor to generate higher codebook tokens
            predictor_input = torch.cat((past_hidden, last_id_hidden), dim=1)
            predictor_result = self.code_predictor.generate(
                inputs_embeds=predictor_input,
                max_new_tokens=self.num_code_groups - 1,
                do_sample=subtalker_dosample,
                top_k=subtalker_top_k,
                top_p=subtalker_top_p,
                temperature=subtalker_temperature,
            )
            
            # Combine all codebook tokens: [first_codebook, higher_codebooks...]
            codec_ids = torch.cat((last_predicted_token, predictor_result.sequences), dim=-1)
            
            # Sum embeddings from all codebooks for the input
            codec_hiddens = [last_id_hidden]
            for i in range(self.num_code_groups - 1):
                codec_embed = self.code_predictor.codec_embedding[i](
                    predictor_result.sequences[..., i:i+1]
                )
                codec_hiddens.append(codec_embed)
            
            # Sum all codebook embeddings
            inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
            
            # Add trailing text embeddings
            if trailing_text_hidden is not None:
                if generation_step < trailing_text_hidden.shape[1]:
                    inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step:generation_step+1, :]
                elif tts_pad_embed is not None:
                    inputs_embeds = inputs_embeds + tts_pad_embed
        
        # Forward through transformer
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)
        
        return TalkerOutput(
            logits=logits,
            hidden_state=hidden_states[:, -1:, :],
            codec_ids=codec_ids,
            past_key_values=outputs.past_key_values,
        )
    
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        trailing_text_hidden: Optional[torch.Tensor] = None,
        tts_pad_embed: Optional[torch.Tensor] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        eos_token_id: Optional[int] = None,
        suppress_tokens: Optional[list] = None,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> TalkerGenerateOutput:
        """
        Generate codec tokens autoregressively.
        
        Args:
            inputs_embeds: Initial input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            trailing_text_hidden: Text embeddings to add during generation [batch_size, text_len, hidden_size]
            tts_pad_embed: Padding embedding to use when text is exhausted [1, 1, hidden_size]
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens before EOS can stop generation
            do_sample: Whether to sample or use greedy decoding
            top_k, top_p, temperature: Sampling parameters
            repetition_penalty: Penalty for repeated tokens
            eos_token_id: Token ID that signals end of generation
            suppress_tokens: Token IDs to suppress during generation
            subtalker_*: Sampling parameters for the code predictor
            output_hidden_states: Whether to return hidden states
            
        Returns:
            TalkerGenerateOutput with generated sequences and all codec IDs
        """
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        
        if eos_token_id is None:
            eos_token_id = self.config.codec_eos_token_id
        
        # Initialize cache
        past_key_values = DynamicCache()
        
        # Reset rope deltas for new generation
        self.rope_deltas = None
        
        # === PREFILL PHASE ===
        prefill_output = self.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            is_prefill=True,
        )
        
        past_key_values = prefill_output.past_key_values
        past_hidden = prefill_output.hidden_state
        
        # Sample first token
        logits = prefill_output.logits[:, -1, :]
        first_token = self._sample_next_token(
            logits, None, do_sample, top_k, top_p, temperature,
            repetition_penalty, suppress_tokens, 0, min_new_tokens, eos_token_id
        )
        
        # Track generated tokens (keep 2D shape [batch, 1] for embedding lookup)
        generated_tokens = [first_token.unsqueeze(-1)]
        all_codec_ids = []
        all_hidden_states = [(prefill_output.hidden_state, None)] if output_hidden_states else []
        
        # Update attention mask for generation
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
            ], dim=-1)
        
        generation_step = 0
        
        # === GENERATION LOOP ===
        for step in range(1, max_new_tokens):
            cache_position = torch.tensor(
                [past_key_values.get_seq_length()], device=device
            )
            
            # Placeholder embeddings - will be replaced in forward() during generation
            step_embeds = torch.zeros(
                (batch_size, 1, self.config.hidden_size),
                device=device,
                dtype=inputs_embeds.dtype,
            )
            
            # Forward pass - the actual input embeddings are computed inside forward()
            # when is_prefill=False, using the code predictor and trailing text
            output = self.forward(
                inputs_embeds=step_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                past_hidden=past_hidden,
                cache_position=cache_position,
                use_cache=True,
                subtalker_dosample=subtalker_dosample,
                subtalker_top_k=subtalker_top_k,
                subtalker_top_p=subtalker_top_p,
                subtalker_temperature=subtalker_temperature,
                is_prefill=False,
                last_predicted_token=generated_tokens[-1],
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                generation_step=generation_step,
            )
            
            past_key_values = output.past_key_values
            past_hidden = output.hidden_state
            
            if output.codec_ids is not None:
                all_codec_ids.append(output.codec_ids)
            
            if output_hidden_states:
                all_hidden_states.append((output.hidden_state, output.codec_ids))
            
            # Sample next token
            logits = output.logits[:, -1, :]
            # Build 2D sequence for repetition penalty: [batch, seq_len]
            generated_sequence = torch.cat(generated_tokens, dim=1)
            next_token = self._sample_next_token(
                logits, generated_sequence, do_sample, top_k, top_p, temperature,
                repetition_penalty, suppress_tokens, step, min_new_tokens, eos_token_id
            )
            
            generated_tokens.append(next_token.unsqueeze(-1))
            generation_step += 1
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
                ], dim=-1)
            
            # Check for EOS
            if step >= min_new_tokens and (next_token == eos_token_id).all():
                break
        
        # Concatenate all generated tokens: [batch, seq_len]
        sequences = torch.cat(generated_tokens, dim=1)
        
        return TalkerGenerateOutput(
            sequences=sequences,
            all_codec_ids=all_codec_ids,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )
    
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        generated_sequence: Optional[torch.Tensor],
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        suppress_tokens: Optional[list],
        step: int,
        min_new_tokens: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        """Sample the next token from logits."""
        # Apply repetition penalty
        if generated_sequence is not None and repetition_penalty != 1.0:
            for i in range(logits.shape[0]):
                for token in generated_sequence[i].unique():
                    if logits[i, token] < 0:
                        logits[i, token] *= repetition_penalty
                    else:
                        logits[i, token] /= repetition_penalty
        
        # Suppress tokens
        if suppress_tokens is not None:
            for token_id in suppress_tokens:
                logits[:, token_id] = float('-inf')
        
        # Prevent EOS before min_new_tokens
        if step < min_new_tokens:
            logits[:, eos_token_id] = float('-inf')
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Sample or greedy
        if do_sample:
            return sample_top_k_top_p(logits, top_k, top_p)
        else:
            return torch.argmax(logits, dim=-1)
    
    def load_original_state_dict(self, state_dict: dict):
        """
        Load weights from the original Qwen3TTSTalkerForConditionalGenerationStandalone model.
        
        Handles key remapping between the original and refactored model structures.
        """
        new_state_dict = {}
        
        for key, value in state_dict.items():
            new_key = key
            
            # Handle code_predictor weights
            if key.startswith("code_predictor."):
                # The code predictor has its own load method
                # We'll collect these separately and load them after
                continue
            
            new_state_dict[new_key] = value
        
        # Load main model weights
        self.load_state_dict(new_state_dict, strict=False)
        
        # Load code predictor weights
        code_predictor_state_dict = {
            k.replace("code_predictor.", ""): v
            for k, v in state_dict.items()
            if k.startswith("code_predictor.")
        }
        if code_predictor_state_dict:
            self.code_predictor.load_original_state_dict(code_predictor_state_dict)
