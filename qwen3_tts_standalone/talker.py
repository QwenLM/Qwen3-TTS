# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simplified Talker model for Qwen3-TTS.

This is a refactored version of Qwen3TTSTalkerForConditionalGenerationStandalone
with improved readability and explicit generation logic.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .code_predictor import CodePredictor
from .configuration import TalkerConfig
from .layers import ResizeMLP
from .transformer import SemanticTransformer
from .utils import (
    DynamicCache,
    sample_top_k_top_p,
)


@dataclass
class TalkerOutput:
    """Output from a single Talker forward pass."""

    logits: Tensor  # [batch_size, seq_len, vocab_size]
    hidden_state: Tensor  # [batch_size, seq_len, hidden_size]
    past_key_values: Optional[DynamicCache]


@dataclass
class TalkerGenerateOutput:
    """Output from Talker.generate()."""

    sequences: Tensor  # [batch_size, seq_len] - first codebook tokens
    all_codec_ids: list[Tensor]  # List of [batch_size, num_code_groups] tensors per step
    hidden_states: Optional[tuple[tuple[Tensor, Optional[Tensor]], ...]]


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

    config: TalkerConfig
    vocab_size: int
    num_code_groups: int
    model: SemanticTransformer
    text_projection: ResizeMLP
    codec_head: nn.Linear
    code_predictor: CodePredictor
    rope_deltas: Optional[Tensor]

    def __init__(self, config: TalkerConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_code_groups = config.num_code_groups

        # Core transformer model
        self.model = SemanticTransformer(config)

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

    @property
    def codec_embedding(self) -> nn.Embedding:
        """Codec token embedding layer."""
        return self.model.codec_embedding

    @property
    def text_embedding(self) -> nn.Embedding:
        """Text token embedding layer."""
        return self.model.text_embedding

    def _compute_rope_position_ids(
        self,
        attention_mask: Tensor,
        cache_position: Optional[Tensor] = None,
        batch_size: int = 1,
        seq_length: int = 1,
    ) -> Tensor:
        """
        Compute 3D rope position IDs for the talker model.

        Args:
            attention_mask: Attention mask [batch, seq_len]
            cache_position: Current position in cache
            batch_size: Batch size
            seq_length: Sequence length

        Returns:
            Position IDs tensor [3, batch, seq_len] for multimodal RoPE
        """
        if attention_mask is not None:
            if cache_position is None or cache_position[0] == 0 or self.rope_deltas is None:
                # Initial position computation
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids = attention_mask.float().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                )
                max_position_ids = (
                    position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                )
                rope_deltas = (
                    max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
                )
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
        inputs_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[Tensor] = None,
        use_cache: bool = True,
    ) -> TalkerOutput:
        """
        Single forward pass of the Talker transformer.

        This is a pure transformer forward pass. The code predictor is called
        separately after sampling the first codebook token.

        Args:
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            past_key_values: KV cache from previous steps
            cache_position: Current position in cache
            use_cache: Whether to use KV caching

        Returns:
            TalkerOutput with logits, hidden states, and updated cache
        """
        batch_size = inputs_embeds.shape[0]
        seq_length = inputs_embeds.shape[1]

        # Compute position IDs
        position_ids = self._compute_rope_position_ids(
            attention_mask, cache_position, batch_size, seq_length
        )

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
            hidden_state=hidden_states,
            past_key_values=outputs.past_key_values,
        )

    def predict_higher_codebooks(
        self,
        hidden_state: Tensor,
        first_codebook_token: Tensor,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ) -> Tensor:
        """
        Predict higher codebook tokens using the code predictor.

        This method takes the hidden state from the transformer and the sampled
        first codebook token, then predicts the remaining codebook tokens.

        Args:
            hidden_state: Hidden state from transformer [batch_size, 1, hidden_size]
            first_codebook_token: Sampled first codebook token [batch_size, 1]
            do_sample: Whether to sample or use greedy decoding
            top_k, top_p, temperature: Sampling parameters

        Returns:
            Full codec IDs tensor [batch_size, num_code_groups] containing all codebook tokens
        """
        # Get embedding of the first codebook token
        first_token_embed = self.codec_embedding(first_codebook_token)

        # Concatenate hidden state and first token embedding as input to code predictor
        predictor_input = torch.cat((hidden_state, first_token_embed), dim=1)

        # Generate higher codebook tokens
        predictor_result = self.code_predictor.generate(
            inputs_embeds=predictor_input,
            max_new_tokens=self.num_code_groups - 1,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

        # Combine first codebook + higher codebooks: [batch_size, num_code_groups]
        codec_ids = torch.cat((first_codebook_token, predictor_result.sequences), dim=-1)

        return codec_ids

    def compute_codec_embeddings(
        self,
        codec_ids: Tensor,
    ) -> Tensor:
        """
        Compute summed embeddings from all codebook tokens.

        This computes the input embedding for the next transformer step by
        summing embeddings from all codebook tokens.

        Args:
            codec_ids: Full codec IDs [batch_size, num_code_groups]

        Returns:
            Summed codec embeddings [batch_size, 1, hidden_size]
        """
        codec_hiddens: list[Tensor] = []

        # First codebook uses the main codec embedding
        codec_hiddens.append(
            self.codec_embedding(codec_ids[:, 0:1])
        )

        # Higher codebooks use code predictor's embeddings
        for i in range(self.num_code_groups - 1):
            codec_embed = self.code_predictor.codec_embedding[i](
                codec_ids[:, i + 1 : i + 2]
            )
            codec_hiddens.append(codec_embed)

        # Sum all codebook embeddings: [batch_size, num_code_groups, hidden] -> [batch_size, 1, hidden]
        return torch.cat(codec_hiddens, dim=1).sum(dim=1, keepdim=True)

    def forward_sub_talker_finetune(
        self,
        codec_ids: Tensor,
        talker_hidden_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass for finetuning the sub-talker (code predictor).

        This matches the original
        Qwen3TTSTalkerForConditionalGeneration.forward_sub_talker_finetune().

        The method builds input embeddings for the code predictor by concatenating:
        - The talker hidden state at each codec frame
        - Embeddings of codebook tokens 0 through N-2

        Then calls the code predictor's forward_finetune to compute logits and
        cross-entropy loss on the higher codebook predictions.

        Args:
            codec_ids: All codebook tokens [N, num_code_groups].
                Each row contains the full set of codebook IDs for one codec frame.
            talker_hidden_states: Hidden states from the main talker [N, hidden_size].
                One hidden state per codec frame.

        Returns:
            Tuple of (logits, loss) where:
            - logits: [N, num_code_groups - 1, vocab_size]
            - loss: scalar cross-entropy loss
        """
        assert len(codec_ids.shape) == 2
        assert len(talker_hidden_states.shape) == 2
        assert codec_ids.shape[0] == talker_hidden_states.shape[0]
        assert talker_hidden_states.shape[1] == self.config.hidden_size
        assert codec_ids.shape[1] == self.num_code_groups

        # Build input embeddings:
        # [hidden_state, embed(cb0), embed(cb1), ..., embed(cb_{N-2})]
        sub_talker_inputs_embeds: list[Tensor] = [
            talker_hidden_states.unsqueeze(1)
        ]

        for i in range(self.num_code_groups - 1):
            if i == 0:
                # First codebook uses the main codec embedding
                sub_talker_inputs_embeds.append(
                    self.codec_embedding(codec_ids[:, :1])
                )
            else:
                # Higher codebooks use code predictor's embeddings
                sub_talker_inputs_embeds.append(
                    self.code_predictor.codec_embedding[i - 1](
                        codec_ids[:, i : i + 1]
                    )
                )

        sub_talker_inputs_embeds_cat = torch.cat(sub_talker_inputs_embeds, dim=1)

        # Forward through code predictor with labels
        sub_talker_outputs = self.code_predictor.forward_finetune(
            inputs_embeds=sub_talker_inputs_embeds_cat,
            labels=codec_ids[:, 1:],
        )

        return sub_talker_outputs.logits, sub_talker_outputs.loss

    def generate(
        self,
        inputs_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        trailing_text_hidden: Optional[Tensor] = None,
        tts_pad_embed: Optional[Tensor] = None,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 2,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        eos_token_id: Optional[int] = None,
        suppress_tokens: Optional[list[int]] = None,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> TalkerGenerateOutput:
        """
        Generate codec tokens autoregressively.

        The generation flow is:
        1. Prefill: Run transformer on initial embeddings, sample first token,
           then predict all codebook tokens for that step
        2. Generation loop: Use previous step's codec embeddings + text as input,
           run transformer, sample first token, predict all codebook tokens

        This ensures all codebook tokens are predicted using the CURRENT hidden state,
        rather than the previous step's hidden state.

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
        # 1. Run transformer forward on initial embeddings
        prefill_output = self.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = prefill_output.past_key_values
        hidden_state = prefill_output.hidden_state[:, -1:, :]  # [batch, 1, hidden]

        # 2. Sample first codebook token from logits
        logits = prefill_output.logits[:, -1, :]
        first_token = self._sample_next_token(
            logits,
            None,
            do_sample,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            suppress_tokens,
            0,
            min_new_tokens,
            eos_token_id,
        )
        first_token_2d = first_token.unsqueeze(-1)  # [batch, 1]

        # 3. Predict higher codebook tokens using current hidden state
        first_codec_ids = self.predict_higher_codebooks(
            hidden_state=hidden_state,
            first_codebook_token=first_token_2d,
            do_sample=subtalker_dosample,
            top_k=subtalker_top_k,
            top_p=subtalker_top_p,
            temperature=subtalker_temperature,
        )

        # Track generated tokens and codec IDs
        generated_tokens: list[Tensor] = [first_token_2d]
        all_codec_ids: list[Tensor] = [first_codec_ids]
        all_hidden_states: list[tuple[Tensor, Optional[Tensor]]] = (
            [(hidden_state, first_codec_ids)] if output_hidden_states else []
        )

        # Store last codec IDs for computing next step's input embeddings
        last_codec_ids = first_codec_ids

        # Update attention mask for generation
        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device),
                ],
                dim=-1,
            )

        # === GENERATION LOOP ===
        for step in range(1, max_new_tokens):
            cache_position = torch.tensor(
                [past_key_values.get_seq_length()], device=device
            )

            # 1. Compute input embeddings from previous step's codec IDs
            step_embeds = self.compute_codec_embeddings(last_codec_ids)

            # Add trailing text embeddings (streaming text input)
            generation_step = step - 1  # 0-indexed for text
            if trailing_text_hidden is not None:
                if generation_step < trailing_text_hidden.shape[1]:
                    step_embeds = (
                        step_embeds
                        + trailing_text_hidden[:, generation_step : generation_step + 1, :]
                    )
                elif tts_pad_embed is not None:
                    step_embeds = step_embeds + tts_pad_embed

            # 2. Run transformer forward
            output = self.forward(
                inputs_embeds=step_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )

            past_key_values = output.past_key_values
            hidden_state = output.hidden_state[:, -1:, :]  # [batch, 1, hidden]

            # 3. Sample first codebook token
            logits = output.logits[:, -1, :]
            generated_sequence = torch.cat(generated_tokens, dim=1)
            next_token = self._sample_next_token(
                logits,
                generated_sequence,
                do_sample,
                top_k,
                top_p,
                temperature,
                repetition_penalty,
                suppress_tokens,
                step,
                min_new_tokens,
                eos_token_id,
            )
            next_token_2d = next_token.unsqueeze(-1)

            # 4. Predict higher codebook tokens using current hidden state
            codec_ids = self.predict_higher_codebooks(
                hidden_state=hidden_state,
                first_codebook_token=next_token_2d,
                do_sample=subtalker_dosample,
                top_k=subtalker_top_k,
                top_p=subtalker_top_p,
                temperature=subtalker_temperature,
            )

            # Track results
            generated_tokens.append(next_token_2d)
            all_codec_ids.append(codec_ids)
            last_codec_ids = codec_ids

            if output_hidden_states:
                all_hidden_states.append((hidden_state, codec_ids))

            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (batch_size, 1), dtype=attention_mask.dtype, device=device
                        ),
                    ],
                    dim=-1,
                )

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
        logits: Tensor,
        generated_sequence: Optional[Tensor],
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        suppress_tokens: Optional[list[int]],
        step: int,
        min_new_tokens: int,
        eos_token_id: int,
    ) -> Tensor:
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
                logits[:, token_id] = float("-inf")

        # Prevent EOS before min_new_tokens
        if step < min_new_tokens:
            logits[:, eos_token_id] = float("-inf")

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Sample or greedy
        if do_sample:
            return sample_top_k_top_p(logits, top_k, top_p)
        else:
            return torch.argmax(logits, dim=-1)

    def load_original_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """
        Load weights from the original Qwen3TTSTalkerForConditionalGeneration model.

        Handles key remapping between the original and refactored model structures.
        """
        new_state_dict: dict[str, Tensor] = {}

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
