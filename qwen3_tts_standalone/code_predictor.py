# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simplified CodePredictor for Qwen3 TTS.

This module predicts codebook layers 1 through (num_code_groups - 1) given:
- A hidden state from the Talker model
- The codebook 0 token embedding

The prediction is autoregressive: each codebook layer conditions on all previous layers.
Each step uses a different embedding layer (for input) and lm_head (for output).
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .configuration import CodePredictorConfig
from .layers import DecoderLayer, RMSNorm, RotaryEmbedding
from .utils import (
    DynamicCache,
    create_causal_mask,
    sample_top_k_top_p,
)


@dataclass
class CodePredictorOutput:
    """Output from CodePredictor.generate()."""

    sequences: Tensor  # [batch, num_code_groups - 1] predicted codebook tokens


@dataclass
class CodePredictorFinetuneOutput:
    """Output from CodePredictor.forward_finetune()."""

    logits: Tensor  # [batch, num_code_groups - 1, vocab_size]
    loss: Optional[Tensor]  # scalar loss (if labels provided)


def _causal_lm_loss(
    logits: Tensor, labels: Tensor, vocab_size: int
) -> Tensor:
    """
    Compute ForCausalLMLoss matching the transformers library implementation.

    This applies the standard next-token-prediction shift:
    shift_logits = logits[..., :-1, :] predicts shift_labels = labels[..., 1:]

    Args:
        logits: Predicted logits [batch, seq_len, vocab_size]
        labels: Target labels [batch, seq_len]
        vocab_size: Vocabulary size for reshaping

    Returns:
        Scalar cross-entropy loss.
    """
    logits = logits.float()
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(shift_logits.device)

    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    return loss


class CodePredictor(nn.Module):
    """
    Predicts higher codebook layers (1 to num_code_groups-1) autoregressively.

    Architecture:
    - codec_embedding[i]: Embeds tokens from codebook (i+1) for input to next step
    - transformer layers: Standard transformer decoder stack
    - lm_head[i]: Predicts tokens for codebook (i+1)

    The autoregressive loop:
        Step 0: [hidden, embed_0(cb0)] -> lm_head[0] -> cb1
        Step 1: [hidden, embed_0(cb0), embed_1(cb1)] -> lm_head[1] -> cb2
        ...and so on
    """

    config: CodePredictorConfig
    num_codebooks_to_predict: int
    codec_embedding: nn.ModuleList
    input_projection: nn.Module
    layers: nn.ModuleList
    norm: RMSNorm
    rotary_emb: RotaryEmbedding
    lm_head: nn.ModuleList

    def __init__(
        self,
        config: CodePredictorConfig,
        embedding_dim: int,
    ) -> None:
        """
        Args:
            config: CodePredictor configuration
            embedding_dim: Dimension of input embeddings (typically talker's hidden_size)
        """
        super().__init__()
        self.config = config
        self.num_codebooks_to_predict = config.num_code_groups - 1

        # Embedding layers: one per codebook we receive as input (codebooks 1 to N-2)
        # codec_embedding[i] embeds codebook (i+1) tokens
        self.codec_embedding = nn.ModuleList(
            [
                nn.Embedding(config.vocab_size, embedding_dim)
                for _ in range(self.num_codebooks_to_predict)
            ]
        )

        # Projection if hidden sizes differ
        if config.hidden_size != embedding_dim:
            self.input_projection = nn.Linear(embedding_dim, config.hidden_size, bias=True)
        else:
            self.input_projection = nn.Identity()

        # Transformer decoder layers (reuse shared implementation)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)

        # Output heads: one per codebook we predict (codebooks 1 to N-1)
        # lm_head[i] predicts codebook (i+1) tokens
        self.lm_head = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                for _ in range(self.num_codebooks_to_predict)
            ]
        )

    def load_original_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """
        Load weights from the original Qwen3TTSCodePredictorForConditionalGeneration model.

        Handles key remapping:
        - model.X -> X (removes 'model.' prefix)
        - small_to_mtp_projection -> input_projection
        """
        new_state_dict: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            new_key = key
            # Remove 'model.' prefix
            if new_key.startswith("model."):
                new_key = new_key[6:]
            # Rename projection layer
            if new_key.startswith("small_to_mtp_projection"):
                new_key = new_key.replace("small_to_mtp_projection", "input_projection")
            new_state_dict[new_key] = value

        self.load_state_dict(new_state_dict)

    def forward_finetune(
        self,
        inputs_embeds: Tensor,
        labels: Optional[Tensor] = None,
    ) -> CodePredictorFinetuneOutput:
        """
        Forward pass for finetuning the code predictor.

        This matches the original
        Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward_finetune().

        The method:
        1. Projects input embeddings through input_projection
        2. Runs a full (non-cached) transformer forward pass
        3. Extracts logits from each position using the respective lm_head
        4. Computes ForCausalLMLoss (with standard shift) if labels are provided

        Note: The loss uses the standard ForCausalLMLoss from transformers which
        applies a shift (logits[:-1] predicts labels[1:]). This matches the
        original implementation's behavior.

        Args:
            inputs_embeds: Input embeddings [batch, num_code_groups, embedding_dim].
                Contains [talker_hidden_state, embed(cb0), embed(cb1), ..., embed(cb_{N-2})]
            labels: Target codebook tokens [batch, num_code_groups - 1].
                Contains [cb1, cb2, ..., cb_{N-1}]

        Returns:
            CodePredictorFinetuneOutput with logits and optional loss.
        """
        # Project to model hidden size (equivalent to small_to_mtp_projection)
        hidden_states = self.input_projection(inputs_embeds)

        # Run through transformer without caching
        cache_position = torch.arange(
            hidden_states.shape[1], device=hidden_states.device
        )
        hidden_states = self._transformer_forward(
            hidden_states, cache=None, cache_position=cache_position
        )

        # Compute logits: for each codebook i (1..N-1), apply lm_head[i-1]
        # to the hidden state at position i
        logits = []
        for i in range(1, self.config.num_code_groups):
            logits.append(self.lm_head[i - 1](hidden_states[:, i]))
        logits = torch.stack(logits, dim=1)  # [batch, num_code_groups - 1, vocab_size]

        # Compute loss matching ForCausalLMLoss (with shift)
        loss = None
        if labels is not None:
            loss = _causal_lm_loss(logits, labels, self.config.vocab_size)

        return CodePredictorFinetuneOutput(logits=logits, loss=loss)

    def generate(
        self,
        inputs_embeds: Tensor,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        **kwargs: Any,  # Ignored - for API compatibility
    ) -> CodePredictorOutput:
        """
        Generate codebook tokens 1 through (num_code_groups - 1).

        Args:
            inputs_embeds: Initial embeddings [batch, 2, hidden_size]
                           Contains [talker_hidden, codebook_0_embedding]
            max_new_tokens: Number of codebooks to predict (should be num_code_groups - 1)
            do_sample: Whether to sample (True) or use greedy decoding (False)
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold

        Returns:
            CodePredictorOutput with sequences of shape [batch, max_new_tokens]

        The generation follows this pattern (matching the original implementation):

        Step 0 (prefill):
            - generation_steps = 0 (inputs_embeds.shape[1] - 2)
            - lm_head[0] predicts codebook 1
            - Returns generation_steps = 1

        Step N (N >= 1):
            - Embed previous token with codec_embedding[N-1]
            - lm_head[N] predicts codebook N+1
            - Returns generation_steps = N+1
        """
        device = inputs_embeds.device

        # Project input embeddings to model hidden size
        hidden_states = self.input_projection(inputs_embeds)

        # Initialize KV cache
        cache = DynamicCache()

        # Track generated tokens
        generated_tokens: list[Tensor] = []

        # generation_steps tracks which step we're on (matches original's logic)
        # During prefill: generation_steps = inputs_embeds.shape[1] - 2 = 0
        generation_steps = 0

        # Prefill: process initial embeddings through transformer
        cache_position = torch.arange(hidden_states.shape[1], device=device)
        hidden_states = self._transformer_forward(hidden_states, cache, cache_position)

        # First prediction: use lm_head[generation_steps] = lm_head[0]
        head_idx = min(generation_steps, len(self.lm_head) - 1)
        next_token = self._predict_token(
            hidden_states[:, -1, :], head_idx, do_sample, temperature, top_k, top_p
        )
        generated_tokens.append(next_token)
        generation_steps += 1  # Now = 1

        # Generate remaining codebooks
        for _ in range(1, max_new_tokens):
            # Embed the previous token with codec_embedding[generation_steps - 1]
            embed_idx = min(generation_steps - 1, len(self.codec_embedding) - 1)
            next_embed = self.codec_embedding[embed_idx](next_token.unsqueeze(-1))
            next_embed = self.input_projection(next_embed)

            # Forward through transformer
            seq_len = inputs_embeds.shape[1] + generation_steps
            cache_position = torch.tensor([seq_len - 1], device=device)
            hidden_states = self._transformer_forward(next_embed, cache, cache_position)

            # Predict with lm_head[generation_steps]
            head_idx = min(generation_steps, len(self.lm_head) - 1)
            next_token = self._predict_token(
                hidden_states[:, -1, :], head_idx, do_sample, temperature, top_k, top_p
            )
            generated_tokens.append(next_token)
            generation_steps += 1

        # Stack all generated tokens: [batch, num_codebooks_to_predict]
        sequences = torch.stack(generated_tokens, dim=1)

        return CodePredictorOutput(sequences=sequences)

    def _predict_token(
        self,
        hidden_state: Tensor,
        head_idx: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tensor:
        """Predict next token from hidden state using specified lm_head."""
        logits = self.lm_head[head_idx](hidden_state)

        if temperature != 1.0:
            logits = logits / temperature

        if do_sample:
            return sample_top_k_top_p(logits, top_k, top_p)
        else:
            return torch.argmax(logits, dim=-1)

    def _transformer_forward(
        self,
        hidden_states: Tensor,
        cache: DynamicCache,
        cache_position: Tensor,
    ) -> Tensor:
        """Run hidden_states through transformer layers with caching."""
        position_ids = cache_position.unsqueeze(0)

        # Create causal mask
        causal_mask = create_causal_mask(
            input_embeds=hidden_states,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=cache,
        )

        # Create rotary embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Run through decoder layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        return hidden_states
