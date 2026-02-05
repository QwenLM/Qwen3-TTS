# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone generation utilities for autoregressive text generation.

This module provides a minimal replacement for transformers.generation.GenerationMixin.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from .cache import DynamicCache


@dataclass
class GenerateOutput:
    """Output from the generate method."""
    sequences: torch.LongTensor
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None


class GenerationMixin:
    """
    A minimal standalone mixin class that provides the generate() method.
    
    This is a simplified replacement for transformers.generation.GenerationMixin.
    It supports:
    - Autoregressive generation with greedy or sampling decoding
    - Top-k and top-p (nucleus) sampling
    - Temperature scaling
    - EOS token stopping
    - Max new tokens limit
    """
    
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict_in_generate: bool = False,
        use_cache: bool = True,
        **model_kwargs,
    ) -> GenerateOutput:
        """
        Generate sequences using autoregressive decoding.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            inputs_embeds: Input embeddings (alternative to input_ids)
            attention_mask: Attention mask
            max_new_tokens: Maximum number of new tokens to generate
            max_length: Maximum total length (deprecated, use max_new_tokens)
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            temperature: Sampling temperature
            top_k: Top-k filtering value
            top_p: Top-p (nucleus) filtering value
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention weights
            return_dict_in_generate: Whether to return a GenerateOutput
            use_cache: Whether to use KV caching
            **model_kwargs: Additional model-specific arguments
        
        Returns:
            GenerateOutput containing generated sequences and optionally hidden states
        """
        # Get config values
        config = getattr(self, "config", None)
        if pad_token_id is None:
            pad_token_id = getattr(config, "pad_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(config, "eos_token_id", None)
        if max_new_tokens is None:
            max_new_tokens = getattr(config, "max_new_tokens", 100)
        
        # Determine batch size and device
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            cur_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
            cur_len = 0
            # Create dummy input_ids for tracking
            input_ids = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        # Initialize cache if using caching
        past_key_values = model_kwargs.get("past_key_values", None)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
            model_kwargs["past_key_values"] = past_key_values
        
        # Track hidden states if requested
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        # Prepare initial model inputs
        model_kwargs["use_cache"] = use_cache
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["output_attentions"] = output_attentions
        
        # Set eos_token_id if provided via kwargs
        if eos_token_id is None and pad_token_id is not None:
            eos_token_id = pad_token_id
        
        # Generation loop
        for step in range(max_new_tokens):
            # Prepare inputs for this step
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                inputs_embeds=inputs_embeds if step == 0 else None,
                attention_mask=attention_mask,
                **model_kwargs,
            )
            
            # Forward pass
            outputs = self(**model_inputs)
            
            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Sample or greedy decode
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update model kwargs for next iteration
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1
            )
            
            # Collect hidden states
            if output_hidden_states and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                all_hidden_states.append(outputs.hidden_states)
            
            # Collect attentions
            if output_attentions and hasattr(outputs, "attentions") and outputs.attentions is not None:
                all_attentions.append(outputs.attentions)
            
            # Update attention mask if needed
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
                ], dim=-1)
            
            # Check for EOS token
            if eos_token_id is not None:
                if (next_tokens == eos_token_id).all():
                    break
        
        # Prepare output
        if return_dict_in_generate:
            return GenerateOutput(
                sequences=input_ids,
                hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
                attentions=tuple(all_attentions) if all_attentions else None,
            )
        else:
            return GenerateOutput(sequences=input_ids)
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        **kwargs,
    ) -> dict:
        """
        Prepare inputs for generation. Override in subclass for custom behavior.
        
        Default implementation passes through input_ids and all kwargs.
        """
        return {"input_ids": input_ids, **kwargs}
    
    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: dict,
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict:
        """
        Update model kwargs for the next generation step.
        
        Updates past_key_values and cache_position.
        """
        # Update past_key_values
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            model_kwargs["past_key_values"] = outputs.past_key_values
        
        # Update cache_position
        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        
        return model_kwargs
