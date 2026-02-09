# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone text processor for Qwen3-TTS.

This is a minimal processor that wraps a tokenizer without requiring
the transformers library.
"""

from typing import Dict, List, Optional, Union

import torch


class Processor:
    """
    Minimal text processor for Qwen3-TTS.
    
    Wraps a tokenizer (Qwen2TokenizerFast from tokenizers library or tiktoken)
    and provides a simple interface for text processing.
    """
    
    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: A tokenizer instance (e.g., from tokenizers library or tiktoken)
        """
        self.tokenizer = tokenizer
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        Load processor from a pretrained model path.
        
        Args:
            pretrained_model_name_or_path: Path to the model directory or HuggingFace repo
            **kwargs: Additional arguments passed to tokenizer loading
        
        Returns:
            Qwen3TTSProcessor instance
        """
        # Try to load tokenizer using tokenizers library first (no transformers needed)
        try:
            from tokenizers import Tokenizer
            import os
            from huggingface_hub import hf_hub_download
            
            # Check if it's a local path
            if os.path.isdir(pretrained_model_name_or_path):
                tokenizer_path = os.path.join(pretrained_model_name_or_path, "tokenizer.json")
            else:
                # Download from HuggingFace Hub
                tokenizer_path = hf_hub_download(
                    pretrained_model_name_or_path,
                    "tokenizer.json",
                    **{k: v for k, v in kwargs.items() if k in ['cache_dir', 'token', 'revision']}
                )
            
            tokenizer = Tokenizer.from_file(tokenizer_path)
            return cls(_TokenizerWrapper(tokenizer))
        except Exception:
            pass
        
        # Fall back to transformers tokenizer if available
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
            return cls(tokenizer)
        except ImportError:
            raise ImportError(
                "Could not load tokenizer. Please install either 'tokenizers' or 'transformers' library."
            )
    
    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = "pt",
        padding: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text input.
        
        Args:
            text: Text string or list of text strings to tokenize
            return_tensors: Return format ("pt" for PyTorch tensors)
            padding: Whether to pad sequences
            **kwargs: Additional tokenizer arguments
        
        Returns:
            Dictionary with 'input_ids' and optionally 'attention_mask'
        """
        if text is None:
            raise ValueError("You need to specify a `text` input to process.")
        
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        if hasattr(self.tokenizer, 'encode_batch'):
            # tokenizers library
            encodings = self.tokenizer.encode_batch(text)
            input_ids = [enc.ids for enc in encodings]
        else:
            # transformers tokenizer
            result = self.tokenizer(
                text,
                padding=padding,
                return_tensors=return_tensors,
                **kwargs
            )
            return dict(result)
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            if padding:
                # Pad to max length
                max_len = max(len(ids) for ids in input_ids)
                padded = []
                attention_mask = []
                pad_id = 0  # Default pad token
                for ids in input_ids:
                    pad_len = max_len - len(ids)
                    padded.append(ids + [pad_id] * pad_len)
                    attention_mask.append([1] * len(ids) + [0] * pad_len)
                return {
                    "input_ids": torch.tensor(padded, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
            else:
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                }
        else:
            return {"input_ids": input_ids}
    
    def decode(self, token_ids, skip_special_tokens: bool = True, **kwargs):
        """Decode token IDs to text."""
        if hasattr(self.tokenizer, 'decode'):
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        raise NotImplementedError("Tokenizer does not support decode")
    
    def batch_decode(self, token_ids_list, skip_special_tokens: bool = True, **kwargs):
        """Decode a batch of token ID sequences to text."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens, **kwargs) for ids in token_ids_list]


class _TokenizerWrapper:
    """Wrapper to give tokenizers.Tokenizer a consistent interface."""
    
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
    
    def encode_batch(self, texts):
        return self._tokenizer.encode_batch(texts)
    
    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


__all__ = [
    "Processor",
]
