# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone utility functions.

This module provides minimal replacements for various transformers utilities.
"""

from functools import wraps

from huggingface_hub import hf_hub_download


def can_return_tuple(func):
    """
    Decorator to convert model output to tuple if return_dict=False.
    
    This is a standalone replacement for transformers.utils.can_return_tuple.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return_dict = getattr(getattr(self, "config", None), "return_dict", True)
        return_dict_passed = kwargs.pop("return_dict", return_dict)
        if return_dict_passed is not None:
            return_dict = return_dict_passed
        output = func(self, *args, **kwargs)
        if not return_dict and not isinstance(output, tuple):
            output = output.to_tuple()
        return output
    return wrapper


def cached_file(
    repo_id: str,
    filename: str,
    **kwargs,
) -> str:
    """
    Download and cache a file from a HuggingFace repository.
    
    This is a standalone replacement for transformers.utils.hub.cached_file.
    
    Args:
        repo_id: The repository ID (e.g., "organization/model-name")
        filename: The filename to download
        **kwargs: Additional arguments passed to hf_hub_download
    
    Returns:
        Local path to the cached file
    """
    return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
