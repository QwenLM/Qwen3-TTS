# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for Qwen3-TTS standalone models.
"""

import huggingface_hub
from huggingface_hub import snapshot_download


def download_weights_from_hf_specific(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    """Download model weights from Hugging Face Hub.
    
    Args:
        model_name_or_path: The model name or path.
        cache_dir: The cache directory to store the model weights.
        allow_patterns: The allowed patterns for the weight files.
        revision: The revision of the model.
        ignore_patterns: The patterns to filter out the weight files.

    Returns:
        The path to the downloaded model weights.
    """
    assert len(allow_patterns) > 0
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE

    for allow_pattern in allow_patterns:
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns=allow_pattern,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_only,
        )
    return hf_folder


__all__ = ["download_weights_from_hf_specific"]
