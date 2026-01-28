# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""Device detection and management utilities for cross-platform support.

Provides intelligent device selection and device-agnostic operations for PyTorch
models. Supports CUDA, Apple Metal Performance Shaders (MPS), and CPU with
automatic fallback (MPS > CUDA > CPU).
"""

import warnings
from typing import Optional, Union

import torch


def get_optimal_device(preferred_device: Optional[str] = None) -> str:
    """
    Detect optimal PyTorch device with intelligent fallback.

    Priority order:
    1. User-specified device (if valid)
    2. MPS (Apple Silicon) if available
    3. CUDA if available
    4. CPU (fallback)

    Args:
        preferred_device: User-specified device string (e.g., "cuda:0", "mps", "cpu").
            If not specified, auto-detects optimal device.

    Returns:
        Device string suitable for device_map parameter in model loading.

    Example:
        >>> device = get_optimal_device()  # Auto-detect
        >>> device = get_optimal_device("cuda:1")  # Prefer specific GPU
    """
    # If user explicitly specifies device, validate and return
    if preferred_device is not None:
        preferred = str(preferred_device).lower().strip()

        # Validate user preference
        if preferred.startswith("cuda"):
            if not torch.cuda.is_available():
                warnings.warn(
                    f"CUDA device '{preferred}' requested but CUDA not available. "
                    "Falling back to auto-detection.",
                    RuntimeWarning,
                )
            else:
                return preferred_device
        elif preferred == "mps":
            if not torch.backends.mps.is_available():
                warnings.warn(
                    "MPS device requested but not available. Falling back to auto-detection.",
                    RuntimeWarning,
                )
            else:
                return "mps"
        elif preferred == "cpu":
            return "cpu"
        else:
            # Let it pass through for other device types
            return preferred_device

    # Auto-detection with priority order
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda:0"
    else:
        warnings.warn(
            "No GPU detected. Using CPU. Performance will be significantly slower.",
            RuntimeWarning,
        )
        return "cpu"


def get_optimal_dtype(
    device: Union[str, torch.device], preferred_dtype: Optional[torch.dtype] = None
) -> torch.dtype:
    """
    Select optimal dtype for given device.

    Args:
        device: Device string or torch.device object.
        preferred_dtype: User-specified dtype preference.

    Returns:
        Recommended torch.dtype.

    Note:
        bfloat16 is recommended for all modern hardware (best quality/performance).
        float16 is also supported as a fallback.
        MPS has good bfloat16 support in PyTorch 2.0+.
    """
    if preferred_dtype is not None:
        return preferred_dtype

    # bfloat16 is optimal for all modern hardware
    return torch.bfloat16


def get_attention_implementation(
    device: Union[str, torch.device], preferred: Optional[str] = None
) -> Optional[str]:
    """
    Select appropriate attention implementation for device.

    Args:
        device: Device string or torch.device object.
        preferred: User preference (e.g., "flash_attention_2", "eager", "sdpa").

    Returns:
        Attention implementation string or None for default implementation.

    Note:
        FlashAttention 2 is CUDA-only and will be skipped on MPS/CPU.
        If user requests FlashAttention on non-CUDA device, a warning is issued
        and default attention is used instead.
    """
    device_str = str(device).lower()

    # If user explicitly requests FlashAttention on non-CUDA, warn them
    if preferred == "flash_attention_2":
        if "cuda" not in device_str:
            warnings.warn(
                f"FlashAttention 2 is CUDA-only and not available on device '{device}'. "
                "Using default attention implementation.",
                RuntimeWarning,
            )
            return None  # Let transformers choose default
        return "flash_attention_2"

    # Return user preference if specified
    if preferred is not None:
        return preferred

    # Auto-select: FlashAttention on CUDA if available, otherwise default
    if "cuda" in device_str:
        # Check if flash-attn is actually installed
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            return None

    return None  # Use default for MPS/CPU


def device_synchronize(device: Union[str, torch.device, None] = None) -> None:
    """
    Device-agnostic synchronization for accurate timing measurements.

    Ensures all previous GPU operations are complete before returning.
    Required for accurate timing measurements across different device types.

    Args:
        device: Device to synchronize (inferred from default if None).

    Example:
        >>> device_synchronize(device)
        >>> t0 = time.time()
        >>> # ... model inference ...
        >>> device_synchronize(device)
        >>> t1 = time.time()
        >>> print(f"Inference time: {t1 - t0:.3f}s")
    """
    if device is None:
        # Try to infer from default device
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            # MPS synchronization (available in PyTorch 2.0+)
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        return

    device_str = str(device).lower()

    if "cuda" in device_str:
        torch.cuda.synchronize()
    elif "mps" in device_str:
        # MPS synchronization (available in PyTorch 2.0+)
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    # CPU doesn't need explicit synchronization (operations are synchronous)


def get_device_info(device: Union[str, torch.device]) -> str:
    """
    Get human-readable device information.

    Args:
        device: Device to describe.

    Returns:
        Descriptive string about the device.

    Example:
        >>> device = "mps"
        >>> print(get_device_info(device))
        "Apple Metal Performance Shaders (MPS) - Apple Silicon GPU"
    """
    device_str = str(device).lower()

    if "cuda" in device_str:
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(device)
                return f"CUDA GPU: {gpu_name}"
            except Exception:
                return f"CUDA GPU: {device}"
        return "CUDA (not available)"
    elif "mps" in device_str:
        if torch.backends.mps.is_available():
            return "Apple Metal Performance Shaders (MPS) - Apple Silicon GPU"
        return "MPS (not available)"
    elif "cpu" in device_str:
        return "CPU"
    else:
        return f"Device: {device}"
