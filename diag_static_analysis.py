#!/usr/bin/env python3
"""
PHASE 1, STEP 2: Static Analysis
Inspects source code of key functions without executing them.
Runs in ~0.5 hours (pure Python, no GPU required)

Output: Prints source code of key functions. Saves to diag_static_analysis_results.txt
Decision: Look for unexpected differences between 4.x and 5.x source.
"""

import inspect
import sys

# CRITICAL: Apply RoPE patch FIRST
from qwen_tts.core.rope_utils import patch_rope_init_functions
patch_rope_init_functions()

print("=" * 80)
print("PHASE 1, STEP 2: STATIC ANALYSIS")
print("=" * 80)

# 1. create_causal_mask source
print("\n" + "=" * 80)
print("1. create_causal_mask source code")
print("=" * 80)
try:
    from transformers.masking_utils import create_causal_mask
    source = inspect.getsource(create_causal_mask)
    print(source[:2000])  # First 2000 chars
    if len(source) > 2000:
        print(f"\n... [truncated, total {len(source)} chars] ...\n")
        print(source[-500:])  # Last 500 chars
except Exception as e:
    print(f"[ERROR] Failed: {e}")

# 2. dynamic_rope_update source
print("\n" + "=" * 80)
print("2. dynamic_rope_update source code")
print("=" * 80)
try:
    from transformers.modeling_rope_utils import dynamic_rope_update
    source = inspect.getsource(dynamic_rope_update)
    print(source[:2000])
    if len(source) > 2000:
        print(f"\n... [truncated, total {len(source)} chars] ...\n")
        print(source[-500:])
except Exception as e:
    print(f"[ERROR] Failed: {e}")

# 3. Check inv_freq computation precision
print("\n" + "=" * 80)
print("3. inv_freq Computation Precision Test")
print("=" * 80)
print("""
Comparing two ways to compute inv_freq:
  A) Native: torch.arange(0, 128, 2, dtype=torch.float32) / 128
  B) Patched: torch.arange(0, 128, 2, dtype=torch.int64).float() / 128

Expected: Bit-identical (both should compute exactly the same)
""")
try:
    import torch

    # Method A: Native float32
    inv_freq_native = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))

    # Method B: int64 then convert to float32
    inv_freq_patched = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.int64).float() / 128))

    max_diff = (inv_freq_native - inv_freq_patched).abs().max().item()
    mean_diff = (inv_freq_native - inv_freq_patched).abs().mean().item()

    print(f"[OK] Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    if max_diff > 1e-7:
        print(f"  [WARN] Differences detected! This could be a numerical drift issue.")
    else:
        print(f"  [OK] Precision is identical (within float32 epsilon)")

    # Also check the actual rope_utils implementation
    from qwen_tts.core.rope_utils import _compute_default_rope_parameters
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig

    try:
        config = Qwen3TTSConfig()
        inv_freq_rope_utils, _ = _compute_default_rope_parameters(config, device="cpu")

        # The computation in rope_utils uses: torch.arange(0, dim, 2, dtype=torch.int64).to(..., dtype=torch.float)
        # Let's verify this matches
        dim = int(config.hidden_size // config.num_attention_heads * config.partial_rotary_factor)
        inv_freq_expected = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))

        max_diff_rope = (inv_freq_rope_utils - inv_freq_expected).abs().max().item()
        print(f"\n[OK] rope_utils implementation matches expected: max_diff = {max_diff_rope:.2e}")
    except AttributeError:
        print(f"\n[WARN] Could not verify rope_utils config (missing attributes - expected in test)")

except Exception as e:
    print(f"[ERROR] Failed: {e}")
    import traceback
    traceback.print_exc()

# 4. Check if create_causal_mask position_ids parameter affects mask
print("\n" + "=" * 80)
print("4. create_causal_mask position_ids Analysis")
print("=" * 80)
print("""
If create_causal_mask has a position_ids parameter:
- In 4.x: position_ids was not used
- In 5.x: position_ids might affect the mask construction

Check: Does the function use position_ids to modify mask values?
""")
try:
    from transformers.masking_utils import create_causal_mask
    import re

    source = inspect.getsource(create_causal_mask)
    has_position_ids_param = "position_ids" in inspect.signature(create_causal_mask).parameters

    if has_position_ids_param:
        # Count uses of position_ids in the function
        uses = len(re.findall(r'\bposition_ids\b', source))
        print(f"[OK] position_ids parameter found")
        print(f"  Used {uses} times in function body")
        if uses > 0:
            print(f"  [WARN] position_ids is used in mask construction - may corrupt decode!")
        else:
            print(f"  [OK] position_ids is unused - safe")
    else:
        print(f"[OK] No position_ids parameter - safe")

except Exception as e:
    print(f"[ERROR] Failed: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Key findings to look for:
1. inv_freq precision: If max_diff > 1e-7, numerical drift is happening
2. create_causal_mask: If position_ids is used in mask construction, decode may fail
3. dynamic_rope_update: Check if decorator alters inv_freq behavior
4. DynamicCache.update: Check if cache interface changed significantly

Next step: Run diag_rope_tensors.py to compare actual RoPE tensors
""")
