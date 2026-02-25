#!/usr/bin/env python3
"""
PHASE 1, STEP 1: Environment Check
Captures structural differences between transformers 4.x and 5.x
Runs in ~0.1 hours (pure Python, no GPU required)

Output: Prints environment differences. Saves to diag_env_check_results.txt
Decision: If differences found, they are root cause candidates.
"""

import inspect
import sys
from typing import Any, Dict

print("=" * 80)
print("PHASE 1, STEP 1: ENVIRONMENT CHECK")
print("=" * 80)

# CRITICAL: Apply RoPE patch FIRST, before any other imports
print("\n[PATCH] Applying RoPE patch...")
try:
    from qwen_tts.core.rope_utils import patch_rope_init_functions
    patch_rope_init_functions()
    print("[PATCH] RoPE patch applied")
except Exception as e:
    print(f"[WARN] Could not apply RoPE patch: {e}")

# 1. Transformers version
try:
    import transformers
    tf_version = transformers.__version__
    print(f"\n[OK] transformers.__version__ = {tf_version}")
except Exception as e:
    print(f"\n[ERROR] Failed to import transformers: {e}")
    sys.exit(1)

# 2. ROPE_INIT_FUNCTIONS registration (did our patch work?)
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    has_default = "default" in ROPE_INIT_FUNCTIONS
    print(f"[OK] ROPE_INIT_FUNCTIONS keys: {list(ROPE_INIT_FUNCTIONS.keys())}")
    print(f"  -> 'default' registered: {has_default}")
    if not has_default:
        print("  [WARN] 'default' not in ROPE_INIT_FUNCTIONS - patch may have failed!")
except Exception as e:
    print(f"[ERROR] Failed to check ROPE_INIT_FUNCTIONS: {e}")

# 3. create_causal_mask signature (does it have position_ids param?)
try:
    from transformers.masking_utils import create_causal_mask
    sig = inspect.signature(create_causal_mask)
    params = list(sig.parameters.keys())
    print(f"\n[OK] create_causal_mask signature:")
    print(f"  Parameters: {params}")
    has_position_ids = "position_ids" in params
    print(f"  -> Has 'position_ids' param: {has_position_ids}")
    if has_position_ids:
        print("  [WARN] position_ids param found - may affect decode mask construction!")
except Exception as e:
    print(f"\n[ERROR] Failed to inspect create_causal_mask: {e}")

# 4. dynamic_rope_update signature/decorator behavior
try:
    from transformers.modeling_rope_utils import dynamic_rope_update
    sig = inspect.signature(dynamic_rope_update)
    params = list(sig.parameters.keys())
    print(f"\n[OK] dynamic_rope_update signature:")
    print(f"  Parameters: {params}")
    is_decorator = callable(dynamic_rope_update)
    print(f"  -> Is decorator: {is_decorator}")
except Exception as e:
    print(f"\n[ERROR] Failed to inspect dynamic_rope_update: {e}")

# 5. DynamicCache.update interface
try:
    from transformers.cache_utils import DynamicCache
    sig = inspect.signature(DynamicCache.update)
    params = list(sig.parameters.keys())
    print(f"\n[OK] DynamicCache.update signature:")
    print(f"  Parameters: {params}")
except Exception as e:
    print(f"\n[ERROR] Failed to inspect DynamicCache.update: {e}")

# 6. Qwen3TTS model config - check attn implementation
try:
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    config = Qwen3TTSConfig()
    attn_impl = getattr(config, "_attn_implementation", "not_set")
    print(f"\n[OK] Qwen3TTSConfig._attn_implementation = {attn_impl}")
    if attn_impl == "sdpa":
        print("  [WARN] _attn_implementation is 'sdpa' - may override local eager attention!")
except Exception as e:
    print(f"\n[ERROR] Failed to check Qwen3TTSConfig: {e}")

# 7. ALL_ATTENTION_FUNCTIONS availability
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    print(f"\n[OK] ALL_ATTENTION_FUNCTIONS keys: {list(ALL_ATTENTION_FUNCTIONS.keys())}")
    has_eager = "eager" in ALL_ATTENTION_FUNCTIONS
    print(f"  -> 'eager' available: {has_eager}")
except Exception as e:
    print(f"\n[ERROR] Failed to check ALL_ATTENTION_FUNCTIONS: {e}")

# 8. Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
If any [WARN] WARNING is printed above:
1. 'default' not in ROPE_INIT_FUNCTIONS -> patch_rope_init_functions() not called
2. 'position_ids' in create_causal_mask -> transformers 5.x may corrupt decode mask
3. _attn_implementation is 'sdpa' -> may override local eager attention (USE EAGER!)
4. dynamic_rope_update signature changed -> may affect inv_freq mutation

Next step: Run diag_static_analysis.py for source code inspection
""")

print("\nEnvironment check complete!")
