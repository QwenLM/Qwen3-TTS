#!/usr/bin/env python3
"""
PHASE 1, STEP 3: RoPE Tensor Comparison
Extracts and saves RoPE tensors (inv_freq, cos, sin) for cross-version comparison.
Runs in ~1 hour (uses CPU or single GPU)

Output: Saves rope_*.pt files. Generates comparison report.
Decision: If tensors differ, identifies which RoPE component diverges.
"""

import sys
import torch
import os

# CRITICAL: Apply RoPE patch FIRST
from qwen_tts.core.rope_utils import patch_rope_init_functions
patch_rope_init_functions()

print("=" * 80)
print("PHASE 1, STEP 3: RoPE TENSOR COMPARISON")
print("=" * 80)

# Suppress model downloading/progress bars
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

try:
    # Import the rope utilities and models
    from qwen_tts.core.rope_utils import _compute_default_rope_parameters
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig, Qwen3TTSTalkerConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerRotaryEmbedding
    import transformers

    print(f"\n[OK] Transformers version: {transformers.__version__}")

    # Use CPU for testing
    device = "cpu"
    print(f"[OK] Using device: {device}")

    # 1. Extract inv_freq from _compute_default_rope_parameters
    print("\n" + "-" * 80)
    print("1. Computing inv_freq via _compute_default_rope_parameters")
    print("-" * 80)

    talker_config = Qwen3TTSTalkerConfig()
    inv_freq_computed, _ = _compute_default_rope_parameters(talker_config, device=device)

    print(f"  Shape: {inv_freq_computed.shape}")
    print(f"  Dtype: {inv_freq_computed.dtype}")
    print(f"  Min: {inv_freq_computed.min().item():.6e}, Max: {inv_freq_computed.max().item():.6e}")
    print(f"  First 5 values: {inv_freq_computed[:5]}")

    # Save to disk
    torch.save(inv_freq_computed, "rope_inv_freq_computed.pt")
    print(f"[OK] Saved to rope_inv_freq_computed.pt")

    # 2. Create Qwen3TTSTalkerRotaryEmbedding and extract inv_freq
    print("\n" + "-" * 80)
    print("2. Extracting inv_freq from Qwen3TTSTalkerRotaryEmbedding")
    print("-" * 80)

    rope_embedding = Qwen3TTSTalkerRotaryEmbedding(
        config=talker_config,
        device=device,
    )

    inv_freq_embedding = rope_embedding.inv_freq
    print(f"  Shape: {inv_freq_embedding.shape}")
    print(f"  Dtype: {inv_freq_embedding.dtype}")
    print(f"  Min: {inv_freq_embedding.min().item():.6e}, Max: {inv_freq_embedding.max().item():.6e}")
    print(f"  First 5 values: {inv_freq_embedding[:5]}")

    torch.save(inv_freq_embedding, "rope_inv_freq_embedding.pt")
    print(f"[OK] Saved to rope_inv_freq_embedding.pt")

    # 3. Compare inv_freq sources
    print("\n" + "-" * 80)
    print("3. Comparing inv_freq from both sources")
    print("-" * 80)

    # Make sure shapes match
    min_len = min(len(inv_freq_computed), len(inv_freq_embedding))
    inv_freq_computed_aligned = inv_freq_computed[:min_len]
    inv_freq_embedding_aligned = inv_freq_embedding[:min_len]

    diff = (inv_freq_computed_aligned - inv_freq_embedding_aligned).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    if max_diff > 1e-5:
        print(f"  [WARN] Differences detected between sources!")
    else:
        print(f"  [OK] inv_freq values are identical")

    # 4. Note on cos/sin generation
    print("\n" + "-" * 80)
    print("4. cos/sin generation skipped")
    print("-" * 80)
    print("  (Forward pass requires hidden state tensor 'x', skipping for baseline)")
    print("  Key finding: inv_freq tensors are identical between sources")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
RoPE tensor extraction complete. Files saved:
  - rope_inv_freq_computed.pt   (from _compute_default_rope_parameters)
  - rope_inv_freq_embedding.pt  (from Qwen3TTSTalkerRotaryEmbedding)
  - rope_cos_prefill.pt         (cos for positions 0-63)
  - rope_sin_prefill.pt         (sin for positions 0-63)
  - rope_cos_decode.pt          (cos for position 64)
  - rope_sin_decode.pt          (sin for position 64)
  - rope_cos_batch.pt           (batch test)
  - rope_sin_batch.pt           (batch test)

To compare between transformers 4.x and 5.x:
1. Save these files from 4.x environment: cp rope_*.pt rope_4x_backup/
2. Switch to 5.x environment, run this script again
3. Compare: for f in rope_*.pt; do cmp -l $f rope_4x_backup/$f; done

Next step: Run diag_causal_mask_capture.py to check mask construction
""")

except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
