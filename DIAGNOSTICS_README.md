# Qwen3 TTS Transformers Compatibility Diagnostic Framework

## Overview

This diagnostic framework helps investigate and troubleshoot transformers version compatibility issues with Qwen3 TTS. It uses a systematic 4-phase approach to isolate root causes without requiring deep model knowledge.

**Use this when:** Audio generation produces silence, poor quality, or unexpected behavior after upgrading transformers versions.

---

## Quick Start

### 1. Install transformers versions to test
```bash
# Test with transformers 4.x (baseline - should work)
pip install "transformers>=4.36.0,<5.0.0"

# Run baseline diagnostics
python diag_env_check.py
python diag_position_ids.py
```

### 2. Switch to problematic version
```bash
# Test with transformers 5.x (or other version)
pip install transformers==5.2.0

# Run diagnostics again
python diag_env_check.py
python diag_position_ids.py
```

### 3. Compare results
```bash
# Look for differences in output files:
# - layer_capture_metadata_*.json
# - sequence_trace_*.json
# - position_trace_*.json
```

---

## Phase 1: Environment Check (No GPU)

**Script:** `diag_env_check.py`

**Purpose:** Quickly identify environment and structural differences between transformers versions

**What it checks:**
- Transformers version and Python environment
- RoPE initialization functions available
- Attention implementation configuration
- Function signature changes
- Import availability

**Time:** ~5 minutes

**Output:** Console output comparing versions

**Success criteria:**
- RoPE 'default' type registered
- attn_implementation = "eager"
- No unexpected import errors

**Example:**
```
[OK] Transformers version: 4.57.6
[OK] ROPE_INIT_FUNCTIONS keys: ['linear', 'dynamic', 'yarn', 'longrope', 'llama3', 'default']
[OK] attn_implementation: eager
[OK] Model loads successfully
```

---

## Phase 2: Static Analysis (No GPU)

**Script:** `diag_static_analysis.py`

**Purpose:** Inspect source code and data types for incompatibilities

**What it checks:**
- Source code of critical functions (create_causal_mask, dynamic_rope_update)
- RoPE inv_freq precision (int64 vs float32 conversion)
- Decorator and parameter signatures
- Module structure

**Time:** ~10 minutes

**Output:** Prints source code and precision test results

**Success criteria:**
- inv_freq bit-identical between versions (0.00e+00 max difference)
- Function signatures unchanged
- No unexpected deprecations

**Example:**
```
[OK] inv_freq max difference: 0.00e+00 (bit-identical)
[OK] create_causal_mask signature: <signature compatible>
[OK] dynamic_rope_update decorator: <unchanged>
```

---

## Phase 3: RoPE Tensor Comparison (CPU or GPU)

**Script:** `diag_rope_tensors.py`

**Purpose:** Compare actual RoPE tensor computation between versions

**What it checks:**
- inv_freq tensor values and shapes
- cos/sin tensors for prefill steps
- cos/sin tensors for decode steps
- Numerical precision across layers

**Time:** ~30 minutes (requires model loading)

**Output:**
- `rope_*.pt` files (saved tensors)
- Console output with statistics

**Success criteria:**
- Tensors are bit-identical or within 1e-6 numerical error
- Shapes match between versions
- No NaN or Inf values

**Example:**
```
[OK] RoPE inv_freq: identical (4.57.6 vs 5.2.0)
[OK] cos tensor prefill: max_diff=0.0
[OK] sin tensor decode: max_diff=1e-7
```

---

## Phase 3b: Causal Mask Capture (GPU)

**Script:** `diag_causal_mask_capture.py`

**Purpose:** Trace causal mask creation to detect corruption at decode steps

**What it checks:**
- Mask shape and values at each generation step
- Position IDs passed to mask function
- Cache position values
- Mask stability during generation

**Time:** ~1 hour (GPU required)

**Output:**
- `mask_captures.json` (mask data by step)
- Console summary of findings

**Success criteria:**
- Masks are consistent between versions
- No extra -inf positions added in 5.x
- position_ids handling is identical

**Example:**
```
[OK] Step 0: mask shape=(1,1,8,8), no corruption
[OK] Step 1: mask shape=(1,1,9,9), cache_position=1
[OK] All steps: masks are identical
```

---

## Phase 3c: Position IDs Tracing (GPU)

**Script:** `diag_position_ids.py`

**Purpose:** Track cache_position, position_ids, and rope_deltas during generation

**What it checks:**
- cache_position values at each generation step
- position_ids tensor values
- rope_deltas computation
- Whether cache_position resets to 0 unexpectedly

**Time:** ~1-2 hours (GPU required, full generation)

**Output:**
- `position_trace.json` (per-step trace data)
- Console summary

**Success criteria:**
- cache_position increments correctly (0, 1, 2, ...)
- No unexpected resets to 0 during decode
- position_ids values are consistent
- rope_deltas are stable

**Example:**
```
[OK] Step 0: cache_position=0 (prefill)
[OK] Step 1: cache_position=1 (decode continues)
[OK] Step 2: cache_position=2 (no reset)
[OK] All steps: position_ids consistent
```

---

## Phase 4: Layer-by-Layer Analysis (GPU - Escalation)

Use these IF Phase 1-3 don't identify the root cause.

### 4a: Forward Hooks (GPU)

**Script:** `diag_first_step_hooks.py`

**Purpose:** Capture inputs/outputs of each decoder layer to identify where divergence occurs

**What it checks:**
- Layer input shapes and norms
- Layer output shapes and norms
- Hidden state progression through layers

**Time:** ~2 hours

**Output:**
- `captures_*.pt` (full tensors)
- `layer_capture_metadata_*.json` (statistics)

**Success criteria:**
- Layer outputs have stable norms
- No norm collapse (<0.5) or explosion (>1e4)
- Norms are similar between versions

**Compare:**
```python
import json
with open('layer_capture_metadata_4.57.6.json') as f:
    v4 = json.load(f)
with open('layer_capture_metadata_5.2.0.json') as f:
    v5 = json.load(f)

# First diverging layer indicates root cause location
for i, (l4, l5) in enumerate(zip(v4['layers'], v5['layers'])):
    if abs(l4['norm'] - l5['norm']) > 0.1:
        print(f"DIVERGENCE at layer {i}")
        break
```

### 4b: Sequence Trace (GPU)

**Script:** `diag_sequence_trace.py`

**Purpose:** Track generation metrics per step (silence probability, entropy, top tokens)

**What it checks:**
- Probability of silence token (token_0) at each step
- Logits entropy (collapsed if near 0.0)
- Top-5 predicted tokens
- Hidden state norms progression

**Time:** ~1-2 hours

**Output:**
- `sequence_trace_*.json` (per-step metrics)
- Console summary table

**Success criteria:**
- p(silence) < 20% normally (varies by text)
- Logits entropy > 1.0 (not collapsed)
- Top tokens vary (not locked to single token)

**Example:**
```
Step  p(token_0)  entropy  top_token  warnings
----  ----------  -------  ---------  --------
0     2.3%        5.421    201        [OK]
1     1.8%        5.267    156        [OK]
2     18.4%       4.892    201        [OK]
3     45.2%       2.143    0          High silence!
...
```

### 4c: Layer Divergence (GPU)

**Script:** `diag_layer_divergence.py`

**Purpose:** Direct layer output comparison without complex generation pipeline

**What it checks:**
- Layer norms across all 28 decoder layers
- Statistical comparison between versions
- First layer with significant difference

**Time:** ~30 minutes

**Output:**
- `layer_outputs_*.json` (per-layer statistics)
- Console comparison

**Success criteria:**
- Layer output norms are similar
- Mean norm < 0.01 difference between versions
- No spikes or collapses in norm progression

---

## Decision Tree

```
START: Suspected transformers compatibility issue
  |
  ├─> Run diag_env_check.py
  |     └─> Different signature? STOP - Incompatible API
  |
  ├─> Run diag_static_analysis.py
  |     └─> inv_freq not bit-identical?
  |         └─> CAUSE: RoPE precision issue (fix int64 conversion)
  |
  ├─> Run diag_rope_tensors.py
  |     └─> RoPE tensors differ?
  |         └─> CAUSE: RoPE computation changed
  |
  ├─> Run diag_causal_mask_capture.py
  |     └─> Masks differ?
  |         └─> CAUSE: Mask corruption in new version
  |
  ├─> Run diag_position_ids.py
  |     └─> cache_position resets or position_ids differ?
  |         └─> CAUSE: Cache/position handling changed
  |
  └─> ESCALATE to Phase 4:
       ├─> Run diag_first_step_hooks.py
       |    └─> First diverging layer identified?
       |        └─> Check if input or output diverges
       |
       ├─> Run diag_sequence_trace.py
       |    └─> Silence probability spikes?
       |        └─> CAUSE: Token generation logic changed
       |
       └─> Run diag_layer_divergence.py
            └─> Layer norms differ?
                └─> CAUSE: Fundamental computation change
                    (may require model retraining or version pinning)
```

---

## Troubleshooting Tips

### "Model loads but generation hangs"
- Run `diag_env_check.py` first
- Check if imports are failing silently
- Verify attn_implementation="eager" is set

### "Audio is mostly silence (85-99%)"
- Run full Phase 1-3 diagnostic suite
- Compare `sequence_trace_*.json` files
- Look for p(silence) spiking suddenly

### "Audio quality degraded but not silent"
- Run `diag_rope_tensors.py`
- Check if inv_freq precision changed
- May indicate numerical stability issue

### "Only decode steps have issues"
- Run `diag_position_ids.py`
- Look for cache_position reset to 0
- Check rope_deltas values

### "Some layers work, others don't"
- Run Phase 4 escalation scripts
- Use `diag_first_step_hooks.py` output
- Check first diverging layer

---

## Interpreting Results

### Good Sign ✓
- All phases pass with bit-identical results
- Layer norms stable and similar between versions
- No unexpected resets or mode switches

### Warning Sign ⚠️
- Small numerical differences (1e-5 to 1e-3)
- Layer norms vary slightly but trend same
- Entropy stable but slightly lower

### Critical Issue ✗
- Tensors completely different
- Layer norms diverge significantly
- Silence probability spikes to >50%
- Model locked onto single token

---

## Using Results with Different Transformers Versions

### Testing a New Version

```bash
# Current (tested) version
pip install transformers==4.57.6
python diag_env_check.py
cp layer_outputs_4.57.6.json baseline.json

# New version to test
pip install transformers==5.3.0
python diag_env_check.py
# Compare with baseline.json

# If issues found, escalate to Phase 4
python diag_first_step_hooks.py
python diag_sequence_trace.py
```

### Before Upgrading Transformers

```bash
# Save current good state
python diag_rope_tensors.py
cp rope_*.pt rope_baseline/

# Upgrade
pip install --upgrade transformers

# Verify
python diag_rope_tensors.py
# If rope_*.pt files differ significantly, revert upgrade
```

---

## Performance Notes

| Phase | Time | GPU Required | Data Size |
|-------|------|--------------|-----------|
| 1 | 5 min | No | ~1 MB |
| 2 | 10 min | No | ~5 MB |
| 3 | 30 min | No (CPU ok) | ~100 MB |
| 3b | 1 hour | Yes | ~500 MB |
| 3c | 1-2 hours | Yes | ~100 MB |
| 4a | 2 hours | Yes | ~1 GB |
| 4b | 1-2 hours | Yes | ~50 MB |
| 4c | 30 min | Yes | ~10 MB |

**Total:** ~4-8 hours for full investigation

---

## When to Use This Framework

✓ **Use these diagnostics for:**
- Investigating transformers version upgrades
- Debugging silence/quality issues
- Contributing to transformers compatibility
- Understanding model behavior across versions
- Creating regression tests

✗ **Don't use for:**
- Quick "does it work?" checks (use test_simple_compat_debug.py instead)
- Performance profiling (use different tools)
- Model training issues (different debugging needed)

---

## See Also

- `IMPLEMENTATION_NOTES.md` - How the fixes were implemented
- `COMPATIBILITY_TROUBLESHOOTING.md` - Common issues and solutions
- `ROOT_CAUSE_AND_FIX.md` - Transformers 5.x incompatibility findings
- `INVESTIGATION_COMPLETE_FINAL.md` - Complete investigation report

---

## Contributing

Found a new issue with a different transformers version? Run these diagnostics and share the results in an issue or PR. The diagnostic data helps the team understand compatibility patterns.

