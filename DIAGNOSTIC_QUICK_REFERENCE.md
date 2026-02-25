# Quick Reference: Root Cause Decision Tree

## One-Minute Summary

Qwen3 TTS generates 67-99% silence in transformers 5.2.0. The 5 Phase 1 diagnostics narrow down which of **5 top suspects** is the culprit.

### What to Run

```bash
# In transformers 4.57.3:
bash run_phase1_diagnostics.sh  # or .bat on Windows
# Archive baseline: mkdir rope_4x_backup && cp rope_*.pt rope_4x_backup/

# In transformers 5.2.0:
bash run_phase1_diagnostics.sh  # or .bat on Windows
# Compare results with baseline
```

---

## The 5 Top Suspects (Priority Order)

### 🔴 Suspect 1: cache_position Reset (Highest Probability)
**Detection:** Run `diag_position_ids.py`
- ✅ **Found if:** position_trace.json shows `cache_position == [0]` on steps > 1
- **Impact:** Re-enters prefill branch, corrupts position computation
- **Quick Fix:**
  ```python
  # In modeling_qwen3_tts.py, override:
  def _update_model_kwargs_for_generation(self, outputs, model_kwargs, ...):
      # Preserve cache_position to prevent reset
  ```
- **Verification:** silence_ratio drops below 15%

### 🟡 Suspect 2: create_causal_mask position_ids Corruption
**Detection:** Run `diag_causal_mask_capture.py`
- ✅ **Found if:** mask_captures.json shows `position_ids` field in 5.x but not 4.x
- **Impact:** Decode mask has wrong -inf positions, breaks attention
- **Quick Fix:**
  ```python
  # In modeling_qwen3_tts.py line ~1513:
  mask = create_causal_mask(
      target_length=...,
      dtype=...,
      device=...,
      # Don't pass position_ids - let it be None
  )
  ```
- **Verification:** silence_ratio drops below 15%

### 🟡 Suspect 3: _attn_implementation Auto-Select SDPA
**Detection:** Run `diag_env_check.py`
- ✅ **Found if:** Output shows `_attn_implementation = "sdpa"`
- **Impact:** SDPA math differs from local eager implementation
- **Quick Fix:**
  ```python
  # In inference/qwen3_tts_model.py, change:
  model = Qwen3TTSForConditionalGeneration.from_pretrained(
      model_id,
      attn_implementation="eager",  # Force eager, don't use SDPA
      ...
  )
  ```
- **Verification:** silence_ratio drops below 15%

### 🟡 Suspect 4: RoPE inv_freq Numerical Drift
**Detection:** Run `diag_rope_tensors.py`
- ✅ **Found if:** Output shows `Max difference: > 1e-7` in inv_freq comparison
- **Impact:** Rotary embeddings wrong, positions decoded incorrectly
- **Quick Fix:**
  ```python
  # In core/rope_utils.py line 47, change:
  inv_freq = 1.0 / (base ** (
      torch.arange(0, dim, 2, dtype=torch.float32)  # Use float32 directly
      / dim
  ))
  ```
- **Verification:** silence_ratio drops below 15%

### 🟡 Suspect 5: dynamic_rope_update Decorator Mutation
**Detection:** Run `diag_rope_tensors.py`
- ✅ **Found if:** rope_cos_*.pt and rope_sin_*.pt differ despite same inv_freq
- **Impact:** Decorator mutates inv_freq during inference
- **Quick Fix:** Investigate decorator behavior in transformers.modeling_rope_utils
- **Verification:** silence_ratio drops below 15%

---

## Diagnostic Script Outputs (Key Files)

| Script | Output Files | What to Look For |
|--------|--------------|------------------|
| diag_env_check.py | Console output | ⚠ warnings about sdpa, missing 'default', position_ids param |
| diag_static_analysis.py | Console output | Differences in source code or inv_freq precision |
| diag_rope_tensors.py | rope_*.pt files | tensor.pt file differences between 4.x and 5.x |
| diag_causal_mask_capture.py | mask_captures.json | position_ids field appearing (should be None or missing) |
| diag_position_ids.py | position_trace.json | cache_position values resetting to [0] during decode |

---

## Analysis Workflow

### Step 1: Run Baseline (Transformers 4.x)
```bash
# Verify it works
python test_simple_compat_debug.py  # Should show silence_ratio < 15%

# Run diagnostics
bash run_phase1_diagnostics.sh

# Archive results
mkdir rope_4x_backup
cp rope_*.pt rope_4x_backup/
cp mask_captures.json mask_captures_4x.json
cp position_trace.json position_trace_4x.json
```

### Step 2: Switch to 5.x and Run Again
```bash
pip install "transformers==5.2.0"
python test_simple_compat_debug.py  # Should show silence_ratio > 67%

bash run_phase1_diagnostics.sh

# Now compare outputs
```

### Step 3: Compare and Identify Root Cause

Use this checklist:

```
☐ diag_env_check.py
  ☐ Does 5.x show "_attn_implementation = sdpa"?
    YES -> Suspect 3 (SDPA)
  ☐ Does 5.x show "position_ids param: True"?
    YES -> Suspect 2 (create_causal_mask position_ids)

☐ diag_static_analysis.py
  ☐ Does inv_freq precision test show "max_diff > 1e-7"?
    YES -> Suspect 4 (numerical drift)

☐ diag_rope_tensors.py
  ☐ Compare rope_*.pt files:
    cmp rope_4x_backup/rope_inv_freq_computed.pt rope_inv_freq_computed.pt
    cmp rope_4x_backup/rope_cos_decode.pt rope_cos_decode.pt
  ☐ Do inv_freq tensors differ?
    YES -> Suspect 4 (RoPE numerical drift)
  ☐ Do cos/sin differ despite same inv_freq?
    YES -> Suspect 5 (dynamic_rope_update)

☐ diag_causal_mask_capture.py
  ☐ Compare JSON:
    diff mask_captures_4x.json mask_captures.json
  ☐ Does 5.x show position_ids unexpectedly?
    YES -> Suspect 2 (create_causal_mask position_ids)

☐ diag_position_ids.py
  ☐ Compare JSON:
    diff position_trace_4x.json position_trace.json
  ☐ Does 5.x show "cache_position": [0] on steps > 1?
    YES -> Suspect 1 (cache_position reset) ⚠ HIGHEST PRIORITY
```

---

## Testing the Fix

Once you identify and apply a fix:

```bash
# Install the version with the fix
pip install -e .

# Test it
python test_simple_compat_debug.py

# Expected: silence_ratio < 15% (was > 67%)
```

---

## If No Root Cause Found in Phase 1

Run Phase 2 (Layer-by-layer analysis):
- `diag_first_step_hooks.py` - Capture all layer outputs and binary-search divergence
- `diag_sequence_trace.py` - Track sequence-level metrics (entropy, token probs)

These are slower (~8-10 hours total) but will pinpoint exact layer.

---

## Key Insights

1. **Silence = Locked Attention:** If the model keeps outputting silence tokens, it's because attention is converging to the wrong position (same token every step).

2. **Position IDs = Everything:** The RoPE/position system is critical. Any reset, corruption, or drift here breaks the model.

3. **cache_position = State Machine:** If cache_position resets, the model thinks it's starting over (prefill), destroying autoregressive state.

4. **Numerical Precision Matters:** Even tiny diffs in inv_freq (< 1e-5) can accumulate through 32 layers.

---

## Common Fixes

### Fix for Suspect 1 (cache_position Reset)
**File:** `qwen_tts/core/models/modeling_qwen3_tts.py`
**Location:** `_update_model_kwargs_for_generation` method (~line 1805)
```python
def _update_model_kwargs_for_generation(self, outputs, model_kwargs, ...):
    # Make sure cache_position increments, doesn't reset to 0
    if "cache_position" in model_kwargs:
        model_kwargs["cache_position"] = model_kwargs["cache_position"] + 1
    return model_kwargs
```

### Fix for Suspect 2 (position_ids Corruption)
**File:** `qwen_tts/core/models/modeling_qwen3_tts.py`
**Location:** `Qwen3TTSTalkerModel.forward()` method (~line 1513)
```python
# Remove position_ids when calling create_causal_mask
mask = create_causal_mask(
    target_length=seq_length,
    dtype=hidden_states.dtype,
    device=hidden_states.device,
    # DO NOT pass position_ids
)
```

### Fix for Suspect 3 (SDPA Override)
**File:** `qwen_tts/inference/qwen3_tts_model.py`
**Location:** `from_pretrained()` call (~line 50-60)
```python
model = Qwen3TTSForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="eager",  # Force eager
    ...
)
```

### Fix for Suspect 4 (RoPE Numerical Drift)
**File:** `qwen_tts/core/rope_utils.py`
**Location:** `_compute_default_rope_parameters()` (~line 47)
```python
# Use float32 directly, not int64 conversion
inv_freq = 1.0 / (base ** (
    torch.arange(0, dim, 2, dtype=torch.float32).to(device=device)
    / dim
))
```

---

## Need Help?

1. **Check console output** for ⚠ warnings
2. **Read the .log files** in diag_results_* folder
3. **Compare JSON files** line-by-line
4. **Check DIAGNOSTIC_PLAN.md** for full workflow

Good luck! 🔍

