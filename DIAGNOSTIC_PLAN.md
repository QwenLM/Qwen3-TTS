# Root Cause Analysis: Qwen3 TTS Transformers 5.x Silence Bug

## Quick Start

This directory contains 5 diagnostic scripts (Phase 1) designed to systematically identify why Qwen3 TTS generates 67-99% silence with transformers 5.2.0 but works perfectly with 4.57.3.

### Execution Order (Minimum Viable Investigation)

Run these scripts **in order** in both transformers 4.x and 5.x environments:

```bash
# Step 1: Environment Check (no GPU, ~5 minutes)
python diag_env_check.py

# Step 2: Static Analysis (no GPU, ~10 minutes)
python diag_static_analysis.py

# Step 3: RoPE Tensor Comparison (CPU ok, ~1 hour)
python diag_rope_tensors.py

# Step 4: Causal Mask Capture (requires GPU, ~1 hour)
python diag_causal_mask_capture.py

# Step 5: Position IDs Trace (requires GPU, ~1 hour)
python diag_position_ids.py
```

**Total estimated time: ~3.5 hours per environment (7 hours total for both)**

---

## What Each Script Does

### 1. `diag_env_check.py` ✅ Created
**Purpose:** Quick environment snapshot - captures structural differences
**Output:** Prints to console (check for ⚠ warnings)
**Key checks:**
- Transformers version
- ROPE_INIT_FUNCTIONS registration ('default' present?)
- create_causal_mask signature (has position_ids param?)
- dynamic_rope_update signature
- DynamicCache.update interface
- Qwen3TTS config (attn_implementation setting)
- ALL_ATTENTION_FUNCTIONS availability

**If found:** SDPA override, missing 'default' rope, unexpected position_ids param

---

### 2. `diag_static_analysis.py` ✅ Created
**Purpose:** Inspect source code without executing
**Output:** Prints source excerpts + precision test results
**Key checks:**
- create_causal_mask source (does it use position_ids?)
- dynamic_rope_update source (decorator behavior?)
- inv_freq numerical precision (int64 vs float32 arange)
- rope_utils implementation match

**If found:** Numerical drift, position_ids usage in mask, decorator mutations

---

### 3. `diag_rope_tensors.py` ✅ Created
**Purpose:** Extract and compare RoPE tensors across versions
**Output:** Saves `rope_*.pt` files
**Files created:**
- `rope_inv_freq_computed.pt` - from _compute_default_rope_parameters
- `rope_inv_freq_embedding.pt` - from Qwen3TTSTalkerRotaryEmbedding
- `rope_cos_prefill.pt`, `rope_sin_prefill.pt` - positions 0-63
- `rope_cos_decode.pt`, `rope_sin_decode.pt` - position 64
- `rope_cos_batch.pt`, `rope_sin_batch.pt` - batch test

**If found:** RoPE tensor divergence (identifies which component)

---

### 4. `diag_causal_mask_capture.py` ✅ Created
**Purpose:** Capture mask construction during generation
**Output:** `mask_captures.json` with all calls to create_causal_mask
**Key captures:**
- Input parameters (target_length, sliding_window, position_ids)
- Output shape and sample values
- position_ids usage (unexpected in 5.x?)
- -inf patterns in masks

**If found:** Mask corruption during decode steps

---

### 5. `diag_position_ids.py` ✅ Created
**Purpose:** Trace cache_position, rope_deltas, and position_ids per step
**Output:** `position_trace.json` with per-forward call trace
**Key traces:**
- cache_position values (should increment, not reset!)
- position_ids presence/values
- rope_deltas state
- Code predictor position tracking

**If found:** cache_position reset to 0 on decode steps (prefill re-entry!)

---

## Expected Root Causes (Priority Order)

### 1. 🔴 **cache_position Reset (Highest Probability)**
- **Symptom:** diag_position_ids.py shows `cache_position == 0` on steps > 1
- **Impact:** Re-enters prefill branch, corrupts rope_deltas and position computation
- **Fix:** Override `_update_model_kwargs_for_generation` to preserve cache_position
- **Evidence:** Silence tokens locked in -> same position = same hidden state = silence

### 2. 🟡 **create_causal_mask position_ids Corruption**
- **Symptom:** diag_causal_mask_capture.py shows unexpected position_ids in 5.x
- **Impact:** Decode mask has wrong -inf positions
- **Fix:** Pass `position_ids=None` explicitly in mask call
- **Evidence:** Wrong mask = wrong attention = garbage hidden states

### 3. 🟡 **_attn_implementation Auto-Select SDPA**
- **Symptom:** diag_env_check.py shows `_attn_implementation == "sdpa"` in 5.x
- **Impact:** SDPA may behave differently than local eager implementation
- **Fix:** Force `attn_implementation="eager"` in from_pretrained
- **Evidence:** Different attention math = different outputs

### 4. 🟡 **RoPE inv_freq Numerical Drift**
- **Symptom:** diag_rope_tensors.py shows max_diff > 1e-7 between 4.x and 5.x
- **Impact:** Rotary embeddings wrong -> positions decoded incorrectly
- **Fix:** Use `dtype=torch.float32` directly instead of int64 conversion
- **Evidence:** Accumulated error through all layers = bad embeddings

### 5. 🟡 **dynamic_rope_update Decorator Mutation**
- **Symptom:** diag_rope_tensors.py shows cos/sin differ despite same inv_freq
- **Impact:** Decorator modifies inv_freq during inference
- **Fix:** Investigate/neutralize decorator behavior
- **Evidence:** RoPE values wrong despite correct frequencies

---

## Workflow for Investigation

### Phase 1A: Transformers 4.x (Baseline)
```bash
# Confirm 4.x works first
python -c "import transformers; print(f'Version: {transformers.__version__}')"

# Run all 5 scripts
python diag_env_check.py
python diag_static_analysis.py
python diag_rope_tensors.py           # Creates rope_*.pt files
python diag_causal_mask_capture.py    # Creates mask_captures.json
python diag_position_ids.py           # Creates position_trace.json

# Archive baseline
mkdir rope_4x_backup
cp rope_*.pt rope_4x_backup/
cp mask_captures.json mask_captures_4x.json
cp position_trace.json position_trace_4x.json
```

### Phase 1B: Transformers 5.x (Compare)
```bash
# Switch to 5.x
pip install "transformers==5.2.0"  # or use the exact failing version

# Verify silence bug exists
python test_simple_compat_debug.py  # Should show silence_ratio > 67%

# Run all 5 scripts again
python diag_env_check.py
python diag_static_analysis.py
python diag_rope_tensors.py
python diag_causal_mask_capture.py
python diag_position_ids.py

# Compare outputs
diff <(python -c "import json; print(json.dumps(json.load(open('position_trace_4x.json')), indent=2))") \
     <(python -c "import json; print(json.dumps(json.load(open('position_trace.json')), indent=2))")
```

### Decision Tree

```
START: 67-99% silence in transformers 5.x
  |
  v
Is _attn_implementation "sdpa" in 5.x? (diag_env_check)
  YES -> ROOT CAUSE: SDPA replacing local eager attention
         FIX: attn_implementation="eager" in from_pretrained()
  NO  -> Continue
  |
  v
Does cache_position reset to 0 on decode steps? (diag_position_ids)
  YES -> ROOT CAUSE: GenerationMixin resets cache_position, re-enters prefill
         FIX: Override _update_model_kwargs_for_generation
  NO  -> Continue
  |
  v
Is position_ids unexpectedly passed to create_causal_mask? (diag_causal_mask_capture)
  YES -> ROOT CAUSE: position_ids corrupts decode mask
         FIX: Pass position_ids=None explicitly
  NO  -> Continue
  |
  v
Do inv_freq tensors differ? (diag_rope_tensors)
  YES -> ROOT CAUSE: RoPE numerical drift
         FIX: Change int64 arange to float32 in rope_utils.py
  NO  -> Do cos/sin differ despite same inv_freq?
    YES -> ROOT CAUSE: dynamic_rope_update mutates inv_freq
           FIX: Investigate/neutralize decorator
    NO  -> Continue (RoPE is fine)
  |
  v
Layer-by-layer analysis needed (Phase 2)
```

---

## After Root Cause Found

Once a root cause is identified:

1. **Implement fix** in the appropriate file
2. **Verify fix:**
   ```bash
   pip install "transformers==5.2.0"
   python test_simple_compat_debug.py
   ```
3. **Expected success:** silence_ratio < 15%
4. **Commit fix:**
   ```bash
   git add -A
   git commit -m "fix: [root cause title and brief description]"
   ```

---

## Files Modified (If Phase 2 Needed)

If Phase 1 doesn't identify root cause, Phase 2 will:

| Step | File | Module |
|------|------|--------|
| 6 | `diag_first_step_hooks.py` | Layer-by-layer hook comparison |
| 7 | `diag_sequence_trace.py` | Sequence-level metrics (entropy, token probs) |

These are more expensive (~8-10 hours total) but can pinpoint exact layer divergence.

---

## Key Files Under Investigation

| File | Role |
|------|------|
| `qwen_tts/core/models/modeling_qwen3_tts.py` | Mask creation (L1513), position_ids (L1696-1714), attn dispatch (L790-803), _update_model_kwargs_for_generation (L1805) |
| `qwen_tts/core/rope_utils.py` | RoPE patch - int64 arange on L47 is a suspect |
| `qwen_tts/core/models/configuration_qwen3_tts.py` | _attn_implementation defaults, rope_config_validation |
| `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` | If trace leads to decoder side |

---

## Notes

- **GPU strongly recommended** for Steps 4-5 (mask capture, position trace)
- **Each script saves output files** for cross-version comparison
- **Decision tree is not exhaustive** - other issues are possible
- **Time estimates** are conservative; actual time may vary

---

## Questions or Issues?

- Check output files (*.json, *.pt)
- Review print statements for ⚠ warnings
- Compare 4.x vs 5.x results line-by-line
- If stuck, run Phase 2 diagnostics

