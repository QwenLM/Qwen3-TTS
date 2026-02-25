# Qwen3 TTS Transformers Compatibility Troubleshooting Guide

## Quick Diagnosis

### Is Your Issue Here?

**A. Audio is mostly silence (85-99%)**
→ Go to: [Transformers 5.x Incompatibility](#transformers-5x-incompatibility)

**B. Model won't load (ImportError)**
→ Go to: [Import Errors](#import-errors)

**C. Generation hangs or times out**
→ Go to: [Generation Issues](#generation-issues)

**D. Audio quality is poor but not silent**
→ Go to: [Audio Quality Issues](#audio-quality-issues)

**E. Different results between runs**
→ Go to: [Reproducibility Issues](#reproducibility-issues)

---

## Transformers 5.x Incompatibility

### Symptom
```
Audio output is 85-99% silence, regardless of parameters
```

### Root Cause
Transformers 5.2.0+ has fundamental incompatible changes in:
- Attention mechanism computation
- Token logit generation
- Model inference behavior

This **cannot be fixed** with API-level patches and requires model retraining.

### Solution

#### Option 1: Use Transformers 4.x (Recommended)

```bash
# The project is already configured for this
pip install -e .

# Or explicitly:
pip install "transformers>=4.36.0,<5.0.0"

# Verify
python -c "import transformers; print(transformers.__version__)"
# Expected output: 4.57.6 or similar
```

**Result:** Perfect audio quality ✓

#### Option 2: Check Your Installation

```bash
# Check what's installed
pip show transformers | grep Version

# If it shows 5.x, reinstall:
pip uninstall transformers -y
pip install "transformers>=4.36.0,<5.0.0"

# If pyproject.toml conflict:
pip install -e . --force-reinstall
```

#### Option 3: Wait for Qwen Model Update

- Qwen may release a model trained for transformers 5.x
- When available, we can upgrade
- Check repo releases/notes regularly

### Verification Script

```bash
python test_simple_compat_debug.py
```

Expected output:
```
Test 1: Config A (0.8 temp)
  Silence ratio: 2.1%
  Max amplitude: 0.156
  Duration: 1.23s
  ✓ Success

Test 2: Config B (greedy)
  Silence ratio: 1.8%
  Max amplitude: 0.189
  Duration: 1.18s
  ✓ Success

Test 3: Config C (lower tokens)
  Silence ratio: 0.9%
  Max amplitude: 0.201
  Duration: 0.58s
  ✓ Success
```

If silence ratio > 50%: You have transformers 5.x installed (see Solution above)

---

## Import Errors

### Error: `ImportError: cannot import name 'check_model_inputs'`

**Cause:** Using transformers 5.2.0+, which removed this function

**Solution:**
```bash
pip install "transformers>=4.36.0,<5.0.0"
```

**Explanation:** The code was updated to remove the redundant decorator, but only works with transformers 4.x

**If it persists after reinstalling:**
```bash
pip uninstall transformers -y
pip cache purge
pip install "transformers>=4.36.0,<5.0.0"
```

---

### Error: `KeyError: 'default'` in ROPE_INIT_FUNCTIONS

**Cause:** RoPE patch not applied or applied too late

**Solution:**
```bash
# This should be automatic, but check:
python -c "
from qwen_tts.core.rope_utils import patch_rope_init_functions
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
patch_rope_init_functions()
print('default' in ROPE_INIT_FUNCTIONS)  # Should print True
"
```

**If it prints False:** Contact support with diagnostic output:
```bash
python diag_env_check.py > diagnostics.txt
# Share diagnostics.txt
```

---

### Error: `RuntimeError: RoPE 'default' type not registered!`

**Cause:** Assertion caught missing RoPE patch (this is good - early failure is better than silent failure)

**Solution:**
1. Check transformers version:
   ```bash
   pip show transformers | grep Version
   ```

2. If 5.x, downgrade to 4.x:
   ```bash
   pip install "transformers>=4.36.0,<5.0.0"
   ```

3. If 4.x, check patch is applied:
   ```bash
   python diag_env_check.py
   ```

4. If still fails, try:
   ```bash
   pip install -e . --force-reinstall
   ```

---

## Generation Issues

### Issue: Model loads but generation hangs

**Symptoms:**
- Model loads successfully
- `model.generate_defaults()` starts but never completes
- Process uses CPU/GPU but makes no progress

**Solution:**

1. **Check transformers version first:**
   ```bash
   pip show transformers | grep Version
   ```
   If 5.x → See [Transformers 5.x Incompatibility](#transformers-5x-incompatibility)

2. **Check available memory:**
   ```bash
   # GPU memory
   python -c "import torch; print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

   # System memory
   python -c "import psutil; print(f'System memory: {psutil.virtual_memory().total / 1e9:.1f} GB')"
   ```

   If low, try smaller batch:
   ```python
   model.generate_defaults(text="Hi", max_new_tokens=50)  # Shorter text
   ```

3. **Check if it's an OOM:**
   ```bash
   # Use system memory instead of GPU
   model = Qwen3TTSModel.from_pretrained(
       "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
       device_map="cpu",  # Force CPU
       dtype=torch.float32
   )
   ```

4. **Run diagnostic:**
   ```bash
   timeout 60 python diag_env_check.py
   # If it completes within 60s, import is fine
   ```

### Issue: Generation fails with CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions (in order of preferred):**

1. **Use smaller model (if available):**
   ```python
   model = Qwen3TTSModel.from_pretrained(
       "Qwen/Qwen3-TTS-12Hz-0.6B-Base",  # Smaller model
       device_map="cuda",
       dtype=torch.bfloat16
   )
   ```

2. **Use 8-bit quantization (if supported):**
   ```python
   model = Qwen3TTSModel.from_pretrained(
       "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
       device_map="cuda",
       load_in_8bit=True
   )
   ```

3. **Clear CUDA cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   torch.cuda.reset_peak_memory_stats()
   ```

4. **Use CPU instead:**
   ```python
   model = Qwen3TTSModel.from_pretrained(
       "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
       device_map="cpu",
       dtype=torch.float32  # Must be float32 for CPU
   )
   ```

5. **Reduce input length:**
   ```python
   # Instead of full text
   text = "Hello world, this is a test."

   # Use shorter text
   text = "Hello"

   wavs, sr = model.generate_defaults(text=text)
   ```

---

## Audio Quality Issues

### Issue: Audio is present but quality is poor

**Symptoms:**
- Audio isn't silent (silence ratio < 50%)
- But quality is noticeably degraded
- May sound robotic, choppy, or distorted

**Possible Causes:**

1. **Wrong model:**
   ```python
   # Check you're using the right model
   print(model.model)  # Should show Qwen3-TTS-1.7B
   ```

2. **Low precision (bfloat16 on CPU):**
   ```python
   # CPU must use float32
   model = Qwen3TTSModel.from_pretrained(
       ...,
       device_map="cpu",
       dtype=torch.float32  # ← Important
   )
   ```

3. **Generation parameters:**
   ```python
   # These affect quality - try defaults first
   wavs, sr = model.generate_defaults(text=text)

   # If using custom params, try temperature 0.7-0.8
   # Lower = more consistent but less natural
   # Higher = more varied but less accurate
   ```

4. **Audio encoding issue:**
   ```python
   import soundfile as sf
   import numpy as np

   # Check if audio is valid
   print(f"Max amplitude: {np.max(np.abs(audio))}")
   print(f"RMS: {np.sqrt(np.mean(audio**2))}")

   # Save to verify
   sf.write("test_output.wav", audio, sr)
   # Listen to test_output.wav
   ```

### Issue: Different audio output on repeated runs

**Symptoms:**
- Same text generates different audio each time
- Inconsistent quality or content

**Causes & Solutions:**

1. **Using random seed (expected behavior):**
   ```python
   # Set seed for reproducibility
   import torch
   torch.manual_seed(42)

   wavs1, sr = model.generate_defaults(text="Hi")
   torch.manual_seed(42)
   wavs2, sr = model.generate_defaults(text="Hi")

   # wavs1 and wavs2 should be identical
   ```

2. **Using do_sample=True (sampling is random):**
   ```python
   # For consistent output, use greedy decoding
   wavs, sr = model.generate_defaults(
       text="Hi",
       do_sample=False,  # Greedy (deterministic)
       temperature=1.0   # Ignored when do_sample=False
   )
   ```

3. **Using temperature > 0 with do_sample=True:**
   ```python
   # Lower temperature = more consistent
   wavs, sr = model.generate_defaults(
       text="Hi",
       do_sample=True,
       temperature=0.5  # Lower = more consistent (was 0.8)
   )
   ```

---

## Reproducibility Issues

### Making Generation Reproducible

```python
import torch
import numpy as np

# Set all seeds
torch.manual_seed(42)
np.random.seed(42)

# Load model
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda"
)

# Generate - will be reproducible
wavs, sr = model.generate_defaults(
    text="Hello world",
    do_sample=False,  # Greedy (deterministic)
)

# Same seed + same params = identical output
torch.manual_seed(42)
wavs2, sr2 = model.generate_defaults(
    text="Hello world",
    do_sample=False,
)
# wavs == wavs2 ✓
```

---

## Diagnostic Flowchart

```
Issue: Audio problem
  │
  ├─ Transformers version?
  │   ├─ 5.x? → Use 4.x (see section above)
  │   └─ 4.x? → Continue
  │
  ├─ Silence ratio?
  │   ├─ >85%? → Definitely transformers 5.x issue
  │   ├─ 50-85%? → Check generation parameters
  │   ├─ <50%? → Quality issue (below)
  │   └─ 0-10%? → Working correctly ✓
  │
  ├─ Generation speed?
  │   ├─ Hangs? → Check GPU/CPU memory
  │   ├─ OOM? → Use smaller model or GPU/CPU switch
  │   └─ Normal? → Continue
  │
  └─ Audio quality?
      ├─ Robotic/distorted? → Check precision/model
      ├─ Good content, poor voice? → Check model loading
      └─ Perfect? → Issue solved ✓
```

---

## Running Diagnostics

### Quick Diagnosis (5 minutes)

```bash
# Step 1: Check environment
python diag_env_check.py > env_check.txt

# Step 2: Check transformers details
python diag_static_analysis.py > static_analysis.txt

# Step 3: Quick test
python test_simple_compat_debug.py > test_output.txt

# Results:
# - If Config A, B, C all show low silence (<20%): ✓ Working
# - If all show >85% silence: transformers 5.x installed
# - If mixed results: Check test_output.txt for errors
```

### Full Diagnosis (30 minutes)

```bash
# All Phase 1-3 diagnostics
bash run_phase1_diagnostics.sh > full_diagnostics.log

# Outputs:
# - layer_capture_metadata_*.json
# - sequence_trace_*.json
# - position_trace_*.json

# Share full_diagnostics.log if you need support
```

### Deep Diagnosis (2+ hours)

```bash
# Phase 4 escalation (for transformers version issues)
python diag_first_step_hooks.py
python diag_sequence_trace.py
python diag_layer_divergence.py
```

---

## Getting Help

### Before Posting an Issue

1. **Run diagnostics:**
   ```bash
   python diag_env_check.py > diagnostics.txt
   python test_simple_compat_debug.py >> diagnostics.txt
   ```

2. **Check transformers version:**
   ```bash
   pip show transformers | grep Version >> diagnostics.txt
   ```

3. **Include in issue:**
   - Contents of `diagnostics.txt`
   - Python version: `python --version`
   - GPU info (if applicable): `nvidia-smi`
   - Your code snippet that fails

### Common Support Responses

**Q: Why doesn't transformers 5.x work?**
A: See [Transformers 5.x Incompatibility](#transformers-5x-incompatibility) - fundamental model incompatibility, not a bug.

**Q: Can you make it work with 5.x?**
A: Would require model retraining by Qwen team. Use 4.x or wait for Qwen to release a 5.x-trained model.

**Q: When will 5.x be supported?**
A: When Qwen releases a model trained for transformers 5.x. Check releases periodically.

**Q: My issue isn't here**
A: Run full diagnostics and create an issue with outputs attached.

---

## Quick Reference: Commands

```bash
# Check what you have
pip show transformers
python -c "import torch; print(torch.cuda.is_available())"

# Fix transformers 4.x
pip uninstall transformers -y
pip install "transformers>=4.36.0,<5.0.0"

# Test it works
python test_simple_compat_debug.py

# Reinstall if something is broken
pip install -e . --force-reinstall

# Run diagnostics
python diag_env_check.py
bash run_phase1_diagnostics.sh
```

---

## Summary Table

| Issue | Cause | Fix |
|-------|-------|-----|
| Silence (85-99%) | transformers 5.x | Downgrade to 4.x |
| ImportError check_model_inputs | transformers 5.x | Downgrade to 4.x |
| KeyError 'default' | Patch not applied | Reinstall with `pip install -e .` |
| Generation hangs | Low memory | Use smaller model or switch to CPU |
| CUDA OOM | Model too large | Use smaller model or 8-bit |
| Poor quality | Wrong dtype on CPU | Use float32 on CPU, bfloat16 on GPU |
| Non-reproducible | Random sampling | Use `do_sample=False` |

