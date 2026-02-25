#!/usr/bin/env python3
"""
PHASE 4, STEP 7: Per-Step Sequence Trace and Metrics
Tracks generation metrics at each step to identify where silence occurs.
Measures: hidden state norms, logit entropy, silence token probability, top tokens.

Runs in ~1-2 hours (requires GPU, full generation)
Output: Saves sequence_trace_*.json with per-step metrics
Decision: Where p(silence) spikes or logits collapse identifies root cause timing
"""

import sys
import os
import torch
import json
from typing import Any, Dict, List
from scipy.special import softmax
import numpy as np

# CRITICAL: Apply RoPE patch FIRST
from qwen_tts.core.rope_utils import patch_rope_init_functions
patch_rope_init_functions()

print("=" * 80)
print("PHASE 4, STEP 7: PER-STEP SEQUENCE TRACE AND METRICS")
print("=" * 80)

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

try:
    import transformers
    print(f"\n[OK] Transformers version: {transformers.__version__}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OK] Using device: {device}")

    # Load model
    print("\nLoading model...")
    from qwen_tts import Qwen3TTSModel

    try:
        inference_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        print("[OK] Inference model loaded")

        # Navigate to the actual talker model
        talker_model = inference_model.model.talker.model
        print(f"[OK] Talker model ready")

        # Patch talker forward to capture per-step metrics
        original_talker_forward = talker_model.forward
        generation_steps = []
        step_counter = [0]  # Use list to allow modification in nested function

        def patched_talker_forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            cache_position=None,
            past_key_values=None,
            hidden_states=None,
            output_hidden_states=False,
            **kwargs,
        ):
            """Wrapper that logs per-step metrics"""

            step_idx = step_counter[0]
            step_counter[0] += 1

            # Call original
            outputs = original_talker_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )

            # Extract logits and hidden states
            logits = outputs.logits if hasattr(outputs, "logits") else None
            last_hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else None

            step_info = {
                "step_idx": step_idx,
                "input_ids_shape": str(input_ids.shape) if input_ids is not None else "None",
                "cache_position_value": (
                    cache_position.tolist() if cache_position is not None else None
                ),
            }

            # Log hidden state norm
            if last_hidden is not None:
                step_info["hidden_state_norm"] = float(last_hidden.norm().item())
                step_info["hidden_state_mean"] = float(last_hidden.mean().item())
                step_info["hidden_state_std"] = float(last_hidden.std().item())

            # Log logits
            if logits is not None:
                logits_float = logits.float()

                # Entropy
                logits_np = logits_float.detach().cpu().numpy().flatten()
                probs = softmax(logits_np)
                entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-10)))
                step_info["logits_entropy"] = float(entropy)

                # Top-5 tokens and probs
                top_5_indices = np.argsort(logits_np)[-5:][::-1]
                top_5_probs = probs[top_5_indices]
                step_info["top_5_tokens"] = top_5_indices.tolist()
                step_info["top_5_probs"] = top_5_probs.tolist()

                # Silence token (token 0) probability
                p_silence = float(probs[0])
                step_info["p_token_0"] = p_silence
                step_info["p_token_0_pct"] = p_silence * 100

                # Flag if high silence probability
                if p_silence > 0.5:
                    step_info["WARN"] = "High silence probability (>50%)"
                elif p_silence > 0.2:
                    step_info["WARN"] = "Moderate silence probability (>20%)"

            generation_steps.append(step_info)
            return outputs

        talker_model.forward = patched_talker_forward
        print("[OK] Patched talker forward for per-step logging")

        # Run generation
        text = "Hello world, this is a test"  # Longer text for more steps
        language = "English"

        print(f"\nGenerating audio for: '{text}'...")
        with torch.no_grad():
            try:
                wavs, sr = inference_model.generate_defaults(
                    text=text,
                    language=language,
                    speaker="default",
                )
                audio = wavs[0] if len(wavs) > 0 else None
                if audio is not None:
                    print(f"[OK] Generation succeeded, audio shape: {audio.shape}")
                else:
                    print("[WARN] No audio generated")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARN] OOM during generation, but trace was captured")
                else:
                    raise
            except Exception as e:
                print(f"[WARN] Error during generation: {e}")
                print("  Proceeding with captured steps...")

    except Exception as e:
        print(f"[WARN] Could not run full generation: {e}")
        print("  Trace may be incomplete")

    # Analyze trace
    print("\n" + "=" * 80)
    print("SEQUENCE TRACE SUMMARY")
    print("=" * 80)

    print(f"\nCaptured {len(generation_steps)} generation steps")

    # Print per-step summary
    print("\nPer-step metrics:")
    print("-" * 80)
    print(f"{'Step':<6} {'p(token_0)':<12} {'entropy':<10} {'top_token':<10} {'warnings':<30}")
    print("-" * 80)

    high_silence_count = 0
    for step in generation_steps:
        step_idx = step.get("step_idx", "?")
        p_silence = step.get("p_token_0_pct", "?")
        entropy = step.get("logits_entropy", "?")
        top_token = (
            step["top_5_tokens"][0] if "top_5_tokens" in step else "?"
        )
        warn = step.get("WARN", "")

        if isinstance(p_silence, float):
            p_silence_str = f"{p_silence:.1f}%"
        else:
            p_silence_str = str(p_silence)

        if isinstance(entropy, float):
            entropy_str = f"{entropy:.3f}"
        else:
            entropy_str = str(entropy)

        print(
            f"{step_idx:<6} {p_silence_str:<12} {entropy_str:<10} {str(top_token):<10} {warn:<30}"
        )

        if "High silence" in warn or "Moderate silence" in warn:
            high_silence_count += 1

    # Summary statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)

    if generation_steps:
        p_silences = [s.get("p_token_0_pct", 0) for s in generation_steps if "p_token_0_pct" in s]
        if p_silences:
            print(f"\nSilence token (token_0) probability:")
            print(f"  Min: {min(p_silences):.1f}%")
            print(f"  Max: {max(p_silences):.1f}%")
            print(f"  Mean: {np.mean(p_silences):.1f}%")
            print(f"  Steps with >50% silence: {sum(1 for p in p_silences if p > 50)}")
            print(f"  Steps with >20% silence: {sum(1 for p in p_silences if p > 20)}")

        entropies = [s.get("logits_entropy", 0) for s in generation_steps if "logits_entropy" in s]
        if entropies:
            print(f"\nLogits entropy:")
            print(f"  Min: {min(entropies):.4f}")
            print(f"  Max: {max(entropies):.4f}")
            print(f"  Mean: {np.mean(entropies):.4f}")
            print(f"  (Near 0.0 = model locked to single token, >5.0 = normal)")

        norms = [s.get("hidden_state_norm", 0) for s in generation_steps if "hidden_state_norm" in s]
        if norms:
            print(f"\nHidden state norms:")
            print(f"  Min: {min(norms):.4f}")
            print(f"  Max: {max(norms):.4f}")
            print(f"  Mean: {np.mean(norms):.4f}")
            print(f"  (Collapse <0.5 or explosion >1e4 = unstable)")

    # Save trace
    print("\n" + "=" * 80)
    print("SAVING TRACE")
    print("=" * 80)

    trace_data = {
        "transformers_version": transformers.__version__,
        "num_steps": len(generation_steps),
        "steps": generation_steps,
        "summary": {
            "high_silence_steps": high_silence_count,
            "total_steps": len(generation_steps),
        },
    }

    with open(f"sequence_trace_{transformers.__version__}.json", "w") as f:
        json.dump(trace_data, f, indent=2)
    print(f"[OK] Saved trace to sequence_trace_{transformers.__version__}.json")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)

    print("""
To compare with other transformers versions:

1. Save this run's trace:
   cp sequence_trace_*.json sequence_trace_latest.json

2. Switch to other transformers version, run this script again

3. Compare traces:
   python -c "
import json
with open('sequence_trace_4.57.6.json') as f:
    trace_4x = json.load(f)
with open('sequence_trace_5.2.0.json') as f:
    trace_5x = json.load(f)

print('Transformers 4.x:')
print(f'  Steps: {trace_4x[\"num_steps\"]}')
print(f'  High silence steps: {trace_4x[\"summary\"][\"high_silence_steps\"]}')

print('Transformers 5.x:')
print(f'  Steps: {trace_5x[\"num_steps\"]}')
print(f'  High silence steps: {trace_5x[\"summary\"][\"high_silence_steps\"]}')
   "

4. Look for:
   - Divergence in p(token_0) between versions at specific steps
   - Logits entropy collapse at different points
   - Hidden state norm differences

5. Match this trace with diag_first_step_hooks.py output to pinpoint the exact
   layer where divergence occurs.
""")

except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
