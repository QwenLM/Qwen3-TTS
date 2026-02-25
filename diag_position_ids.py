#!/usr/bin/env python3
"""
PHASE 1, STEP 5: Position IDs and cache_position Tracing
Instruments the model to log cache_position, rope_deltas, and position_ids per step.
Runs in ~1 hour (requires GPU, uses test generation)

Output: Saves position_trace.json with per-step trace data
Decision: If cache_position resets to 0 during decode, identifies root cause.
"""

import sys
import os
import torch
import json
from typing import Any, Dict, List

# CRITICAL: Apply RoPE patch FIRST
from qwen_tts.core.rope_utils import patch_rope_init_functions
patch_rope_init_functions()

print("=" * 80)
print("PHASE 1, STEP 5: POSITION IDS AND CACHE_POSITION TRACING")
print("=" * 80)

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

try:
    import transformers
    print(f"\n[OK] Transformers version: {transformers.__version__}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OK] Using device: {device}")

    # Patch Qwen3TTSTalkerModel.forward to log position_ids info
    import qwen_tts.core.models.modeling_qwen3_tts as modeling_module

    original_talker_forward = modeling_module.Qwen3TTSTalkerModel.forward
    forward_calls = []

    def patched_talker_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        cache_position=None,
        past_key_values=None,
        hidden_states=None,
        **kwargs,
    ):
        """Wrapper that logs position_ids and cache_position"""

        call_info = {
            "forward_call_idx": len(forward_calls),
            "position_ids_info": {},
            "cache_position_info": {},
            "input_info": {},
        }

        # Log position_ids
        if position_ids is not None:
            call_info["position_ids_info"] = {
                "shape": str(position_ids.shape),
                "dtype": str(position_ids.dtype),
                "min": position_ids.min().item(),
                "max": position_ids.max().item(),
                "sample": position_ids.flatten()[:10].cpu().tolist(),
            }
        else:
            call_info["position_ids_info"]["value"] = "None"

        # Log cache_position
        if cache_position is not None:
            call_info["cache_position_info"] = {
                "shape": str(cache_position.shape),
                "dtype": str(cache_position.dtype),
                "value": cache_position.flatten().cpu().tolist(),
                "is_reset_to_zero": cache_position[0].item() == 0,
            }
        else:
            call_info["cache_position_info"]["value"] = "None"

        # Log input shapes
        if input_ids is not None:
            call_info["input_info"]["input_ids_shape"] = str(input_ids.shape)
        if hidden_states is not None:
            call_info["input_info"]["hidden_states_shape"] = str(hidden_states.shape)

        # Log rope_deltas if it exists
        if hasattr(self, "rope_deltas"):
            rope_deltas = self.rope_deltas
            call_info["rope_deltas"] = {
                "shape": str(rope_deltas.shape),
                "value": rope_deltas.flatten()[:5].cpu().tolist(),
            }

        forward_calls.append(call_info)

        # Call original
        return original_talker_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            **kwargs,
        )

    modeling_module.Qwen3TTSTalkerModel.forward = patched_talker_forward
    print("[OK] Patched Qwen3TTSTalkerModel.forward")

    # Also patch the code predictor to trace its position_ids
    original_code_pred_forward = modeling_module.Qwen3TTSTalkerCodePredictorModel.forward
    code_pred_calls = []

    def patched_code_pred_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        cache_position=None,
        past_key_values=None,
        **kwargs,
    ):
        """Wrapper that logs code predictor position_ids"""

        if cache_position is not None and cache_position[0].item() == 0:
            code_pred_calls.append({
                "cache_position_reset": True,
                "position_ids_present": position_ids is not None,
            })

        return original_code_pred_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            **kwargs,
        )

    modeling_module.Qwen3TTSTalkerCodePredictorModel.forward = patched_code_pred_forward
    print("[OK] Patched Qwen3TTSTalkerCodePredictorModel.forward")

    # Load model
    print("\nLoading model...")
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSForConditionalGeneration

    try:
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            "QwenLM/Qwen3-TTS-1B",
            attn_implementation="eager",
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        model.eval()
        print("[OK] Model loaded")

        # Minimal generation
        text = "Hi"
        speaker_name = "default"

        print(f"\nGenerating audio for: '{text}' (max 5 tokens for trace)...")
        with torch.no_grad():
            try:
                audio = model.generate(
                    text=text,
                    speaker_name=speaker_name,
                    max_new_tokens=5,
                )
                print(f"[OK] Generation succeeded, audio shape: {audio.shape}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARN] OOM during generation, but trace was captured")
                else:
                    raise

    except Exception as e:
        print(f"[WARN] Could not run full generation: {e}")
        print("  Trace may be incomplete, but will show import-time calls")

    # Save trace
    print("\n" + "=" * 80)
    print("TRACE RESULTS")
    print("=" * 80)

    print(f"\nQwen3TTSTalkerModel.forward calls: {len(forward_calls)}")
    for i, call in enumerate(forward_calls):
        cache_pos = call["cache_position_info"]
        is_reset = cache_pos.get("is_reset_to_zero", False)
        marker = "[WARN]" if is_reset else "[OK]"
        print(f"  {marker} Call {i}: cache_position={cache_pos.get('value', 'None')}")

    print(f"\nQwen3TTSTalkerCodePredictorModel.forward calls with cache_position reset: {len(code_pred_calls)}")
    for i, call in enumerate(code_pred_calls):
        print(f"  Call {i}: {call}")

    # Check for critical issue
    reset_count = sum(1 for c in code_pred_calls if c.get("cache_position_reset"))
    if reset_count > 0:
        print(f"\n[CRITICAL] cache_position was reset {reset_count} times")
        print("  This triggers prefill branch during decode, corrupting position_ids!")

    # Save full trace
    trace_data = {
        "transformers_version": transformers.__version__,
        "talker_model_calls": forward_calls,
        "code_predictor_resets": code_pred_calls,
        "total_forward_calls": len(forward_calls),
        "cache_position_reset_count": reset_count,
    }

    with open("position_trace.json", "w") as f:
        json.dump(trace_data, f, indent=2)
    print(f"\n[OK] Saved trace to position_trace.json")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if len(forward_calls) > 0:
        # Check if cache_position starts at 0
        first_cache_pos = forward_calls[0]["cache_position_info"].get("value")
        if first_cache_pos == [0]:
            print("[OK] Prefill step: cache_position starts at 0 (expected)")

        # Check if cache_position resets to 0 during decode
        for i, call in enumerate(forward_calls[1:], 1):
            cache_pos = call["cache_position_info"].get("value")
            if cache_pos == [0]:
                print(f"[WARN] cache_position reset to 0 at step {i} (should increment!)")
                print(f"  This causes prefill branch to re-enter, corrupting position_ids!")
                break
        else:
            print("[OK] cache_position increments correctly during decode")

    print("""
To compare with transformers 5.x:
1. Save this file: cp position_trace.json position_trace_4x.json
2. Switch to 5.x, run this script again
3. Compare traces: diff position_trace_4x.json position_trace.json

Key differences to look for:
- cache_position resets during decode steps
- position_ids appearing unexpectedly
- rope_deltas values changing

Next step: Run diag_first_step_hooks.py if above scripts don't identify root cause
""")

except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
