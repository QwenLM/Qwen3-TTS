#!/usr/bin/env python3
"""
PHASE 1, STEP 4: Causal Mask Capture
Monkey-patches create_causal_mask to capture inputs/outputs during generation.
Runs in ~1 hour (requires GPU, uses test generation)

Output: Saves mask_captures.json with all captured masks
Decision: If masks differ between 4.x and 5.x, identifies the problem step.
"""

import sys
import os
import torch
import json

# CRITICAL: Apply RoPE patch FIRST
from qwen_tts.core.rope_utils import patch_rope_init_functions
patch_rope_init_functions()

print("=" * 80)
print("PHASE 1, STEP 4: CAUSAL MASK CAPTURE")
print("=" * 80)

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

try:
    import transformers
    print(f"\n[OK] Transformers version: {transformers.__version__}")

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OK] Using device: {device}")

    # Import and patch create_causal_mask BEFORE loading models
    from transformers import masking_utils
    original_create_causal_mask = masking_utils.create_causal_mask

    captures = []
    call_count = 0

    def patched_create_causal_mask(**kwargs):
        """Wrapper that captures inputs and outputs"""
        global call_count
        call_count += 1

        # Capture input parameters
        capture_dict = {
            "call_idx": call_count,
            "input_shapes": {},
            "input_values": {},
        }

        # Log key parameters
        for key in ["target_length", "dtype", "device", "sliding_window"]:
            if key in kwargs:
                val = kwargs[key]
                if isinstance(val, torch.Tensor):
                    capture_dict["input_shapes"][key] = str(val.shape)
                else:
                    capture_dict["input_values"][key] = str(val)

        # Log position_ids if present
        if "position_ids" in kwargs:
            pos_ids = kwargs["position_ids"]
            capture_dict["position_ids"] = {
                "shape": str(pos_ids.shape),
                "dtype": str(pos_ids.dtype),
                "value": pos_ids.cpu().tolist() if pos_ids.numel() < 100 else "truncated",
            }

        # Call original function
        result = original_create_causal_mask(**kwargs)

        # Capture output
        if result is not None:
            capture_dict["output_shape"] = str(result.shape)
            capture_dict["output_dtype"] = str(result.dtype)
            capture_dict["output_sample"] = {
                "top_left": result[:min(4, result.shape[0]), :min(4, result.shape[1])].cpu().tolist(),
            }
            # Check for unexpected -inf values at decode positions
            if result.shape[0] > 1:
                last_row = result[-1, :]
                inf_count = (last_row == float("-inf")).sum().item()
                capture_dict["last_row_inf_count"] = inf_count

        captures.append(capture_dict)
        return result

    # Monkey-patch
    masking_utils.create_causal_mask = patched_create_causal_mask
    print("[OK] Patched create_causal_mask")

    # Now import and load model (uses patched version)
    print("\nLoading model (this will trigger mask creation)...")
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSForConditionalGeneration

    # Use a minimal test to trigger mask creation without full generation
    print("\n" + "-" * 80)
    print("Attempting minimal generation to capture masks")
    print("-" * 80)

    try:
        # Try to load from a local path or HF hub
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            "QwenLM/Qwen3-TTS-1B",
            attn_implementation="eager",  # Ensure we use eager to bypass SDPA
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        model.eval()
        print("[OK] Model loaded")

        # Create minimal input
        text = "Hello world"
        speaker_name = "default"

        # Try generation with a very small max_tokens to minimize runtime
        print(f"\nGenerating audio for: '{text}' (max 10 tokens for test)...")
        with torch.no_grad():
            try:
                audio = model.generate(
                    text=text,
                    speaker_name=speaker_name,
                    max_new_tokens=10,  # Minimal generation
                )
                print(f"[OK] Generation succeeded, audio shape: {audio.shape}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[WARN] OOM during generation (expected on small GPUs), but masks were captured")
                else:
                    raise

    except Exception as e:
        print(f"[WARN] Could not load full model: {e}")
        print("  This is expected if model weights are not available")
        print("  Mask capture will be based on import-time mask creation only")

    # Save captures
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nTotal create_causal_mask calls captured: {call_count}")

    for i, cap in enumerate(captures):
        print(f"\nCall {i + 1}:")
        print(f"  Target length: {cap['input_values'].get('target_length', 'unknown')}")
        print(f"  Sliding window: {cap['input_values'].get('sliding_window', 'false')}")
        if "position_ids" in cap:
            print(f"  Position IDs: {cap['position_ids']['shape']}")
        print(f"  Output shape: {cap.get('output_shape', 'None')}")
        if "last_row_inf_count" in cap:
            print(f"  Last row -inf count: {cap['last_row_inf_count']}")

    # Save full capture data
    with open("mask_captures.json", "w") as f:
        json.dump(captures, f, indent=2)
    print(f"\n[OK] Saved detailed captures to mask_captures.json")

    # Summary checks
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    unexpected_position_ids = sum(1 for c in captures if "position_ids" in c)
    if unexpected_position_ids > 0:
        print(f"[WARN] {unexpected_position_ids} calls have position_ids")
        print("  This is unexpected for transformers 4.x")
        print("  In 5.x, this may cause mask corruption during decode")

    print("""
To compare with transformers 5.x:
1. Save this file: cp mask_captures.json mask_captures_4x.json
2. Switch to 5.x, run this script again
3. Compare the JSON: diff mask_captures_4x.json mask_captures.json

Key differences to look for:
- Different number of calls
- position_ids appearing unexpectedly
- Output shapes differing
- Unexpected -inf values in decode steps

Next step: Run diag_position_ids.py to trace cache_position behavior
""")

except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
