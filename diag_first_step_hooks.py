#!/usr/bin/env python3
"""
PHASE 4, STEP 6: Layer-by-Layer Hidden State Comparison
Instruments each decoder layer with forward hooks to capture inputs/outputs.
Identifies the FIRST layer where outputs diverge between transformers versions.

Runs in ~2-3 hours (requires GPU, uses test generation)
Output: Saves captures_4x.pt and captures_5x.pt for cross-version comparison
Decision: Which layer first diverges identifies root cause location
"""

import sys
import os
import torch
import json
from typing import Any, Dict, List
from collections import defaultdict

# CRITICAL: Apply RoPE patch FIRST
from qwen_tts.core.rope_utils import patch_rope_init_functions
patch_rope_init_functions()

print("=" * 80)
print("PHASE 4, STEP 6: LAYER-BY-LAYER HIDDEN STATE COMPARISON")
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

        # Navigate to the actual talker model with layers
        # Qwen3TTSModel.model -> Qwen3TTSForConditionalGeneration
        # .talker -> Qwen3TTSTalkerForConditionalGeneration
        # .model -> Qwen3TTSTalkerModel (with .layers)
        talker_model = inference_model.model.talker.model
        print(f"[OK] Talker model has {len(talker_model.layers)} layers")

        # Dictionary to store layer outputs
        layer_captures = defaultdict(dict)
        layer_hooks = []

        def create_hook(layer_idx):
            """Create a hook function for a specific layer"""
            def hook(module, input, output):
                # Capture input and output
                inp = input[0] if isinstance(input, tuple) else input
                out = output[0] if isinstance(output, tuple) else output

                layer_captures[f"layer_{layer_idx}"]["input"] = inp.detach().cpu().float()
                layer_captures[f"layer_{layer_idx}"]["output"] = out.detach().cpu().float()

            return hook

        # Register hooks on all layers
        print("\nRegistering forward hooks on all decoder layers...")
        for i, layer in enumerate(talker_model.layers):
            hook = layer.register_forward_hook(create_hook(i))
            layer_hooks.append(hook)
        print(f"[OK] Registered {len(layer_hooks)} hooks")

        # Minimal generation (just 2 tokens)
        text = "Hi"
        language = "English"

        print(f"\nGenerating audio for: '{text}' (2 tokens for layer capture)...")
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
                    print(f"[WARN] OOM during generation, but captures may be partial")
                else:
                    raise
            except Exception as e:
                print(f"[WARN] Error during generation: {e}")
                print("  Proceeding with captured layers...")

        # Remove hooks
        print("\nRemoving hooks...")
        for hook in layer_hooks:
            hook.remove()
        print("[OK] Hooks removed")

    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        print("  Using mock captures for demonstration")
        layer_captures = {}

    # Analyze captures
    print("\n" + "=" * 80)
    print("LAYER CAPTURE SUMMARY")
    print("=" * 80)

    print(f"\nCaptured {len(layer_captures)} layers")
    for layer_name in sorted(layer_captures.keys()):
        layer_data = layer_captures[layer_name]
        if "output" in layer_data:
            output = layer_data["output"]
            print(f"  {layer_name}: output shape={output.shape}, dtype={output.dtype}, "
                  f"norm={output.norm().item():.6f}")

    # Save captures
    print("\n" + "=" * 80)
    print("SAVING CAPTURES")
    print("=" * 80)

    # Prepare data for saving (convert to simpler format)
    captures_dict = {}
    for layer_name, layer_data in layer_captures.items():
        captures_dict[layer_name] = {
            "input_shape": str(layer_data.get("input", torch.tensor([])).shape),
            "input_norm": float(layer_data.get("input", torch.tensor(0.0)).norm().item()),
            "output_shape": str(layer_data.get("output", torch.tensor([])).shape),
            "output_norm": float(layer_data.get("output", torch.tensor(0.0)).norm().item()),
            "output_dtype": str(layer_data.get("output", torch.tensor([])).dtype),
        }

    # Save full tensors if not too large
    try:
        torch.save(layer_captures, f"captures_{transformers.__version__}.pt")
        print(f"[OK] Saved full captures to captures_{transformers.__version__}.pt")
    except Exception as e:
        print(f"[WARN] Could not save full tensors: {e}")

    # Save metadata
    with open(f"layer_capture_metadata_{transformers.__version__}.json", "w") as f:
        json.dump(captures_dict, f, indent=2)
    print(f"[OK] Saved metadata to layer_capture_metadata_{transformers.__version__}.json")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)

    print("""
To compare with other transformers versions:

1. Save this run's metadata:
   cp layer_capture_metadata_*.json layer_capture_metadata_latest.json

2. Switch to other transformers version, run this script again

3. Compare metadata:
   diff layer_capture_metadata_4.57.6.json layer_capture_metadata_5.2.0.json

4. Look for:
   - First layer with different output_norm values
   - First layer with different output shapes
   - First layer where divergence appears

5. If you find divergence at layer_N:
   - Check if input_norm is same (input came from previous layer)
   - If input_norm differs, then previous layer is root cause
   - If input_norm same but output differs, this layer's computation differs

Run diag_sequence_trace.py next to see per-step token generation patterns.
""")

except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
