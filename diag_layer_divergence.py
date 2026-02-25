#!/usr/bin/env python3
"""
PHASE 4 SIMPLIFIED: Direct Layer Output Comparison
Captures layer outputs by directly calling the talker model's forward pass.
Compares outputs between transformers versions to identify divergence.

Output: JSON file with layer-by-layer norms and statistics
Decision: First diverging layer identifies root cause location
"""

import sys
import os
import torch
import json
from collections import defaultdict

# CRITICAL: Apply RoPE patch FIRST
from qwen_tts.core.rope_utils import patch_rope_init_functions
patch_rope_init_functions()

print("=" * 80)
print("PHASE 4 SIMPLIFIED: LAYER OUTPUT COMPARISON")
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

    inference_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    print("[OK] Model loaded")

    # Navigate to the talker model
    talker_model = inference_model.model.talker.model
    print(f"[OK] Talker model has {len(talker_model.layers)} layers")

    # Create dummy input tensors
    batch_size = 1
    seq_len = 8
    hidden_dim = talker_model.config.hidden_size

    print(f"\n[OK] Creating dummy inputs (batch={batch_size}, seq_len={seq_len}, hidden={hidden_dim})")

    # Create test input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)

    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  hidden_states shape: {hidden_states.shape}")
    print(f"  attention_mask shape: {attention_mask.shape}")

    # Capture layer outputs
    layer_captures = {}
    layer_norms = []

    print(f"\nCapturing layer outputs...")

    with torch.no_grad():
        # Forward pass through each layer manually
        current_hidden = hidden_states.clone()

        for layer_idx, layer in enumerate(talker_model.layers):
            try:
                # Simple forward pass through layer
                outputs = layer(
                    current_hidden,
                    attention_mask=attention_mask,
                    output_attentions=False,
                )

                # Extract output hidden state
                if isinstance(outputs, tuple):
                    next_hidden = outputs[0]
                else:
                    next_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs

                # Compute statistics
                norm = float(next_hidden.norm().item())
                mean = float(next_hidden.mean().item())
                std = float(next_hidden.std().item())

                layer_norms.append({
                    "layer": layer_idx,
                    "norm": norm,
                    "mean": mean,
                    "std": std,
                    "shape": str(next_hidden.shape),
                    "dtype": str(next_hidden.dtype),
                })

                if layer_idx % 5 == 0 or layer_idx == len(talker_model.layers) - 1:
                    print(f"  Layer {layer_idx:2d}: norm={norm:.6f}, mean={mean:.6f}, std={std:.6f}")

                # Update hidden state for next layer
                current_hidden = next_hidden

            except Exception as e:
                print(f"  [ERROR] Layer {layer_idx}: {e}")
                layer_norms.append({
                    "layer": layer_idx,
                    "error": str(e),
                })

    # Save results
    print(f"\n[OK] Captured {len(layer_norms)} layers")

    results = {
        "transformers_version": transformers.__version__,
        "device": str(device),
        "model": "Qwen3-TTS-12Hz-1.7B-Base",
        "num_layers": len(layer_norms),
        "layers": layer_norms,
    }

    # Compute statistics across layers
    valid_norms = [l["norm"] for l in layer_norms if "norm" in l]
    if valid_norms:
        results["statistics"] = {
            "norm_min": min(valid_norms),
            "norm_max": max(valid_norms),
            "norm_mean": sum(valid_norms) / len(valid_norms),
        }
        print(f"\n[STATS] Layer norms:")
        print(f"  Min: {results['statistics']['norm_min']:.6f}")
        print(f"  Max: {results['statistics']['norm_max']:.6f}")
        print(f"  Mean: {results['statistics']['norm_mean']:.6f}")

    filename = f"layer_outputs_{transformers.__version__}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved to {filename}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Run this script on transformers 4.57.6:
   pip install "transformers>=4.36.0,<5.0.0"
   python diag_layer_divergence.py

2. Run on transformers 5.2.0:
   pip install transformers==5.2.0
   python diag_layer_divergence.py

3. Compare results:
   python -c "
import json
with open('layer_outputs_4.57.6.json') as f:
    v4 = json.load(f)
with open('layer_outputs_5.2.0.json') as f:
    v5 = json.load(f)

print('Transformers 4.57.6:')
print(f'  Mean norm: {v4[\"statistics\"][\"norm_mean\"]:.6f}')
print(f'  Range: {v4[\"statistics\"][\"norm_min\"]:.6f} - {v4[\"statistics\"][\"norm_max\"]:.6f}')

print('Transformers 5.2.0:')
print(f'  Mean norm: {v5[\"statistics\"][\"norm_mean\"]:.6f}')
print(f'  Range: {v5[\"statistics\"][\"norm_min\"]:.6f} - {v5[\"statistics\"][\"norm_max\"]:.6f}')

print()
print('Comparing layer outputs:')
for i, (l4, l5) in enumerate(zip(v4['layers'], v5['layers'])):
    if 'norm' in l4 and 'norm' in l5:
        diff = abs(l4['norm'] - l5['norm'])
        if diff > 0.01:
            print(f'  Layer {i}: 4.x={l4[\"norm\"]:.6f}, 5.x={l5[\"norm\"]:.6f}, diff={diff:.6f} <<<')
   "
""")

except Exception as e:
    print(f"\n[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
