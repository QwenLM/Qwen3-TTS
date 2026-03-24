#!/usr/bin/env python3
"""Export reference values from Python implementation for Rust validation.

This script loads the real model and exports intermediate tensor values
at key checkpoints so we can validate our Rust implementation.
"""

import json
import numpy as np
import torch
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def export_reference_values():
    """Export reference values for validation."""

    device = torch.device("cpu")
    output_dir = Path("test_data/reference_values")
    output_dir.mkdir(exist_ok=True)

    # Load model weights
    print("Loading model weights...")
    from safetensors.torch import load_file
    weights = load_file("test_data/model/model.safetensors")

    # Convert to float32 for consistency
    weights = {k: v.float() for k, v in weights.items()}

    # Test input: "Hello"
    # Using known token IDs from our tokenizer
    input_ids = torch.tensor([[9707, 11, 419, 374, 264]], dtype=torch.long)  # "Hello, this is a"
    batch_size, seq_len = input_ids.shape

    print(f"Input shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids.tolist()}")

    # ===== 1. Text Embedding =====
    print("\n=== Text Embedding ===")
    text_embed_weight = weights["talker.model.text_embedding.weight"]
    print(f"Text embedding weight shape: {text_embed_weight.shape}")

    text_embeddings = torch.nn.functional.embedding(input_ids, text_embed_weight)
    print(f"Text embeddings shape: {text_embeddings.shape}")

    save_tensor(output_dir / "text_embeddings.bin", text_embeddings)

    # ===== 2. Text Projection =====
    print("\n=== Text Projection ===")
    proj_fc1_w = weights["talker.text_projection.linear_fc1.weight"]
    proj_fc1_b = weights["talker.text_projection.linear_fc1.bias"]
    proj_fc2_w = weights["talker.text_projection.linear_fc2.weight"]
    proj_fc2_b = weights["talker.text_projection.linear_fc2.bias"]

    print(f"FC1 weight: {proj_fc1_w.shape}, FC2 weight: {proj_fc2_w.shape}")

    # Project: fc1 -> silu -> fc2
    hidden = torch.nn.functional.linear(text_embeddings, proj_fc1_w, proj_fc1_b)
    hidden = torch.nn.functional.silu(hidden)
    projected = torch.nn.functional.linear(hidden, proj_fc2_w, proj_fc2_b)

    print(f"Projected shape: {projected.shape}")
    save_tensor(output_dir / "projected.bin", projected)

    # ===== 3. RMS Norm (input_layernorm) =====
    print("\n=== RMS Norm ===")
    input_ln_weight = weights["talker.model.layers.0.input_layernorm.weight"]
    eps = 1e-6

    def rms_norm(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + eps)
        return weight * x_norm

    normed = rms_norm(projected, input_ln_weight, eps)
    print(f"Normed shape: {normed.shape}")
    save_tensor(output_dir / "after_input_ln.bin", normed)

    # ===== 4. Q, K, V Projections =====
    print("\n=== QKV Projections ===")
    q_proj_w = weights["talker.model.layers.0.self_attn.q_proj.weight"]
    k_proj_w = weights["talker.model.layers.0.self_attn.k_proj.weight"]
    v_proj_w = weights["talker.model.layers.0.self_attn.v_proj.weight"]
    q_norm_w = weights["talker.model.layers.0.self_attn.q_norm.weight"]
    k_norm_w = weights["talker.model.layers.0.self_attn.k_norm.weight"]

    # Config values from model
    hidden_size = 1024
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128

    # Q projection
    q = torch.nn.functional.linear(normed, q_proj_w)
    print(f"Q after proj: {q.shape}")

    # Reshape to (batch, seq, num_heads, head_dim)
    q = q.view(batch_size, seq_len, num_heads, head_dim)

    # Q norm (per-head RMS norm)
    q = rms_norm(q, q_norm_w, eps)

    # Transpose to (batch, num_heads, seq, head_dim)
    q = q.transpose(1, 2)
    print(f"Q final shape: {q.shape}")

    # K projection
    k = torch.nn.functional.linear(normed, k_proj_w)
    k = k.view(batch_size, seq_len, num_kv_heads, head_dim)
    k = rms_norm(k, k_norm_w, eps)
    k = k.transpose(1, 2)
    print(f"K final shape: {k.shape}")

    # V projection (no norm!)
    v = torch.nn.functional.linear(normed, v_proj_w)
    v = v.view(batch_size, seq_len, num_kv_heads, head_dim)
    v = v.transpose(1, 2)
    print(f"V final shape: {v.shape}")

    save_tensor(output_dir / "q_states.bin", q)
    save_tensor(output_dir / "k_states.bin", k)
    save_tensor(output_dir / "v_states.bin", v)

    # ===== 5. RoPE =====
    print("\n=== RoPE ===")
    # For simplicity, we'll compute basic RoPE without the multimodal sections
    # Since for pure text, all three position IDs are the same, standard RoPE works

    rope_theta = 1000000.0
    mrope_section = [24, 20, 20]  # Total = 64 dims per rotation (half of head_dim=128)

    # Compute rotary embeddings
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)  # [1, seq_len]

    # Create frequency matrix
    dim = head_dim
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # Compute sin/cos
    freqs = torch.outer(position_ids.float().squeeze(), inv_freq)  # [seq_len, dim/2]
    cos = freqs.cos()
    sin = freqs.sin()

    # For text-only, all 3 modality positions are the same
    # So we can stack the same cos/sin for each section
    cos_full = cos.repeat(1, 2)  # [seq_len, head_dim]
    sin_full = sin.repeat(1, 2)

    print(f"cos shape: {cos_full.shape}")

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Apply RoPE
    # cos/sin shape needs to be [1, 1, seq_len, head_dim] for broadcasting
    cos_emb = cos_full.unsqueeze(0).unsqueeze(0)
    sin_emb = sin_full.unsqueeze(0).unsqueeze(0)

    q_rot = (q * cos_emb) + (rotate_half(q) * sin_emb)
    k_rot = (k * cos_emb) + (rotate_half(k) * sin_emb)

    print(f"Q after RoPE: {q_rot.shape}")
    save_tensor(output_dir / "q_rope.bin", q_rot)
    save_tensor(output_dir / "k_rope.bin", k_rot)
    save_tensor(output_dir / "cos_emb.bin", cos_emb)
    save_tensor(output_dir / "sin_emb.bin", sin_emb)

    # ===== 6. Attention =====
    print("\n=== Attention ===")

    # Repeat KV for GQA
    def repeat_kv(x, n_rep):
        if n_rep == 1:
            return x
        batch, n_kv_heads, seq, hd = x.shape
        x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq, hd)
        return x.reshape(batch, n_kv_heads * n_rep, seq, hd)

    n_rep = num_heads // num_kv_heads
    k_expanded = repeat_kv(k_rot, n_rep)
    v_expanded = repeat_kv(v, n_rep)
    print(f"K expanded: {k_expanded.shape}")

    # Attention scores
    scaling = head_dim ** -0.5
    attn_weights = torch.matmul(q_rot, k_expanded.transpose(2, 3)) * scaling
    print(f"Attention weights shape: {attn_weights.shape}")

    # Causal mask
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    attn_weights = attn_weights + causal_mask

    # Softmax
    attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    print(f"Attention probs shape: {attn_probs.shape}")

    # Apply attention
    attn_output = torch.matmul(attn_probs, v_expanded)
    print(f"Attention output shape: {attn_output.shape}")

    save_tensor(output_dir / "attn_weights.bin", attn_weights)
    save_tensor(output_dir / "attn_probs.bin", attn_probs)
    save_tensor(output_dir / "attn_output.bin", attn_output)

    # ===== 7. O Projection =====
    print("\n=== O Projection ===")
    o_proj_w = weights["talker.model.layers.0.self_attn.o_proj.weight"]

    # Reshape: (batch, num_heads, seq, head_dim) -> (batch, seq, num_heads * head_dim)
    attn_output_flat = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
    print(f"Attention output flat shape: {attn_output_flat.shape}")

    attn_proj = torch.nn.functional.linear(attn_output_flat, o_proj_w)
    print(f"After O proj: {attn_proj.shape}")

    # Residual
    hidden_after_attn = projected + attn_proj
    print(f"After residual: {hidden_after_attn.shape}")

    save_tensor(output_dir / "attn_output_flat.bin", attn_output_flat)
    save_tensor(output_dir / "after_o_proj.bin", attn_proj)
    save_tensor(output_dir / "after_attn_residual.bin", hidden_after_attn)

    # ===== 8. MLP =====
    print("\n=== MLP ===")
    post_attn_ln_w = weights["talker.model.layers.0.post_attention_layernorm.weight"]
    gate_proj_w = weights["talker.model.layers.0.mlp.gate_proj.weight"]
    up_proj_w = weights["talker.model.layers.0.mlp.up_proj.weight"]
    down_proj_w = weights["talker.model.layers.0.mlp.down_proj.weight"]

    # Post-attention layer norm
    mlp_input = rms_norm(hidden_after_attn, post_attn_ln_w, eps)
    print(f"MLP input shape: {mlp_input.shape}")

    # SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
    gate = torch.nn.functional.linear(mlp_input, gate_proj_w)
    up = torch.nn.functional.linear(mlp_input, up_proj_w)
    mlp_hidden = torch.nn.functional.silu(gate) * up
    mlp_output = torch.nn.functional.linear(mlp_hidden, down_proj_w)

    print(f"MLP output shape: {mlp_output.shape}")

    # Residual
    layer_output = hidden_after_attn + mlp_output
    print(f"Layer output shape: {layer_output.shape}")

    save_tensor(output_dir / "mlp_input.bin", mlp_input)
    save_tensor(output_dir / "mlp_output.bin", mlp_output)
    save_tensor(output_dir / "layer_0_output.bin", layer_output)

    # ===== 9. Full Forward Pass Through All 28 Layers =====
    print("\n=== Full Forward Pass (28 layers) ===")

    num_layers = 28
    hidden = projected.clone()  # Start from projected text embeddings

    for layer_idx in range(num_layers):
        # Input LayerNorm
        input_ln_w = weights[f"talker.model.layers.{layer_idx}.input_layernorm.weight"]
        normed = rms_norm(hidden, input_ln_w, eps)

        # QKV projections
        q_proj_w = weights[f"talker.model.layers.{layer_idx}.self_attn.q_proj.weight"]
        k_proj_w = weights[f"talker.model.layers.{layer_idx}.self_attn.k_proj.weight"]
        v_proj_w = weights[f"talker.model.layers.{layer_idx}.self_attn.v_proj.weight"]
        q_norm_w = weights[f"talker.model.layers.{layer_idx}.self_attn.q_norm.weight"]
        k_norm_w = weights[f"talker.model.layers.{layer_idx}.self_attn.k_norm.weight"]

        q = torch.nn.functional.linear(normed, q_proj_w)
        q = q.view(batch_size, seq_len, num_heads, head_dim)
        q = rms_norm(q, q_norm_w, eps)
        q = q.transpose(1, 2)

        k = torch.nn.functional.linear(normed, k_proj_w)
        k = k.view(batch_size, seq_len, num_kv_heads, head_dim)
        k = rms_norm(k, k_norm_w, eps)
        k = k.transpose(1, 2)

        v = torch.nn.functional.linear(normed, v_proj_w)
        v = v.view(batch_size, seq_len, num_kv_heads, head_dim)
        v = v.transpose(1, 2)

        # RoPE
        q_rot = (q * cos_emb) + (rotate_half(q) * sin_emb)
        k_rot = (k * cos_emb) + (rotate_half(k) * sin_emb)

        # Attention
        k_expanded = repeat_kv(k_rot, n_rep)
        v_expanded = repeat_kv(v, n_rep)

        attn_weights = torch.matmul(q_rot, k_expanded.transpose(2, 3)) * scaling
        attn_weights = attn_weights + causal_mask
        attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_probs, v_expanded)

        # O projection
        o_proj_w = weights[f"talker.model.layers.{layer_idx}.self_attn.o_proj.weight"]
        attn_output_flat = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_proj = torch.nn.functional.linear(attn_output_flat, o_proj_w)

        # Residual
        hidden = hidden + attn_proj

        # MLP
        post_ln_w = weights[f"talker.model.layers.{layer_idx}.post_attention_layernorm.weight"]
        mlp_input = rms_norm(hidden, post_ln_w, eps)

        gate_w = weights[f"talker.model.layers.{layer_idx}.mlp.gate_proj.weight"]
        up_w = weights[f"talker.model.layers.{layer_idx}.mlp.up_proj.weight"]
        down_w = weights[f"talker.model.layers.{layer_idx}.mlp.down_proj.weight"]

        gate = torch.nn.functional.linear(mlp_input, gate_w)
        up = torch.nn.functional.linear(mlp_input, up_w)
        mlp_hidden = torch.nn.functional.silu(gate) * up
        mlp_output = torch.nn.functional.linear(mlp_hidden, down_w)

        # Residual
        hidden = hidden + mlp_output

        if layer_idx % 7 == 0 or layer_idx == num_layers - 1:
            print(f"  Layer {layer_idx}: mean={hidden.mean().item():.6f}")

    print(f"After all layers: {hidden.shape}")
    save_tensor(output_dir / "after_all_layers.bin", hidden)

    # ===== 10. Final Norm =====
    print("\n=== Final Norm ===")
    final_norm_w = weights["talker.model.norm.weight"]
    final_hidden = rms_norm(hidden, final_norm_w, eps)
    print(f"After final norm: {final_hidden.shape}, mean={final_hidden.mean().item():.6f}")
    save_tensor(output_dir / "after_final_norm.bin", final_hidden)

    # ===== 11. Codec Head =====
    print("\n=== Codec Head ===")
    codec_head_w = weights["talker.codec_head.weight"]
    print(f"Codec head weight: {codec_head_w.shape}")

    # The codec head projects to vocab_size (3072 = 3 * 1024 for 3 codebook groups)
    logits = torch.nn.functional.linear(final_hidden, codec_head_w)
    print(f"Logits shape: {logits.shape}")
    save_tensor(output_dir / "codec_logits.bin", logits)

    # Get predictions
    predictions = logits.argmax(dim=-1)
    print(f"Predictions: {predictions}")

    # ===== Save metadata =====
    metadata = {
        "input_ids": input_ids.tolist(),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "rope_theta": rope_theta,
        "eps": eps,
        "num_layers": num_layers,
        "predictions": predictions.tolist(),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n=== Saved reference values to {output_dir} ===")


def save_tensor(path, tensor):
    """Save tensor as raw float32 binary."""
    np.array(tensor.detach().cpu().numpy(), dtype=np.float32).tofile(path)
    print(f"  Saved {path.name}: {tensor.shape}")


if __name__ == "__main__":
    export_reference_values()
