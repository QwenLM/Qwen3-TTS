//! Validation tests comparing Rust implementation against Python reference values
//!
//! This file contains tests that load pre-computed reference values from the Python
//! implementation and verify our Rust implementation produces identical results.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

const REFERENCE_DIR: &str = "test_data/reference_values";
const MODEL_PATH: &str = "test_data/model/model.safetensors";

/// Load a reference tensor from binary file
fn load_reference(name: &str, shape: &[usize], device: &Device) -> Result<Tensor> {
    let path = Path::new(REFERENCE_DIR).join(name);
    let bytes = std::fs::read(&path)?;
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(Tensor::from_vec(floats, shape, device)?)
}

/// Load model weights
fn load_weights(device: &Device) -> Result<HashMap<String, Tensor>> {
    let tensors: HashMap<String, Tensor> =
        candle_core::safetensors::load(Path::new(MODEL_PATH), device)?;
    // Convert BF16 to F32
    let tensors: HashMap<String, Tensor> = tensors
        .into_iter()
        .map(|(name, tensor)| {
            let converted = if tensor.dtype() == DType::BF16 {
                tensor.to_dtype(DType::F32).unwrap()
            } else {
                tensor
            };
            (name, converted)
        })
        .collect();
    Ok(tensors)
}

/// Check if reference values exist
fn reference_available() -> bool {
    Path::new(REFERENCE_DIR).join("metadata.json").exists()
}

/// Compare two tensors with tolerance
fn tensors_close(a: &Tensor, b: &Tensor, rtol: f64, atol: f64) -> Result<bool> {
    let diff = (a - b)?.abs()?;
    let threshold = (b.abs()? * rtol)?.broadcast_add(&Tensor::new(&[atol as f32], a.device())?)?;
    // Check if all diff values are <= threshold
    // We compute (diff - threshold) and check if max <= 0
    let over = (diff - threshold)?;
    let max_over: f32 = over.flatten_all()?.max(0)?.to_scalar()?;
    Ok(max_over <= 0.0)
}

/// Print tensor comparison statistics
fn compare_tensors(name: &str, rust: &Tensor, python: &Tensor) -> Result<()> {
    let diff = (rust - python)?;
    let abs_diff = diff.abs()?;
    let max_diff: f32 = abs_diff.flatten_all()?.max(0)?.to_scalar()?;
    let mean_diff: f32 = abs_diff.flatten_all()?.mean_all()?.to_scalar()?;

    let rust_mean: f32 = rust.flatten_all()?.mean_all()?.to_scalar()?;
    let python_mean: f32 = python.flatten_all()?.mean_all()?.to_scalar()?;

    println!(
        "  {}: max_diff={:.6}, mean_diff={:.6}, rust_mean={:.6}, python_mean={:.6}",
        name, max_diff, mean_diff, rust_mean, python_mean
    );

    if max_diff > 1e-4 {
        println!("    WARNING: max_diff > 1e-4!");
    }

    Ok(())
}

/// Linear projection for 3D input tensors
/// Handles the case where x is [batch, seq, features] and weight is [out, in]
fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let dims = x.dims();
    if dims.len() == 3 {
        let (batch, seq, features) = (dims[0], dims[1], dims[2]);
        // Flatten to 2D: [batch * seq, features]
        let x_2d = x.reshape((batch * seq, features))?;
        // Matmul: [batch * seq, features] @ [features, out] = [batch * seq, out]
        let out_2d = x_2d.matmul(&weight.t()?)?;
        let out_features = out_2d.dim(1)?;
        // Reshape back to 3D: [batch, seq, out]
        let out_3d = out_2d.reshape((batch, seq, out_features))?;
        // Add bias if present
        match bias {
            Some(b) => Ok(out_3d.broadcast_add(b)?),
            None => Ok(out_3d),
        }
    } else {
        // 2D case: just matmul directly
        let out = x.matmul(&weight.t()?)?;
        match bias {
            Some(b) => Ok(out.broadcast_add(b)?),
            None => Ok(out),
        }
    }
}

/// RMS Norm implementation (matches Python exactly)
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    // variance = x.pow(2).mean(-1, keepdim=True)
    let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    // x_norm = x * torch.rsqrt(variance + eps)
    let x_norm = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
    // return weight * x_norm
    Ok(x_norm.broadcast_mul(weight)?)
}

/// Rotate half for RoPE
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(candle_core::D::Minus1)? / 2;
    let x1 = x.narrow(candle_core::D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(candle_core::D::Minus1, half_dim, half_dim)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)?)
}

/// Apply RoPE
fn apply_rope(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
    let q_rot = q.broadcast_mul(cos)?.broadcast_add(&rotate_half(q)?.broadcast_mul(sin)?)?;
    let k_rot = k.broadcast_mul(cos)?.broadcast_add(&rotate_half(k)?.broadcast_mul(sin)?)?;
    Ok((q_rot, k_rot))
}

/// Repeat KV for GQA
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (batch, n_kv_heads, seq_len, head_dim) = x.dims4()?;
    // x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq, hd)
    let x = x.unsqueeze(2)?;
    let x = x.expand((batch, n_kv_heads, n_rep, seq_len, head_dim))?;
    Ok(x.reshape((batch, n_kv_heads * n_rep, seq_len, head_dim))?)
}

// ============================================================================
// TESTS
// ============================================================================

#[test]
fn test_text_embedding() -> Result<()> {
    if !reference_available() {
        eprintln!("Reference values not found. Run: python3 tools/export_reference_values.py");
        return Ok(());
    }

    let device = Device::Cpu;
    let weights = load_weights(&device)?;

    println!("\n=== Text Embedding Validation ===");

    // Input: [9707, 11, 419, 374, 264] = "Hello, this is a"
    let input_ids = Tensor::new(&[9707u32, 11, 419, 374, 264], &device)?;

    // Get embedding weight
    let embed_weight = weights
        .get("talker.model.text_embedding.weight")
        .ok_or_else(|| anyhow::anyhow!("text_embedding not found"))?;

    // Look up embeddings
    let rust_embeddings = embed_weight.index_select(&input_ids, 0)?;
    let rust_embeddings = rust_embeddings.unsqueeze(0)?; // Add batch dim

    // Load Python reference
    let python_embeddings = load_reference("text_embeddings.bin", &[1, 5, 2048], &device)?;

    compare_tensors("text_embeddings", &rust_embeddings, &python_embeddings)?;

    assert!(tensors_close(&rust_embeddings, &python_embeddings, 1e-5, 1e-6)?);
    println!("  PASS!");

    Ok(())
}

#[test]
fn test_text_projection() -> Result<()> {
    if !reference_available() {
        return Ok(());
    }

    let device = Device::Cpu;
    let weights = load_weights(&device)?;

    println!("\n=== Text Projection Validation ===");

    // Load input (text embeddings from Python)
    let embeddings = load_reference("text_embeddings.bin", &[1, 5, 2048], &device)?;

    // Get projection weights
    let fc1_w = weights.get("talker.text_projection.linear_fc1.weight").unwrap();
    let fc1_b = weights.get("talker.text_projection.linear_fc1.bias").unwrap();
    let fc2_w = weights.get("talker.text_projection.linear_fc2.weight").unwrap();
    let fc2_b = weights.get("talker.text_projection.linear_fc2.bias").unwrap();

    // Project: fc1 -> silu -> fc2
    let hidden = linear(&embeddings, fc1_w, Some(fc1_b))?;
    let hidden = candle_nn::ops::silu(&hidden)?;
    let rust_projected = linear(&hidden, fc2_w, Some(fc2_b))?;

    // Load Python reference
    let python_projected = load_reference("projected.bin", &[1, 5, 1024], &device)?;

    compare_tensors("projected", &rust_projected, &python_projected)?;

    assert!(tensors_close(&rust_projected, &python_projected, 1e-5, 1e-6)?);
    println!("  PASS!");

    Ok(())
}

#[test]
fn test_rms_norm() -> Result<()> {
    if !reference_available() {
        return Ok(());
    }

    let device = Device::Cpu;
    let weights = load_weights(&device)?;

    println!("\n=== RMS Norm Validation ===");

    // Load input (projected from Python)
    let projected = load_reference("projected.bin", &[1, 5, 1024], &device)?;

    // Get layernorm weight
    let ln_weight = weights
        .get("talker.model.layers.0.input_layernorm.weight")
        .unwrap();

    // Apply RMS norm
    let rust_normed = rms_norm(&projected, ln_weight, 1e-6)?;

    // Load Python reference
    let python_normed = load_reference("after_input_ln.bin", &[1, 5, 1024], &device)?;

    compare_tensors("after_input_ln", &rust_normed, &python_normed)?;

    assert!(tensors_close(&rust_normed, &python_normed, 1e-5, 1e-6)?);
    println!("  PASS!");

    Ok(())
}

#[test]
fn test_qkv_projections() -> Result<()> {
    if !reference_available() {
        return Ok(());
    }

    let device = Device::Cpu;
    let weights = load_weights(&device)?;

    println!("\n=== QKV Projections Validation ===");

    let batch_size = 1usize;
    let seq_len = 5usize;
    let num_heads = 16usize;
    let num_kv_heads = 8usize;
    let head_dim = 128usize;

    // Load input (normed from Python)
    let normed = load_reference("after_input_ln.bin", &[1, 5, 1024], &device)?;

    // Get weights
    let q_proj_w = weights.get("talker.model.layers.0.self_attn.q_proj.weight").unwrap();
    let k_proj_w = weights.get("talker.model.layers.0.self_attn.k_proj.weight").unwrap();
    let v_proj_w = weights.get("talker.model.layers.0.self_attn.v_proj.weight").unwrap();
    let q_norm_w = weights.get("talker.model.layers.0.self_attn.q_norm.weight").unwrap();
    let k_norm_w = weights.get("talker.model.layers.0.self_attn.k_norm.weight").unwrap();

    // Q: proj -> reshape -> norm -> transpose
    let q = linear(&normed, q_proj_w, None)?;
    let q = q.reshape((batch_size, seq_len, num_heads, head_dim))?;
    let q = rms_norm(&q, q_norm_w, 1e-6)?;
    let rust_q = q.transpose(1, 2)?;

    // K: proj -> reshape -> norm -> transpose
    let k = linear(&normed, k_proj_w, None)?;
    let k = k.reshape((batch_size, seq_len, num_kv_heads, head_dim))?;
    let k = rms_norm(&k, k_norm_w, 1e-6)?;
    let rust_k = k.transpose(1, 2)?;

    // V: proj -> reshape -> transpose (NO norm!)
    let v = linear(&normed, v_proj_w, None)?;
    let v = v.reshape((batch_size, seq_len, num_kv_heads, head_dim))?;
    let rust_v = v.transpose(1, 2)?;

    // Load Python references
    let python_q = load_reference("q_states.bin", &[1, 16, 5, 128], &device)?;
    let python_k = load_reference("k_states.bin", &[1, 8, 5, 128], &device)?;
    let python_v = load_reference("v_states.bin", &[1, 8, 5, 128], &device)?;

    compare_tensors("q_states", &rust_q, &python_q)?;
    compare_tensors("k_states", &rust_k, &python_k)?;
    compare_tensors("v_states", &rust_v, &python_v)?;

    assert!(tensors_close(&rust_q, &python_q, 1e-4, 1e-5)?);
    assert!(tensors_close(&rust_k, &python_k, 1e-4, 1e-5)?);
    assert!(tensors_close(&rust_v, &python_v, 1e-4, 1e-5)?);
    println!("  PASS!");

    Ok(())
}

#[test]
fn test_rope() -> Result<()> {
    if !reference_available() {
        return Ok(());
    }

    let device = Device::Cpu;

    println!("\n=== RoPE Validation ===");

    let seq_len = 5usize;
    let head_dim = 128usize;
    let rope_theta = 1000000.0f64;

    // Load Q and K states from Python (pre-RoPE)
    let q = load_reference("q_states.bin", &[1, 16, 5, 128], &device)?;
    let k = load_reference("k_states.bin", &[1, 8, 5, 128], &device)?;

    // Compute RoPE
    let inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), &device)?;

    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let positions = Tensor::new(positions.as_slice(), &device)?;

    // freqs = outer(positions, inv_freq)
    let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    // Repeat for full head_dim
    let cos = Tensor::cat(&[&cos, &cos], 1)?;
    let sin = Tensor::cat(&[&sin, &sin], 1)?;

    // Shape: [1, 1, seq_len, head_dim]
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    // Apply RoPE
    let (rust_q_rope, rust_k_rope) = apply_rope(&q, &k, &cos, &sin)?;

    // Load Python references
    let python_q_rope = load_reference("q_rope.bin", &[1, 16, 5, 128], &device)?;
    let python_k_rope = load_reference("k_rope.bin", &[1, 8, 5, 128], &device)?;

    compare_tensors("q_rope", &rust_q_rope, &python_q_rope)?;
    compare_tensors("k_rope", &rust_k_rope, &python_k_rope)?;

    assert!(tensors_close(&rust_q_rope, &python_q_rope, 1e-4, 1e-5)?);
    assert!(tensors_close(&rust_k_rope, &python_k_rope, 1e-4, 1e-5)?);
    println!("  PASS!");

    Ok(())
}

#[test]
fn test_attention() -> Result<()> {
    if !reference_available() {
        return Ok(());
    }

    let device = Device::Cpu;

    println!("\n=== Attention Validation ===");

    let seq_len = 5usize;
    let num_heads = 16usize;
    let num_kv_heads = 8usize;
    let head_dim = 128usize;

    // Load Q, K, V after RoPE
    let q = load_reference("q_rope.bin", &[1, 16, 5, 128], &device)?;
    let k = load_reference("k_rope.bin", &[1, 8, 5, 128], &device)?;
    let v = load_reference("v_states.bin", &[1, 8, 5, 128], &device)?;

    // Repeat KV for GQA
    let n_rep = num_heads / num_kv_heads;
    let k = repeat_kv(&k, n_rep)?;
    let v = repeat_kv(&v, n_rep)?;

    // Attention scores
    let scaling = (head_dim as f64).powf(-0.5);
    let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scaling, 0.0)?;

    // Causal mask
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), &device)?;
    let attn_weights = attn_weights.broadcast_add(&mask)?;

    // Softmax
    let attn_probs = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;

    // Apply attention
    let rust_attn_output = attn_probs.matmul(&v)?;

    // Load Python references
    let python_attn_weights = load_reference("attn_weights.bin", &[1, 16, 5, 5], &device)?;
    let python_attn_probs = load_reference("attn_probs.bin", &[1, 16, 5, 5], &device)?;
    let python_attn_output = load_reference("attn_output.bin", &[1, 16, 5, 128], &device)?;

    compare_tensors("attn_weights", &attn_weights, &python_attn_weights)?;
    compare_tensors("attn_probs", &attn_probs, &python_attn_probs)?;
    compare_tensors("attn_output", &rust_attn_output, &python_attn_output)?;

    assert!(tensors_close(&attn_weights, &python_attn_weights, 1e-4, 1e-5)?);
    assert!(tensors_close(&attn_probs, &python_attn_probs, 1e-4, 1e-5)?);
    assert!(tensors_close(&rust_attn_output, &python_attn_output, 1e-4, 1e-5)?);
    println!("  PASS!");

    Ok(())
}

#[test]
fn test_o_projection_and_residual() -> Result<()> {
    if !reference_available() {
        return Ok(());
    }

    let device = Device::Cpu;
    let weights = load_weights(&device)?;

    println!("\n=== O Projection & Residual Validation ===");

    let batch_size = 1usize;
    let seq_len = 5usize;
    let num_heads = 16usize;
    let head_dim = 128usize;

    // Load attention output
    let attn_output = load_reference("attn_output.bin", &[1, 16, 5, 128], &device)?;

    // Reshape: (batch, num_heads, seq, head_dim) -> (batch, seq, num_heads * head_dim)
    let attn_flat = attn_output
        .transpose(1, 2)?
        .reshape((batch_size, seq_len, num_heads * head_dim))?;

    // O projection
    let o_proj_w = weights.get("talker.model.layers.0.self_attn.o_proj.weight").unwrap();
    let rust_after_o = linear(&attn_flat, o_proj_w, None)?;

    // Residual
    let projected = load_reference("projected.bin", &[1, 5, 1024], &device)?;
    let rust_after_residual = (&projected + &rust_after_o)?;

    // Load Python references
    let python_after_o = load_reference("after_o_proj.bin", &[1, 5, 1024], &device)?;
    let python_after_residual = load_reference("after_attn_residual.bin", &[1, 5, 1024], &device)?;

    compare_tensors("after_o_proj", &rust_after_o, &python_after_o)?;
    compare_tensors("after_attn_residual", &rust_after_residual, &python_after_residual)?;

    assert!(tensors_close(&rust_after_o, &python_after_o, 1e-4, 1e-5)?);
    assert!(tensors_close(&rust_after_residual, &python_after_residual, 1e-4, 1e-5)?);
    println!("  PASS!");

    Ok(())
}

#[test]
fn test_mlp() -> Result<()> {
    if !reference_available() {
        return Ok(());
    }

    let device = Device::Cpu;
    let weights = load_weights(&device)?;

    println!("\n=== MLP Validation ===");

    // Load input (after attention residual)
    let hidden = load_reference("after_attn_residual.bin", &[1, 5, 1024], &device)?;

    // Get MLP weights
    let post_ln_w = weights.get("talker.model.layers.0.post_attention_layernorm.weight").unwrap();
    let gate_w = weights.get("talker.model.layers.0.mlp.gate_proj.weight").unwrap();
    let up_w = weights.get("talker.model.layers.0.mlp.up_proj.weight").unwrap();
    let down_w = weights.get("talker.model.layers.0.mlp.down_proj.weight").unwrap();

    // Post-attention layer norm
    let mlp_input = rms_norm(&hidden, post_ln_w, 1e-6)?;

    // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
    let gate = linear(&mlp_input, gate_w, None)?;
    let up = linear(&mlp_input, up_w, None)?;
    let mlp_hidden = candle_nn::ops::silu(&gate)?.mul(&up)?;
    let rust_mlp_output = linear(&mlp_hidden, down_w, None)?;

    // Residual
    let rust_layer_output = (&hidden + &rust_mlp_output)?;

    // Load Python references
    let python_mlp_input = load_reference("mlp_input.bin", &[1, 5, 1024], &device)?;
    let python_mlp_output = load_reference("mlp_output.bin", &[1, 5, 1024], &device)?;
    let python_layer_output = load_reference("layer_0_output.bin", &[1, 5, 1024], &device)?;

    compare_tensors("mlp_input", &mlp_input, &python_mlp_input)?;
    compare_tensors("mlp_output", &rust_mlp_output, &python_mlp_output)?;
    compare_tensors("layer_0_output", &rust_layer_output, &python_layer_output)?;

    assert!(tensors_close(&mlp_input, &python_mlp_input, 1e-4, 1e-5)?);
    assert!(tensors_close(&rust_mlp_output, &python_mlp_output, 1e-4, 1e-5)?);
    assert!(tensors_close(&rust_layer_output, &python_layer_output, 1e-4, 1e-5)?);
    println!("  PASS!");

    Ok(())
}

#[test]
fn test_full_layer_0() -> Result<()> {
    if !reference_available() {
        eprintln!("Reference values not found. Run: python3 tools/export_reference_values.py");
        return Ok(());
    }

    let device = Device::Cpu;
    let weights = load_weights(&device)?;

    println!("\n=== Full Layer 0 End-to-End Validation ===");

    let batch_size = 1usize;
    let seq_len = 5usize;
    let num_heads = 16usize;
    let num_kv_heads = 8usize;
    let head_dim = 128usize;
    let rope_theta = 1000000.0f64;

    // Start from text embeddings (ground truth)
    let text_embeddings = load_reference("text_embeddings.bin", &[1, 5, 2048], &device)?;

    // ===== Text Projection =====
    let fc1_w = weights.get("talker.text_projection.linear_fc1.weight").unwrap();
    let fc1_b = weights.get("talker.text_projection.linear_fc1.bias").unwrap();
    let fc2_w = weights.get("talker.text_projection.linear_fc2.weight").unwrap();
    let fc2_b = weights.get("talker.text_projection.linear_fc2.bias").unwrap();

    let hidden = linear(&text_embeddings, fc1_w, Some(fc1_b))?;
    let hidden = candle_nn::ops::silu(&hidden)?;
    let projected = linear(&hidden, fc2_w, Some(fc2_b))?;

    // ===== Input LayerNorm =====
    let input_ln_w = weights.get("talker.model.layers.0.input_layernorm.weight").unwrap();
    let normed = rms_norm(&projected, input_ln_w, 1e-6)?;

    // ===== QKV Projections with QK Norm =====
    let q_proj_w = weights.get("talker.model.layers.0.self_attn.q_proj.weight").unwrap();
    let k_proj_w = weights.get("talker.model.layers.0.self_attn.k_proj.weight").unwrap();
    let v_proj_w = weights.get("talker.model.layers.0.self_attn.v_proj.weight").unwrap();
    let q_norm_w = weights.get("talker.model.layers.0.self_attn.q_norm.weight").unwrap();
    let k_norm_w = weights.get("talker.model.layers.0.self_attn.k_norm.weight").unwrap();

    let q = linear(&normed, q_proj_w, None)?;
    let q = q.reshape((batch_size, seq_len, num_heads, head_dim))?;
    let q = rms_norm(&q, q_norm_w, 1e-6)?;
    let q = q.transpose(1, 2)?;

    let k = linear(&normed, k_proj_w, None)?;
    let k = k.reshape((batch_size, seq_len, num_kv_heads, head_dim))?;
    let k = rms_norm(&k, k_norm_w, 1e-6)?;
    let k = k.transpose(1, 2)?;

    let v = linear(&normed, v_proj_w, None)?;
    let v = v.reshape((batch_size, seq_len, num_kv_heads, head_dim))?;
    let v = v.transpose(1, 2)?;

    // ===== RoPE =====
    let inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), &device)?;
    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let positions = Tensor::new(positions.as_slice(), &device)?;
    let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    let cos = Tensor::cat(&[&freqs.cos()?, &freqs.cos()?], 1)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = Tensor::cat(&[&freqs.sin()?, &freqs.sin()?], 1)?.unsqueeze(0)?.unsqueeze(0)?;

    let (q, k) = apply_rope(&q, &k, &cos, &sin)?;

    // ===== Attention =====
    let n_rep = num_heads / num_kv_heads;
    let k = repeat_kv(&k, n_rep)?;
    let v = repeat_kv(&v, n_rep)?;

    let scaling = (head_dim as f64).powf(-0.5);
    let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scaling, 0.0)?;

    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), &device)?;
    let attn_weights = attn_weights.broadcast_add(&mask)?;
    let attn_probs = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
    let attn_output = attn_probs.matmul(&v)?;

    // ===== O Projection & Residual =====
    let o_proj_w = weights.get("talker.model.layers.0.self_attn.o_proj.weight").unwrap();
    let attn_flat = attn_output.transpose(1, 2)?.reshape((batch_size, seq_len, num_heads * head_dim))?;
    let after_o = linear(&attn_flat, o_proj_w, None)?;
    let hidden = (&projected + &after_o)?;

    // ===== MLP =====
    let post_ln_w = weights.get("talker.model.layers.0.post_attention_layernorm.weight").unwrap();
    let gate_w = weights.get("talker.model.layers.0.mlp.gate_proj.weight").unwrap();
    let up_w = weights.get("talker.model.layers.0.mlp.up_proj.weight").unwrap();
    let down_w = weights.get("talker.model.layers.0.mlp.down_proj.weight").unwrap();

    let mlp_input = rms_norm(&hidden, post_ln_w, 1e-6)?;
    let gate = linear(&mlp_input, gate_w, None)?;
    let up = linear(&mlp_input, up_w, None)?;
    let mlp_hidden = candle_nn::ops::silu(&gate)?.mul(&up)?;
    let mlp_output = linear(&mlp_hidden, down_w, None)?;

    let rust_layer_output = (&hidden + &mlp_output)?;

    // ===== Compare =====
    let python_layer_output = load_reference("layer_0_output.bin", &[1, 5, 1024], &device)?;

    compare_tensors("layer_0_output (end-to-end)", &rust_layer_output, &python_layer_output)?;

    assert!(tensors_close(&rust_layer_output, &python_layer_output, 1e-4, 1e-5)?);
    println!("  FULL LAYER 0 END-TO-END PASS!");

    Ok(())
}

#[test]
fn test_full_forward_28_layers() -> Result<()> {
    if !reference_available() {
        eprintln!("Reference values not found. Run: python3 tools/export_reference_values.py");
        return Ok(());
    }

    let device = Device::Cpu;
    let weights = load_weights(&device)?;

    println!("\n=== Full 28-Layer Forward Pass Validation ===");

    let batch_size = 1usize;
    let seq_len = 5usize;
    let num_heads = 16usize;
    let num_kv_heads = 8usize;
    let head_dim = 128usize;
    let num_layers = 28usize;
    let rope_theta = 1000000.0f64;

    // Start from text embeddings
    let text_embeddings = load_reference("text_embeddings.bin", &[1, 5, 2048], &device)?;

    // Text Projection
    let fc1_w = weights.get("talker.text_projection.linear_fc1.weight").unwrap();
    let fc1_b = weights.get("talker.text_projection.linear_fc1.bias").unwrap();
    let fc2_w = weights.get("talker.text_projection.linear_fc2.weight").unwrap();
    let fc2_b = weights.get("talker.text_projection.linear_fc2.bias").unwrap();

    let proj = linear(&text_embeddings, fc1_w, Some(fc1_b))?;
    let proj = candle_nn::ops::silu(&proj)?;
    let mut hidden = linear(&proj, fc2_w, Some(fc2_b))?;

    // Precompute RoPE
    let inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), &device)?;
    let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let positions = Tensor::new(positions.as_slice(), &device)?;
    let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    let cos = Tensor::cat(&[&freqs.cos()?, &freqs.cos()?], 1)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = Tensor::cat(&[&freqs.sin()?, &freqs.sin()?], 1)?.unsqueeze(0)?.unsqueeze(0)?;

    // Precompute causal mask
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    let causal_mask = Tensor::from_vec(mask_data, (seq_len, seq_len), &device)?;

    let n_rep = num_heads / num_kv_heads;
    let scaling = (head_dim as f64).powf(-0.5);

    // Run through all layers
    for layer_idx in 0..num_layers {
        // Input LayerNorm
        let input_ln_w = weights
            .get(&format!("talker.model.layers.{}.input_layernorm.weight", layer_idx))
            .unwrap();
        let normed = rms_norm(&hidden, input_ln_w, 1e-6)?;

        // QKV
        let q_proj_w = weights
            .get(&format!("talker.model.layers.{}.self_attn.q_proj.weight", layer_idx))
            .unwrap();
        let k_proj_w = weights
            .get(&format!("talker.model.layers.{}.self_attn.k_proj.weight", layer_idx))
            .unwrap();
        let v_proj_w = weights
            .get(&format!("talker.model.layers.{}.self_attn.v_proj.weight", layer_idx))
            .unwrap();
        let q_norm_w = weights
            .get(&format!("talker.model.layers.{}.self_attn.q_norm.weight", layer_idx))
            .unwrap();
        let k_norm_w = weights
            .get(&format!("talker.model.layers.{}.self_attn.k_norm.weight", layer_idx))
            .unwrap();

        let q = linear(&normed, q_proj_w, None)?;
        let q = q.reshape((batch_size, seq_len, num_heads, head_dim))?;
        let q = rms_norm(&q, q_norm_w, 1e-6)?;
        let q = q.transpose(1, 2)?;

        let k = linear(&normed, k_proj_w, None)?;
        let k = k.reshape((batch_size, seq_len, num_kv_heads, head_dim))?;
        let k = rms_norm(&k, k_norm_w, 1e-6)?;
        let k = k.transpose(1, 2)?;

        let v = linear(&normed, v_proj_w, None)?;
        let v = v.reshape((batch_size, seq_len, num_kv_heads, head_dim))?;
        let v = v.transpose(1, 2)?;

        // RoPE
        let (q, k) = apply_rope(&q, &k, &cos, &sin)?;

        // Attention
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(scaling, 0.0)?;
        let attn_weights = attn_weights.broadcast_add(&causal_mask)?;
        let attn_probs = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
        let attn_output = attn_probs.matmul(&v)?;

        // O Projection & Residual
        let o_proj_w = weights
            .get(&format!("talker.model.layers.{}.self_attn.o_proj.weight", layer_idx))
            .unwrap();
        let attn_flat = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, num_heads * head_dim))?;
        let after_o = linear(&attn_flat, o_proj_w, None)?;
        hidden = (&hidden + &after_o)?;

        // MLP
        let post_ln_w = weights
            .get(&format!("talker.model.layers.{}.post_attention_layernorm.weight", layer_idx))
            .unwrap();
        let gate_w = weights
            .get(&format!("talker.model.layers.{}.mlp.gate_proj.weight", layer_idx))
            .unwrap();
        let up_w = weights
            .get(&format!("talker.model.layers.{}.mlp.up_proj.weight", layer_idx))
            .unwrap();
        let down_w = weights
            .get(&format!("talker.model.layers.{}.mlp.down_proj.weight", layer_idx))
            .unwrap();

        let mlp_input = rms_norm(&hidden, post_ln_w, 1e-6)?;
        let gate = linear(&mlp_input, gate_w, None)?;
        let up = linear(&mlp_input, up_w, None)?;
        let mlp_hidden = candle_nn::ops::silu(&gate)?.mul(&up)?;
        let mlp_output = linear(&mlp_hidden, down_w, None)?;

        hidden = (&hidden + &mlp_output)?;

        if layer_idx % 7 == 0 {
            let mean: f32 = hidden.flatten_all()?.mean_all()?.to_scalar()?;
            println!("  Layer {}: mean={:.6}", layer_idx, mean);
        }
    }

    // Load Python reference
    let python_after_layers = load_reference("after_all_layers.bin", &[1, 5, 1024], &device)?;

    compare_tensors("after_all_layers", &hidden, &python_after_layers)?;

    // Use slightly larger tolerance for accumulated error over 28 layers
    assert!(tensors_close(&hidden, &python_after_layers, 1e-3, 1e-4)?);
    println!("  28-LAYER FORWARD PASS!");

    Ok(())
}

#[test]
fn test_final_norm_and_codec_head() -> Result<()> {
    if !reference_available() {
        return Ok(());
    }

    let device = Device::Cpu;
    let weights = load_weights(&device)?;

    println!("\n=== Final Norm & Codec Head Validation ===");

    // Load output after all layers
    let after_layers = load_reference("after_all_layers.bin", &[1, 5, 1024], &device)?;

    // Final norm
    let final_norm_w = weights.get("talker.model.norm.weight").unwrap();
    let rust_final = rms_norm(&after_layers, final_norm_w, 1e-6)?;

    let python_final = load_reference("after_final_norm.bin", &[1, 5, 1024], &device)?;
    compare_tensors("after_final_norm", &rust_final, &python_final)?;
    assert!(tensors_close(&rust_final, &python_final, 1e-5, 1e-6)?);

    // Codec head
    let codec_head_w = weights.get("talker.codec_head.weight").unwrap();
    let rust_logits = linear(&rust_final, codec_head_w, None)?;

    let python_logits = load_reference("codec_logits.bin", &[1, 5, 3072], &device)?;
    compare_tensors("codec_logits", &rust_logits, &python_logits)?;
    assert!(tensors_close(&rust_logits, &python_logits, 1e-4, 1e-5)?);

    // Check predictions match
    let rust_preds = rust_logits.argmax(candle_core::D::Minus1)?;
    let rust_preds_vec: Vec<u32> = rust_preds.flatten_all()?.to_vec1()?;
    println!("  Rust predictions: {:?}", rust_preds_vec);

    // Expected from Python: [1501, 1231, 1732, 1353, 963]
    let expected = vec![1501u32, 1231, 1732, 1353, 963];
    assert_eq!(rust_preds_vec, expected, "Predictions should match Python");

    println!("  FINAL NORM & CODEC HEAD PASS!");

    Ok(())
}

#[test]
fn test_model_module_matches_reference() -> Result<()> {
    // This test uses the actual qwen3_tts model module to verify it matches Python
    use qwen3_tts::models::config::Qwen3TTSConfig;
    use qwen3_tts::models::qwen3_tts::{RotaryEmbedding, DecoderLayer};
    use candle_nn::VarBuilder;

    if !reference_available() {
        return Ok(());
    }

    let device = Device::Cpu;
    let weights = load_weights(&device)?;

    println!("\n=== Model Module Validation ===");

    // Create config matching the 0.6B model talker
    // Key insight: head_dim = 128, not hidden_size/num_heads = 64
    // So q_proj output = num_heads * head_dim = 16 * 128 = 2048
    let config = Qwen3TTSConfig {
        hidden_size: 1024,
        num_attention_heads: 16,
        num_key_value_heads: Some(8),
        head_dim_override: Some(128),  // Explicitly set head_dim
        intermediate_size: 3072,
        num_hidden_layers: 28,
        vocab_size: 3072,
        rms_norm_eps: 1e-6,
        rope_theta: 1000000.0,
        max_position_embeddings: 8192,
        num_codebook_groups: 16,
        ..Default::default()
    };

    // Load weights into VarBuilder for model initialization
    let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, &device);

    // Test a single decoder layer from the model module
    let layer_vb = vb.pp("talker.model.layers.0");
    let layer = DecoderLayer::new(&config, layer_vb)?;

    // Create RoPE using config.head_dim()
    let rope = RotaryEmbedding::new(config.head_dim(), 512, config.rope_theta, &device)?;

    // Get projected input (matches what Python exports)
    let projected = load_reference("projected.bin", &[1, 5, 1024], &device)?;

    // Create causal mask
    let seq_len = 5;
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), &device)?;

    // Run layer forward
    let output = layer.forward(&projected, &rope, Some(&mask), None, 0)?;

    // Compare with Python reference
    let python_output = load_reference("layer_0_output.bin", &[1, 5, 1024], &device)?;

    compare_tensors("layer_0_from_module", &output, &python_output)?;

    // Use tolerance for accumulated floating point differences
    let diff = (&output - &python_output)?.abs()?;
    let max_diff: f32 = diff.flatten_all()?.max(0)?.to_scalar()?;
    println!("  Model module layer 0 max_diff: {:.6}", max_diff);

    // Allow slightly larger tolerance since we're testing the full layer
    assert!(max_diff < 1e-3, "Model module output should match Python within 1e-3");
    println!("  MODEL MODULE LAYER 0 PASS!");

    Ok(())
}
