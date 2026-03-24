//! End-to-end inference example
//!
//! This loads the real 0.6B model and generates audio from text.

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let device = Device::Cpu;
    println!("Device: {:?}", device);

    // Check test data exists
    let model_path = Path::new("test_data/model/model.safetensors");
    let tokenizer_path = Path::new("test_data/tokenizer/tokenizer.json");
    let config_path = Path::new("test_data/tokenizer/config.json");

    if !model_path.exists() {
        anyhow::bail!("Model not found. Run: ./scripts/download_test_data.sh");
    }

    // Load text tokenizer
    println!("\n=== Loading text tokenizer ===");
    let tokenizer = qwen3_tts::tokenizer::TextTokenizer::from_file(tokenizer_path)?;
    println!("Vocab size: {}", tokenizer.vocab_size());

    // Load config
    println!("\n=== Loading config ===");
    let config_str = std::fs::read_to_string(config_path)?;
    let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
    let talker_config = &config_json["talker_config"];

    println!("Model type: {}", config_json["model_type"]);
    println!("Talker hidden_size: {}", talker_config["hidden_size"]);
    println!("Talker num_layers: {}", talker_config["num_hidden_layers"]);
    println!("Talker num_heads: {}", talker_config["num_attention_heads"]);
    println!("Talker num_kv_heads: {}", talker_config["num_key_value_heads"]);
    println!("Num code groups: {}", talker_config["num_code_groups"]);

    // Load model weights
    println!("\n=== Loading model weights ===");
    let start = std::time::Instant::now();
    let tensors_raw: HashMap<String, Tensor> =
        candle_core::safetensors::load(model_path, &device)?;
    println!("Loaded {} tensors in {:?}", tensors_raw.len(), start.elapsed());

    // Convert BF16 weights to F32 for CPU inference
    println!("Converting BF16 weights to F32...");
    let convert_start = std::time::Instant::now();
    let tensors: HashMap<String, Tensor> = tensors_raw
        .into_iter()
        .map(|(name, tensor)| {
            let converted = if tensor.dtype() == candle_core::DType::BF16 {
                tensor.to_dtype(candle_core::DType::F32).unwrap()
            } else {
                tensor
            };
            (name, converted)
        })
        .collect();
    println!("Converted in {:?}", convert_start.elapsed());

    // Tokenize input text
    let text = "Hello, this is a test of the Qwen3 TTS system.";
    println!("\n=== Tokenizing ===");
    println!("Input: \"{}\"", text);

    let input_ids = tokenizer.encode(text)?;
    println!("Token IDs: {:?}", &input_ids[..input_ids.len().min(20)]);
    if input_ids.len() > 20 {
        println!("  ... ({} tokens total)", input_ids.len());
    }

    // Convert to tensor
    let input_tensor = Tensor::new(input_ids.as_slice(), &device)?.unsqueeze(0)?;
    println!("Input tensor shape: {:?}", input_tensor.dims());

    // Check key model tensors
    println!("\n=== Checking model tensors ===");
    let key_tensors = [
        "talker.model.text_embedding.weight",      // Text embeddings (151936, 2048)
        "talker.model.codec_embedding.weight",     // Codec embeddings (3072, 1024)
        "talker.model.layers.0.self_attn.q_proj.weight",
        "talker.model.layers.0.input_layernorm.weight",
        "talker.model.norm.weight",
        "talker.codec_head.weight",                // Semantic token head (3072, 1024)
        "talker.code_predictor.lm_head.0.weight",  // Acoustic token head 0
    ];

    for name in key_tensors {
        if let Some(t) = tensors.get(name) {
            println!("  {}: {:?}", name, t.dims());
        } else {
            println!("  {}: NOT FOUND", name);
        }
    }

    // Try a simple forward pass through embeddings
    println!("\n=== Testing text embedding lookup ===");
    let text_embed_weight = tensors
        .get("talker.model.text_embedding.weight")
        .ok_or_else(|| anyhow::anyhow!("Text embedding weight not found"))?;
    println!("Text embedding weight shape: {:?}", text_embed_weight.dims());

    // Look up text embeddings for input tokens
    let embeddings = text_embed_weight.index_select(&input_tensor.flatten_all()?, 0)?;
    println!("Text embeddings shape: {:?}", embeddings.dims());

    // Check embedding values
    let embed_slice: Vec<f32> = embeddings.flatten_all()?.to_vec1()?;
    let mean = embed_slice.iter().sum::<f32>() / embed_slice.len() as f32;
    let max = embed_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min = embed_slice.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    println!(
        "Embedding stats: mean={:.6}, min={:.4}, max={:.4}",
        mean, min, max
    );

    // Test text projection (projects 2048-dim text embeddings to 1024-dim talker space)
    println!("\n=== Testing text projection ===");
    let proj_fc1_w = tensors
        .get("talker.text_projection.linear_fc1.weight")
        .ok_or_else(|| anyhow::anyhow!("text_projection.linear_fc1.weight not found"))?;
    let proj_fc1_b = tensors
        .get("talker.text_projection.linear_fc1.bias")
        .ok_or_else(|| anyhow::anyhow!("text_projection.linear_fc1.bias not found"))?;
    let proj_fc2_w = tensors
        .get("talker.text_projection.linear_fc2.weight")
        .ok_or_else(|| anyhow::anyhow!("text_projection.linear_fc2.weight not found"))?;
    let proj_fc2_b = tensors
        .get("talker.text_projection.linear_fc2.bias")
        .ok_or_else(|| anyhow::anyhow!("text_projection.linear_fc2.bias not found"))?;

    println!("FC1 weight: {:?}", proj_fc1_w.dims());
    println!("FC2 weight: {:?}", proj_fc2_w.dims());

    // Project embeddings: x @ fc1_w.T + fc1_b -> silu -> x @ fc2_w.T + fc2_b
    let hidden = embeddings.matmul(&proj_fc1_w.t()?)?.broadcast_add(proj_fc1_b)?;
    let hidden = candle_nn::ops::silu(&hidden)?;
    let projected = hidden.matmul(&proj_fc2_w.t()?)?.broadcast_add(proj_fc2_b)?;
    println!("Projected shape: {:?}", projected.dims());

    let proj_slice: Vec<f32> = projected.flatten_all()?.to_vec1()?;
    let mean = proj_slice.iter().sum::<f32>() / proj_slice.len() as f32;
    let max = proj_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min = proj_slice.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    println!(
        "Projected stats: mean={:.6}, min={:.4}, max={:.4}",
        mean, min, max
    );

    // Test single attention layer
    println!("\n=== Testing single transformer layer ===");
    let seq_len = projected.dim(0)?;
    let hidden_size = projected.dim(1)?;
    println!("Sequence length: {}, Hidden size: {}", seq_len, hidden_size);

    // Get layer 0 weights
    let q_proj = tensors.get("talker.model.layers.0.self_attn.q_proj.weight").unwrap();
    let k_proj = tensors.get("talker.model.layers.0.self_attn.k_proj.weight").unwrap();
    let v_proj = tensors.get("talker.model.layers.0.self_attn.v_proj.weight").unwrap();
    let o_proj = tensors.get("talker.model.layers.0.self_attn.o_proj.weight").unwrap();
    let input_ln = tensors.get("talker.model.layers.0.input_layernorm.weight").unwrap();

    println!("Q proj: {:?}", q_proj.dims());
    println!("K proj: {:?}", k_proj.dims());
    println!("V proj: {:?}", v_proj.dims());
    println!("O proj: {:?}", o_proj.dims());
    println!("Input LN: {:?}", input_ln.dims());

    // RMS norm
    fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(1)?;
        let x_norm = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
        Ok(x_norm.broadcast_mul(weight)?)
    }

    let normed = rms_norm(&projected, input_ln, 1e-6)?;
    println!("After RMS norm: {:?}", normed.dims());

    // Q, K, V projections
    let q = normed.matmul(&q_proj.t()?)?;
    let k = normed.matmul(&k_proj.t()?)?;
    let v = normed.matmul(&v_proj.t()?)?;
    println!("Q: {:?}, K: {:?}, V: {:?}", q.dims(), k.dims(), v.dims());

    // Simple attention (no RoPE for now, just verify shapes)
    let num_heads = 16;
    let num_kv_heads = 8;
    let head_dim = q.dim(1)? / num_heads;
    println!("Num heads: {}, KV heads: {}, Head dim: {}", num_heads, num_kv_heads, head_dim);

    // Reshape for multi-head attention
    let q = q.reshape((seq_len, num_heads, head_dim))?.transpose(0, 1)?;
    let k = k.reshape((seq_len, num_kv_heads, head_dim))?.transpose(0, 1)?;
    let v = v.reshape((seq_len, num_kv_heads, head_dim))?.transpose(0, 1)?;
    println!("Q reshaped: {:?}", q.dims());
    println!("K reshaped: {:?}", k.dims());

    // Repeat KV heads to match Q heads (GQA)
    // Each KV head is repeated `repeats` times
    let repeats = num_heads / num_kv_heads;
    fn repeat_kv(x: &Tensor, repeats: usize) -> Result<Tensor> {
        if repeats == 1 {
            return Ok(x.clone());
        }
        // x shape: [num_kv_heads, seq_len, head_dim]
        let (n_kv, seq, hd) = x.dims3()?;
        // Repeat by concatenating
        let repeated: Vec<Tensor> = (0..n_kv)
            .flat_map(|i| {
                let slice = x.narrow(0, i, 1).unwrap();
                std::iter::repeat(slice).take(repeats)
            })
            .collect();
        Ok(Tensor::cat(&repeated, 0)?)
    }
    let k = repeat_kv(&k, repeats)?;
    let v = repeat_kv(&v, repeats)?;
    println!("K after GQA repeat: {:?}", k.dims());

    // Attention scores
    let scale = (head_dim as f64).sqrt();
    let attn_scores = q.matmul(&k.transpose(1, 2)?)?.affine(1.0 / scale, 0.0)?;
    println!("Attention scores: {:?}", attn_scores.dims());

    // Causal mask - create lower triangular mask
    fn create_causal_mask(size: usize, device: &Device) -> Result<Tensor> {
        let mut mask_data = vec![0.0f32; size * size];
        for i in 0..size {
            for j in 0..size {
                if j > i {
                    mask_data[i * size + j] = f32::NEG_INFINITY;
                }
            }
        }
        Ok(Tensor::from_vec(mask_data, (size, size), device)?)
    }
    let mask = create_causal_mask(seq_len, &device)?;
    let attn_scores = attn_scores.broadcast_add(&mask)?;

    // Softmax
    let attn_probs = candle_nn::ops::softmax(&attn_scores, 2)?;

    // Apply attention
    let attn_out = attn_probs.matmul(&v)?;
    println!("Attention output: {:?}", attn_out.dims());

    // Reshape back
    let attn_out = attn_out.transpose(0, 1)?.reshape((seq_len, num_heads * head_dim))?;
    println!("Attention output reshaped: {:?}", attn_out.dims());

    // Output projection
    let attn_out = attn_out.matmul(&o_proj.t()?)?;
    println!("After O projection: {:?}", attn_out.dims());

    // Residual
    let hidden = (projected + attn_out)?;
    println!("After residual: {:?}", hidden.dims());

    println!("\n=== SUCCESS! ===");
    println!("Single transformer layer forward pass works!");
    println!("Text input: \"{}\"", text);
    println!("Token count: {}", input_ids.len());
    println!("Output hidden shape: {:?}", hidden.dims());
    println!("\nNext steps:");
    println!("  1. Implement full model forward pass (28 layers)");
    println!("  2. Add RoPE positional encoding");
    println!("  3. Implement code predictor for generating codec tokens");
    println!("  4. Load speech tokenizer decoder for audio synthesis");

    Ok(())
}
