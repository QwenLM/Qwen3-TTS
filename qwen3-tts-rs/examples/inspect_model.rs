//! Inspect model tensor structure

use std::collections::HashMap;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model_path = Path::new("test_data/model/model.safetensors");

    if !model_path.exists() {
        anyhow::bail!("Model not found. Run: ./scripts/download_test_data.sh");
    }

    println!("Loading model from {:?}", model_path);

    let tensors = candle_core::safetensors::load(model_path, &candle_core::Device::Cpu)?;

    println!("\n=== Model has {} tensors ===\n", tensors.len());

    // Group by prefix
    let mut groups: HashMap<String, Vec<(String, Vec<usize>)>> = HashMap::new();
    for (name, tensor) in &tensors {
        let prefix = name.split('.').take(2).collect::<Vec<_>>().join(".");
        groups
            .entry(prefix)
            .or_default()
            .push((name.clone(), tensor.dims().to_vec()));
    }

    println!("=== Tensor groups ===");
    for prefix in groups.keys().collect::<std::collections::BTreeSet<_>>() {
        let tensors = &groups[prefix];
        println!("\n{}  ({} tensors)", prefix, tensors.len());
        for (name, shape) in tensors.iter().take(5) {
            println!("  {}: {:?}", name, shape);
        }
        if tensors.len() > 5 {
            println!("  ... and {} more", tensors.len() - 5);
        }
    }

    // Key tensors for understanding the architecture
    println!("\n=== Key architecture tensors ===");
    let key_patterns = [
        "embed_tokens",
        "lm_head",
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.mlp.gate_proj",
        "norm.weight",
        "code_embed",
    ];

    for pattern in key_patterns {
        for (name, tensor) in &tensors {
            if name.contains(pattern) {
                println!("  {}: {:?}", name, tensor.dims());
                break;
            }
        }
    }

    // Find all embedding-related tensors
    println!("\n=== All embedding/vocab-related tensors ===");
    let mut embed_tensors: Vec<_> = tensors
        .iter()
        .filter(|(name, _)| {
            name.to_lowercase().contains("embed")
                || name.contains("vocab")
                || name.contains("codec_head")
                || name.contains("lm_head")
        })
        .collect();
    embed_tensors.sort_by_key(|(name, _)| name.clone());

    for (name, tensor) in embed_tensors {
        println!("  {}: {:?}", name, tensor.dims());
    }

    Ok(())
}
