//! Check layer 0 structure
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let tensors = candle_core::safetensors::load(
        Path::new("test_data/model/model.safetensors"),
        &candle_core::Device::Cpu,
    )?;

    println!("=== talker.model.layers.0 tensors ===");
    let mut layer0: Vec<_> = tensors
        .iter()
        .filter(|(n, _)| n.starts_with("talker.model.layers.0."))
        .collect();
    layer0.sort_by_key(|(n, _)| n.clone());

    for (name, tensor) in layer0 {
        println!("  {}: {:?}", name, tensor.dims());
    }

    println!("\n=== Config from file ===");
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("test_data/tokenizer/config.json")?,
    )?;
    let talker = &config["talker_config"];
    println!("rope_theta: {}", talker["rope_theta"]);
    println!("rope_scaling: {}", talker["rope_scaling"]);

    Ok(())
}
