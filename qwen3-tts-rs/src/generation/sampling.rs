//! Token sampling strategies for autoregressive generation

use anyhow::Result;
use candle_core::{DType, IndexOp, Tensor, D};

/// Configuration for autoregressive generation
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
    /// Sampling temperature (1.0 = no change, <1.0 = more focused, >1.0 = more random)
    pub temperature: f64,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f64,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f64,
    /// End-of-sequence token ID (generation stops when this token is sampled)
    pub eos_token_id: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 2048,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            eos_token_id: None,
        }
    }
}

/// Sample next token from logits
///
/// # Arguments
/// * `logits` - Logits tensor of shape [batch, vocab_size]
/// * `config` - Generation configuration
///
/// # Returns
/// Token indices of shape [batch]
pub fn sample(logits: &Tensor, config: &GenerationConfig) -> Result<Tensor> {
    let logits = logits.to_dtype(DType::F32)?;

    // Apply temperature
    let logits = if config.temperature != 1.0 && config.temperature > 0.0 {
        (logits / config.temperature)?
    } else {
        logits
    };

    // For simplicity, if temperature is very low, use greedy
    if config.temperature < 0.01 {
        return greedy_sample(&logits);
    }

    // Convert to probabilities
    let probs = candle_nn::ops::softmax_last_dim(&logits)?;

    // Sample from distribution
    multinomial_sample(&probs)
}

/// Compute cumulative sum along last dimension
fn cumulative_sum(x: &Tensor) -> Result<Tensor> {
    let (batch, len) = x.dims2()?;
    let mut results = Vec::with_capacity(batch);

    for b in 0..batch {
        let row: Vec<f32> = x.i(b)?.to_vec1()?;
        let mut cumsum = Vec::with_capacity(len);
        let mut sum = 0.0f32;
        for v in row {
            sum += v;
            cumsum.push(sum);
        }
        results.push(cumsum);
    }

    let flat: Vec<f32> = results.into_iter().flatten().collect();
    Ok(Tensor::new(flat.as_slice(), x.device())?.reshape((batch, len))?)
}

/// Sample from probability distribution using multinomial sampling
fn multinomial_sample(probs: &Tensor) -> Result<Tensor> {
    let (batch, vocab) = probs.dims2()?;

    // Use cumulative distribution for sampling
    let cumsum = cumulative_sum(probs)?;

    // Generate uniform random values
    let uniform: Vec<f32> = (0..batch).map(|_| rand_f32()).collect();
    let uniform = Tensor::new(uniform.as_slice(), probs.device())?.unsqueeze(1)?;

    // Find first index where cumsum >= uniform
    let mask = cumsum.ge(&uniform.broadcast_as(cumsum.shape())?)?;

    // Convert mask to f32 for operations
    let mask_f32 = mask.to_dtype(DType::F32)?;

    // Use a trick: multiply by position and find first nonzero
    let positions: Vec<f32> = (0..vocab).map(|i| i as f32 + 1.0).collect();
    let positions = Tensor::new(positions.as_slice(), probs.device())?
        .unsqueeze(0)?
        .broadcast_as(mask_f32.shape())?;

    // Where mask is true, use position; else use large value
    let large =
        Tensor::new(&[vocab as f32 + 1.0], probs.device())?.broadcast_as(mask_f32.shape())?;
    let masked_positions = mask.where_cond(&positions, &large)?;

    // Argmin gives first True position
    Ok(masked_positions.argmin(D::Minus1)?)
}

/// Generate a random f32 in [0, 1)
fn rand_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64;
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);

    // LCG with seed and counter
    let state = seed
        .wrapping_add(count)
        .wrapping_mul(1103515245)
        .wrapping_add(12345);
    (state as f32) / (u64::MAX as f32)
}

/// Apply repetition penalty to logits
pub fn apply_repetition_penalty(
    logits: &Tensor,
    input_ids: &Tensor,
    penalty: f64,
) -> Result<Tensor> {
    if (penalty - 1.0).abs() < 1e-9 {
        return Ok(logits.clone());
    }

    let (batch, vocab) = logits.dims2()?;
    let input_ids_vec: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
    let mut logits_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;

    // Apply penalty to previously generated tokens
    for token_id in input_ids_vec {
        let idx = token_id as usize;
        if idx < vocab {
            for b in 0..batch {
                let i = b * vocab + idx;
                if logits_vec[i] > 0.0 {
                    logits_vec[i] /= penalty as f32;
                } else {
                    logits_vec[i] *= penalty as f32;
                }
            }
        }
    }

    Ok(Tensor::new(logits_vec.as_slice(), logits.device())?.reshape((batch, vocab))?)
}

/// Greedy sampling (argmax)
pub fn greedy_sample(logits: &Tensor) -> Result<Tensor> {
    Ok(logits.argmax(D::Minus1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_generation_config_default() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_new_tokens, 2048);
        assert!((config.temperature - 0.7).abs() < 1e-6);
        assert_eq!(config.top_k, 50);
        assert!((config.top_p - 0.9).abs() < 1e-6);
        assert!((config.repetition_penalty - 1.0).abs() < 1e-6);
        assert_eq!(config.eos_token_id, None);
    }

    #[test]
    fn test_generation_config_custom() {
        let config = GenerationConfig {
            max_new_tokens: 512,
            temperature: 0.5,
            top_k: 10,
            top_p: 0.8,
            repetition_penalty: 1.2,
            eos_token_id: Some(151670),
        };
        assert_eq!(config.max_new_tokens, 512);
        assert!((config.temperature - 0.5).abs() < 1e-6);
        assert_eq!(config.top_k, 10);
        assert_eq!(config.eos_token_id, Some(151670));
    }

    #[test]
    fn test_cumulative_sum() {
        let device = Device::Cpu;
        let x = Tensor::new(&[[0.1f32, 0.2, 0.3, 0.4]], &device).unwrap();
        let cumsum = cumulative_sum(&x).unwrap();
        let result: Vec<f32> = cumsum.flatten_all().unwrap().to_vec1().unwrap();
        assert!((result[0] - 0.1).abs() < 1e-5);
        assert!((result[1] - 0.3).abs() < 1e-5);
        assert!((result[2] - 0.6).abs() < 1e-5);
        assert!((result[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cumulative_sum_batch() {
        let device = Device::Cpu;
        let x = Tensor::new(
            &[[0.25f32, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]],
            &device,
        )
        .unwrap();
        let cumsum = cumulative_sum(&x).unwrap();
        let result: Vec<f32> = cumsum.flatten_all().unwrap().to_vec1().unwrap();
        // First row
        assert!((result[0] - 0.25).abs() < 1e-5);
        assert!((result[1] - 0.50).abs() < 1e-5);
        assert!((result[2] - 0.75).abs() < 1e-5);
        assert!((result[3] - 1.00).abs() < 1e-5);
        // Second row
        assert!((result[4] - 0.1).abs() < 1e-5);
        assert!((result[5] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_greedy_sample() {
        let device = Device::Cpu;
        // Logits where position 2 has highest value
        let logits = Tensor::new(&[[1.0f32, 2.0, 5.0, 1.0]], &device).unwrap();
        let result = greedy_sample(&logits).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 2); // Index of max
    }

    #[test]
    fn test_greedy_sample_batch() {
        let device = Device::Cpu;
        let logits = Tensor::new(
            &[[1.0f32, 5.0, 2.0], [3.0, 1.0, 2.0], [1.0, 2.0, 10.0]],
            &device,
        )
        .unwrap();
        let result = greedy_sample(&logits).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 1); // Max at position 1
        assert_eq!(idx[1], 0); // Max at position 0
        assert_eq!(idx[2], 2); // Max at position 2
    }

    #[test]
    fn test_sample_very_low_temperature() {
        let device = Device::Cpu;
        // With very low temperature, should act like greedy
        let logits = Tensor::new(&[[1.0f32, 10.0, 2.0, 1.0]], &device).unwrap();
        let config = GenerationConfig {
            temperature: 0.001,
            ..Default::default()
        };
        let result = sample(&logits, &config).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 1); // Should pick the highest
    }

    #[test]
    fn test_sample_normal_temperature() {
        let device = Device::Cpu;
        // With normal temperature, sampling should work
        let logits = Tensor::new(&[[1.0f32, 1.0, 1.0, 1.0]], &device).unwrap();
        let config = GenerationConfig::default();
        let result = sample(&logits, &config).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        // Should return a valid index
        assert!(idx[0] < 4);
    }

    #[test]
    fn test_sample_temperature_one() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[2.0f32, 2.0, 2.0]], &device).unwrap();
        let config = GenerationConfig {
            temperature: 1.0,
            ..Default::default()
        };
        let result = sample(&logits, &config).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert!(idx[0] < 3);
    }

    #[test]
    fn test_apply_repetition_penalty_no_penalty() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[1.0f32, 2.0, 3.0]], &device).unwrap();
        let input_ids = Tensor::new(&[0u32], &device).unwrap();
        let result = apply_repetition_penalty(&logits, &input_ids, 1.0).unwrap();
        // With penalty 1.0, should be unchanged
        let original: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        let penalized: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!((original[0] - penalized[0]).abs() < 1e-5);
        assert!((original[1] - penalized[1]).abs() < 1e-5);
        assert!((original[2] - penalized[2]).abs() < 1e-5);
    }

    #[test]
    fn test_apply_repetition_penalty_with_penalty() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[2.0f32, 3.0, 4.0]], &device).unwrap();
        let input_ids = Tensor::new(&[0u32], &device).unwrap();
        let penalty = 2.0;
        let result = apply_repetition_penalty(&logits, &input_ids, penalty).unwrap();
        let penalized: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // Token 0 had positive logit, should be divided by penalty
        assert!((penalized[0] - 1.0).abs() < 1e-5); // 2.0 / 2.0 = 1.0
                                                    // Others unchanged
        assert!((penalized[1] - 3.0).abs() < 1e-5);
        assert!((penalized[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_apply_repetition_penalty_negative_logit() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[-2.0f32, 3.0, 4.0]], &device).unwrap();
        let input_ids = Tensor::new(&[0u32], &device).unwrap();
        let penalty = 2.0;
        let result = apply_repetition_penalty(&logits, &input_ids, penalty).unwrap();
        let penalized: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // Token 0 had negative logit, should be multiplied by penalty
        assert!((penalized[0] - (-4.0)).abs() < 1e-5); // -2.0 * 2.0 = -4.0
    }

    #[test]
    fn test_rand_f32_range() {
        // Test that random values are in [0, 1)
        for _ in 0..100 {
            let r = rand_f32();
            assert!(r >= 0.0);
            assert!(r < 1.0);
        }
    }

    #[test]
    fn test_rand_f32_variability() {
        // Test that random values vary
        let values: Vec<f32> = (0..10).map(|_| rand_f32()).collect();
        let unique: std::collections::HashSet<u32> = values.iter().map(|v| v.to_bits()).collect();
        // Should have some variation (not all the same)
        assert!(unique.len() > 1);
    }

    #[test]
    fn test_multinomial_sample_deterministic_probs() {
        let device = Device::Cpu;
        // Probability of 1.0 on one token
        let probs = Tensor::new(&[[0.0f32, 1.0, 0.0, 0.0]], &device).unwrap();
        let result = multinomial_sample(&probs).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 1); // Should always pick index 1
    }

    #[test]
    fn test_sample_with_batch() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[10.0f32, 1.0, 1.0], [1.0, 10.0, 1.0]], &device).unwrap();
        let config = GenerationConfig {
            temperature: 0.001, // Very low temp for deterministic
            ..Default::default()
        };
        let result = sample(&logits, &config).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 0);
        assert_eq!(idx[1], 1);
    }
}
