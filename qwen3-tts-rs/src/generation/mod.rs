//! Generation and sampling utilities for Qwen3-TTS
//!
//! This module provides:
//! - Sampling strategies (greedy, top-k, top-p, temperature)
//! - Generation configuration
//! - Repetition penalty

mod sampling;

pub use sampling::{apply_repetition_penalty, greedy_sample, sample, GenerationConfig};
