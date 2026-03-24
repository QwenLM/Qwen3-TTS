//! Audio processing utilities for Qwen3-TTS
//!
//! This module provides:
//! - WAV file I/O
//! - Audio resampling
//! - Mel-spectrogram computation
//! - Audio normalization

mod io;
mod mel;
pub mod resample;

pub use io::{load_wav, save_wav, AudioBuffer};
pub use mel::{MelConfig, MelSpectrogram};
pub use resample::{resample, resample_to_24k, Resampler};

/// Standard sample rate used by Qwen3-TTS
pub const SAMPLE_RATE: u32 = 24000;

/// Number of mel bands for speaker encoder
pub const N_MELS: usize = 128;
