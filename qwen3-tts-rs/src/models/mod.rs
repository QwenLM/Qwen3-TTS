//! Neural network models for Qwen3-TTS
//!
//! This module contains:
//! - `config`: Model configuration
//! - `qwen3_tts`: Main TTS model (Talker)
//! - `speaker`: Speaker encoder (ECAPA-TDNN)
//! - `codec`: Audio codec for encoding/decoding

pub mod codec;
pub mod config;
pub mod qwen3_tts;
pub mod speaker;

pub use config::Qwen3TTSConfig;
pub use qwen3_tts::Qwen3TTSModel;
pub use speaker::SpeakerEncoder;
