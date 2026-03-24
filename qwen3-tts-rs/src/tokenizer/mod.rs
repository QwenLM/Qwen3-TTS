//! Text tokenization for Qwen3-TTS
//!
//! Uses HuggingFace's tokenizers library (written in Rust) for
//! Qwen2TokenizerFast compatibility.

mod text;

pub use text::TextTokenizer;
