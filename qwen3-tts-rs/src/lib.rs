//! # Qwen3-TTS
//!
//! Pure Rust inference for Qwen3-TTS text-to-speech model.
//!
//! ## Features
//!
//! - **CPU**: Default, with optional MKL/Accelerate for faster BLAS
//! - **CUDA**: NVIDIA GPU acceleration
//! - **Metal**: Apple Silicon GPU acceleration
//!
//! ## Example
//!
//! ```rust,ignore
//! use qwen3_tts::{Qwen3TTS, Config};
//!
//! let model = Qwen3TTS::from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")?;
//! let audio = model.synthesize("Hello, world!", None)?;
//! audio.save("output.wav")?;
//! ```

pub mod audio;
pub mod generation;
pub mod models;
pub mod tokenizer;

use anyhow::Result;
use candle_core::{Device, Tensor};

/// Re-exports for convenience
pub use audio::AudioBuffer;
pub use models::config::Qwen3TTSConfig;
pub use models::Qwen3TTSModel;

/// Main TTS interface
pub struct Qwen3TTS {
    /// The underlying model
    model: Qwen3TTSModel,
    /// Text tokenizer
    text_tokenizer: tokenizer::TextTokenizer,
    /// Audio codec for decoding generated tokens to waveform
    audio_codec: models::codec::AudioCodec,
    /// Device to run inference on
    device: Device,
}

impl Qwen3TTS {
    /// Load a model from a HuggingFace model ID or local path
    pub fn from_pretrained(model_id: &str, device: Device) -> Result<Self> {
        tracing::info!("Loading Qwen3-TTS from: {}", model_id);

        // Load config
        let config = Qwen3TTSConfig::from_pretrained(model_id)?;

        // Load text tokenizer
        let text_tokenizer = tokenizer::TextTokenizer::from_pretrained(model_id)?;

        // Load main TTS model
        let model = Qwen3TTSModel::from_pretrained(model_id, &config, &device)?;

        // Load audio codec (tokenizer)
        let codec_id = config.tokenizer_model_id.as_deref().unwrap_or(model_id);
        let audio_codec = models::codec::AudioCodec::from_pretrained(codec_id, &device)?;

        Ok(Self {
            model,
            text_tokenizer,
            audio_codec,
            device,
        })
    }

    /// Synthesize speech from text
    pub fn synthesize(&self, text: &str, options: Option<SynthesisOptions>) -> Result<AudioBuffer> {
        let options = options.unwrap_or_default();

        // Tokenize text
        let input_ids = self.text_tokenizer.encode(text)?;
        let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)?.unsqueeze(0)?; // Add batch dimension

        // Generate codec tokens
        let codec_tokens = self.model.generate(
            &input_tensor,
            options.speaker_embedding.as_ref(),
            &generation::GenerationConfig {
                max_new_tokens: options.max_length,
                temperature: options.temperature,
                top_k: options.top_k,
                top_p: options.top_p,
                repetition_penalty: options.repetition_penalty,
                eos_token_id: options.eos_token_id,
            },
        )?;

        // Decode codec tokens to audio waveform
        let waveform = self.audio_codec.decode(&codec_tokens)?;

        AudioBuffer::from_tensor(waveform, 24000)
    }

    /// Get the device this model is running on
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Audio end-of-sequence token ID for Qwen3-TTS
pub const AUDIO_EOS_TOKEN_ID: u32 = 151670;

/// Options for speech synthesis
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// Maximum number of tokens to generate
    pub max_length: usize,
    /// Sampling temperature (higher = more random)
    pub temperature: f64,
    /// Top-k sampling
    pub top_k: usize,
    /// Top-p (nucleus) sampling
    pub top_p: f64,
    /// Repetition penalty
    pub repetition_penalty: f64,
    /// Optional speaker embedding for voice cloning
    pub speaker_embedding: Option<Tensor>,
    /// Language code (e.g., "en", "zh")
    pub language: Option<String>,
    /// End-of-sequence token ID (defaults to audio_end token 151670)
    pub eos_token_id: Option<u32>,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_length: 2048,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.0,
            speaker_embedding: None,
            language: None,
            eos_token_id: Some(AUDIO_EOS_TOKEN_ID),
        }
    }
}

/// Select the best available device
pub fn auto_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::cuda_if_available(0) {
            if device.is_cuda() {
                tracing::info!("Using CUDA device");
                return Ok(device);
            }
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            tracing::info!("Using Metal device");
            return Ok(device);
        }
    }

    tracing::info!("Using CPU device");
    Ok(Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_options_default() {
        let options = SynthesisOptions::default();
        assert_eq!(options.max_length, 2048);
        assert!((options.temperature - 0.7).abs() < 1e-6);
        assert_eq!(options.top_k, 50);
        assert!((options.top_p - 0.9).abs() < 1e-6);
        assert!((options.repetition_penalty - 1.0).abs() < 1e-6);
        assert!(options.speaker_embedding.is_none());
        assert!(options.language.is_none());
        assert_eq!(options.eos_token_id, Some(AUDIO_EOS_TOKEN_ID));
    }

    #[test]
    fn test_synthesis_options_custom() {
        let options = SynthesisOptions {
            max_length: 512,
            temperature: 0.5,
            top_k: 10,
            top_p: 0.8,
            repetition_penalty: 1.2,
            speaker_embedding: None,
            language: Some("en".to_string()),
            eos_token_id: Some(AUDIO_EOS_TOKEN_ID),
        };
        assert_eq!(options.max_length, 512);
        assert!((options.temperature - 0.5).abs() < 1e-6);
        assert_eq!(options.language, Some("en".to_string()));
        assert_eq!(options.eos_token_id, Some(151670));
    }

    #[test]
    fn test_synthesis_options_clone() {
        let options = SynthesisOptions::default();
        let cloned = options.clone();
        assert_eq!(cloned.max_length, options.max_length);
        assert_eq!(cloned.top_k, options.top_k);
    }

    #[test]
    fn test_synthesis_options_debug() {
        let options = SynthesisOptions::default();
        let debug_str = format!("{:?}", options);
        assert!(debug_str.contains("max_length"));
        assert!(debug_str.contains("2048"));
    }

    #[test]
    fn test_auto_device() {
        // Should always succeed on CPU
        let device = auto_device().unwrap();
        // Just verify it returns a valid device
        assert!(
            matches!(device, Device::Cpu)
                || matches!(device, Device::Cuda(_))
                || matches!(device, Device::Metal(_))
        );
    }

    #[test]
    fn test_audio_buffer_reexport() {
        // Verify re-exports work
        let buffer = AudioBuffer::new(vec![0.0f32; 100], 24000);
        assert_eq!(buffer.sample_rate, 24000);
    }

    #[test]
    fn test_config_reexport() {
        // Verify config re-export works
        let config = Qwen3TTSConfig::default();
        assert_eq!(config.model_type, "qwen3_tts");
    }
}
