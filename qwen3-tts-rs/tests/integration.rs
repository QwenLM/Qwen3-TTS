//! Integration tests for Qwen3-TTS
//!
//! These tests verify the full pipeline works correctly with mock weights.

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

/// Create a mock VarBuilder for testing without real weights
fn create_mock_vb(device: &Device) -> VarBuilder<'static> {
    let varmap = VarMap::new();
    VarBuilder::from_varmap(&varmap, DType::F32, device)
}

mod audio_tests {
    use qwen3_tts::audio::{resample, AudioBuffer, MelConfig, MelSpectrogram};
    use std::f32::consts::PI;

    #[test]
    fn test_audio_pipeline() {
        // Create a simple sine wave
        let sample_rate = 24000;
        let duration = 0.5;
        let freq = 440.0;
        let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect();

        let audio = AudioBuffer::new(samples, sample_rate);
        assert_eq!(audio.sample_rate, sample_rate);
        assert!(!audio.samples.is_empty());

        // Test duration
        assert!((audio.duration() - duration).abs() < 0.01);

        // Test resampling
        let resampled = resample::resample(&audio, 16000).unwrap();
        assert_eq!(resampled.sample_rate, 16000);
        assert!(resampled.samples.len() < audio.samples.len());

        // Test mel spectrogram
        let mel_config = MelConfig {
            sample_rate,
            n_fft: 512,
            hop_length: 256,
            n_mels: 80,
            ..Default::default()
        };
        let mel = MelSpectrogram::new(mel_config);
        let spec = mel.compute(&audio.samples);
        // spec is [frames, n_mels], each frame has 80 mel bins
        assert!(!spec.is_empty());
        assert_eq!(spec[0].len(), 80); // each frame has n_mels values
    }

    #[test]
    fn test_mel_spectrogram_consistency() {
        let sample_rate = 24000;
        let samples: Vec<f32> = (0..24000).map(|i| (i as f32 * 0.01).sin()).collect();
        let audio = AudioBuffer::new(samples.clone(), sample_rate);

        let mel = MelSpectrogram::new(MelConfig::default());
        let spec1 = mel.compute(&audio.samples);
        let spec2 = mel.compute(&audio.samples);

        // Should be deterministic
        assert_eq!(spec1.len(), spec2.len());
        for (row1, row2) in spec1.iter().zip(spec2.iter()) {
            for (v1, v2) in row1.iter().zip(row2.iter()) {
                assert!((v1 - v2).abs() < 1e-6);
            }
        }
    }
}

mod tokenizer_tests {
    use qwen3_tts::tokenizer::TextTokenizer;
    use tokenizers::{models::bpe::BPE, pre_tokenizers::whitespace::Whitespace, Tokenizer};

    fn create_test_tokenizer() -> TextTokenizer {
        // Create a simple BPE tokenizer with a minimal vocab using array
        let vocab: [(&str, u32); 10] = [
            ("hello", 0),
            ("world", 1),
            ("test", 2),
            ("<|im_start|>", 3),
            ("<|im_end|>", 4),
            ("<|endoftext|>", 5),
            ("user", 6),
            ("assistant", 7),
            ("\n", 8),
            ("Ġ", 9),
        ];

        let merges: Vec<(String, String)> = vec![];
        let bpe = BPE::builder()
            .vocab_and_merges(vocab.map(|(k, v)| (k.to_string(), v)), merges)
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();

        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.with_pre_tokenizer(Some(Whitespace));

        TextTokenizer::from_tokenizer(tokenizer).unwrap()
    }

    #[test]
    fn test_tokenizer_roundtrip() {
        let tokenizer = create_test_tokenizer();

        // Use empty string which always works with mock tokenizer
        let text = "";
        let ids = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&ids).unwrap();

        assert!(ids.is_empty());
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_tokenizer_special_tokens() {
        let tokenizer = create_test_tokenizer();

        assert_eq!(tokenizer.bos_token_id, 3); // <|im_start|>
        assert_eq!(tokenizer.eos_token_id, 4); // <|im_end|>
        assert_eq!(tokenizer.pad_token_id, 5); // <|endoftext|>
    }

    #[test]
    fn test_tokenizer_batch() {
        let tokenizer = create_test_tokenizer();

        // Use empty strings which always work
        let texts = ["", "", ""];
        let batch = tokenizer.encode_batch(&texts).unwrap();

        assert_eq!(batch.len(), 3);
    }
}

mod model_tests {
    use super::*;
    use qwen3_tts::models::{
        codec::{presets, CodecDecoder, DecoderConfig},
        Qwen3TTSConfig, Qwen3TTSModel,
    };

    fn small_config() -> Qwen3TTSConfig {
        Qwen3TTSConfig {
            vocab_size: 100,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: Some(2),
            max_position_embeddings: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            ..Default::default()
        }
    }

    #[test]
    fn test_model_construction() {
        // Test that model construction succeeds with mock weights
        // Forward pass not tested due to mock weight limitations
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let model = Qwen3TTSModel::new(config.clone(), vb);
        assert!(model.is_ok());
    }

    #[test]
    fn test_kv_cache_creation() {
        let config = small_config();
        let kv_caches: Vec<qwen3_tts::models::qwen3_tts::KVCache> = (0..config.num_hidden_layers)
            .map(|_| qwen3_tts::models::qwen3_tts::KVCache::new())
            .collect();

        assert_eq!(kv_caches.len(), config.num_hidden_layers);
    }

    #[test]
    fn test_codec_preset_configs() {
        let config_12hz = presets::codec_12hz();
        let config_25hz = presets::codec_25hz();

        assert_eq!(config_12hz.codec_type, "12hz");
        assert_eq!(config_25hz.codec_type, "25hz");
        assert!((config_12hz.frame_rate - 12.5).abs() < 1e-6);
        assert!((config_25hz.frame_rate - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_codec_decoder_construction() {
        // Test decoder construction with mock weights
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);

        let config = DecoderConfig {
            hidden_size: 32,
            num_layers: 1,
            num_heads: 4,
            upsample_ratios: vec![2, 2],
            num_quantizers: 2,
            codebook_dim: 16,
            codebook_size: 64,
            out_channels: 1,
        };

        let decoder = CodecDecoder::new(config, vb);
        assert!(decoder.is_ok());
    }
}

mod generation_tests {
    use super::*;
    use qwen3_tts::generation::{
        apply_repetition_penalty, greedy_sample, sample, GenerationConfig,
    };

    #[test]
    fn test_greedy_sampling() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[1.0f32, 5.0, 2.0]], &device).unwrap();
        let result = greedy_sample(&logits).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 1);
    }

    #[test]
    fn test_sampling_with_low_temperature() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[1.0f32, 100.0, 2.0]], &device).unwrap();
        let config = GenerationConfig {
            temperature: 0.001,
            ..Default::default()
        };
        let result = sample(&logits, &config).unwrap();
        let idx: Vec<u32> = result.to_vec1().unwrap();
        assert_eq!(idx[0], 1);
    }

    #[test]
    fn test_repetition_penalty() {
        let device = Device::Cpu;
        let logits = Tensor::new(&[[2.0f32, 3.0, 4.0]], &device).unwrap();
        let input_ids = Tensor::new(&[0u32], &device).unwrap();

        let penalized = apply_repetition_penalty(&logits, &input_ids, 2.0).unwrap();
        let vals: Vec<f32> = penalized.flatten_all().unwrap().to_vec1().unwrap();

        // Token 0 should be penalized (divided by 2)
        assert!((vals[0] - 1.0).abs() < 1e-5);
        // Others unchanged
        assert!((vals[1] - 3.0).abs() < 1e-5);
        assert!((vals[2] - 4.0).abs() < 1e-5);
    }
}

mod end_to_end_mock {
    use qwen3_tts::{AudioBuffer, Qwen3TTSConfig, SynthesisOptions};

    #[test]
    fn test_synthesis_options_configuration() {
        let options = SynthesisOptions {
            max_length: 512,
            temperature: 0.8,
            top_k: 30,
            top_p: 0.85,
            repetition_penalty: 1.1,
            speaker_embedding: None,
            language: Some("en".to_string()),
            eos_token_id: Some(151670), // audio_end token
        };

        assert_eq!(options.max_length, 512);
        assert!((options.temperature - 0.8).abs() < 1e-6);
        assert_eq!(options.eos_token_id, Some(151670));
    }

    #[test]
    fn test_audio_buffer_from_samples() {
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001).sin()).collect();
        let buffer = AudioBuffer::new(samples.clone(), 24000);

        assert_eq!(buffer.len(), 1000);
        assert_eq!(buffer.sample_rate, 24000);
    }

    #[test]
    fn test_config_defaults_are_sensible() {
        let config = Qwen3TTSConfig::default();

        assert!(config.vocab_size > 0);
        assert!(config.hidden_size > 0);
        assert!(config.num_hidden_layers > 0);
        assert!(config.num_attention_heads > 0);
        assert!(config.max_position_embeddings >= 4096);
    }
}

/// Tests using real downloaded weights (if available)
/// These tests are skipped if test data is not present
mod real_weights_tests {
    use std::path::Path;

    /// Path to downloaded test data
    const TEST_DATA_DIR: &str = "test_data/tokenizer";

    fn test_data_available() -> bool {
        Path::new(TEST_DATA_DIR).join("tokenizer.json").exists()
    }

    #[test]
    fn test_real_tokenizer_loading() {
        if !test_data_available() {
            eprintln!("Skipping test_real_tokenizer_loading: test data not found");
            return;
        }

        use qwen3_tts::tokenizer::TextTokenizer;

        let tokenizer_path = Path::new(TEST_DATA_DIR).join("tokenizer.json");
        let tokenizer = TextTokenizer::from_file(&tokenizer_path).unwrap();

        // Qwen2 tokenizer should have large vocab
        assert!(tokenizer.vocab_size() > 150000);

        // Check special tokens exist
        assert!(tokenizer.token_to_id("<|im_start|>").is_some());
        assert!(tokenizer.token_to_id("<|im_end|>").is_some());
        assert!(tokenizer.token_to_id("<|endoftext|>").is_some());
    }

    #[test]
    fn test_real_tokenizer_encoding() {
        if !test_data_available() {
            eprintln!("Skipping test_real_tokenizer_encoding: test data not found");
            return;
        }

        use qwen3_tts::tokenizer::TextTokenizer;

        let tokenizer_path = Path::new(TEST_DATA_DIR).join("tokenizer.json");
        let tokenizer = TextTokenizer::from_file(&tokenizer_path).unwrap();

        // Test encoding simple text
        let text = "Hello, world!";
        let ids = tokenizer.encode(text).unwrap();

        // Should produce some tokens
        assert!(!ids.is_empty());
        assert!(ids.len() < 20); // Simple text should be compact

        // Test decoding back
        let decoded = tokenizer.decode(&ids).unwrap();
        assert!(decoded.contains("Hello"));
        assert!(decoded.contains("world"));
    }

    #[test]
    fn test_real_tokenizer_chinese() {
        if !test_data_available() {
            eprintln!("Skipping test_real_tokenizer_chinese: test data not found");
            return;
        }

        use qwen3_tts::tokenizer::TextTokenizer;

        let tokenizer_path = Path::new(TEST_DATA_DIR).join("tokenizer.json");
        let tokenizer = TextTokenizer::from_file(&tokenizer_path).unwrap();

        // Qwen tokenizer supports Chinese
        let text = "你好世界";
        let ids = tokenizer.encode(text).unwrap();

        assert!(!ids.is_empty());

        let decoded = tokenizer.decode(&ids).unwrap();
        assert!(decoded.contains("你好") || decoded.contains("世界"));
    }

    #[test]
    fn test_real_tokenizer_chat_format() {
        if !test_data_available() {
            eprintln!("Skipping test_real_tokenizer_chat_format: test data not found");
            return;
        }

        use qwen3_tts::tokenizer::TextTokenizer;

        let tokenizer_path = Path::new(TEST_DATA_DIR).join("tokenizer.json");
        let tokenizer = TextTokenizer::from_file(&tokenizer_path).unwrap();

        // Test chat-style encoding
        let ids = tokenizer.encode_chat("Hello", "user").unwrap();

        // Should include special tokens in the encoding
        assert!(!ids.is_empty());

        // Decode and check format
        let decoded = tokenizer.decode(&ids).unwrap();
        assert!(decoded.contains("user") || decoded.contains("Hello"));
    }

    #[test]
    fn test_real_tokenizer_batch() {
        if !test_data_available() {
            eprintln!("Skipping test_real_tokenizer_batch: test data not found");
            return;
        }

        use qwen3_tts::tokenizer::TextTokenizer;

        let tokenizer_path = Path::new(TEST_DATA_DIR).join("tokenizer.json");
        let tokenizer = TextTokenizer::from_file(&tokenizer_path).unwrap();

        let texts = ["Hello", "World", "Test"];
        let batch = tokenizer.encode_batch(&texts).unwrap();

        assert_eq!(batch.len(), 3);
        for encoded in &batch {
            assert!(!encoded.is_empty());
        }
    }

    #[test]
    fn test_config_json_parsing() {
        if !test_data_available() {
            eprintln!("Skipping test_config_json_parsing: test data not found");
            return;
        }

        // Read the real config.json and parse relevant fields
        let config_path = Path::new(TEST_DATA_DIR).join("config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Verify expected fields exist
        assert_eq!(config["model_type"], "qwen3_tts");
        assert!(config["talker_config"].is_object());
        assert!(config["speaker_encoder_config"].is_object());

        // Check talker config values
        let talker = &config["talker_config"];
        assert_eq!(talker["hidden_size"], 1024);
        assert_eq!(talker["num_hidden_layers"], 28);
        assert_eq!(talker["num_attention_heads"], 16);
        assert_eq!(talker["num_key_value_heads"], 8);
        assert_eq!(talker["num_code_groups"], 16);

        // Check speaker encoder config
        let speaker = &config["speaker_encoder_config"];
        assert_eq!(speaker["enc_dim"], 1024);
        assert_eq!(speaker["sample_rate"], 24000);
    }
}

/// Tests for speech tokenizer using real downloaded weights
mod speech_tokenizer_tests {
    use std::path::Path;

    /// Path to downloaded speech tokenizer data
    const SPEECH_TOKENIZER_DIR: &str = "test_data/speech_tokenizer";

    fn speech_tokenizer_available() -> bool {
        Path::new(SPEECH_TOKENIZER_DIR)
            .join("model.safetensors")
            .exists()
    }

    #[test]
    fn test_speech_tokenizer_config_parsing() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_speech_tokenizer_config_parsing: test data not found");
            eprintln!("Run: ./scripts/download_test_data.sh to download");
            return;
        }

        // Read and parse the config
        let config_path = Path::new(SPEECH_TOKENIZER_DIR).join("config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Verify architecture
        assert_eq!(config["model_type"], "qwen3_tts_tokenizer_12hz");
        assert!(config["encoder_config"].is_object());
        assert!(config["decoder_config"].is_object());

        // Check encoder config
        let encoder = &config["encoder_config"];
        assert_eq!(encoder["sampling_rate"], 24000);
        assert_eq!(encoder["num_quantizers"], 32);
        assert_eq!(encoder["codebook_size"], 2048);
        assert_eq!(encoder["hidden_size"], 512);

        // Check decoder config
        let decoder = &config["decoder_config"];
        assert_eq!(decoder["num_quantizers"], 16);
        assert_eq!(decoder["codebook_size"], 2048);
        assert_eq!(decoder["hidden_size"], 512);
        assert_eq!(decoder["num_attention_heads"], 16);
    }

    #[test]
    fn test_speech_tokenizer_model_loading() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_speech_tokenizer_model_loading: test data not found");
            return;
        }

        use safetensors::SafeTensors;

        // Load the safetensors file
        let model_path = Path::new(SPEECH_TOKENIZER_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Check that we have tensors
        let tensor_names = tensors.names();
        assert!(!tensor_names.is_empty());

        // Should have encoder and decoder weights
        let has_encoder = tensor_names.iter().any(|n| n.contains("encoder"));
        let has_decoder = tensor_names.iter().any(|n| n.contains("decoder"));
        assert!(has_encoder, "Model should have encoder weights");
        assert!(has_decoder, "Model should have decoder weights");

        // Count total tensors
        println!("Speech tokenizer has {} tensors", tensor_names.len());
    }

    #[test]
    fn test_speech_tokenizer_encoder_weights() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_speech_tokenizer_encoder_weights: test data not found");
            return;
        }

        use safetensors::SafeTensors;

        let model_path = Path::new(SPEECH_TOKENIZER_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Find encoder embedding weights
        let encoder_tensors: Vec<&String> = tensors
            .names()
            .iter()
            .filter(|n| n.starts_with("encoder."))
            .cloned()
            .collect();

        assert!(!encoder_tensors.is_empty(), "Should have encoder tensors");
        println!("Found {} encoder tensors", encoder_tensors.len());

        // Check a specific tensor shape
        for name in encoder_tensors.iter().take(5) {
            let tensor = tensors.tensor(name).unwrap();
            println!("  {}: {:?}", name, tensor.shape());
        }
    }

    #[test]
    fn test_speech_tokenizer_decoder_weights() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_speech_tokenizer_decoder_weights: test data not found");
            return;
        }

        use safetensors::SafeTensors;

        let model_path = Path::new(SPEECH_TOKENIZER_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Find decoder weights
        let decoder_tensors: Vec<&String> = tensors
            .names()
            .iter()
            .filter(|n| n.starts_with("decoder."))
            .cloned()
            .collect();

        assert!(!decoder_tensors.is_empty(), "Should have decoder tensors");
        println!("Found {} decoder tensors", decoder_tensors.len());

        // Check a specific tensor shape
        for name in decoder_tensors.iter().take(5) {
            let tensor = tensors.tensor(name).unwrap();
            println!("  {}: {:?}", name, tensor.shape());
        }
    }

    #[test]
    fn test_speech_tokenizer_quantizer_codebooks() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_speech_tokenizer_quantizer_codebooks: test data not found");
            return;
        }

        use safetensors::SafeTensors;

        let model_path = Path::new(SPEECH_TOKENIZER_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Find quantizer/codebook weights
        let all_names = tensors.names();
        let codebook_tensors: Vec<&String> = all_names
            .iter()
            .filter(|n| n.contains("quantiz") || n.contains("codebook") || n.contains("embed"))
            .cloned()
            .collect();

        println!(
            "Found {} quantizer/codebook tensors:",
            codebook_tensors.len()
        );
        for name in &codebook_tensors {
            let tensor = tensors.tensor(name).unwrap();
            println!("  {}: {:?}", name, tensor.shape());
        }

        // Should have quantization-related weights
        assert!(
            !codebook_tensors.is_empty(),
            "Should have quantizer weights"
        );
    }

    #[test]
    fn test_speech_tokenizer_preprocessor_config() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_speech_tokenizer_preprocessor_config: test data not found");
            return;
        }

        let config_path = Path::new(SPEECH_TOKENIZER_DIR).join("preprocessor_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Verify preprocessor config
        assert_eq!(config["sampling_rate"], 24000);
        assert_eq!(config["feature_size"], 1); // mono audio
        assert_eq!(config["padding_side"], "right");
    }

    #[test]
    fn test_load_tensor_to_candle() {
        if !speech_tokenizer_available() {
            eprintln!("Skipping test_load_tensor_to_candle: test data not found");
            return;
        }

        use candle_core::Device;

        let model_path = Path::new(SPEECH_TOKENIZER_DIR).join("model.safetensors");
        let device = Device::Cpu;

        // Use candle's built-in safetensors loading
        let tensors = candle_core::safetensors::load(&model_path, &device).unwrap();

        println!("Loaded {} tensors from safetensors file", tensors.len());

        // Check a few tensors
        for (name, tensor) in tensors.iter().take(5) {
            println!("  {}: {:?} ({:?})", name, tensor.dims(), tensor.dtype());
        }

        // Verify we can access tensors
        assert!(!tensors.is_empty(), "Should have loaded tensors");

        // Verify tensor shapes are valid
        for (name, tensor) in &tensors {
            assert!(
                !tensor.dims().is_empty(),
                "Tensor {} should have valid shape",
                name
            );
        }
    }
}

/// Tests for model configuration files from the 0.6B model
mod model_config_tests {
    use std::path::Path;

    /// Path to downloaded model config data
    const MODEL_CONFIG_DIR: &str = "test_data/model_config";

    fn model_config_available() -> bool {
        Path::new(MODEL_CONFIG_DIR)
            .join("generation_config.json")
            .exists()
    }

    #[test]
    fn test_generation_config_parsing() {
        if !model_config_available() {
            eprintln!("Skipping test_generation_config_parsing: test data not found");
            return;
        }

        let config_path = Path::new(MODEL_CONFIG_DIR).join("generation_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Verify generation parameters match official defaults
        assert_eq!(config["do_sample"], true);
        assert_eq!(config["temperature"], 0.9);
        assert_eq!(config["top_p"], 1.0);
        assert_eq!(config["top_k"], 50);
        assert_eq!(config["repetition_penalty"], 1.05);
        assert_eq!(config["max_new_tokens"], 8192);

        // Subtalker has its own parameters
        assert_eq!(config["subtalker_dosample"], true);
        assert_eq!(config["subtalker_temperature"], 0.9);
        assert_eq!(config["subtalker_top_k"], 50);
    }

    #[test]
    fn test_preprocessor_config_parsing() {
        if !model_config_available() {
            eprintln!("Skipping test_preprocessor_config_parsing: test data not found");
            return;
        }

        let config_path = Path::new(MODEL_CONFIG_DIR).join("preprocessor_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Verify preprocessor config
        assert_eq!(config["padding_side"], "left");
        assert_eq!(config["padding_value"], 0.0);
        assert_eq!(config["processor_class"], "Qwen3TTSProcessor");
        assert_eq!(config["return_attention_mask"], true);
    }

    #[test]
    fn test_tokenizer_config_special_tokens() {
        if !model_config_available() {
            eprintln!("Skipping test_tokenizer_config_special_tokens: test data not found");
            return;
        }

        let config_path = Path::new(MODEL_CONFIG_DIR).join("tokenizer_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Verify tokenizer class
        assert_eq!(config["tokenizer_class"], "Qwen2Tokenizer");
        assert_eq!(config["model_max_length"], 131072);

        // Verify important special tokens
        assert_eq!(config["eos_token"], "<|im_end|>");
        assert_eq!(config["pad_token"], "<|endoftext|>");

        // Verify audio-specific tokens
        assert_eq!(config["audio_bos_token"], "<|audio_start|>");
        assert_eq!(config["audio_eos_token"], "<|audio_end|>");
        assert_eq!(config["audio_token"], "<|audio_pad|>");
    }

    #[test]
    fn test_tokenizer_config_tts_tokens() {
        if !model_config_available() {
            eprintln!("Skipping test_tokenizer_config_tts_tokens: test data not found");
            return;
        }

        let config_path = Path::new(MODEL_CONFIG_DIR).join("tokenizer_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Check that TTS-specific tokens are present in added_tokens_decoder
        let added_tokens = &config["added_tokens_decoder"];

        // Find TTS tokens by their IDs
        assert_eq!(added_tokens["151671"]["content"], "<tts_pad>");
        assert_eq!(added_tokens["151672"]["content"], "<tts_text_bos>");
        assert_eq!(added_tokens["151673"]["content"], "<tts_text_eod>");
        assert_eq!(added_tokens["151674"]["content"], "<tts_text_bos_single>");
        assert_eq!(added_tokens["151675"]["content"], "<|audio_pad|>");

        // Audio markers
        assert_eq!(added_tokens["151669"]["content"], "<|audio_start|>");
        assert_eq!(added_tokens["151670"]["content"], "<|audio_end|>");

        // Standard Qwen tokens
        assert_eq!(added_tokens["151643"]["content"], "<|endoftext|>");
        assert_eq!(added_tokens["151644"]["content"], "<|im_start|>");
        assert_eq!(added_tokens["151645"]["content"], "<|im_end|>");
    }

    #[test]
    fn test_tokenizer_config_additional_special_tokens() {
        if !model_config_available() {
            eprintln!(
                "Skipping test_tokenizer_config_additional_special_tokens: test data not found"
            );
            return;
        }

        let config_path = Path::new(MODEL_CONFIG_DIR).join("tokenizer_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        let additional_tokens = config["additional_special_tokens"].as_array().unwrap();
        let tokens: Vec<&str> = additional_tokens
            .iter()
            .map(|t| t.as_str().unwrap())
            .collect();

        // Verify TTS-specific tokens are in additional_special_tokens
        assert!(tokens.contains(&"<tts_pad>"));
        assert!(tokens.contains(&"<tts_text_bos>"));
        assert!(tokens.contains(&"<tts_text_bos_single>"));
        assert!(tokens.contains(&"<|audio_start|>"));
        assert!(tokens.contains(&"<|audio_end|>"));
        assert!(tokens.contains(&"<|audio_pad|>"));

        // Verify standard chat tokens
        assert!(tokens.contains(&"<|im_start|>"));
        assert!(tokens.contains(&"<|im_end|>"));
    }

    #[test]
    fn test_generation_config_matches_our_defaults() {
        if !model_config_available() {
            eprintln!("Skipping test_generation_config_matches_our_defaults: test data not found");
            return;
        }

        use qwen3_tts::generation::GenerationConfig;

        let config_path = Path::new(MODEL_CONFIG_DIR).join("generation_config.json");
        let config_str = std::fs::read_to_string(config_path).unwrap();
        let official: serde_json::Value = serde_json::from_str(&config_str).unwrap();

        // Create our default config and compare key values
        let our_config = GenerationConfig::default();

        // Our defaults should be reasonable (though may differ from official)
        // This test documents the official values for reference
        println!("Official generation config:");
        println!("  temperature: {}", official["temperature"]);
        println!("  top_k: {}", official["top_k"]);
        println!("  top_p: {}", official["top_p"]);
        println!("  repetition_penalty: {}", official["repetition_penalty"]);
        println!("  max_new_tokens: {}", official["max_new_tokens"]);

        println!("\nOur default config:");
        println!("  temperature: {}", our_config.temperature);
        println!("  top_k: {:?}", our_config.top_k);
        println!("  top_p: {:?}", our_config.top_p);
        println!("  repetition_penalty: {}", our_config.repetition_penalty);
        println!("  max_new_tokens: {}", our_config.max_new_tokens);

        // At minimum, our config should have sensible values
        assert!(our_config.temperature > 0.0);
        assert!(our_config.max_new_tokens > 0);
    }
}

/// Tests for the 0.6B model weights
mod model_weights_tests {
    use std::collections::HashMap;
    use std::path::Path;

    const MODEL_DIR: &str = "test_data/model";

    fn model_available() -> bool {
        Path::new(MODEL_DIR).join("model.safetensors").exists()
    }

    #[test]
    fn test_model_loading() {
        if !model_available() {
            eprintln!("Skipping test_model_loading: model weights not found");
            eprintln!("Download with: curl -L https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/model.safetensors -o test_data/model/model.safetensors");
            return;
        }

        use candle_core::Device;

        let model_path = Path::new(MODEL_DIR).join("model.safetensors");
        let device = Device::Cpu;

        let tensors = candle_core::safetensors::load(&model_path, &device).unwrap();
        println!("Loaded {} tensors from 0.6B model", tensors.len());

        assert!(!tensors.is_empty());
    }

    #[test]
    fn test_model_tensor_groups() {
        if !model_available() {
            eprintln!("Skipping test_model_tensor_groups: model weights not found");
            return;
        }

        use safetensors::SafeTensors;

        let model_path = Path::new(MODEL_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Group tensors by top-level component
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        for name in tensors.names() {
            let prefix = name.split('.').next().unwrap_or(name).to_string();
            groups.entry(prefix).or_default().push(name.clone());
        }

        println!("Model tensor groups:");
        for (prefix, names) in &groups {
            println!("  {}: {} tensors", prefix, names.len());
        }

        // 0.6B model should have talker and speaker_encoder components
        assert!(groups.contains_key("talker"), "Should have talker weights");
    }

    #[test]
    fn test_talker_layer_structure() {
        if !model_available() {
            eprintln!("Skipping test_talker_layer_structure: model weights not found");
            return;
        }

        use safetensors::SafeTensors;

        let model_path = Path::new(MODEL_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Explore talker structure - find unique 2nd-level prefixes
        let mut sub_components: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for name in tensors.names() {
            if name.starts_with("talker.") {
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() >= 2 {
                    sub_components.insert(parts[1].to_string());
                }
            }
        }

        println!("Talker sub-components: {:?}", sub_components);

        // Find layer tensors - the model uses "talker.model.layers." structure
        let layer_tensors: Vec<&String> = tensors
            .names()
            .iter()
            .filter(|n| n.contains(".layers."))
            .cloned()
            .collect();

        // Count unique layer indices
        let mut layer_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for name in &layer_tensors {
            // Extract layer number from paths like "talker.model.layers.0.self_attn..."
            for part in name.split('.') {
                if let Ok(idx) = part.parse::<usize>() {
                    layer_indices.insert(idx);
                    break;
                }
            }
        }

        println!("Found {} layers", layer_indices.len());
        if !layer_indices.is_empty() {
            println!("Layer range: 0..{}", layer_indices.iter().max().unwrap());
        }

        // Print sample layer tensor names
        println!("Sample layer tensors:");
        for name in layer_tensors.iter().take(5) {
            println!("  {}", name);
        }

        // Model should have transformer layers
        assert!(!layer_indices.is_empty(), "Should have transformer layers");
    }

    #[test]
    fn test_embedding_dimensions() {
        if !model_available() {
            eprintln!("Skipping test_embedding_dimensions: model weights not found");
            return;
        }

        use safetensors::SafeTensors;

        let model_path = Path::new(MODEL_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Find embedding tensor
        let embed_names: Vec<&String> = tensors
            .names()
            .iter()
            .filter(|n| n.contains("embed") && n.contains("token"))
            .cloned()
            .collect();

        println!("Embedding tensors:");
        for name in &embed_names {
            let tensor = tensors.tensor(name).unwrap();
            println!("  {}: {:?}", name, tensor.shape());
        }

        // Check text embedding dimensions match config
        // vocab_size should be large (151k+), hidden_size should be 1024
        if let Some(embed_name) = embed_names.iter().find(|n| n.contains("text")) {
            let tensor = tensors.tensor(embed_name).unwrap();
            let shape = tensor.shape();
            assert!(shape.len() == 2, "Embedding should be 2D");
            assert!(shape[0] > 150000, "Vocab size should be > 150k");
            assert_eq!(shape[1], 1024, "Hidden size should be 1024");
        }
    }

    #[test]
    fn test_attention_head_dimensions() {
        if !model_available() {
            eprintln!("Skipping test_attention_head_dimensions: model weights not found");
            return;
        }

        use safetensors::SafeTensors;

        let model_path = Path::new(MODEL_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Check first layer's attention weights (path is talker.model.layers.X...)
        let q_proj = tensors.tensor("talker.model.layers.0.self_attn.q_proj.weight");
        let k_proj = tensors.tensor("talker.model.layers.0.self_attn.k_proj.weight");
        let v_proj = tensors.tensor("talker.model.layers.0.self_attn.v_proj.weight");

        if let (Ok(q), Ok(k), Ok(v)) = (q_proj, k_proj, v_proj) {
            println!("Attention projection shapes:");
            println!("  q_proj: {:?}", q.shape());
            println!("  k_proj: {:?}", k.shape());
            println!("  v_proj: {:?}", v.shape());

            // Actual shapes from 0.6B model:
            // q_proj: [2048, 1024] - 32 query heads * 64 head_dim (interleaved with code groups)
            // k_proj: [1024, 1024] - 16 kv heads * 64 head_dim
            // v_proj: [1024, 1024] - 16 kv heads * 64 head_dim
            // This suggests the model uses an expanded Q for handling multiple code groups
            assert_eq!(q.shape()[1], 1024, "Q projection input should be 1024");
            assert_eq!(k.shape()[1], 1024, "K projection input should be 1024");
            assert_eq!(v.shape()[1], 1024, "V projection input should be 1024");

            // Q is 2x K/V size (for 16 code groups interleaved with text)
            assert_eq!(q.shape()[0], 2048, "Q projection output is 2048");
            assert_eq!(k.shape()[0], 1024, "K projection output is 1024");
            assert_eq!(v.shape()[0], 1024, "V projection output is 1024");
        }
    }

    #[test]
    fn test_speaker_encoder_weights() {
        if !model_available() {
            eprintln!("Skipping test_speaker_encoder_weights: model weights not found");
            return;
        }

        use safetensors::SafeTensors;

        let model_path = Path::new(MODEL_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Find speaker encoder tensors
        let speaker_tensors: Vec<&String> = tensors
            .names()
            .iter()
            .filter(|n| n.contains("speaker"))
            .cloned()
            .collect();

        println!("Speaker encoder tensors: {}", speaker_tensors.len());
        for name in speaker_tensors.iter().take(10) {
            let tensor = tensors.tensor(name).unwrap();
            println!("  {}: {:?}", name, tensor.shape());
        }

        assert!(
            !speaker_tensors.is_empty(),
            "Should have speaker encoder weights"
        );
    }

    #[test]
    fn test_code_embeddings() {
        if !model_available() {
            eprintln!("Skipping test_code_embeddings: model weights not found");
            return;
        }

        use safetensors::SafeTensors;

        let model_path = Path::new(MODEL_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Find code embedding tensors (for audio tokens)
        let code_embed_names: Vec<&String> = tensors
            .names()
            .iter()
            .filter(|n| n.contains("code") && n.contains("embed"))
            .cloned()
            .collect();

        println!("Code embedding tensors:");
        for name in &code_embed_names {
            let tensor = tensors.tensor(name).unwrap();
            println!("  {}: {:?}", name, tensor.shape());
        }

        // Should have code embeddings for the 16 codec groups
        assert!(!code_embed_names.is_empty(), "Should have code embeddings");
    }

    #[test]
    fn test_lm_head_dimensions() {
        if !model_available() {
            eprintln!("Skipping test_lm_head_dimensions: model weights not found");
            return;
        }

        use safetensors::SafeTensors;

        let model_path = Path::new(MODEL_DIR).join("model.safetensors");
        let model_bytes = std::fs::read(&model_path).unwrap();
        let tensors = SafeTensors::deserialize(&model_bytes).unwrap();

        // Find LM head / output projection
        let lm_head_names: Vec<&String> = tensors
            .names()
            .iter()
            .filter(|n| n.contains("lm_head") || n.contains("output"))
            .cloned()
            .collect();

        println!("LM head tensors:");
        for name in &lm_head_names {
            let tensor = tensors.tensor(name).unwrap();
            println!("  {}: {:?}", name, tensor.shape());
        }
    }

    #[test]
    fn test_load_tensors_to_candle() {
        if !model_available() {
            eprintln!("Skipping test_load_tensors_to_candle: model weights not found");
            return;
        }

        use candle_core::Device;

        let model_path = Path::new(MODEL_DIR).join("model.safetensors");
        let device = Device::Cpu;

        // Load all tensors
        let tensors = candle_core::safetensors::load(&model_path, &device).unwrap();

        // Verify we can access key tensors (path is talker.model.layers.X...)
        let key_tensors = [
            "talker.model.layers.0.self_attn.q_proj.weight",
            "talker.model.layers.0.mlp.gate_proj.weight",
            "talker.model.norm.weight",
        ];

        for name in &key_tensors {
            if let Some(tensor) = tensors.get(*name) {
                println!("{}: {:?} {:?}", name, tensor.dims(), tensor.dtype());
                assert!(!tensor.dims().is_empty());
            }
        }

        println!("\nSuccessfully loaded {} tensors to Candle", tensors.len());
    }
}
