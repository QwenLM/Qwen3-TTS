//! Audio I/O utilities

use anyhow::{Context, Result};
use candle_core::Tensor;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;

/// Audio buffer holding raw waveform data
#[derive(Debug, Clone)]
pub struct AudioBuffer {
    /// Mono audio samples in [-1.0, 1.0] range
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
}

impl AudioBuffer {
    /// Create a new audio buffer
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Create from a Candle tensor (assumed shape: [samples] or [1, samples])
    pub fn from_tensor(tensor: Tensor, sample_rate: u32) -> Result<Self> {
        let tensor = if tensor.dims().len() == 2 {
            tensor.squeeze(0)?
        } else {
            tensor
        };

        let samples: Vec<f32> = tensor.to_vec1()?;
        Ok(Self::new(samples, sample_rate))
    }

    /// Convert to a Candle tensor
    pub fn to_tensor(&self, device: &candle_core::Device) -> Result<Tensor> {
        Ok(Tensor::new(self.samples.as_slice(), device)?)
    }

    /// Duration in seconds
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Save to WAV file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        save_wav(path, &self.samples, self.sample_rate)
    }

    /// Load from WAV file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        load_wav(path)
    }

    /// Normalize audio to [-1.0, 1.0] range
    pub fn normalize(&mut self) {
        let max_abs = self.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        if max_abs > 0.0 && max_abs != 1.0 {
            for sample in &mut self.samples {
                *sample /= max_abs;
            }
        }
    }

    /// Apply peak normalization to a target dB level
    pub fn normalize_db(&mut self, target_db: f32) {
        let target_amplitude = 10.0f32.powf(target_db / 20.0);
        let max_abs = self.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        if max_abs > 0.0 {
            let scale = target_amplitude / max_abs;
            for sample in &mut self.samples {
                *sample *= scale;
            }
        }
    }
}

/// Load a WAV file into an AudioBuffer
pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<AudioBuffer> {
    let path = path.as_ref();
    let reader = WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()?,
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    // Convert to mono by averaging channels
    let mono_samples = if channels > 1 {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    Ok(AudioBuffer::new(mono_samples, sample_rate))
}

/// Save samples to a WAV file
pub fn save_wav<P: AsRef<Path>>(path: P, samples: &[f32], sample_rate: u32) -> Result<()> {
    let path = path.as_ref();
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("Failed to create WAV file: {}", path.display()))?;

    for &sample in samples {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use tempfile::tempdir;

    #[test]
    fn test_audio_buffer_new() {
        let samples = vec![0.1, 0.2, 0.3];
        let buffer = AudioBuffer::new(samples.clone(), 16000);
        assert_eq!(buffer.samples, samples);
        assert_eq!(buffer.sample_rate, 16000);
    }

    #[test]
    fn test_audio_buffer_duration() {
        let buffer = AudioBuffer::new(vec![0.0; 24000], 24000);
        assert!((buffer.duration() - 1.0).abs() < 1e-6);

        let buffer2 = AudioBuffer::new(vec![0.0; 48000], 24000);
        assert!((buffer2.duration() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_audio_buffer_len_and_empty() {
        let buffer = AudioBuffer::new(vec![0.0; 100], 24000);
        assert_eq!(buffer.len(), 100);
        assert!(!buffer.is_empty());

        let empty_buffer = AudioBuffer::new(vec![], 24000);
        assert_eq!(empty_buffer.len(), 0);
        assert!(empty_buffer.is_empty());
    }

    #[test]
    fn test_normalize() {
        let mut buffer = AudioBuffer::new(vec![0.5, -0.25, 0.1], 24000);
        buffer.normalize();
        assert!((buffer.samples[0] - 1.0).abs() < 1e-6);
        assert!((buffer.samples[1] - (-0.5)).abs() < 1e-6);
        assert!((buffer.samples[2] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_already_normalized() {
        let mut buffer = AudioBuffer::new(vec![1.0, -1.0, 0.5], 24000);
        buffer.normalize();
        assert!((buffer.samples[0] - 1.0).abs() < 1e-6);
        assert!((buffer.samples[1] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_silence() {
        let mut buffer = AudioBuffer::new(vec![0.0, 0.0, 0.0], 24000);
        buffer.normalize();
        // Should not panic, samples remain zero
        assert!((buffer.samples[0]).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_db() {
        let mut buffer = AudioBuffer::new(vec![0.5, -0.5, 0.25], 24000);
        buffer.normalize_db(-6.0); // -6 dB ≈ 0.5 amplitude
        let max_abs = buffer
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!((max_abs - 0.501187).abs() < 0.01); // 10^(-6/20) ≈ 0.501
    }

    #[test]
    fn test_to_tensor() {
        let buffer = AudioBuffer::new(vec![0.1, 0.2, 0.3], 24000);
        let device = Device::Cpu;
        let tensor = buffer.to_tensor(&device).unwrap();
        assert_eq!(tensor.dims(), &[3]);
        let values: Vec<f32> = tensor.to_vec1().unwrap();
        assert!((values[0] - 0.1).abs() < 1e-6);
        assert!((values[1] - 0.2).abs() < 1e-6);
        assert!((values[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_from_tensor_1d() {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[0.1f32, 0.2, 0.3], &device).unwrap();
        let buffer = AudioBuffer::from_tensor(tensor, 24000).unwrap();
        assert_eq!(buffer.samples.len(), 3);
        assert_eq!(buffer.sample_rate, 24000);
    }

    #[test]
    fn test_from_tensor_2d() {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[[0.1f32, 0.2, 0.3]], &device).unwrap();
        let buffer = AudioBuffer::from_tensor(tensor, 24000).unwrap();
        assert_eq!(buffer.samples.len(), 3);
    }

    #[test]
    fn test_save_and_load_wav() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wav");

        let original = AudioBuffer::new(vec![0.1, 0.2, -0.3, 0.4, -0.5], 24000);
        original.save(&path).unwrap();

        let loaded = AudioBuffer::load(&path).unwrap();
        assert_eq!(loaded.sample_rate, 24000);
        assert_eq!(loaded.samples.len(), 5);

        for (a, b) in original.samples.iter().zip(loaded.samples.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_save_wav_function() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test2.wav");

        let samples = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        save_wav(&path, &samples, 16000).unwrap();

        assert!(path.exists());
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = load_wav("/nonexistent/path/to/file.wav");
        assert!(result.is_err());
    }
}
