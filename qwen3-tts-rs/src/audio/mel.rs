//! Mel-spectrogram computation
//!
//! Implementation based on librosa's mel spectrogram computation,
//! optimized for the Qwen3-TTS speaker encoder requirements.

use anyhow::Result;
use candle_core::{Device, Tensor};
use num_complex::Complex;
use rustfft::{num_complex::Complex as FftComplex, FftPlanner};
use std::f32::consts::PI;

/// Configuration for mel spectrogram computation
#[derive(Debug, Clone)]
pub struct MelConfig {
    /// Sample rate of input audio
    pub sample_rate: u32,
    /// FFT window size
    pub n_fft: usize,
    /// Hop length between frames
    pub hop_length: usize,
    /// Window length (defaults to n_fft)
    pub win_length: Option<usize>,
    /// Number of mel bands
    pub n_mels: usize,
    /// Minimum frequency for mel filterbank
    pub fmin: f32,
    /// Maximum frequency for mel filterbank (defaults to sample_rate / 2)
    pub fmax: Option<f32>,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            n_fft: 400,
            hop_length: 160,
            win_length: None,
            n_mels: 128,
            fmin: 0.0,
            fmax: None,
        }
    }
}

/// Mel-spectrogram extractor
pub struct MelSpectrogram {
    config: MelConfig,
    /// Precomputed mel filterbank
    mel_basis: Vec<Vec<f32>>,
    /// Precomputed Hann window
    window: Vec<f32>,
}

impl MelSpectrogram {
    /// Create a new mel spectrogram extractor
    pub fn new(config: MelConfig) -> Self {
        let win_length = config.win_length.unwrap_or(config.n_fft);
        let fmax = config.fmax.unwrap_or(config.sample_rate as f32 / 2.0);

        // Compute mel filterbank
        let mel_basis = Self::create_mel_filterbank(
            config.sample_rate,
            config.n_fft,
            config.n_mels,
            config.fmin,
            fmax,
        );

        // Compute Hann window
        let window = Self::hann_window(win_length);

        Self {
            config,
            mel_basis,
            window,
        }
    }

    /// Compute mel spectrogram from audio samples
    pub fn compute(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        // Compute STFT
        let stft = self.stft(samples);

        // Compute power spectrogram
        let power_spec: Vec<Vec<f32>> = stft
            .iter()
            .map(|frame| frame.iter().map(|c| c.norm_sqr()).collect())
            .collect();

        // Apply mel filterbank
        self.apply_mel_filterbank(&power_spec)
    }

    /// Compute mel spectrogram and return as tensor
    pub fn compute_tensor(&self, samples: &[f32], device: &Device) -> Result<Tensor> {
        let mel = self.compute(samples);
        let n_frames = mel.len();
        let n_mels = self.config.n_mels;

        // Flatten and create tensor
        let flat: Vec<f32> = mel.into_iter().flatten().collect();
        let tensor = Tensor::new(flat.as_slice(), device)?
            .reshape((n_frames, n_mels))?
            .transpose(0, 1)?; // [n_mels, n_frames]

        Ok(tensor)
    }

    /// Compute log mel spectrogram
    pub fn compute_log(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mel = self.compute(samples);
        mel.into_iter()
            .map(|frame| frame.into_iter().map(|v| (v.max(1e-10)).ln()).collect())
            .collect()
    }

    /// Short-time Fourier transform
    fn stft(&self, samples: &[f32]) -> Vec<Vec<Complex<f32>>> {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;
        let win_length = self.window.len();

        // Pad signal
        let pad_length = n_fft / 2;
        let mut padded = vec![0.0f32; pad_length];
        padded.extend_from_slice(samples);
        padded.extend(vec![0.0f32; pad_length]);

        // Setup FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_fft);

        let n_frames = (padded.len() - n_fft) / hop_length + 1;
        let mut result = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let start = i * hop_length;
            let _end = start + n_fft;

            // Apply window and prepare FFT input
            let mut buffer: Vec<FftComplex<f32>> = (0..n_fft)
                .map(|j| {
                    let sample = if j < win_length && start + j < padded.len() {
                        padded[start + j] * self.window[j]
                    } else {
                        0.0
                    };
                    FftComplex::new(sample, 0.0)
                })
                .collect();

            // Perform FFT
            fft.process(&mut buffer);

            // Take positive frequencies only (n_fft/2 + 1)
            let frame: Vec<Complex<f32>> = buffer
                .iter()
                .take(n_fft / 2 + 1)
                .map(|c| Complex::new(c.re, c.im))
                .collect();

            result.push(frame);
        }

        result
    }

    /// Apply mel filterbank to power spectrogram
    fn apply_mel_filterbank(&self, power_spec: &[Vec<f32>]) -> Vec<Vec<f32>> {
        power_spec
            .iter()
            .map(|frame| {
                self.mel_basis
                    .iter()
                    .map(|filter| filter.iter().zip(frame.iter()).map(|(f, p)| f * p).sum())
                    .collect()
            })
            .collect()
    }

    /// Create mel filterbank matrix
    fn create_mel_filterbank(
        sample_rate: u32,
        n_fft: usize,
        n_mels: usize,
        fmin: f32,
        fmax: f32,
    ) -> Vec<Vec<f32>> {
        let n_freqs = n_fft / 2 + 1;

        // Mel scale conversion functions
        let hz_to_mel = |f: f32| 2595.0 * (1.0 + f / 700.0).log10();
        let mel_to_hz = |m: f32| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0);

        // Create mel points
        let mel_min = hz_to_mel(fmin);
        let mel_max = hz_to_mel(fmax);
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();

        // Convert to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

        // Convert to FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&f| ((n_fft + 1) as f32 * f / sample_rate as f32).floor() as usize)
            .collect();

        // Create filterbank
        let mut filterbank = vec![vec![0.0f32; n_freqs]; n_mels];

        for i in 0..n_mels {
            let left = bin_points[i];
            let center = bin_points[i + 1];
            let right = bin_points[i + 2];

            // Rising slope
            let rising_denom = (center - left).max(1) as f32;
            for (j, val) in filterbank[i].iter_mut().enumerate().take(center).skip(left) {
                if j < n_freqs {
                    *val = (j - left) as f32 / rising_denom;
                }
            }

            // Falling slope
            let falling_denom = (right - center).max(1) as f32;
            for (j, val) in filterbank[i]
                .iter_mut()
                .enumerate()
                .take(right)
                .skip(center)
            {
                if j < n_freqs {
                    *val = (right - j) as f32 / falling_denom;
                }
            }
        }

        filterbank
    }

    /// Create Hann window
    fn hann_window(length: usize) -> Vec<f32> {
        (0..length)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / length as f32).cos()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_mel_config_default() {
        let config = MelConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.n_fft, 400);
        assert_eq!(config.hop_length, 160);
        assert_eq!(config.n_mels, 128);
        assert!((config.fmin - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mel_config_custom() {
        let config = MelConfig {
            sample_rate: 16000,
            n_fft: 512,
            hop_length: 256,
            win_length: Some(512),
            n_mels: 80,
            fmin: 80.0,
            fmax: Some(7600.0),
        };
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.fmax, Some(7600.0));
    }

    #[test]
    fn test_hann_window() {
        let window = MelSpectrogram::hann_window(4);
        assert_eq!(window.len(), 4);
        assert!((window[0] - 0.0).abs() < 1e-6);
        assert!((window[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hann_window_larger() {
        let window = MelSpectrogram::hann_window(256);
        assert_eq!(window.len(), 256);
        // Window starts near zero
        assert!(window[0] < 0.01);
        // Peak at center
        assert!(window[128] > 0.99);
        // Values are symmetric around center
        assert!((window[1] - window[255]).abs() < 0.01);
        assert!((window[64] - window[192]).abs() < 0.01);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let mel = MelSpectrogram::new(MelConfig::default());
        assert_eq!(mel.mel_basis.len(), 128);
        assert_eq!(mel.mel_basis[0].len(), 201); // n_fft/2 + 1 = 400/2 + 1
    }

    #[test]
    fn test_mel_filterbank_triangular() {
        let mel = MelSpectrogram::new(MelConfig {
            n_mels: 4,
            ..Default::default()
        });
        // Each filter should have triangular shape (non-negative values)
        for filter in &mel.mel_basis {
            for &val in filter {
                assert!(val >= 0.0);
            }
        }
    }

    #[test]
    fn test_compute_mel_silence() {
        let mel = MelSpectrogram::new(MelConfig::default());
        let samples = vec![0.0f32; 24000];
        let result = mel.compute(&samples);
        assert!(!result.is_empty());
        assert_eq!(result[0].len(), 128);
        // Silence should produce very small values
        for frame in &result {
            for &val in frame {
                assert!(val < 1e-6);
            }
        }
    }

    #[test]
    fn test_compute_mel_sine_wave() {
        let mel = MelSpectrogram::new(MelConfig::default());
        // Generate 440Hz sine wave (A4 note)
        let samples: Vec<f32> = (0..24000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let result = mel.compute(&samples);
        assert!(!result.is_empty());
        // Should have non-zero energy
        let total_energy: f32 = result.iter().flat_map(|frame| frame.iter()).sum();
        assert!(total_energy > 0.0);
    }

    #[test]
    fn test_compute_log_mel() {
        let mel = MelSpectrogram::new(MelConfig::default());
        let samples: Vec<f32> = (0..24000)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 24000.0).sin())
            .collect();
        let result = mel.compute_log(&samples);
        assert!(!result.is_empty());
        // Log mel should have negative values (log of small numbers)
        let has_negative = result
            .iter()
            .flat_map(|frame| frame.iter())
            .any(|&v| v < 0.0);
        assert!(has_negative);
    }

    #[test]
    fn test_compute_tensor() {
        let mel = MelSpectrogram::new(MelConfig::default());
        let samples = vec![0.0f32; 4800]; // 0.2 seconds
        let device = Device::Cpu;
        let tensor = mel.compute_tensor(&samples, &device).unwrap();
        // Shape should be [n_mels, n_frames]
        assert_eq!(tensor.dims()[0], 128);
    }

    #[test]
    fn test_stft_output_frames() {
        let mel = MelSpectrogram::new(MelConfig {
            n_fft: 400,
            hop_length: 160,
            ..Default::default()
        });
        // With padding, number of frames should be predictable
        let samples = vec![0.0f32; 1600]; // 10 hops
        let result = mel.compute(&samples);
        assert!(!result.is_empty());
    }
}
