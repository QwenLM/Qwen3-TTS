//! Audio resampling using rubato
//!
//! Provides high-quality resampling for converting between different sample rates.

use anyhow::{Context, Result};
use rubato::{
    FastFixedIn, PolynomialDegree, Resampler as RubatoResampler, SincFixedIn,
    SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use super::AudioBuffer;

/// Resampling quality preset
#[derive(Debug, Clone, Copy, Default)]
pub enum ResampleQuality {
    /// Fast resampling, lower quality
    Fast,
    /// Balanced speed and quality
    #[default]
    Normal,
    /// High quality, slower
    High,
}

/// Audio resampler
pub struct Resampler {
    quality: ResampleQuality,
}

impl Resampler {
    /// Create a new resampler
    pub fn new(quality: ResampleQuality) -> Self {
        Self { quality }
    }

    /// Resample audio to a target sample rate
    pub fn resample(&self, audio: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
        if audio.sample_rate == target_rate {
            return Ok(audio.clone());
        }

        let ratio = target_rate as f64 / audio.sample_rate as f64;

        match self.quality {
            ResampleQuality::Fast => self.resample_fast(audio, target_rate, ratio),
            ResampleQuality::Normal | ResampleQuality::High => {
                self.resample_sinc(audio, target_rate, ratio)
            }
        }
    }

    /// Fast polynomial resampling
    fn resample_fast(
        &self,
        audio: &AudioBuffer,
        target_rate: u32,
        ratio: f64,
    ) -> Result<AudioBuffer> {
        let chunk_size = 1024;

        let mut resampler = FastFixedIn::<f32>::new(
            ratio,
            1.0,
            PolynomialDegree::Cubic,
            chunk_size,
            1, // mono
        )
        .context("Failed to create fast resampler")?;

        let output = self.process_chunks(&mut resampler, &audio.samples, chunk_size)?;
        Ok(AudioBuffer::new(output, target_rate))
    }

    /// High-quality sinc resampling
    fn resample_sinc(
        &self,
        audio: &AudioBuffer,
        target_rate: u32,
        ratio: f64,
    ) -> Result<AudioBuffer> {
        let chunk_size = 1024;

        let params = SincInterpolationParameters {
            sinc_len: if matches!(self.quality, ResampleQuality::High) {
                256
            } else {
                128
            },
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: if matches!(self.quality, ResampleQuality::High) {
                256
            } else {
                128
            },
            window: WindowFunction::BlackmanHarris2,
        };

        let mut resampler = SincFixedIn::<f32>::new(
            ratio, 1.0, params, chunk_size, 1, // mono
        )
        .context("Failed to create sinc resampler")?;

        let output = self.process_chunks(&mut resampler, &audio.samples, chunk_size)?;
        Ok(AudioBuffer::new(output, target_rate))
    }

    /// Process audio in chunks through the resampler
    fn process_chunks<R: RubatoResampler<f32>>(
        &self,
        resampler: &mut R,
        samples: &[f32],
        chunk_size: usize,
    ) -> Result<Vec<f32>> {
        let mut output = Vec::new();
        let mut pos = 0;

        while pos < samples.len() {
            let end = (pos + chunk_size).min(samples.len());
            let chunk = &samples[pos..end];

            // Pad last chunk if needed
            let input: Vec<Vec<f32>> = if chunk.len() < chunk_size {
                let mut padded = chunk.to_vec();
                padded.resize(chunk_size, 0.0);
                vec![padded]
            } else {
                vec![chunk.to_vec()]
            };

            let result = resampler
                .process(&input, None)
                .context("Resampling failed")?;

            if let Some(channel) = result.first() {
                output.extend_from_slice(channel);
            }

            pos += chunk_size;
        }

        Ok(output)
    }
}

impl Default for Resampler {
    fn default() -> Self {
        Self::new(ResampleQuality::Normal)
    }
}

/// Convenience function to resample audio
pub fn resample(audio: &AudioBuffer, target_rate: u32) -> Result<AudioBuffer> {
    Resampler::default().resample(audio, target_rate)
}

/// Resample to Qwen3-TTS's native 24kHz
pub fn resample_to_24k(audio: &AudioBuffer) -> Result<AudioBuffer> {
    resample(audio, 24000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_resample_quality_default() {
        let resampler = Resampler::default();
        // Should use Normal quality
        assert!(matches!(resampler.quality, ResampleQuality::Normal));
    }

    #[test]
    fn test_resample_quality_variants() {
        let _fast = Resampler::new(ResampleQuality::Fast);
        let _normal = Resampler::new(ResampleQuality::Normal);
        let _high = Resampler::new(ResampleQuality::High);
    }

    #[test]
    fn test_no_resample_needed() {
        let audio = AudioBuffer::new(vec![0.0; 1000], 24000);
        let result = resample(&audio, 24000).unwrap();
        assert_eq!(result.sample_rate, 24000);
        assert_eq!(result.len(), audio.len());
    }

    #[test]
    fn test_downsample() {
        // 48kHz -> 24kHz (half)
        let audio = AudioBuffer::new(vec![0.0; 4800], 48000);
        let result = resample(&audio, 24000).unwrap();
        assert_eq!(result.sample_rate, 24000);
        // Should be approximately half the samples
        assert!(result.len() > 2000 && result.len() < 3000);
    }

    #[test]
    fn test_upsample() {
        // 16kHz -> 24kHz (1.5x)
        let audio = AudioBuffer::new(vec![0.0; 1600], 16000);
        let result = resample(&audio, 24000).unwrap();
        assert_eq!(result.sample_rate, 24000);
        // Should be approximately 1.5x the samples
        assert!(result.len() > 2000 && result.len() < 3000);
    }

    #[test]
    fn test_resample_to_24k() {
        let audio = AudioBuffer::new(vec![0.0; 1600], 16000);
        let result = resample_to_24k(&audio).unwrap();
        assert_eq!(result.sample_rate, 24000);
    }

    #[test]
    fn test_resample_fast_quality() {
        let resampler = Resampler::new(ResampleQuality::Fast);
        let audio = AudioBuffer::new(vec![0.0; 2048], 48000);
        let result = resampler.resample(&audio, 24000).unwrap();
        assert_eq!(result.sample_rate, 24000);
    }

    #[test]
    fn test_resample_high_quality() {
        let resampler = Resampler::new(ResampleQuality::High);
        let audio = AudioBuffer::new(vec![0.0; 2048], 48000);
        let result = resampler.resample(&audio, 24000).unwrap();
        assert_eq!(result.sample_rate, 24000);
    }

    #[test]
    fn test_resample_preserves_sine_wave() {
        // Create a low frequency sine wave that should survive resampling
        let freq = 100.0; // 100 Hz - well below Nyquist for both sample rates
        let audio = AudioBuffer::new(
            (0..4800)
                .map(|i| (2.0 * PI * freq * i as f32 / 48000.0).sin())
                .collect(),
            48000,
        );

        let result = resample(&audio, 24000).unwrap();

        // Check that output has non-zero values
        let max_val = result
            .samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(max_val > 0.5); // Sine wave should maintain amplitude
    }

    #[test]
    fn test_resample_empty_audio() {
        let audio = AudioBuffer::new(vec![], 24000);
        let result = resample(&audio, 48000).unwrap();
        // Should handle empty gracefully
        assert_eq!(result.sample_rate, 48000);
    }

    #[test]
    fn test_resample_small_audio() {
        let audio = AudioBuffer::new(vec![0.5, -0.5], 24000);
        let result = resample(&audio, 48000).unwrap();
        assert_eq!(result.sample_rate, 48000);
    }
}
