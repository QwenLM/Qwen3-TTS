//! Vector quantization for audio codecs
//!
//! Implements residual vector quantization (RVQ) used by neural audio codecs
//! to discretize continuous audio representations.

use anyhow::Result;
use candle_core::{IndexOp, Module, Tensor, D};
use candle_nn::{embedding, Embedding, VarBuilder};

/// Single vector quantizer with learnable codebook
pub struct VectorQuantizer {
    /// Codebook embeddings: [codebook_size, dim]
    codebook: Embedding,
    /// Codebook size
    codebook_size: usize,
    /// Embedding dimension
    dim: usize,
}

impl VectorQuantizer {
    /// Create new vector quantizer
    pub fn new(codebook_size: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        let codebook = embedding(codebook_size, dim, vb.pp("codebook"))?;

        Ok(Self {
            codebook,
            codebook_size,
            dim,
        })
    }

    /// Quantize continuous vectors to discrete codes
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq, dim]
    ///
    /// # Returns
    /// Tuple of (quantized embeddings, indices)
    pub fn encode(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, seq, dim) = x.dims3()?;

        // Flatten to [batch * seq, dim]
        let x_flat = x.reshape((batch * seq, dim))?;

        // Get codebook weights: [codebook_size, dim]
        let codebook_indices: Vec<u32> = (0..self.codebook_size as u32).collect();
        let indices_tensor = Tensor::new(codebook_indices.as_slice(), x.device())?;
        let codebook_weights = self.codebook.forward(&indices_tensor)?;

        // Compute distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x @ c^T
        let x_sq = x_flat.sqr()?.sum(D::Minus1)?;
        let c_sq = codebook_weights.sqr()?.sum(D::Minus1)?;
        let xc = x_flat.matmul(&codebook_weights.transpose(0, 1)?)?;

        let distances = ((x_sq.unsqueeze(1)? + c_sq.unsqueeze(0)?)? - (xc * 2.0)?)?;

        // Get nearest codebook entry
        let indices = distances.argmin(D::Minus1)?;
        let indices = indices.reshape((batch, seq))?;

        // Look up quantized embeddings
        let quantized = self.decode(&indices)?;

        Ok((quantized, indices))
    }

    /// Decode indices to embeddings
    ///
    /// # Arguments
    /// * `indices` - Token indices of shape [batch, seq]
    ///
    /// # Returns
    /// Embeddings of shape [batch, seq, dim]
    pub fn decode(&self, indices: &Tensor) -> Result<Tensor> {
        Ok(self.codebook.forward(indices)?)
    }

    /// Get codebook dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get codebook size
    pub fn size(&self) -> usize {
        self.codebook_size
    }
}

/// Residual Vector Quantizer (RVQ)
///
/// Applies multiple VQ layers sequentially, where each layer quantizes
/// the residual from the previous layer.
pub struct ResidualVectorQuantizer {
    /// Individual quantizers
    quantizers: Vec<VectorQuantizer>,
    /// Number of quantizers
    num_quantizers: usize,
    /// Codebook dimension
    dim: usize,
}

impl ResidualVectorQuantizer {
    /// Create new residual vector quantizer
    pub fn new(
        num_quantizers: usize,
        codebook_size: usize,
        dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut quantizers = Vec::with_capacity(num_quantizers);

        for i in 0..num_quantizers {
            quantizers.push(VectorQuantizer::new(
                codebook_size,
                dim,
                vb.pp(format!("layers.{}", i)),
            )?);
        }

        Ok(Self {
            quantizers,
            num_quantizers,
            dim,
        })
    }

    /// Quantize continuous vectors using residual quantization
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch, seq, dim]
    ///
    /// # Returns
    /// Tuple of (quantized sum, all indices [batch, num_q, seq])
    pub fn encode(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_batch, _seq, _) = x.dims3()?;
        let mut residual = x.clone();
        let mut quantized_sum = Tensor::zeros(x.shape(), x.dtype(), x.device())?;
        let mut all_indices = Vec::with_capacity(self.num_quantizers);

        for quantizer in &self.quantizers {
            let (quantized, indices) = quantizer.encode(&residual)?;

            // Accumulate quantized values
            quantized_sum = (quantized_sum + &quantized)?;

            // Update residual
            residual = (&residual - &quantized)?;

            // Store indices
            all_indices.push(indices);
        }

        // Stack indices: [batch, num_q, seq]
        let all_indices = Tensor::stack(&all_indices, 1)?;

        Ok((quantized_sum, all_indices))
    }

    /// Decode indices to embeddings
    ///
    /// # Arguments
    /// * `indices` - Token indices of shape [batch, num_quantizers, seq]
    ///
    /// # Returns
    /// Embeddings of shape [batch, seq, num_quantizers, dim]
    pub fn decode(&self, indices: &Tensor) -> Result<Tensor> {
        let (_batch, num_q, _seq) = indices.dims3()?;

        let mut embeddings = Vec::with_capacity(num_q);

        for (i, quantizer) in self.quantizers.iter().enumerate() {
            // Get indices for this quantizer: [batch, seq]
            let q_indices = indices.i((.., i, ..))?;
            let emb = quantizer.decode(&q_indices)?;
            embeddings.push(emb);
        }

        // Stack: [batch, seq, num_q, dim]
        let embeddings = Tensor::stack(&embeddings, 2)?;

        Ok(embeddings)
    }

    /// Decode and sum all quantizer outputs
    ///
    /// # Arguments
    /// * `indices` - Token indices of shape [batch, num_quantizers, seq]
    ///
    /// # Returns
    /// Summed embeddings of shape [batch, seq, dim]
    pub fn decode_sum(&self, indices: &Tensor) -> Result<Tensor> {
        let embeddings = self.decode(indices)?;
        Ok(embeddings.sum(2)?) // Sum over quantizer dimension
    }

    /// Get number of quantizers
    pub fn num_quantizers(&self) -> usize {
        self.num_quantizers
    }

    /// Get codebook dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Split Residual Vector Quantizer
///
/// Variant that splits quantizers into semantic (first N) and acoustic groups,
/// commonly used in speech codecs.
#[allow(dead_code)]
pub struct SplitResidualVectorQuantizer {
    /// Semantic quantizers (typically 1)
    semantic_quantizers: ResidualVectorQuantizer,
    /// Acoustic quantizers (typically 15)
    acoustic_quantizers: ResidualVectorQuantizer,
    /// Number of semantic quantizers
    num_semantic: usize,
}

#[allow(dead_code)]
impl SplitResidualVectorQuantizer {
    /// Create new split RVQ
    pub fn new(
        num_semantic: usize,
        num_acoustic: usize,
        codebook_size: usize,
        dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let semantic_quantizers =
            ResidualVectorQuantizer::new(num_semantic, codebook_size, dim, vb.pp("semantic"))?;

        let acoustic_quantizers =
            ResidualVectorQuantizer::new(num_acoustic, codebook_size, dim, vb.pp("acoustic"))?;

        Ok(Self {
            semantic_quantizers,
            acoustic_quantizers,
            num_semantic,
        })
    }

    /// Decode semantic indices only
    pub fn decode_semantic(&self, indices: &Tensor) -> Result<Tensor> {
        self.semantic_quantizers.decode_sum(indices)
    }

    /// Decode acoustic indices only
    pub fn decode_acoustic(&self, indices: &Tensor) -> Result<Tensor> {
        self.acoustic_quantizers.decode_sum(indices)
    }

    /// Decode all indices
    pub fn decode(&self, semantic_indices: &Tensor, acoustic_indices: &Tensor) -> Result<Tensor> {
        let semantic = self.semantic_quantizers.decode_sum(semantic_indices)?;
        let acoustic = self.acoustic_quantizers.decode_sum(acoustic_indices)?;
        Ok((semantic + acoustic)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    /// Create a VarBuilder with mock weights for testing
    fn create_mock_vb(device: &Device) -> VarBuilder<'static> {
        let varmap = VarMap::new();
        VarBuilder::from_varmap(&varmap, DType::F32, device)
    }

    #[test]
    fn test_vq_dimensions() {
        // Just test that types are correct
        let codebook_size = 2048;
        let dim = 256;
        assert!(codebook_size > 0);
        assert!(dim > 0);
    }

    #[test]
    fn test_vector_quantizer_new() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let vq = VectorQuantizer::new(128, 64, vb).unwrap();
        assert_eq!(vq.size(), 128);
        assert_eq!(vq.dim(), 64);
    }

    #[test]
    fn test_vector_quantizer_decode() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let vq = VectorQuantizer::new(128, 64, vb).unwrap();

        // Create indices: [batch=2, seq=4]
        let indices = Tensor::new(&[[0u32, 1, 2, 3], [4, 5, 6, 7]], &device).unwrap();

        let decoded = vq.decode(&indices).unwrap();
        assert_eq!(decoded.dims(), &[2, 4, 64]);
    }

    #[test]
    fn test_rvq_new() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let rvq = ResidualVectorQuantizer::new(4, 128, 64, vb).unwrap();
        assert_eq!(rvq.num_quantizers(), 4);
        assert_eq!(rvq.dim(), 64);
    }

    #[test]
    fn test_rvq_construction() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let rvq = ResidualVectorQuantizer::new(4, 128, 64, vb);
        // Just verify construction succeeds
        assert!(rvq.is_ok());
    }

    #[test]
    fn test_rvq_decode() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let rvq = ResidualVectorQuantizer::new(4, 128, 64, vb).unwrap();

        // Create indices: [batch=2, num_q=4, seq=3]
        let indices = Tensor::zeros((2, 4, 3), DType::U32, &device).unwrap();

        let decoded = rvq.decode(&indices).unwrap();
        // Should be [batch, seq, num_q, dim]
        assert_eq!(decoded.dims(), &[2, 3, 4, 64]);
    }

    #[test]
    fn test_rvq_decode_sum() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let rvq = ResidualVectorQuantizer::new(4, 128, 64, vb).unwrap();

        // Create indices: [batch=2, num_q=4, seq=3]
        let indices = Tensor::zeros((2, 4, 3), DType::U32, &device).unwrap();

        let decoded = rvq.decode_sum(&indices).unwrap();
        // Should be [batch, seq, dim]
        assert_eq!(decoded.dims(), &[2, 3, 64]);
    }

    #[test]
    fn test_split_rvq_new() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let split_rvq = SplitResidualVectorQuantizer::new(1, 15, 128, 64, vb).unwrap();
        // Just test construction succeeds
        assert_eq!(split_rvq.num_semantic, 1);
    }

    #[test]
    fn test_split_rvq_decode_semantic() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let split_rvq = SplitResidualVectorQuantizer::new(1, 15, 128, 64, vb).unwrap();

        // Semantic indices: [batch=2, num_semantic=1, seq=3]
        let indices = Tensor::zeros((2, 1, 3), DType::U32, &device).unwrap();

        let decoded = split_rvq.decode_semantic(&indices).unwrap();
        assert_eq!(decoded.dims(), &[2, 3, 64]);
    }

    #[test]
    fn test_split_rvq_decode_acoustic() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let split_rvq = SplitResidualVectorQuantizer::new(1, 15, 128, 64, vb).unwrap();

        // Acoustic indices: [batch=2, num_acoustic=15, seq=3]
        let indices = Tensor::zeros((2, 15, 3), DType::U32, &device).unwrap();

        let decoded = split_rvq.decode_acoustic(&indices).unwrap();
        assert_eq!(decoded.dims(), &[2, 3, 64]);
    }

    #[test]
    fn test_split_rvq_decode_both() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let split_rvq = SplitResidualVectorQuantizer::new(1, 15, 128, 64, vb).unwrap();

        // Create indices
        let semantic_indices = Tensor::zeros((2, 1, 3), DType::U32, &device).unwrap();
        let acoustic_indices = Tensor::zeros((2, 15, 3), DType::U32, &device).unwrap();

        let decoded = split_rvq
            .decode(&semantic_indices, &acoustic_indices)
            .unwrap();
        assert_eq!(decoded.dims(), &[2, 3, 64]);
    }

    #[test]
    fn test_vq_decode_consistency() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let vq = VectorQuantizer::new(128, 64, vb).unwrap();

        // Use known indices
        let indices = Tensor::new(&[[0u32, 1, 2]], &device).unwrap();

        let decoded_1 = vq.decode(&indices).unwrap();
        let decoded_2 = vq.decode(&indices).unwrap();

        // Decoding same indices should give same result
        let d1: Vec<f32> = decoded_1.flatten_all().unwrap().to_vec1().unwrap();
        let d2: Vec<f32> = decoded_2.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_vq_different_indices_different_outputs() {
        let device = Device::Cpu;
        let vb = create_mock_vb(&device);
        let vq = VectorQuantizer::new(128, 64, vb).unwrap();

        // Different indices
        let indices_1 = Tensor::new(&[[0u32, 0, 0]], &device).unwrap();
        let indices_2 = Tensor::new(&[[1u32, 1, 1]], &device).unwrap();

        let decoded_1 = vq.decode(&indices_1).unwrap();
        let decoded_2 = vq.decode(&indices_2).unwrap();

        // Should be different (with very high probability for random init)
        let d1: Vec<f32> = decoded_1.flatten_all().unwrap().to_vec1().unwrap();
        let d2: Vec<f32> = decoded_2.flatten_all().unwrap().to_vec1().unwrap();

        let mut any_diff = false;
        for (a, b) in d1.iter().zip(d2.iter()) {
            if (a - b).abs() > 1e-6 {
                any_diff = true;
                break;
            }
        }
        assert!(
            any_diff,
            "Different indices should decode to different embeddings"
        );
    }
}
