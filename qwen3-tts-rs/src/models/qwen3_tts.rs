//! Qwen3-TTS main model implementation
//!
//! This implements the "Talker" model that generates audio codec tokens
//! from text input autoregressively.

use anyhow::Result;
use candle_core::{Device, IndexOp, Module, Tensor, D};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};

use super::config::Qwen3TTSConfig;
use crate::generation::GenerationConfig;

/// Rotary position embedding
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    #[allow(dead_code)]
    dim: usize,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (theta as f32).powf(i as f32 / dim as f32))
            .collect();

        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?.unsqueeze(1)?;

        let freqs = positions.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin, dim })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.i(offset..offset + seq_len)?;
        let sin = self.sin.i(offset..offset + seq_len)?;

        let q_rot = self.rotate(q, &cos, &sin)?;
        let k_rot = self.rotate(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }

    fn rotate(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (_b, _h, _seq, d) = x.dims4()?;
        let x1 = x.narrow(D::Minus1, 0, d / 2)?;
        let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;

        // Broadcast cos/sin to match x dimensions
        // cos/sin are [seq_len, head_dim/2], need [1, 1, seq_len, head_dim/2]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        let cos = cos.broadcast_as(x1.shape())?;
        let sin = sin.broadcast_as(x1.shape())?;

        // Standard RoPE: x * cos + rotate_half(x) * sin
        // where rotate_half([x1, x2]) = [-x2, x1]
        // Result: [x1*cos - x2*sin, x2*cos + x1*sin]
        let rotated = Tensor::cat(
            &[
                &(x1.mul(&cos)? - x2.mul(&sin)?)?,
                &(x2.mul(&cos)? + x1.mul(&sin)?)?,
            ],
            D::Minus1,
        )?;

        Ok(rotated)
    }
}

/// Multi-head attention with grouped-query attention and QK normalization
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn new(config: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads();
        let head_dim = config.head_dim();

        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        // QK normalization: RMSNorm applied per-head after projection
        let q_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RotaryEmbedding,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut KVCache>,
        offset: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, seq, heads, head_dim] for QK norm
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;

        // Apply QK normalization (per-head RMSNorm)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = rope.apply(&q, &k, offset)?;

        // Update KV cache
        let (k, v) = if let Some(cache) = kv_cache {
            let k = cache.update_k(&k)?;
            let v = cache.update_v(&v)?;
            (k, v)
        } else {
            (k, v)
        };

        // Repeat KV heads for GQA
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;

        // Scaled dot-product attention
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;

        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        Ok(self.o_proj.forward(&attn_output)?)
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x.clone());
        }

        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x
            .unsqueeze(2)?
            .expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))?;
        Ok(x)
    }
}

/// MLP block with SwiGLU activation
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    pub fn new(config: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&(gate * up)?)?)
    }
}

/// Transformer decoder layer
pub struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    pub fn new(config: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(config, vb.pp("self_attn"))?,
            mlp: MLP::new(config, vb.pp("mlp"))?,
            input_layernorm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RotaryEmbedding,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut KVCache>,
        offset: usize,
    ) -> Result<Tensor> {
        // Self-attention with residual
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states =
            self.self_attn
                .forward(&hidden_states, rope, attention_mask, kv_cache, offset)?;
        let hidden_states = (residual + hidden_states)?;

        // MLP with residual
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;

        Ok(hidden_states)
    }
}

/// KV cache for efficient autoregressive generation
pub struct KVCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn update_k(&mut self, k: &Tensor) -> Result<Tensor> {
        let k = if let Some(prev_k) = &self.k {
            Tensor::cat(&[prev_k, k], 2)?
        } else {
            k.clone()
        };
        self.k = Some(k.clone());
        Ok(k)
    }

    pub fn update_v(&mut self, v: &Tensor) -> Result<Tensor> {
        let v = if let Some(prev_v) = &self.v {
            Tensor::cat(&[prev_v, v], 2)?
        } else {
            v.clone()
        };
        self.v = Some(v.clone());
        Ok(v)
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

/// Main Qwen3-TTS model
pub struct Qwen3TTSModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: RotaryEmbedding,
    #[allow(dead_code)]
    config: Qwen3TTSConfig,
    device: Device,
}

impl Qwen3TTSModel {
    /// Load model from pretrained weights
    pub fn from_pretrained(
        model_id: &str,
        _config: &Qwen3TTSConfig,
        _device: &Device,
    ) -> Result<Self> {
        // TODO: Implement weight loading from safetensors
        // For now, just return uninitialized error
        anyhow::bail!("Weight loading not yet implemented. Model ID: {}", model_id)
    }

    /// Create model with given weights
    pub fn new(config: Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();

        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(
                &config,
                vb.pp(format!("model.layers.{}", i)),
            )?);
        }

        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;

        let rope = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            &device,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            config,
            device,
        })
    }

    /// Forward pass returning logits
    pub fn forward(
        &self,
        input_ids: &Tensor,
        kv_caches: &mut [KVCache],
        offset: usize,
    ) -> Result<Tensor> {
        let attention_mask = self.create_causal_mask(input_ids.dim(1)?, offset)?;

        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(
                &hidden_states,
                &self.rope,
                Some(&attention_mask),
                Some(&mut kv_caches[i]),
                offset,
            )?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;

        Ok(logits)
    }

    /// Generate codec tokens autoregressively
    pub fn generate(
        &self,
        input_ids: &Tensor,
        _speaker_embedding: Option<&Tensor>,
        config: &GenerationConfig,
    ) -> Result<Tensor> {
        let _batch_size = input_ids.dim(0)?;
        let mut kv_caches: Vec<KVCache> = (0..self.layers.len()).map(|_| KVCache::new()).collect();

        // Process input prompt
        let mut logits = self.forward(input_ids, &mut kv_caches, 0)?;
        let mut generated = input_ids.clone();
        let mut offset = input_ids.dim(1)?;

        // Generate tokens
        for _ in 0..config.max_new_tokens {
            // Get last token logits
            let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;

            // Sample next token
            let next_token = crate::generation::sample(&last_logits, config)?;

            // Check for EOS - stop if all sequences in batch have generated EOS
            if let Some(eos_id) = config.eos_token_id {
                let token_ids: Vec<u32> = next_token.flatten_all()?.to_vec1()?;
                if token_ids.iter().all(|&t| t == eos_id) {
                    break;
                }
            }

            // Append to generated sequence
            let next_token = next_token.unsqueeze(1)?;
            generated = Tensor::cat(&[&generated, &next_token], 1)?;

            // Forward with just the new token
            logits = self.forward(&next_token, &mut kv_caches, offset)?;
            offset += 1;
        }

        Ok(generated)
    }

    fn create_causal_mask(&self, seq_len: usize, offset: usize) -> Result<Tensor> {
        let total_len = offset + seq_len;
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..total_len).map(move |j| {
                    if j <= offset + i {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        Ok(Tensor::new(mask.as_slice(), &self.device)?.reshape((1, 1, seq_len, total_len))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    fn create_mock_vb(device: &Device) -> VarBuilder<'static> {
        let varmap = VarMap::new();
        VarBuilder::from_varmap(&varmap, DType::F32, device)
    }

    fn small_config() -> Qwen3TTSConfig {
        Qwen3TTSConfig {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            ..Default::default()
        }
    }

    #[test]
    fn test_rotary_embedding_creation() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 512, 10000.0, &device).unwrap();
        assert_eq!(rope.dim, 64);
    }

    #[test]
    fn test_rotary_embedding_shape() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(64, 512, 10000.0, &device).unwrap();

        // cos and sin should be [max_seq_len, dim/2]
        assert_eq!(rope.cos.dims()[0], 512);
        assert_eq!(rope.cos.dims()[1], 32); // dim / 2
        assert_eq!(rope.sin.dims()[0], 512);
        assert_eq!(rope.sin.dims()[1], 32);
    }

    #[test]
    fn test_rotary_embedding_apply() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(16, 512, 10000.0, &device).unwrap();

        // q, k: [batch, heads, seq, head_dim]
        let q = Tensor::randn(0.0f32, 1.0, (2, 4, 10, 16), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (2, 4, 10, 16), &device).unwrap();

        let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();

        assert_eq!(q_rot.dims(), q.dims());
        assert_eq!(k_rot.dims(), k.dims());
    }

    #[test]
    fn test_rotary_embedding_with_offset() {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new(16, 512, 10000.0, &device).unwrap();

        let q = Tensor::randn(0.0f32, 1.0, (1, 2, 5, 16), &device).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 5, 16), &device).unwrap();

        let (q_rot, k_rot) = rope.apply(&q, &k, 100).unwrap();

        assert_eq!(q_rot.dims(), &[1, 2, 5, 16]);
        assert_eq!(k_rot.dims(), &[1, 2, 5, 16]);
    }

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new();
        assert!(cache.k.is_none());
        assert!(cache.v.is_none());
    }

    #[test]
    fn test_kv_cache_update() {
        let device = Device::Cpu;
        let mut cache = KVCache::new();

        let k1 = Tensor::randn(0.0f32, 1.0, (1, 2, 4, 16), &device).unwrap();
        let k_out = cache.update_k(&k1).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 4, 16]);

        let k2 = Tensor::randn(0.0f32, 1.0, (1, 2, 3, 16), &device).unwrap();
        let k_out = cache.update_k(&k2).unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 7, 16]); // 4 + 3 = 7
    }

    #[test]
    fn test_kv_cache_reset() {
        let device = Device::Cpu;
        let mut cache = KVCache::new();

        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 4, 16), &device).unwrap();
        cache.update_k(&k).unwrap();
        assert!(cache.k.is_some());

        cache.reset();
        assert!(cache.k.is_none());
        assert!(cache.v.is_none());
    }

    #[test]
    fn test_mlp() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let mlp = MLP::new(&config, vb).unwrap();

        // Input: [batch=2, seq=10, hidden=64]
        let input = Tensor::randn(0.0f32, 1.0, (2, 10, 64), &device).unwrap();
        let output = mlp.forward(&input).unwrap();

        assert_eq!(output.dims(), &[2, 10, 64]);
    }

    #[test]
    fn test_attention() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let attn = Attention::new(&config, vb).unwrap();

        assert_eq!(attn.num_heads, 4);
        assert_eq!(attn.num_kv_heads, 2);
        assert_eq!(attn.head_dim, 16); // 64 / 4 = 16
    }

    #[test]
    fn test_attention_forward() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let attn = Attention::new(&config, vb).unwrap();
        let rope = RotaryEmbedding::new(16, 512, 10000.0, &device).unwrap();

        // Input: [batch=1, seq=10, hidden=64]
        let input = Tensor::randn(0.0f32, 1.0, (1, 10, 64), &device).unwrap();
        let output = attn.forward(&input, &rope, None, None, 0).unwrap();

        assert_eq!(output.dims(), &[1, 10, 64]);
    }

    #[test]
    fn test_attention_with_cache() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let attn = Attention::new(&config, vb).unwrap();
        let rope = RotaryEmbedding::new(16, 512, 10000.0, &device).unwrap();
        let mut cache = KVCache::new();

        // First forward
        let input1 = Tensor::randn(0.0f32, 1.0, (1, 5, 64), &device).unwrap();
        let _out1 = attn
            .forward(&input1, &rope, None, Some(&mut cache), 0)
            .unwrap();

        // Second forward with cache
        let input2 = Tensor::randn(0.0f32, 1.0, (1, 3, 64), &device).unwrap();
        let out2 = attn
            .forward(&input2, &rope, None, Some(&mut cache), 5)
            .unwrap();

        assert_eq!(out2.dims(), &[1, 3, 64]);
    }

    #[test]
    fn test_decoder_layer() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let layer = DecoderLayer::new(&config, vb).unwrap();
        let rope = RotaryEmbedding::new(16, 512, 10000.0, &device).unwrap();
        let mut cache = KVCache::new();

        let input = Tensor::randn(0.0f32, 1.0, (1, 8, 64), &device).unwrap();
        let output = layer
            .forward(&input, &rope, None, Some(&mut cache), 0)
            .unwrap();

        assert_eq!(output.dims(), &[1, 8, 64]);
    }

    #[test]
    fn test_qwen3_tts_model_creation() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let model = Qwen3TTSModel::new(config.clone(), vb).unwrap();

        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.config.hidden_size, 64);
    }

    #[test]
    fn test_qwen3_tts_model_construction_only() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let model = Qwen3TTSModel::new(config.clone(), vb);
        // Just verify construction succeeds
        assert!(model.is_ok());
    }

    #[test]
    fn test_qwen3_tts_kv_caches_creation() {
        // Test that KV caches can be created for a model
        let kv_caches: Vec<KVCache> = (0..2).map(|_| KVCache::new()).collect();
        // Just verify KV caches can be created
        assert_eq!(kv_caches.len(), 2);
    }

    #[test]
    fn test_causal_mask_shape() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let model = Qwen3TTSModel::new(config, vb).unwrap();

        let mask = model.create_causal_mask(5, 0).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 5, 5]);
    }

    #[test]
    fn test_causal_mask_values() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let model = Qwen3TTSModel::new(config, vb).unwrap();

        let mask = model.create_causal_mask(3, 0).unwrap();
        let mask_vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // For a 3x3 causal mask:
        // [0, -inf, -inf]
        // [0, 0, -inf]
        // [0, 0, 0]
        assert_eq!(mask_vals[0], 0.0); // (0,0) - can attend
        assert!(mask_vals[1] == f32::NEG_INFINITY); // (0,1) - cannot attend
        assert_eq!(mask_vals[3], 0.0); // (1,0) - can attend
        assert_eq!(mask_vals[4], 0.0); // (1,1) - can attend
        assert!(mask_vals[5] == f32::NEG_INFINITY); // (1,2) - cannot attend
    }

    #[test]
    fn test_causal_mask_with_offset() {
        let device = Device::Cpu;
        let config = small_config();
        let vb = create_mock_vb(&device);

        let model = Qwen3TTSModel::new(config, vb).unwrap();

        // seq_len=2, offset=3 -> total_len=5
        let mask = model.create_causal_mask(2, 3).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 2, 5]);
    }

    #[test]
    fn test_from_pretrained_not_implemented() {
        let device = Device::Cpu;
        let config = small_config();
        let result = Qwen3TTSModel::from_pretrained("/some/path", &config, &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_repeat_kv_no_repeat() {
        let device = Device::Cpu;
        let config = Qwen3TTSConfig {
            num_attention_heads: 4,
            num_key_value_heads: Some(4), // Same as num_heads
            hidden_size: 64,
            ..small_config()
        };
        let vb = create_mock_vb(&device);

        let attn = Attention::new(&config, vb).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 10, 16), &device).unwrap();
        let repeated = attn.repeat_kv(&x).unwrap();

        // Should be unchanged when n_rep = 1
        assert_eq!(repeated.dims(), x.dims());
    }

    #[test]
    fn test_repeat_kv_with_repeat() {
        let device = Device::Cpu;
        let config = Qwen3TTSConfig {
            num_attention_heads: 8,
            num_key_value_heads: Some(2), // 8/2 = 4x repeat
            hidden_size: 128,
            ..small_config()
        };
        let vb = create_mock_vb(&device);

        let attn = Attention::new(&config, vb).unwrap();

        // [batch=1, kv_heads=2, seq=10, head_dim=16]
        let x = Tensor::randn(0.0f32, 1.0, (1, 2, 10, 16), &device).unwrap();
        let repeated = attn.repeat_kv(&x).unwrap();

        // Should expand to [1, 8, 10, 16]
        assert_eq!(repeated.dims(), &[1, 8, 10, 16]);
    }
}
