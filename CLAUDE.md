# CLAUDE.md - AI Assistant Guide for Qwen3-TTS

This document provides AI assistants with essential information about the Qwen3-TTS codebase structure, development workflows, conventions, and best practices.

## Project Overview

**Qwen3-TTS** is a comprehensive speech synthesis system developed by Alibaba Qwen Team. It provides:
- Voice cloning capabilities (3-second rapid cloning)
- Voice design from natural language descriptions
- Custom voice generation with premium timbres
- Ultra-high-quality human-like speech generation
- Natural language-based voice control
- Multilingual support (10+ languages)

### Key Features
- **12Hz/25Hz Tokenizers**: High-fidelity acoustic compression
- **Streaming Support**: Low-latency streaming generation (97ms end-to-end)
- **Instruction-Based Control**: Control tone, emotion, prosody via natural language
- **Model Sizes**: 0.6B and 1.7B parameter variants
- **Languages**: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## Repository Structure

```
Qwen3-TTS/
├── qwen_tts/              # Main Python package
│   ├── core/              # Core implementations
│   │   ├── models/        # TTS model architecture (configuration, modeling, processing)
│   │   ├── tokenizer_12hz/  # 12Hz tokenizer implementation
│   │   └── tokenizer_25hz/  # 25Hz tokenizer implementation (VQ-based)
│   ├── inference/         # High-level inference APIs
│   │   ├── qwen3_tts_model.py      # Main TTS model interface
│   │   └── qwen3_tts_tokenizer.py  # Tokenizer interface
│   ├── cli/               # CLI tools
│   │   └── demo.py        # Gradio web demo
│   └── __init__.py        # Package exports
├── examples/              # Usage examples
│   ├── test_model_12hz_base.py          # Voice clone examples
│   ├── test_model_12hz_custom_voice.py  # Custom voice examples
│   ├── test_model_12hz_voice_design.py  # Voice design examples
│   └── test_tokenizer_12hz.py           # Tokenizer usage
├── finetuning/            # Fine-tuning workflows
│   ├── prepare_data.py    # Data preparation script
│   ├── sft_12hz.py        # Fine-tuning script
│   ├── dataset.py         # Dataset utilities
│   └── README.md          # Fine-tuning guide
├── assets/                # Documentation assets
├── .github/               # GitHub configuration
│   ├── ISSUE_TEMPLATE/    # Issue templates
│   └── workflows/         # CI/CD workflows
├── README.md              # Main documentation
├── LICENSE                # Apache-2.0 license
├── pyproject.toml         # Python package configuration
└── MANIFEST.in            # Package manifest

```

## Available Models

### Tokenizers
- **Qwen3-TTS-Tokenizer-12Hz**: Encodes/decodes speech into/from codes (16 codebooks, 2048 vocab, 12.5 FPS)
- **Qwen3-TTS-Tokenizer-25Hz**: Alternative tokenizer (25 FPS)

### TTS Models (12Hz Series)

| Model | Size | Features | Streaming | Instruction Control |
|-------|------|----------|-----------|---------------------|
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 1.7B | Voice design from descriptions | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 1.7B | 9 premium timbres + instruction control | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-Base | 1.7B | 3-sec voice clone, fine-tuning | ✅ | ❌ |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 0.6B | 9 premium timbres | ✅ | ❌ |
| Qwen3-TTS-12Hz-0.6B-Base | 0.6B | 3-sec voice clone, fine-tuning | ✅ | ❌ |

### Premium Speakers (CustomVoice Models)
- **Vivian**: Bright young female (Chinese)
- **Serena**: Warm young female (Chinese)
- **Uncle_Fu**: Seasoned male with mellow timbre (Chinese)
- **Dylan**: Beijing male with clear timbre (Chinese - Beijing Dialect)
- **Eric**: Chengdu male with husky brightness (Chinese - Sichuan Dialect)
- **Ryan**: Dynamic male with rhythmic drive (English)
- **Aiden**: Sunny American male (English)
- **Ono_Anna**: Playful Japanese female (Japanese)
- **Sohee**: Warm Korean female (Korean)

## Code Organization

### Package Structure
The `qwen_tts` package follows a modular architecture:

1. **Core Layer** (`qwen_tts/core/`)
   - Model architecture implementations
   - Tokenizer implementations (12Hz and 25Hz variants)
   - Configuration classes
   - Processing utilities

2. **Inference Layer** (`qwen_tts/inference/`)
   - High-level user-facing APIs
   - `Qwen3TTSModel`: Main TTS interface
   - `Qwen3TTSTokenizer`: Audio encoding/decoding interface

3. **CLI Layer** (`qwen_tts/cli/`)
   - Command-line interfaces
   - Gradio web demo

### Key Classes

#### Qwen3TTSModel
Primary interface for TTS generation:
- `from_pretrained()`: Load models from HuggingFace/ModelScope or local paths
- `generate_custom_voice()`: Generate with premium speakers
- `generate_voice_design()`: Generate from natural language descriptions
- `generate_voice_clone()`: Clone voices from reference audio
- `create_voice_clone_prompt()`: Create reusable voice prompts
- `get_supported_speakers()`: List available speakers
- `get_supported_languages()`: List supported languages

#### Qwen3TTSTokenizer
Audio tokenization interface:
- `from_pretrained()`: Load tokenizer
- `encode()`: Convert audio to codes
- `decode()`: Convert codes to audio

## Development Workflows

### Environment Setup

**Recommended Python Version**: 3.12 (supports 3.9-3.13)

```bash
# Create clean environment
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# Install from PyPI
pip install -U qwen-tts

# Install from source (for development)
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
pip install -e .

# Optional: FlashAttention 2 (for GPU memory optimization)
pip install -U flash-attn --no-build-isolation
# OR with limited cores:
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

### Dependencies
Core dependencies (from pyproject.toml):
- `transformers==4.57.3` (HuggingFace Transformers)
- `accelerate==1.12.0` (Multi-GPU support)
- `gradio` (Web UI)
- `librosa`, `torchaudio`, `soundfile`, `sox` (Audio processing)
- `onnxruntime` (ONNX support)
- `einops` (Tensor operations)

### Model Loading Pattern

```python
import torch
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",  # HF model ID or local path
    device_map="cuda:0",                      # Device placement
    dtype=torch.bfloat16,                     # Precision (bfloat16 recommended)
    attn_implementation="flash_attention_2",  # Use FlashAttention if available
)
```

### Inference Patterns

#### Custom Voice Generation
```python
wavs, sr = model.generate_custom_voice(
    text="Your text here",
    language="Chinese",  # or "Auto" for auto-detection
    speaker="Vivian",    # Use model.get_supported_speakers()
    instruct="语气描述",  # Optional instruction
)
```

#### Voice Design
```python
wavs, sr = model.generate_voice_design(
    text="Your text here",
    language="Chinese",
    instruct="描述目标音色特征",  # Natural language voice description
)
```

#### Voice Clone
```python
wavs, sr = model.generate_voice_clone(
    text="Your text here",
    language="English",
    ref_audio="path/to/reference.wav",  # URL, path, base64, or (array, sr)
    ref_text="Reference transcript",
)

# Reusable prompt for efficiency
prompt = model.create_voice_clone_prompt(
    ref_audio="path/to/reference.wav",
    ref_text="Reference transcript",
)
wavs, sr = model.generate_voice_clone(
    text=["Text 1", "Text 2"],
    language=["English", "English"],
    voice_clone_prompt=prompt,  # Reuse across generations
)
```

### Fine-tuning Workflow

Located in `finetuning/` directory:

1. **Prepare JSONL Data**
   ```jsonl
   {"audio":"./data/utt0001.wav","text":"Transcript","ref_audio":"./data/ref.wav"}
   ```

2. **Extract Audio Codes**
   ```bash
   python prepare_data.py \
     --device cuda:0 \
     --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
     --input_jsonl train_raw.jsonl \
     --output_jsonl train_with_codes.jsonl
   ```

3. **Run Fine-tuning**
   ```bash
   python sft_12hz.py \
     --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
     --output_model_path output \
     --train_jsonl train_with_codes.jsonl \
     --batch_size 2 \
     --lr 2e-5 \
     --num_epochs 3 \
     --speaker_name custom_speaker
   ```

4. **Test Fine-tuned Model**
   ```python
   model = Qwen3TTSModel.from_pretrained("output/checkpoint-epoch-2", ...)
   wavs, sr = model.generate_custom_voice(text="...", speaker="custom_speaker")
   ```

### Web Demo

```bash
# Launch Gradio demo
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000

# For Base model with microphone (requires HTTPS)
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --ssl-certfile cert.pem --ssl-keyfile key.pem --no-ssl-verify
```

## Coding Conventions & Best Practices

### Code Style
- **License Headers**: All Python files include Apache-2.0 license headers
- **Copyright**: "Copyright 2026 The Alibaba Qwen team."
- **Encoding**: UTF-8 (`# coding=utf-8`)
- **SPDX**: `# SPDX-License-Identifier: Apache-2.0`

### Import Organization
Standard structure:
```python
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

# Standard library imports
import time
import torch

# Third-party imports
import soundfile as sf

# Local imports
from qwen_tts import Qwen3TTSModel
```

### Model Loading Best Practices
1. **Always use `torch.bfloat16`** for optimal quality/performance
2. **Use FlashAttention 2** when available (requires compatible hardware)
3. **Explicit device placement**: Use `device_map="cuda:0"` or specific device
4. **Batch inference**: Pass lists to generate functions for efficiency

### Audio Handling
- **Output format**: `(wavs, sample_rate)` tuple
- **Sample rate**: Typically 16000 Hz
- **Batch output**: `wavs` is a list of numpy arrays
- **Supported inputs**: Local paths, URLs, base64 strings, (array, sr) tuples

### Language Specification
- Use `"Auto"` for automatic language detection
- Explicitly set language when known for better quality
- Supported: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

### Error Handling & Edge Cases
1. **Reference audio**: 3+ seconds recommended for voice clone
2. **Same ref_audio**: Use identical reference across fine-tuning dataset
3. **Streaming**: All models support streaming mode
4. **x_vector_only_mode**: Falls back to speaker embedding only (lower quality)

## Testing & Validation

### Example Scripts
Run examples to verify installation:
```bash
python examples/test_model_12hz_custom_voice.py
python examples/test_model_12hz_voice_design.py
python examples/test_model_12hz_base.py
python examples/test_tokenizer_12hz.py
```

### Performance Benchmarks
- **Latency**: 97ms end-to-end synthesis (streaming)
- **WER**: 0.77 (Chinese), 1.24 (English) on Seed-TTS test
- **Speaker Similarity**: 0.8+ cosine similarity
- **Quality**: 4.16 UTMOS score

## Common Development Tasks

### Adding New Speakers (Fine-tuning)
1. Collect 10-50 utterances from target speaker
2. Prepare JSONL with consistent `ref_audio`
3. Run `prepare_data.py` to extract codes
4. Fine-tune with `sft_12hz.py`
5. Test with custom speaker name

### Integrating into Applications
```python
from qwen_tts import Qwen3TTSModel
import torch

# Initialize once
tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Generate on demand
def synthesize(text, speaker="Vivian", language="Auto"):
    wavs, sr = tts.generate_custom_voice(
        text=text, speaker=speaker, language=language
    )
    return wavs[0], sr
```

### Batch Processing
```python
texts = ["Text 1", "Text 2", "Text 3"]
languages = ["Chinese", "English", "Japanese"]
speakers = ["Vivian", "Ryan", "Ono_Anna"]

wavs, sr = model.generate_custom_voice(
    text=texts,
    language=languages,
    speaker=speakers,
)

# Save all outputs
for i, wav in enumerate(wavs):
    sf.write(f"output_{i}.wav", wav, sr)
```

## Model Download

Models auto-download from HuggingFace or ModelScope. Manual download:

```bash
# Via ModelScope (recommended for China)
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-Tokenizer-12Hz --local_dir ./models/tokenizer
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ./models/custom

# Via HuggingFace
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./models/tokenizer
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./models/custom
```

Then use local paths:
```python
model = Qwen3TTSModel.from_pretrained("./models/custom", ...)
```

## vLLM Integration

For production deployment, use vLLM-Omni:
```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni/examples/offline_inference/qwen3_tts

# Run inference
python end2end.py --query-type CustomVoice
python end2end.py --query-type VoiceDesign
python end2end.py --query-type Base --mode-tag icl
```

## Documentation & Resources

- **Main README**: `/README.md` - Comprehensive usage guide
- **Fine-tuning Guide**: `/finetuning/README.md` - SFT workflow
- **Examples**: `/examples/` - Code samples for each feature
- **Paper**: https://arxiv.org/abs/2601.15621
- **Blog**: https://qwen.ai/blog?id=qwen3tts-0115
- **HuggingFace**: https://huggingface.co/collections/Qwen/qwen3-tts
- **ModelScope**: https://modelscope.cn/collections/Qwen/Qwen3-TTS

## AI Assistant Guidelines

When working with this codebase:

1. **Read before modifying**: Always read relevant files before suggesting changes
2. **Use examples**: Reference `/examples/` for usage patterns
3. **Check model compatibility**: Verify model variant supports requested features
4. **Respect license**: Maintain Apache-2.0 headers in all files
5. **Test changes**: Suggest running example scripts after modifications
6. **Device awareness**: Consider GPU availability and memory constraints
7. **Batch optimization**: Recommend batch processing for multiple inputs
8. **Language handling**: Use explicit language settings when language is known
9. **Error context**: Check model size, device memory, and input formats
10. **Documentation**: Update README.md if adding features or changing APIs

### Common Pitfalls to Avoid
- Don't use Base models for instruction-based control (not supported)
- Don't omit `ref_text` in voice clone unless using `x_vector_only_mode=True`
- Don't use FP32 dtype (use bfloat16 for quality and memory efficiency)
- Don't assume FlashAttention is available (check hardware compatibility)
- Don't use relative imports outside package structure
- Don't modify core model architectures without understanding tokenizer coupling

### When Suggesting Changes
1. Verify the change aligns with existing code patterns
2. Consider backward compatibility
3. Test with multiple model variants if applicable
4. Update docstrings and type hints
5. Consider performance implications (batching, device placement)
6. Maintain consistency with HuggingFace Transformers conventions

## Version Information

- **Package Version**: 0.0.4 (as of pyproject.toml)
- **Python Support**: 3.9 - 3.13
- **Transformers Version**: 4.57.3
- **Accelerate Version**: 1.12.0
- **Release Date**: January 2026 (Qwen3-TTS-12Hz series)

---

**Last Updated**: 2026-01-23

This guide should be updated when:
- New models are released
- API changes are introduced
- New features are added
- Best practices evolve
- Dependencies are updated