# Voice Cloning Modes: X-Vector Only vs ICL Mode

## Overview

Qwen3-TTS Base models support two modes for voice cloning, each with different trade-offs between quality and convenience:

1. **ICL Mode (In-Context Learning)** - Default, high quality
2. **X-Vector Only Mode** - Simplified, lower quality

## ICL Mode (Default - Recommended)

**Configuration**: `x_vector_only_mode=False` (default)

### What It Does

ICL (In-Context Learning) mode uses both:
- **Speaker embedding** (x-vector): Captures the speaker's voice characteristics
- **Reference audio codes**: Encoded representation of the reference audio
- **Reference text**: Transcript of the reference audio

The model conditions on both the reference text and reference speech codes to generate high-quality voice clones.

### Requirements

- `ref_audio`: Reference audio file (required)
- `ref_text`: Transcript of the reference audio (required)

### Advantages

- **Higher quality**: Better voice cloning fidelity
- **Better prosody**: More natural intonation and rhythm
- **Contextual understanding**: Model learns from the reference audio's speaking style

### Example

```python
wavs, sr = model.generate_voice_clone(
    text="New text to synthesize",
    language="English",
    ref_audio="reference.wav",
    ref_text="Transcript of reference audio",  # Required
    x_vector_only_mode=False,  # ICL mode (default)
)
```

## X-Vector Only Mode

**Configuration**: `x_vector_only_mode=True`

### What It Does

Uses only the speaker embedding (x-vector) extracted from the reference audio. The reference text and audio codes are ignored.

### Requirements

- `ref_audio`: Reference audio file (required)
- `ref_text`: Not required (can be omitted or None)

### Advantages

- **Simpler**: No need to provide reference text transcript
- **Faster setup**: Skip transcription step
- **Convenient**: Useful when transcript is unavailable

### Disadvantages

- **Lower quality**: Reduced voice cloning fidelity
- **Limited effect**: As noted in the UI: "效果有限" (limited effectiveness)
- **Less natural**: May lose some prosodic characteristics

### Example

```python
wavs, sr = model.generate_voice_clone(
    text="New text to synthesize",
    language="English",
    ref_audio="reference.wav",
    ref_text=None,  # Not required in x-vector only mode
    x_vector_only_mode=True,  # X-vector only mode
)
```

## Technical Implementation

### VoiceClonePromptItem Structure

```python
@dataclass
class VoiceClonePromptItem:
    ref_code: Optional[torch.Tensor]      # None in x-vector only mode
    ref_spk_embedding: torch.Tensor       # Always used (x-vector)
    x_vector_only_mode: bool              # Mode flag
    icl_mode: bool                        # Opposite of x_vector_only_mode
    ref_text: Optional[str]               # None in x-vector only mode
```

### Mode Selection Logic

```python
# In create_voice_clone_prompt():
if x_vector_only_mode:
    # X-vector only: ignore ref_text and ref_code
    ref_code = None
    icl_mode = False
else:
    # ICL mode: require ref_text and use ref_code
    if ref_text is None or ref_text == "":
        raise ValueError("ref_text is required when x_vector_only_mode=False")
    ref_code = encoded_audio_codes
    icl_mode = True
```

## When to Use Each Mode

### Use ICL Mode (Default) When:

- Quality is the priority
- You have or can obtain the reference audio transcript
- You want the best possible voice cloning results
- You're fine-tuning or doing production work

### Use X-Vector Only Mode When:

- You don't have the reference audio transcript
- Quick prototyping or testing
- Convenience is more important than quality
- The reference audio is difficult to transcribe

## UI Labels

In the Gradio demo (`qwen_tts/cli/demo.py`):

- **English**: "Use x-vector only"
- **Chinese**: "仅用说话人向量，效果有限，但不用传入参考音频文本"
  - Translation: "Only use speaker vector, limited effectiveness, but no need to provide reference audio text"

## Best Practices

1. **Default to ICL mode** for production use
2. **Provide accurate transcripts** when using ICL mode - accuracy matters
3. **Use x-vector only mode** only when transcripts are unavailable
4. **Test both modes** to understand the quality difference for your use case
5. **Reuse prompts** with `create_voice_clone_prompt()` for efficiency when generating multiple outputs with the same reference

## Code Examples

### Creating Reusable Prompts

```python
# ICL mode (high quality)
prompt_icl = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text="Transcript of reference",
    x_vector_only_mode=False,
)

# X-vector only mode (convenient)
prompt_xvec = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text=None,  # Not needed
    x_vector_only_mode=True,
)

# Reuse prompts for multiple generations
wavs1, sr = model.generate_voice_clone(
    text="First sentence",
    language="English",
    voice_clone_prompt=prompt_icl,
)

wavs2, sr = model.generate_voice_clone(
    text="Second sentence",
    language="English",
    voice_clone_prompt=prompt_icl,
)
```

### Batch Processing with Different Modes

```python
# Mix modes in batch processing
prompt_items = model.create_voice_clone_prompt(
    ref_audio=["ref1.wav", "ref2.wav"],
    ref_text=["Transcript 1", None],
    x_vector_only_mode=[False, True],  # ICL for first, x-vector for second
)
```

## Related Files

- Implementation: `qwen_tts/inference/qwen3_tts_model.py`
- UI: `qwen_tts/cli/demo.py`
- Examples: `examples/test_model_12hz_base.py`
