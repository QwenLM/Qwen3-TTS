# coding=utf-8
# Qwen3-TTS Gradio Demo for HuggingFace Spaces with Zero GPU
# Supports: Voice Design, Voice Clone (Base), TTS (CustomVoice)
import subprocess
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)
import os
import spaces
import gradio as gr
import numpy as np
import torch
from huggingface_hub import snapshot_download

from huggingface_hub import login
HF_TOKEN = os.environ.get('HF_TOKEN')
login(token=HF_TOKEN)

# Global model holders - keyed by (model_type, model_size)
loaded_models = {}

# Model size options
MODEL_SIZES = ["0.6B", "1.7B"]


def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")


def get_model(model_type: str, model_size: str):
    """Get or load a model by type and size."""
    global loaded_models
    key = (model_type, model_size)
    if key not in loaded_models:
        from qwen_tts import Qwen3TTSModel
        model_path = get_model_path(model_type, model_size)
        loaded_models[key] = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda",
            dtype=torch.bfloat16,
            token=HF_TOKEN,
#           attn_implementation="flash_attention_2",
        )
    return loaded_models[key]


def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


# Speaker and language choices for CustomVoice model
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]


@spaces.GPU(duration=60)
def generate_voice_design(text, language, voice_description):
    """Generate speech using Voice Design model (1.7B only)."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not voice_description or not voice_description.strip():
        return None, "Error: Voice description is required."

    try:
        tts = get_model("VoiceDesign", "1.7B")
        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Voice design generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


@spaces.GPU(duration=60)
def generate_voice_clone(ref_audio, ref_text, target_text, language, use_xvector_only, model_size):
    """Generate speech using Base (Voice Clone) model."""
    if not target_text or not target_text.strip():
        return None, "Error: Target text is required."

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "Error: Reference audio is required."

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "Error: Reference text is required when 'Use x-vector only' is not enabled."

    try:
        tts = get_model("Base", model_size)
        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=language,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Voice clone generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


@spaces.GPU(duration=60)
def generate_custom_voice(text, language, speaker, instruct, model_size):
    """Generate speech using CustomVoice model."""
    if not text or not text.strip():
        return None, "Error: Text is required."
    if not speaker:
        return None, "Error: Speaker is required."

    try:
        tts = get_model("CustomVoice", model_size)
        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=language,
            speaker=speaker.lower().replace(" ", "_"),
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "Generation completed successfully!"
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"


# Build Gradio UI
def build_ui():
    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
    )

    css = """
    .gradio-container {max-width: none !important;}
    .tab-content {padding: 20px;}
    """

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Demo") as demo:
        gr.Markdown(
            """
# Qwen3-TTS Demo

A unified Text-to-Speech demo featuring three powerful modes:
- **Voice Design**: Create custom voices using natural language descriptions
- **Voice Clone (Base)**: Clone any voice from a reference audio
- **TTS (CustomVoice)**: Generate speech with predefined speakers and optional style instructions

Built with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team.
"""
        )

        with gr.Tabs():
            # Tab 1: Voice Design (Default, 1.7B only)
            with gr.Tab("Voice Design"):
                gr.Markdown("### Create Custom Voice with Natural Language")
                with gr.Row():
                    with gr.Column(scale=2):
                        design_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to convert to speech...",
                            value="It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
                        )
                        design_language = gr.Dropdown(
                            label="Language",
                            choices=LANGUAGES,
                            value="Auto",
                            interactive=True,
                        )
                        design_instruct = gr.Textbox(
                            label="Voice Description",
                            lines=3,
                            placeholder="Describe the voice characteristics you want...",
                            value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
                        )
                        design_btn = gr.Button("Generate with Custom Voice", variant="primary")

                    with gr.Column(scale=2):
                        design_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        design_status = gr.Textbox(label="Status", lines=2, interactive=False)

                design_btn.click(
                    generate_voice_design,
                    inputs=[design_text, design_language, design_instruct],
                    outputs=[design_audio_out, design_status],
                )

            # Tab 2: Voice Clone (Base)
            with gr.Tab("Voice Clone (Base)"):
                gr.Markdown("### Clone Voice from Reference Audio")
                with gr.Row():
                    with gr.Column(scale=2):
                        clone_ref_audio = gr.Audio(
                            label="Reference Audio (Upload a voice sample to clone)",
                            type="numpy",
                        )
                        clone_ref_text = gr.Textbox(
                            label="Reference Text (Transcript of the reference audio)",
                            lines=2,
                            placeholder="Enter the exact text spoken in the reference audio...",
                        )
                        clone_xvector = gr.Checkbox(
                            label="Use x-vector only (No reference text needed, but lower quality)",
                            value=False,
                        )

                    with gr.Column(scale=2):
                        clone_target_text = gr.Textbox(
                            label="Target Text (Text to synthesize with cloned voice)",
                            lines=4,
                            placeholder="Enter the text you want the cloned voice to speak...",
                        )
                        with gr.Row():
                            clone_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="Auto",
                                interactive=True,
                            )
                            clone_model_size = gr.Dropdown(
                                label="Model Size",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                        clone_btn = gr.Button("Clone & Generate", variant="primary")

                with gr.Row():
                    clone_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                    clone_status = gr.Textbox(label="Status", lines=2, interactive=False)

                clone_btn.click(
                    generate_voice_clone,
                    inputs=[clone_ref_audio, clone_ref_text, clone_target_text, clone_language, clone_xvector, clone_model_size],
                    outputs=[clone_audio_out, clone_status],
                )

            # Tab 3: TTS (CustomVoice)
            with gr.Tab("TTS (CustomVoice)"):
                gr.Markdown("### Text-to-Speech with Predefined Speakers")
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_text = gr.Textbox(
                            label="Text to Synthesize",
                            lines=4,
                            placeholder="Enter the text you want to convert to speech...",
                            value="Hello! Welcome to Text-to-Speech system. This is a demo of our TTS capabilities."
                        )
                        with gr.Row():
                            tts_language = gr.Dropdown(
                                label="Language",
                                choices=LANGUAGES,
                                value="English",
                                interactive=True,
                            )
                            tts_speaker = gr.Dropdown(
                                label="Speaker",
                                choices=SPEAKERS,
                                value="Ryan",
                                interactive=True,
                            )
                        with gr.Row():
                            tts_instruct = gr.Textbox(
                                label="Style Instruction (Optional)",
                                lines=2,
                                placeholder="e.g., Speak in a cheerful and energetic tone",
                            )
                            tts_model_size = gr.Dropdown(
                                label="Model Size",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )
                        tts_btn = gr.Button("Generate Speech", variant="primary")

                    with gr.Column(scale=2):
                        tts_audio_out = gr.Audio(label="Generated Audio", type="numpy")
                        tts_status = gr.Textbox(label="Status", lines=2, interactive=False)

                tts_btn.click(
                    generate_custom_voice,
                    inputs=[tts_text, tts_language, tts_speaker, tts_instruct, tts_model_size],
                    outputs=[tts_audio_out, tts_status],
                )

        gr.Markdown(
            """
---

**Note**: This demo uses HuggingFace Spaces Zero GPU. Each generation has a time limit.
For longer texts, please split them into smaller segments.
"""
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
