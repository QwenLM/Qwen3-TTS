"""
Streamlit demo UI for Qwen3-TTS synthesis testing.

REQ-007: Implements a web interface for testing the TTS API.
"""

import os
import requests
import streamlit as st

# Backend API URL - configurable via environment variable
API_BASE_URL = os.environ.get("TTS_API_URL", "http://qwen3-tts-server:8000")


def get_speakers() -> list[str]:
    """Fetch available speakers from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/v1/tts/speakers", timeout=10)
        response.raise_for_status()
        return response.json().get("speakers", [])
    except requests.RequestException as e:
        st.error(f"Failed to fetch speakers: {e}")
        return []


def get_languages() -> list[str]:
    """Fetch available languages from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/v1/tts/languages", timeout=10)
        response.raise_for_status()
        return response.json().get("languages", [])
    except requests.RequestException as e:
        st.error(f"Failed to fetch languages: {e}")
        return []


def check_health() -> dict:
    """Check API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {"status": "unavailable", "model_loaded": False}


def synthesize_speech(text: str, speaker: str, language: str, instruct: str | None = None) -> bytes | None:
    """Call the TTS synthesis API and return WAV audio bytes."""
    payload = {
        "text": text,
        "speaker": speaker,
        "language": language,
    }
    if instruct:
        payload["instruct"] = instruct

    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/tts/synthesize",
            json=payload,
            timeout=120,  # TTS can take time for long texts
        )
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        st.error(f"Synthesis failed: {e}")
        return None


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Qwen3-TTS Demo",
        page_icon="🎙️",
        layout="centered",
    )

    st.title("🎙️ Qwen3-TTS Demo")
    st.markdown("Test text-to-speech synthesis with Qwen3-TTS model.")

    # Health check indicator in sidebar
    with st.sidebar:
        st.header("API Status")
        health = check_health()
        if health.get("status") == "healthy":
            st.success("✅ Server healthy")
            if health.get("model_loaded"):
                st.success("✅ Model loaded")
            else:
                st.warning("⏳ Model loading...")
        else:
            st.error("❌ Server unavailable")

        st.markdown("---")
        st.markdown(f"**API URL:** `{API_BASE_URL}`")

    # Fetch available options
    speakers = get_speakers()
    languages = get_languages()

    if not speakers or not languages:
        st.warning("Unable to fetch speakers/languages from API. Is the server running?")
        st.stop()

    # Main form
    with st.form("tts_form"):
        # Text input area
        text = st.text_area(
            "Text to synthesize",
            value="Hello! This is a test of the Qwen3 text-to-speech system.",
            height=150,
            help="Enter the text you want to convert to speech.",
        )

        col1, col2 = st.columns(2)

        with col1:
            # Speaker selection dropdown
            speaker = st.selectbox(
                "Speaker",
                options=speakers,
                index=0,
                help="Select the voice for synthesis.",
            )

        with col2:
            # Language selection dropdown
            language = st.selectbox(
                "Language",
                options=languages,
                index=languages.index("english") if "english" in languages else 0,
                help="Select the language for synthesis.",
            )

        # Optional instruct field for voice style customization
        with st.expander("Advanced Options"):
            instruct = st.text_input(
                "Style instruction (optional)",
                value="",
                help="Optional instruction to customize voice style (e.g., 'speak slowly', 'with excitement').",
            )

        # Submit button
        submitted = st.form_submit_button("🔊 Synthesize", use_container_width=True)

    # Handle form submission
    if submitted:
        if not text.strip():
            st.error("Please enter some text to synthesize.")
        else:
            with st.spinner("Synthesizing speech..."):
                audio_bytes = synthesize_speech(
                    text=text,
                    speaker=speaker,
                    language=language,
                    instruct=instruct if instruct.strip() else None,
                )

            if audio_bytes:
                st.success("Synthesis complete!")

                # Audio playback in browser
                st.audio(audio_bytes, format="audio/wav")

                # Download button
                st.download_button(
                    label="📥 Download WAV",
                    data=audio_bytes,
                    file_name="tts_output.wav",
                    mime="audio/wav",
                )

                # Show audio info
                st.caption(f"Audio size: {len(audio_bytes):,} bytes")


if __name__ == "__main__":
    main()
