"""
FastAPI server for Qwen3-TTS REST API.

Provides TTS synthesis endpoint and health/metadata endpoints.
Configured for external access from Meta Quest clients.
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from server.model_manager import get_model_manager


class SynthesizeRequest(BaseModel):
    """Request body for TTS synthesis."""

    text: str = Field(..., description="Text to synthesize")
    speaker: str = Field(..., description="Speaker name (e.g., 'Vivian', 'Ryan')")
    language: str = Field(..., description="Language: 'Chinese', 'English', or 'Auto'")
    instruct: Optional[str] = Field(None, description="Optional style instruction")


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str
    model_loaded: bool


class SpeakersResponse(BaseModel):
    """Response for speakers list endpoint."""

    speakers: list[str]


class LanguagesResponse(BaseModel):
    """Response for languages list endpoint."""

    languages: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for model preloading."""
    manager = get_model_manager()
    manager.load_model()
    yield


app = FastAPI(
    title="Qwen3-TTS REST API",
    description="Text-to-Speech synthesis API using Qwen3-TTS model",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    manager = get_model_manager()
    return HealthResponse(status="healthy", model_loaded=manager.model_loaded)


@app.get("/v1/tts/speakers", response_model=SpeakersResponse)
async def get_speakers() -> SpeakersResponse:
    """Get list of available speakers."""
    manager = get_model_manager()
    speakers = manager.get_supported_speakers()
    return SpeakersResponse(speakers=speakers)


@app.get("/v1/tts/languages", response_model=LanguagesResponse)
async def get_languages() -> LanguagesResponse:
    """Get list of supported languages."""
    manager = get_model_manager()
    languages = manager.get_supported_languages()
    return LanguagesResponse(languages=languages)


@app.post("/v1/tts/synthesize")
async def synthesize(request: SynthesizeRequest) -> Response:
    """
    Synthesize speech from text.

    Returns WAV audio binary data.
    """
    manager = get_model_manager()

    if not manager.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        audio, sample_rate = manager.generate_custom_voice(
            text=request.text,
            speaker=request.speaker,
            language=request.language,
            instruct=request.instruct,
        )
        wav_bytes = manager.audio_to_wav_bytes(audio, sample_rate)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
