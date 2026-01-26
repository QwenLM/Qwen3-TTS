# Qwen3-TTS REST API Server
# Optimized for NVIDIA DGX Spark with Blackwell GPU (sm_121) support

# Use NGC PyTorch container with Blackwell support
FROM nvcr.io/nvidia/pytorch:25.01-py3

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libsox-dev \
    sox \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-server.txt .
COPY pyproject.toml .

# Install server dependencies (PyTorch already included in NGC container)
RUN pip install -r requirements-server.txt

# Build torchaudio from source (NGC's custom torch 2.6.0a0 needs matching torchaudio)
# Using v2.6.0 branch, build wheel and install with --no-deps to skip version check
RUN pip install sox && \
    git clone --depth 1 --branch v2.6.0 https://github.com/pytorch/audio.git /tmp/torchaudio && \
    cd /tmp/torchaudio && \
    BUILD_SOX=0 USE_CUDA=1 python setup.py bdist_wheel && \
    pip install --no-deps dist/*.whl && \
    rm -rf /tmp/torchaudio

# Copy the qwen_tts package
COPY qwen_tts/ ./qwen_tts/

# Install qwen_tts package WITHOUT reinstalling torch (NGC provides compatible version)
# Use pip constraint to prevent PyPI torch from being installed over NGC's torch
RUN echo "torch" > /tmp/constraints.txt && \
    echo "torchvision" >> /tmp/constraints.txt && \
    pip install -c /tmp/constraints.txt gradio && \
    pip install -e . --no-deps

# Copy server code
COPY server/ ./server/

# Expose port for FastAPI server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["python", "-m", "uvicorn", "server.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
