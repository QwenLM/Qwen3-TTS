# Qwen3-TTS REST API Server
# Optimized for NVIDIA DGX Spark with FlashInfer attention backend

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    libsndfile1 \
    libsox-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-server.txt .
COPY pyproject.toml .

# Install PyTorch with CUDA 12.4 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install FlashInfer for optimized attention on DGX Spark
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/

# Install server dependencies
RUN pip install -r requirements-server.txt

# Copy the qwen_tts package
COPY qwen_tts/ ./qwen_tts/

# Install qwen_tts package in editable mode
RUN pip install -e .

# Copy server code
COPY server/ ./server/

# Expose port for FastAPI server
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["python", "-m", "uvicorn", "server.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
