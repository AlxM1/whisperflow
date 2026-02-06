# WhisperFlow Dockerfile
#
# Build:
#   docker build -t whisperflow .
#
# Run the web server:
#   docker run -p 8766:8766 -v whisperflow-data:/data whisperflow
#
# Run the GPU server (requires nvidia-container-toolkit):
#   docker run --gpus all -p 8765:8765 whisperflow \
#     whisperflow-server --host 0.0.0.0 --port 8765

FROM python:3.11-slim

# Install system dependencies:
#   ffmpeg  - audio format conversion used by faster-whisper
#   libsndfile1 - required by sounddevice / soundfile
#   build-essential - needed to compile native Python extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set up a non-root user
RUN useradd --system --create-home --shell /usr/sbin/nologin whisperflow

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY whisperflow/ ./whisperflow/
COPY config/ ./config/

# Install the package with web extras
RUN pip install --no-cache-dir ".[web]"

# Create a data directory for persistent storage
RUN mkdir -p /data && chown whisperflow:whisperflow /data

USER whisperflow

EXPOSE 8766

CMD ["whisperflow-web", "--host", "0.0.0.0", "--port", "8766"]
