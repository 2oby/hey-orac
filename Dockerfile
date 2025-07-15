# Multi-stage build for Raspberry Pi
FROM python:3.9-slim-bullseye as base

# Install system dependencies for audio and OpenWakeWord
RUN apt-get update && apt-get install -y \
    # Audio system dependencies
    alsa-utils \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libsndfile1-dev \
    # Build dependencies
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    # Additional utilities
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY third_party/ ./third_party/

# Create necessary directories
RUN mkdir -p /app/logs /app/recordings

# Set up ALSA configuration for container
RUN echo "pcm.!default {" > /app/.asoundrc && \
    echo "  type pulse" >> /app/.asoundrc && \
    echo "  server unix:/tmp/pulse-socket" >> /app/.asoundrc && \
    echo "}" >> /app/.asoundrc && \
    echo "ctl.!default {" >> /app/.asoundrc && \
    echo "  type pulse" >> /app/.asoundrc && \
    echo "  server unix:/tmp/pulse-socket" >> /app/.asoundrc && \
    echo "}" >> /app/.asoundrc

# Alternative ALSA config for direct hardware access
RUN echo "pcm.!default {" > /app/.asoundrc.hw && \
    echo "  type hw" >> /app/.asoundrc.hw && \
    echo "  card 1" >> /app/.asoundrc.hw && \
    echo "  device 0" >> /app/.asoundrc.hw && \
    echo "}" >> /app/.asoundrc.hw

# Set environment variables
ENV PYTHONPATH=/app
ENV ALSA_CARD=1
ENV AUDIO_DEVICE=/dev/snd

# Default command
CMD ["python3", "src/wake_word_detection.py"]