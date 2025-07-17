# Multi-stage build for Raspberry Pi
FROM python:3.9-slim-bullseye as builder

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

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenWakeWord without dependencies, then handle tflite-runtime
RUN pip install --no-cache-dir --no-deps openwakeword==0.6.0 && \
    (pip install --no-cache-dir tflite-runtime || pip install --no-cache-dir tensorflow>=2.10.0)

# Copy project files for installation
COPY pyproject.toml .
COPY README.md .
COPY src/ ./src/

# Install the package without dependencies
RUN pip install --no-cache-dir --no-deps -e .

# Copy additional resources
COPY third_party/ ./third_party/
COPY models/ ./models/

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

# Add runtime stage
FROM python:3.9-slim-bullseye as runtime

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    alsa-utils \
    libasound2 \
    libportaudio2 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ALSA_CARD=1

# Default command - use the installed CLI
CMD ["hey-orac", "run"]