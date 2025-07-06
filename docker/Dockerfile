FROM python:3.12-slim

# Install system dependencies for audio processing
# Do this first to cache system dependencies separately
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    libasound2-plugins \
    libjack-jackd2-dev \
    libsndfile1 \
    pkg-config \
    alsa-utils \
    python3-pyaudio \
    usbutils \
    libasound2 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Add user to audio group for device access
RUN usermod -a -G audio appuser

# Set working directory
WORKDIR /app

# Create virtual environment as root (to avoid permission issues)
RUN python3 -m venv /app/venv

# Copy requirements and build configuration first for better caching
# This layer will only rebuild if requirements.txt or pyproject.toml changes
COPY --chown=appuser:appuser requirements.txt .
COPY --chown=appuser:appuser pyproject.toml .

# Install Python dependencies in virtual environment with modern build
# This layer will only rebuild if requirements change
RUN /app/venv/bin/pip install --upgrade pip setuptools wheel \
    && /app/venv/bin/pip install --no-cache-dir --use-pep517 -r requirements.txt \
    && /app/venv/bin/pip cache purge \
    && find /app/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /app/venv -name "*.pyc" -delete 2>/dev/null || true

# Install PyAudio with proper ALSA support and force reinstall
RUN /app/venv/bin/pip uninstall -y pyaudio || true \
    && CFLAGS="-I/usr/include/alsa" LDFLAGS="-L/usr/lib/aarch64-linux-gnu -lasound" \
    /app/venv/bin/pip install --no-cache-dir --no-binary=:all: pyaudio \
    && /app/venv/bin/pip install --force-reinstall numpy

# Change ownership of virtual environment to appuser
RUN chown -R appuser:appuser /app/venv

# Switch to non-root user after setup
USER appuser

# Copy application code and configuration
# This layer will rebuild when source code changes
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser third_party/ ./third_party/
COPY --chown=appuser:appuser src/config.yaml ./config.yaml

# Copy ALSA configuration for USB microphone
COPY --chown=appuser:appuser .asoundrc /home/appuser/.asoundrc

# Expose port for health checks (optional)
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH=/app/venv/bin:/home/appuser/.local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
# ALSA configuration for SH-04 USB microphone
ENV ALSA_CARD=0
ENV ALSA_DEVICE=hw:0,0
ENV ALSA_PCM_CARD=0
ENV ALSA_PCM_DEVICE=0

# Health check - temporarily disabled since main loop is commented out
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD /app/venv/bin/python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["/app/venv/bin/python", "src/main.py"] 