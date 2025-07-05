FROM python:3.12-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    libjack-jackd2-dev \
    libsndfile1 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Change ownership of /app to appuser
RUN chown -R appuser:appuser /app

# Create virtual environment as root (to avoid permission issues)
RUN python3 -m venv /app/venv

# Copy requirements first for better caching
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies in virtual environment
RUN /app/venv/bin/pip install --upgrade pip setuptools wheel
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Change ownership of virtual environment to appuser
RUN chown -R appuser:appuser /app/venv

# Switch to non-root user after setup
USER appuser

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser third_party/ ./third_party/

# Copy configuration
COPY --chown=appuser:appuser src/config.yaml ./config.yaml

# Expose port for health checks (optional)
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PATH=/app/venv/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/venv/bin/python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["/app/venv/bin/python", "src/main.py"] 