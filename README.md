# Hey ORAC Wake-Word Module

A Raspberry Pi container for continuous wake-word detection using OpenWakeWord.

## Features

- Real-time wake-word detection with multiple models
- Hot-reloadable configuration
- Web UI for monitoring and configuration
- Audio streaming to configurable endpoint
- Docker containerized for easy deployment
- Prometheus metrics export

## Quick Start

### Running with Docker

```bash
docker run -d --name hey-orac \
  --device /dev/snd \
  -v $(pwd)/config:/config \
  -v $(pwd)/models:/models \
  -p 8000:8000 \
  ghcr.io/2oby/hey-orac:latest
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/2oby/hey-orac.git
cd hey-orac

# Build Docker image
./scripts/build_image.sh

# Run locally
./scripts/run_local.sh
```

## Configuration

Configuration is managed through `/config/settings.json`:

```json
{
  "models": {
    "hey_jarvis": {
      "enabled": true,
      "threshold": 0.5,
      "type": "onnx"
    }
  },
  "audio": {
    "device_index": -1,
    "sample_rate": 16000,
    "chunk_size": 1280
  },
  "transport": {
    "endpoint": "http://orin-nano:8080/stream"
  }
}
```

## Development

### Prerequisites

- Python 3.11+
- Docker
- PortAudio (for local development)

### Setup Development Environment

```bash
# Install dependencies
pip install -e .[dev]

# Run tests
pytest

# Run linting
black src tests
ruff check src tests

# Type checking
mypy src
```

### Running Tests

```bash
# Unit tests
pytest tests/unit

# Integration tests
pytest tests/integration

# Coverage report
pytest --cov=hey_orac --cov-report=html
```

## Architecture

The module follows a modular architecture:

- **Audio Module**: Handles microphone input and ring buffer management
- **Models Module**: Manages wake-word model loading and inference
- **Config Module**: Settings management with hot-reload support
- **Transport Module**: Audio streaming to external endpoints
- **Web Module**: REST API and WebSocket interface

## API Reference

### REST Endpoints

- `GET /api/v1/settings` - Get current configuration
- `PUT /api/v1/settings` - Update configuration
- `GET /api/v1/health` - Health check endpoint
- `GET /metrics` - Prometheus metrics

### WebSocket Events

- `rms` - Real-time audio level updates (5Hz)
- `wake` - Wake-word detection events
- `config_changed` - Configuration update notifications

## License

MIT License - see LICENSE file for details.