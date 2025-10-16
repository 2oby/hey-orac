# Hey ORAC Developer Guide

**Version**: 1.0
**Last Updated**: 2025-10-16
**Target Audience**: Developers, Contributors, System Architects

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Development Environment Setup](#development-environment-setup)
4. [Project Structure](#project-structure)
5. [Core Components](#core-components)
6. [Deployment & Build System](#deployment--build-system)
7. [Configuration Management](#configuration-management)
8. [Testing Strategy](#testing-strategy)
9. [Performance & Monitoring](#performance--monitoring)
10. [Code Quality & Refactoring History](#code-quality--refactoring-history)
11. [Contributing Guidelines](#contributing-guidelines)

---

## System Overview

Hey ORAC is a Raspberry Pi-based wake word detection system that continuously monitors audio input for configurable wake words, captures voice commands, and streams them to external API endpoints for processing.

### Key Capabilities

- **Multi-Model Wake Word Detection**: Supports multiple OpenWakeWord models (ONNX/TFLite)
- **Real-time Web Interface**: Flask/SocketIO-based monitoring and configuration
- **STT Integration**: Seamless integration with ORAC STT service for speech-to-text
- **Audio Processing**: Ring buffer with pre-roll capture and silence detection
- **Hot Configuration Reload**: Change models and settings without restart
- **Docker Deployment**: Containerized with smart build detection

### Technology Stack

- **Runtime**: Python 3.11 on Alpine Linux
- **Audio**: PyAudio (ALSA backend), 16kHz 16-bit mono PCM
- **ML Framework**: OpenWakeWord (ONNX Runtime / TensorFlow Lite)
- **Web**: Flask 3.0 + Flask-SocketIO + gevent
- **Container**: Docker with multi-stage builds
- **IPC**: multiprocessing.Manager for shared state

---

## Architecture

### System Design

The application uses a **modular monolithic architecture** after extensive refactoring (see [CLEANUP.md](CLEANUP.md) for full sprint history).

```
┌─────────────────────────────────────────────────────────────┐
│                     Hey ORAC Application                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐         ┌──────────────────┐          │
│  │  Audio Reader   │         │   Web Server     │          │
│  │    Thread       │────────▶│  (Flask/SocketIO)│          │
│  │  (PyAudio)      │         │   Port 7171      │          │
│  └────────┬────────┘         └────────┬─────────┘          │
│           │                            │                     │
│           │   Multi-Consumer Queues    │                     │
│           │                            │                     │
│  ┌────────▼────────┐         ┌────────▼─────────┐          │
│  │  Main Detection │         │  WebSocket       │          │
│  │      Loop       │────────▶│  Broadcaster     │          │
│  │  (OpenWakeWord) │         │  (5Hz updates)   │          │
│  └────────┬────────┘         └──────────────────┘          │
│           │                                                  │
│           │  Wake Word Detected                             │
│           │                                                  │
│  ┌────────▼────────┐                                        │
│  │ Speech Recorder │                                        │
│  │  (Background    │                                        │
│  │   Thread)       │                                        │
│  └────────┬────────┘                                        │
│           │                                                  │
│           │  HTTP POST                                       │
│           │                                                  │
└───────────┼──────────────────────────────────────────────────┘
            │
            ▼
    ┌───────────────┐
    │  ORAC STT API │
    │  (External)   │
    └───────────────┘
```

### Audio Processing Pipeline

```
Microphone Input (USB/ALSA)
    ↓
Audio Reader Thread (non-blocking)
    ↓
Multi-Consumer Distribution
    ├──▶ Main Loop (wake word detection)
    └──▶ Speech Recorder (post-detection capture)
         ↓
Ring Buffer (10s history, 1s pre-roll)
    ↓
OpenWakeWord Inference (~7ms/model)
    ↓
Detection? → Capture Audio
    ↓
Endpointing (silence detection: 300ms + 400ms grace)
    ↓
WAV Encoding + HTTP POST to endpoint
```

### Inter-Process Communication

| Data Type | Producer | Consumer | Mechanism |
|-----------|----------|----------|-----------|
| RMS values | Audio thread | Web server | `shared_data` dict |
| Detection events | Detection loop | Web UI | Event queue + WebSocket |
| Config changes | Web UI | Detection loop | `config_changed` flag |
| Audio chunks | Audio reader | Multiple consumers | Per-consumer queues |

---

## Development Environment Setup

### Prerequisites

**Development Machine:**
- macOS/Linux/Windows with Docker
- Python 3.11+
- Git
- SSH access to Raspberry Pi

**Raspberry Pi:**
- Raspberry Pi 4 (2GB+ RAM recommended)
- Raspberry Pi OS (Bullseye or later)
- Docker & Docker Compose installed
- USB microphone connected
- SSH enabled with key authentication

### SSH Configuration

Add to `~/.ssh/config`:

```
Host pi
  HostName 192.168.8.99
  User 2oby
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 60
```

Test connection:
```bash
ssh pi "whoami && hostname"
# Expected: 2oby, niederpi
```

### Clone Repository

```bash
git clone https://github.com/your-org/hey-orac.git
cd hey-orac
```

### Local Development (Optional)

For local testing without Pi:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -r requirements.txt

# Run locally (requires audio device)
python -m hey_orac.wake_word_detection
```

---

## Project Structure

```
hey-orac/
├── src/
│   └── hey_orac/
│       ├── __init__.py
│       ├── wake_word_detection.py    # Main entry point
│       ├── constants.py               # Centralized constants (Sprint 6)
│       ├── audio/
│       │   ├── audio_manager.py      # Audio device management
│       │   ├── audio_reader_thread.py # Non-blocking audio capture
│       │   ├── conversion.py         # Audio format conversion (Sprint 7)
│       │   ├── endpointing.py        # Silence detection
│       │   ├── ring_buffer.py        # Pre-roll audio storage
│       │   ├── speech_recorder.py    # Post-detection recording
│       │   └── utils.py              # Audio utilities
│       ├── config/
│       │   └── manager.py            # Settings management
│       ├── models/
│       │   └── wake_detector.py      # OpenWakeWord wrapper
│       ├── transport/
│       │   ├── heartbeat_sender.py   # ORAC STT heartbeat
│       │   └── stt_client.py         # STT API client
│       └── web/
│           ├── app.py                # Flask application factory
│           ├── routes.py             # REST API endpoints
│           ├── broadcaster.py        # WebSocket broadcaster
│           └── static/               # Web UI assets
│               ├── index.html
│               ├── css/
│               └── js/
├── models/
│   ├── openwakeword/                 # Default OpenWakeWord models
│   └── custom/                       # Custom user models
├── config/
│   └── settings.json.template        # Default configuration
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
│   └── deploy_and_test.sh           # Automated deployment
├── docker/
│   └── Dockerfile                   # Multi-stage container build
├── docker-compose.yml               # Service orchestration
├── pyproject.toml                   # Python package metadata
├── requirements.txt                 # Python dependencies
├── CLEANUP.md                       # Refactoring sprint history
├── devlog.md                        # Development chronology
├── DEVELOPER_GUIDE.md              # This file
├── USER_GUIDE.md                    # End-user documentation
└── README.md                        # Quick start guide
```

### Key Directories

- **src/hey_orac/**: All application source code
- **models/**: OpenWakeWord model files (ONNX/TFLite)
- **config/**: Runtime configuration (volume-mounted)
- **scripts/**: Deployment and utility scripts
- **tests/**: Unit and integration tests

---

## Core Components

### 1. Audio Manager (`audio/audio_manager.py`)

Manages PyAudio interface and device discovery.

**Key Methods:**
```python
class AudioManager:
    def find_usb_microphone(self) -> Optional[int]:
        """Locate USB microphone device index."""

    def open_stream(self, device_index: int, **params) -> pyaudio.Stream:
        """Open audio stream with specified parameters."""

    def close(self):
        """Clean up PyAudio resources."""
```

### 2. Audio Reader Thread (`audio/audio_reader_thread.py`)

Non-blocking audio capture with multi-consumer distribution (Sprint 9 fix).

**Architecture:**
```python
class AudioReaderThread:
    def register_consumer(self, name: str) -> queue.Queue:
        """Register new consumer, returns dedicated queue."""

    def unregister_consumer(self, name: str):
        """Remove consumer and clean up queue."""

    def start(self) -> bool:
        """Start audio reading thread."""

    def is_healthy(self) -> bool:
        """Check thread health (heartbeat monitoring)."""
```

**Why Multi-Consumer?**
Previously, a single queue was shared causing choppy audio (main loop and speech recorder competing). Now each consumer gets complete audio stream.

### 3. Audio Conversion (`audio/conversion.py`)

Centralized audio format conversion (Sprint 7).

**Functions:**
```python
def convert_to_openwakeword_format(audio_bytes: bytes) -> np.ndarray:
    """
    Convert raw audio bytes to OpenWakeWord format.
    CRITICAL: Does NOT normalize to [-1, 1] range.
    OpenWakeWord expects raw int16 values as float32.
    """

def convert_to_normalized_format(audio_bytes: bytes) -> np.ndarray:
    """
    Convert and normalize audio for STT.
    Scales to [-1, 1] range for proper speech recognition.
    """
```

**Historical Context:**
The normalization bug (documented in devlog.md 2025-09-11) caused detection failures. OpenWakeWord models are trained on non-normalized int16 values cast to float32.

### 4. Ring Buffer (`audio/ring_buffer.py`)

Circular buffer storing 10 seconds of audio history for pre-roll capture.

**Usage:**
```python
ring_buffer = RingBuffer(sample_rate=16000, channels=1, duration_seconds=10.0)
ring_buffer.write(audio_chunk)  # Continuously write audio
audio_preroll = ring_buffer.read_last_n_seconds(1.0)  # Get 1s pre-roll
```

### 5. Speech Recorder (`audio/speech_recorder.py`)

Post-detection audio capture with silence-based endpointing.

**Flow:**
1. Wake word detected → `start_recording()`
2. Register as audio consumer
3. Collect audio chunks with silence detection
4. Minimum 2s, maximum 15s capture
5. Encode as WAV and POST to endpoint
6. Unregister consumer

**Endpointing:**
- Trailing silence threshold: 300ms
- Grace period: 400ms
- Failsafe timeout: 15 seconds

### 6. Settings Manager (`config/manager.py`)

Thread-safe configuration management with JSON schema validation.

**Key Features:**
- Atomic file writes (`settings.json.tmp` → `os.replace()`)
- Schema validation on load
- Auto-creation from template if missing
- Configuration change signaling

**API:**
```python
settings_manager = SettingsManager(config_path)

# Thread-safe read
with settings_manager.get_config() as config:
    threshold = config.models[0].threshold

# Update and save
settings_manager.update_model_config("hey_orac", threshold=0.6)
settings_manager.save()
```

### 7. Wake Detector (`models/wake_detector.py`)

OpenWakeWord model wrapper.

**Model Management:**
- Loads ONNX or TFLite models
- Validates file format via magic numbers
- Hot-reload with rollback on failure
- Memory management (unload disabled models)

### 8. STT Client (`transport/stt_client.py`)

HTTP client for ORAC STT service integration.

**Features:**
- WAV format conversion (16kHz, 16-bit, mono)
- Multipart/form-data POST
- Configurable timeouts and retries
- Health check endpoint support
- Per-model webhook URLs

### 9. Web Application (`web/`)

Flask-based REST API and WebSocket server.

**REST Endpoints:**
- `GET /api/models` - List all models
- `PUT /api/config/global` - Update global settings
- `PUT /api/config/models/{name}` - Update model config
- `GET /api/health` - Health check

**WebSocket Events:**
- `status_update` - RMS values, listening state, STT health (5Hz)
- `detection` - Wake word detection events
- `config_changed` - Configuration update notifications

### 10. Setup Functions (Sprint 8)

Main function orchestration through extracted setup functions:

```python
def setup_audio_input(...) -> stream
def setup_wake_word_models(...) -> tuple
def setup_heartbeat_sender(...) -> HeartbeatSender
def setup_web_server(...) -> WebSocketBroadcaster
def setup_stt_components(...) -> tuple
```

These reduce main() from 1200+ lines to ~200 lines of clean orchestration.

### 11. Detection Loop (Sprint 9)

Main wake word detection logic extracted to `run_detection_loop()`:

**Responsibilities:**
- Continuous audio processing
- OpenWakeWord inference
- Multi-trigger vs single-trigger modes
- Configuration change detection
- STT health monitoring
- Audio thread health checks
- Detection event handling

---

## Deployment & Build System

### Smart Build Detection

The `deploy_and_test.sh` script automatically chooses optimal build strategy:

```bash
# Full rebuild (~10-15 min) - When dependencies change
if [requirements.txt, pyproject.toml, or Dockerfile changed]:
    docker-compose build --no-cache

# Incremental rebuild (~2-5 min) - When source code changes
elif [src/*.py or models/* changed]:
    docker-compose build

# Cache-only (~30 sec) - When only docs change
else:
    docker-compose build
```

### Deployment Workflow

```bash
# Standard deployment
./scripts/deploy_and_test.sh "Fix wake word detection threshold"

# What it does:
# 1. Commits changes locally
# 2. Pushes to current branch
# 3. SSHs to Pi and pulls latest
# 4. Detects changes and chooses build strategy
# 5. Builds Docker container
# 6. Restarts container
# 7. Shows logs and status
```

### Docker Configuration

**Dockerfile Highlights:**
- Multi-stage build (dependencies cached separately)
- Alpine Linux base (~150MB)
- PyAudio with ALSA support
- Privileged mode for audio device access
- Non-root user for security

**Required Volume Mounts:**
```yaml
volumes:
  - ./config:/config              # Configuration persistence
  - ./models:/models              # Model storage
```

**Required Devices:**
```yaml
devices:
  - /dev/snd:/dev/snd            # ALSA audio access
```

### Performance Metrics

| Metric | Target | Actual (Pi 4) |
|--------|--------|---------------|
| Build time (full) | <15 min | ~10-12 min |
| Build time (incremental) | <5 min | ~3-4 min |
| Build time (cache-only) | <1 min | ~30 sec |
| Container size | <1 GB | ~700 MB |
| Startup time | <10 sec | ~5 sec |

### Monitoring Deployment

```bash
# Check container status
ssh pi "docker ps | grep hey-orac"

# View recent logs
ssh pi "cd ~/hey-orac && docker-compose logs --tail=50 hey-orac"

# Watch logs in real-time
ssh pi "cd ~/hey-orac && docker-compose logs -f hey-orac"

# Check resource usage
ssh pi "docker stats hey-orac"
```

---

## Configuration Management

### Configuration File

**Location:** `/config/settings.json` (inside container)

**Schema:**
```json
{
  "models": [
    {
      "name": "hey_orac",
      "enabled": true,
      "threshold": 0.5,
      "path": "/models/openwakeword/hey_orac_v0.1.tflite",
      "topic": "wake/hey_orac",
      "webhook_url": "http://orac-stt:8080/stt",
      "stt_enabled": true
    }
  ],
  "system": {
    "multi_trigger": false,
    "vad_threshold": 0.5,
    "cooldown": 2.0,
    "rms_filter": 50.0
  },
  "audio": {
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 1280
  },
  "stt": {
    "url": "http://orac-stt:8080/stt/v1/stream",
    "language": "en",
    "enabled": true,
    "preload": true,
    "timeout": 30
  }
}
```

### Configuration Parameters

**Per-Model Settings:**
- `enabled`: Activate/deactivate model
- `threshold`: Detection confidence threshold (0.0-1.0)
- `path`: Model file path (ONNX or TFLite)
- `topic`: MQTT topic for heartbeat
- `webhook_url`: Target API endpoint for audio
- `stt_enabled`: Enable STT transcription

**System Settings:**
- `multi_trigger`: Allow multiple simultaneous detections
- `vad_threshold`: OpenWakeWord VAD sensitivity (0.0-1.0)
- `cooldown`: Seconds between repeat detections
- `rms_filter`: Audio level filter threshold

**Audio Settings:**
- `sample_rate`: Fixed at 16000 Hz
- `channels`: Fixed at 1 (mono)
- `chunk_size`: Samples per buffer (1280 = 80ms @ 16kHz)

**STT Settings:**
- `url`: ORAC STT endpoint
- `language`: Transcription language code
- `enabled`: Global STT enable flag
- `preload`: Preload STT models on startup
- `timeout`: Request timeout in seconds

### Hot Configuration Reload

Configuration changes via web UI trigger reload without restart:

```python
# 1. User changes settings in web UI
# 2. PUT /api/config/global or /api/config/models/{name}
# 3. Settings manager saves to settings.json
# 4. shared_data['config_changed'] = True
# 5. Detection loop detects flag (checked every 1s)
# 6. reload_models_on_config_change() called
# 7. New OpenWakeWord Model instance created
# 8. Test prediction to validate
# 9. Atomically replace old model
# 10. Update heartbeat sender with new models
# 11. Continue detection loop with new config
```

**Rollback on Failure:**
If reload fails (invalid model, prediction test fails), old model remains active and error is logged.

---

## Testing Strategy

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src/hey_orac tests/

# Run specific test file
pytest tests/unit/test_ring_buffer.py
```

**Test Coverage:**
- Ring buffer operations
- Audio format conversion
- Endpointing logic
- Settings manager thread safety
- Model validation

### Integration Tests

```bash
# Test on actual hardware
pytest tests/integration/

# Test with mock audio
pytest tests/integration/ --mock-audio
```

**Integration Tests:**
- End-to-end wake word detection
- Audio reader thread health
- Multi-consumer distribution
- STT client integration
- Web API endpoints

### Manual Testing Checklist

After deployment:

- [ ] Container starts without errors
- [ ] Web UI accessible at port 7171
- [ ] RMS meter updates continuously
- [ ] Wake word detection triggers correctly
- [ ] Model enable/disable works via UI
- [ ] Configuration changes persist after restart
- [ ] STT integration captures and sends audio
- [ ] Multiple models work simultaneously (if enabled)
- [ ] Logs are clean without errors
- [ ] Container restarts gracefully on failure

### Golden Clip Testing

Use known-good audio clips for regression testing:

```bash
# Test with WAV file
python -m hey_orac.wake_word_detection --input-wav tests/audio/hey_orac_sample.wav

# Expected: Detection at specific timestamp with known confidence
```

---

## Performance & Monitoring

### Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Latency (detection → event) | <500ms | ~200ms |
| RMS updates | 5 Hz | 5 Hz |
| Audio jitter | <20ms | ~5ms |
| CPU usage (2 models) | <25% | ~15-20% |
| Memory usage | <250 MB | ~150-200 MB |
| Inference time per model | <10ms | ~7ms |

### Logging

**Structured Logging:**
- JSON format for machine parsing
- ISO 8601 timestamps
- Log levels: DEBUG, INFO, WARNING, ERROR
- Contextual information (RMS, chunk count, model names)

**Log Locations:**
- Container stdout: `docker logs hey-orac`
- Persistent logs: `/logs` volume (if mounted)

**Key Log Messages:**
- `🎤 Starting wake word detection loop` - Detection started
- `🎯 WAKE WORD DETECTED!` - Detection event
- `✅ Models reloaded successfully` - Config reload success
- `❌ Error processing audio data` - Processing error
- `📊 Processed N audio chunks` - Periodic status (every 100 chunks)

### Health Monitoring

**Automated Health Checks:**
- Audio thread heartbeat (every 5s)
- STT endpoint health (every 30s)
- Configuration validation (on load)
- RMS stuck detection (10 consecutive identical values)

**Metrics Endpoints:**
- `/api/health` - JSON health status
- WebSocket `status_update` - Real-time status

**Health Indicators:**
- `is_listening`: Actively processing audio
- `stt_health`: STT endpoint status (connected/partial/disconnected)
- `audio_thread_healthy`: Audio reader thread status

---

## Code Quality & Refactoring History

### Refactoring Sprints

The codebase underwent systematic cleanup documented in [CLEANUP.md](CLEANUP.md):

**Completed Sprints (14/14 = 100%):**

1. ✅ **Sprint 1**: Deleted redundant files (953 lines)
2. ✅ **Sprint 2**: Removed debug print statements
3. ✅ **Sprint 3**: Fixed resource cleanup (`__del__()` → `close()`)
4. ✅ **Sprint 4**: Fixed bare except blocks
5. ✅ **Sprint 5**: Removed unused entry points (277 lines)
6. ✅ **Sprint 6**: Extracted constants (42 constants to `constants.py`)
7. ✅ **Sprint 7**: Consolidated audio conversion (35 lines)
8. ✅ **Sprint 8**: Extracted setup functions (main: 1200→900 lines)
9. ✅ **Sprint 9**: Extracted detection loop (main: 900→200 lines)
10. ✅ **Sprint 10**: Deleted preprocessing module (173 lines)
11. ✅ **Sprint 11**: Standardized configuration (obsolete)
12. ✅ **Sprint 12**: Removed TODOs (only 2 remain, both valuable)
13. ✅ **Sprint 13**: Verified naming conventions
14. ✅ **Sprint 14**: Verified emoji logging standard

**Result:** ~1,438 lines of dead code removed, main() reduced by 83%, zero technical debt.

### Code Quality Standards

**Enforced Standards:**
- ✅ No bare `except:` blocks (use specific exceptions)
- ✅ No direct `__del__()` calls (use `close()` methods)
- ✅ No debug `print()` statements (use structured logging)
- ✅ All magic numbers in `constants.py`
- ✅ Single responsibility principle (functions <100 lines)
- ✅ Type hints on public APIs
- ✅ Docstrings on all public functions

**Emoji Logging Convention:**
- 🎤 Audio/recording operations
- 🎯 Wake word detections
- ✅ Success operations
- ❌ Errors/failures
- 📊 Statistics/metrics
- 🔄 Reload/restart operations
- 🏥 Health checks
- 📞 Network calls

---

## Contributing Guidelines

### Development Workflow

1. **Create Feature Branch:**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Make Changes:**
   - Follow code quality standards
   - Add tests for new functionality
   - Update documentation

3. **Test Locally:**
   ```bash
   pytest tests/
   ```

4. **Deploy to Pi:**
   ```bash
   ./scripts/deploy_and_test.sh "Add feature X"
   ```

5. **Verify on Pi:**
   - Check logs for errors
   - Test feature functionality
   - Verify no regressions

6. **Submit Pull Request:**
   - Clear description of changes
   - Link to related issues
   - Include test results

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] No new TODOs without issues
- [ ] Logging messages use emoji convention
- [ ] No bare except blocks
- [ ] Magic numbers moved to constants
- [ ] Functions have clear single responsibility
- [ ] Type hints on public APIs
- [ ] Deployment tested on actual Pi

### Commit Message Format

```
Component: Brief description (50 chars)

Longer explanation if needed:
- What changed
- Why it changed
- Impact on system

Related: #123
```

**Examples:**
```
Audio: Fix choppy audio with multi-consumer queues

Detection: Add multi-trigger mode support

Config: Implement hot reload for model changes
```

### Branch Strategy

- `main` - Production-ready code
- `code-cleanup` - Refactoring branch (merged)
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Critical production fixes

---

## Troubleshooting

### Common Development Issues

**Build Fails:**
```bash
# Clean Docker cache
docker system prune -a

# Force full rebuild
ssh pi "cd ~/hey-orac && docker-compose build --no-cache"
```

**Audio Device Not Found:**
```bash
# List audio devices
ssh pi "arecord -l"

# Check Docker device access
ssh pi "docker exec hey-orac arecord -l"
```

**Import Errors:**
```bash
# Reinstall in development mode
pip install -e .
```

**Tests Fail on Pi:**
```bash
# Check audio device availability
pytest tests/integration/ --mock-audio
```

### Debugging Tips

**Enable Debug Logging:**
```bash
# In docker-compose.yml:
environment:
  - LOG_LEVEL=DEBUG
```

**Check Shared State:**
```python
# In detection loop or web routes:
logger.debug(f"Shared data: {shared_data}")
```

**Monitor Audio Flow:**
```bash
# Watch RMS values
ssh pi "cd ~/hey-orac && docker-compose logs -f hey-orac | grep '📊'"
```

**Profile Performance:**
```python
import cProfile
cProfile.run('run_detection_loop(...)', 'profile.stats')
```

---

## Additional Resources

- **User Guide**: [USER_GUIDE.md](USER_GUIDE.md) - End-user documentation
- **Sprint History**: [CLEANUP.md](CLEANUP.md) - Complete refactoring history
- **Development Log**: [devlog.md](devlog.md) - Chronological development notes
- **STT API Reference**: [STT_API_REFERENCE.md](STT_API_REFERENCE.md) - ORAC STT API documentation
- **OpenWakeWord**: https://github.com/dscripka/openWakeWord
- **Flask-SocketIO**: https://flask-socketio.readthedocs.io/

---

**Last Updated**: 2025-10-16
**Maintainers**: Development Team
**Questions?** Open an issue on GitHub
