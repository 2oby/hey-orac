# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Sources

- **OpenWakeWord**: https://github.com/dscripka/openWakeWord - Official documentation for the wake word detection engine

## Important Development Constraints

- **USB Microphone Access**: Only the main program in the Docker container has exclusive access to the USB microphone. Cannot create separate test scripts that access audio hardware - must add debugging to existing code instead.
- **Settings System Issues**: Settings persistence and reload mechanism is currently faulty (see CRITICAL_PATH.txt lines 34-47). Use hardcoded values as workaround until settings system is fixed.

## Project Overview

Hey Orac is a containerized wake-word detection service for Raspberry Pi deployment. It's designed as Phase 1a of the ORAC Voice-Control Architecture - a system that monitors audio input for custom wake words and streams detected audio to remote processing endpoints.

**Current Status**: Custom wake word detection working with 86.8% confidence using OpenWakeWord engine.

## Common Commands

Read the file LOCAL_ENVIRONMENT to:
-  learn about the 'deploy_and_test' script which commits to git and then pulls from git to the Pi and tests.
- see how to use ssh pi <command> to ssh into the Rasberry Pi and run commands


### Development
```bash
# Run tests
python -m pytest tests/ -v

# Test with sample audio  
python src/main.py --test-audio tests/sample_orac.wav

# Run the main application
python src/main_new.py

# Test wake word monitor
python src/wake_word_monitor_new.py
```

### Docker Operations
```bash
# Build container
docker build -t hey-orac -f docker/Dockerfile .

# Run container with audio access
docker run --rm --privileged --device /dev/snd:/dev/snd -p 7171:7171 hey-orac

# Start with docker-compose
docker-compose up

# View logs
docker-compose logs -f hey-orac
```

### Deployment
```bash
# Deploy to Raspberry Pi
./scripts/deploy_and_test.sh

# Reset settings
./scripts/reset_settings.sh

# Test shared memory activation
./scripts/test_shared_memory_activation.sh

# Verify activation system
./scripts/verify_activation_system.sh
```

### Testing & Debugging
```bash
# Test logging functionality
python test_logging.py

# Test OpenWakeWord fixes
python test_openwakeword_fixes.py

# Debug audio loop
python debug_audio_loop.py
```

## Code Architecture

### Core Components

**New Implementation (Recommended)**:
- `src/main_new.py` - New main application entry point
- `src/wake_word_monitor_new.py` - Configuration-driven wake word detection with real model loading
- `src/audio_pipeline_new.py` - New audio processing pipeline
- `src/settings_manager.py` - Centralized settings management with file-based IPC

**Legacy Implementation**:
- `src/main.py` - Original main application
- `src/monitor_custom_model.py` - Working custom model detection
- `src/monitor_default_model.py` - Pre-trained model detection  
- `src/audio_pipeline.py` - Original audio processing

### Settings Architecture

Settings are managed through a sophisticated multi-layer system:

1. **tmpfs Storage**: Fast RAM-based settings in `/tmp/settings/config.json`
2. **Permanent Backup**: Persistent storage in `/app/src/settings_backup.json`
3. **Real-time Updates**: File watcher with 1-second polling for GUI changes
4. **Web API Integration**: REST endpoints for web interface configuration

**Key Settings Sections**:
- `wake_word.models.*` - Per-model sensitivity, threshold, API URL, active state
- `audio.*` - Sample rate (16000), chunk size (1280), channels (1), device index
- `detection.*` - Audio level thresholds and amplification
- `volume_monitoring.*` - RMS filter (50), window size, silence thresholds
- `buffer.*` - Pre/post-roll audio capture settings
- `web.*` - Web interface configuration (port 7171)

### Wake Word Detection Flow

1. **Model Discovery**: Scan `third_party/openwakeword/custom_models/` for .onnx/.tflite files
2. **Configuration Loading**: Load per-model settings (sensitivity, threshold, active state)
3. **Detector Initialization**: Create WakeWordDetector instances for active models
4. **Audio Processing**: Stream 1280-sample chunks (80ms at 16kHz) through all active detectors
5. **Detection Handling**: Check cooldown, log results, update shared memory, create detection files

### Web Interface Architecture

- **Backend**: Flask server on port 7171 (`src/web_backend.py`)
- **Frontend**: Dark neon pixel theme in `web/` directory
- **API Endpoints**: RESTful API for model configuration and real-time data
- **Real-time Updates**: Settings changes apply immediately without restart

### Shared Memory IPC

The system uses a sophisticated IPC mechanism via `src/shared_memory_ipc.py`:
- **Audio State**: Current RMS levels for volume meter
- **Activation State**: Detection status and active model information
- **Settings Sync**: Real-time settings updates between processes

## Model Management

### Custom Model Structure
```
third_party/openwakeword/custom_models/
├── Hay--compUta_v_lrg.onnx
├── Hay--compUta_v_lrg.tflite  
├── Hey_computer.onnx
├── Hey_computer.tflite
├── hey-CompUter_lrg.onnx
└── hey-CompUter_lrg.tflite
```

**Model Priority**: ONNX models are preferred over TFLite when both exist.

### Per-Model Configuration
Each model has independent settings:
- **Sensitivity**: Internal model parameter (0.0-1.0) affecting model processing
- **Threshold**: Detection confidence level (0.0-1.0) for trigger conditions  
- **API URL**: Webhook endpoint for detection notifications
- **Active State**: Boolean indicating if model should be loaded and used

## OpenWakeWord Implementation

### Audio Requirements
- **Sample Rate**: 16kHz (CRITICAL - OpenWakeWord requirement)
- **Channels**: Mono (1 channel)
- **Chunk Size**: 1280 samples (80ms at 16kHz)
- **Format**: 16-bit signed integers (np.int16)

### Detection Process
```python
# Audio capture
audio_data = mic_stream.read(1280, exception_on_overflow=False)
audio_np = np.frombuffer(audio_data, dtype=np.int16)

# Process through all active detectors
for model_name, detector in self.active_detectors.items():
    detection_result = detector.process_audio(audio_np)
    if detection_result:
        self._handle_detection(model_name, detector, audio_np)
```

### Model Loading
The system loads custom models through the WakeWordDetector interface:
```python
detector_config = {
    'wake_word': {
        'engine': 'openwakeword',
        'wakeword_models': ['/path/to/model.onnx'],
        'sensitivity': 0.8,
        'threshold': 0.00001  # CRITICAL: Very low threshold required
    }
}
detector.initialize(detector_config)
```

### Critical Threshold Configuration

**IMPORTANT DISCOVERY**: OpenWakeWord default models produce extremely low confidence scores that require ultra-low thresholds:

```python
# WRONG - will never detect anything
threshold = 0.3  # 30% - far too high

# CORRECT - based on actual testing
threshold = 0.00001  # 0.001% - works with default models
```

**Confidence Score Analysis** (from actual testing):
- Silence: `~0.000001` (baseline noise)
- Ambient audio: `0.000017` - `0.000037` (normal baseline)  
- **Highest observed**: `0.000049` (hey_mycroft with synthetic noise)
- **NOTE**: No actual wake words tested yet - these are baseline/quiet room values

**Recommended Thresholds by Model**:
- `hey_mycroft`: `0.00005` (most reliable)
- `alexa`: `0.00003`
- `hey_jarvis`: `0.00002` (less sensitive)
- Timer commands: `0.00001` (very conservative)

## Critical Implementation Notes

### Settings Persistence 
- Settings changes trigger immediate file writes to tmpfs
- Delayed backup mechanism (10-second delay) prevents race conditions
- File locking prevents corruption during concurrent access
- Automatic fallback to defaults if settings become corrupted

### Audio Pipeline Thread Safety
- Multi-threaded architecture with careful resource management
- Graceful shutdown handling for all audio resources
- Exception handling prevents pipeline crashes

### Real-time Model Switching
- GUI changes trigger immediate model reloading
- Old detectors are properly cleaned up before loading new ones
- Settings validation prevents invalid configurations

## Deployment Configuration

### Docker Requirements
```yaml
# docker-compose.yml essentials
privileged: true
devices:
  - /dev/snd:/dev/snd
ports:
  - "7171:7171"
volumes:
  - settings-data:/tmp/settings:rw  # tmpfs for fast settings access
```

### Audio Device Configuration
- USB microphone expected at default device index
- ALSA configuration via environment variables
- Container needs privileged access for audio hardware

### TMPFS Storage
Critical directories mounted as tmpfs for SD card protection:
- `/tmp/settings` - Settings storage (5MB)
- `/tmp/cache` - Temporary cache (50MB)  
- `/tmp/sessions` - Session data (20MB)
- `/tmp/uploads` - File uploads (30MB)

## Testing Strategy

### OpenWakeWord Test Implementation (main_test.py)

**IMPORTANT**: A comprehensive test implementation has been created in `src/main_test.py` that provides complete OpenWakeWord testing functionality.

#### Test Features:
- **Stage 1**: USB microphone access and audio capture verification
- **Stage 2**: OpenWakeWord initialization with default pre-trained models  
- **Stage 3**: Live wake word detection with real-time confidence logging

#### Usage:
```bash
# Deploy and run test (uses Docker container)
./scripts/deploy_and_test.sh "Test OpenWakeWord functionality"

# Monitor test results
ssh pi 'cd ~/hey-orac && docker-compose logs -f hey-orac'
```

#### Key Findings from Testing:

**OpenWakeWord Confidence Scores**: Default models produce very low confidence scores:
- Typical baseline: `0.000017` - `0.000037` (0.0017% - 0.0037%)
- These are **normal** values for OpenWakeWord default models
- Threshold must be set extremely low: `0.00001` (0.001%) for detection

**Available Wake Words** (verified working):
- `alexa` - Most responsive default model
- `hey_mycroft` - Consistently highest baseline confidence (~0.000037)
- `hey_jarvis` - Lower baseline confidence (~0.000002)
- Timer commands: `1_minute_timer`, `5_minute_timer`, etc.
- `weather`

**Audio Pipeline Verification**:
- USB microphone: SH-04 device properly detected and accessed
- Sample rate: 16kHz ✅ (OpenWakeWord requirement)
- Chunk size: 1280 samples ✅ (80ms at 16kHz)
- Audio format: int16 → float32 normalization ✅

### Model Testing
```bash
# RECOMMENDED: Use main_test.py for comprehensive testing
docker-compose up --build -d  # Runs main_test.py

# Legacy testing (if needed)
python src/monitor_custom_model.py
python src/wake_word_monitor_new.py
./scripts/verify_activation_system.sh
```

### Audio Testing
```bash
# Test audio capture
arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 test.wav

# List audio devices (should show SH-04)
arecord -l
```

### Web Interface Testing
- Navigate to `http://localhost:7171` or `http://pi-ip:7171`
- Test model activation/deactivation
- Verify settings persistence
- Check real-time volume meter

## Known Issues & Solutions

### TODO Items from CRITICAL_PATH.txt
1. **Settings Manager Polling**: Replace 1-second polling with inotify/fsevents for immediate file change detection
2. **Web GUI Random Signal**: Remove random signal generator and connect real audio pipeline data
3. **Multiple Concurrent Models**: Implement parallel processing of multiple active models

### Performance Considerations
- Current implementation loads one model at a time
- Future: Load multiple models simultaneously for better performance
- Memory limit: 200MB for efficient Pi deployment
- CPU usage target: <30% on Pi 4

## Configuration Files

### Key Configuration Locations
- `/tmp/settings/config.json` - Active settings (tmpfs)
- `/app/src/settings_backup.json` - Permanent backup
- `pyproject.toml` - Python project configuration
- `requirements.txt` - Python dependencies
- `docker-compose.yml` - Container orchestration

### Default Settings Structure
The settings manager provides comprehensive defaults including audio configuration, wake word models, detection parameters, volume monitoring, web interface settings, and logging configuration.

## Development Workflow

1. **Make changes** in development environment
2. **Test locally** with unit tests and mock audio
3. **Test specific components** using individual test scripts
4. **Deploy to Pi** using `./scripts/deploy_and_test.sh`
5. **Verify functionality** on Pi with real microphone
6. **Check web interface** for real-time monitoring

## Service Architecture

The system runs as a multi-process service:
- **Main Process**: Wake word detection and audio processing
- **Web Backend**: Flask API server for configuration and monitoring
- **Startup Script**: `src/startup.sh` manages both processes with health monitoring
- **Signal Handling**: Graceful shutdown on SIGTERM/SIGINT