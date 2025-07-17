# Current Focus: Implementing Hey ORAC Technical Design - M0 to M2

## Implementation Plan Overview
Following the Technical Design specification, we need to refactor the current test implementation into a production-ready architecture that supports:
- Proper project structure with modular components
- Ring buffer for audio capture with pre-roll capability
- Config file based settings management
- Hot-reload capability for models
- Web API and monitoring capabilities

## Current State Analysis
### What We Have:
- ✅ Working wake word detection with OpenWakeWord
- ✅ Docker container setup with audio device access
- ✅ Basic detection loop processing audio
- ✅ Multiple pre-trained models loading successfully

### What's Missing (per Technical Design):
- ❌ Proper project structure (src/hey_orac/ module hierarchy)
- ❌ Ring buffer implementation for audio capture
- ❌ Settings manager with JSON config file
- ❌ Model manager with hot-reload capability
- ❌ Web API layer (Flask + SocketIO)
- ❌ Inter-thread communication with queues
- ❌ Metrics and monitoring capabilities
- ❌ CI/CD setup (GitHub Actions)
- ❌ Proper testing infrastructure

## Milestone Plan

### M0: Project Bootstrap (Week 30)
1. **Restructure project** to match Technical Design spec
2. **Create Python package** structure (src/hey_orac/)
3. **Set up pyproject.toml** for modern Python packaging
4. **Create .gitignore** file
5. **Set up GitHub Actions** CI pipeline
6. **Create dev container** configuration
7. **Add unit test infrastructure** with pytest

### M1: Baseline Wake Detection (Week 31)
1. **Implement AudioCapture class** with PyAudio
2. **Create RingBuffer class** for audio storage (10s capacity)
3. **Refactor wake detection** into WakeDetector class
4. **Test with "hey jarvis" ONNX model**
5. **Achieve ≥90% recall** on test clips
6. **Implement proper logging** with structlog

### M2: Custom Model Loading (Week 33)
1. **Create SettingsManager** class with JSON schema
2. **Implement ModelManager** with hot-reload capability
3. **Add config validation** with jsonschema
4. **Create /config/settings.json** template
5. **Test model swapping** without restart
6. **Add basic metrics** collection

## Immediate Next Steps
1. Start with M0 - Create proper project structure
2. Set up Python package with pyproject.toml
3. Create GitHub Actions workflow
4. Begin refactoring current code into modular components

## Success Criteria
- M0: pytest runs successfully, Docker image builds in CI
- M1: ≥90% recall on test audio clips with "hey jarvis"
- M2: Can swap to custom TFLite model without container restart