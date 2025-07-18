# Current Focus: M2 - Custom Model Loading Implementation

## Current Status Summary

### ✅ Completed Milestones

#### M0: Project Bootstrap - COMPLETED
- ✅ Created proper project structure (src/hey_orac/ module hierarchy)
- ✅ Set up modern Python packaging with pyproject.toml
- ✅ Created comprehensive .gitignore file
- ✅ Attempted GitHub Actions CI (removed due to permissions)
- ✅ Created test infrastructure with pytest
- ✅ Updated Dockerfile to multi-stage build
- ✅ Created project documentation (README.md)

#### M1: Baseline Wake Detection - COMPLETED
- ✅ Implemented AudioCapture class with USB mic detection
- ✅ Created RingBuffer class for 10s audio storage with pre-roll
- ✅ Built WakeDetector class using OpenWakeWord
- ✅ Implemented SpeechEndpointer for utterance boundaries
- ✅ Created HeyOracApplication orchestrator
- ✅ Built test script for "hey jarvis" model testing
- ✅ Proper logging infrastructure in place
- ✅ **Added recording and testing functionality:**
  - Recording mode: `-record_test` or `-rt` records 10 seconds with countdown
  - Test pipeline mode: `-test_pipeline` or `-tp` tests recorded audio through full pipeline
  - Recorded audio enters pipeline at exact same point as live microphone audio
  - RMS data logging and confidence score verification included

### 🎯 Current Focus: M2 - Custom Model Loading

#### Implementation Tasks for M2:
1. **Create SettingsManager** class (src/hey_orac/config/manager.py)
   - JSON schema validation with jsonschema
   - Atomic file writes with tempfile
   - Change notification system
   - Thread-safe get/update methods

2. **Implement ModelManager** (src/hey_orac/models/manager.py)
   - Dynamic model loading/unloading
   - Support for ONNX and TFLite formats
   - Hot-reload without restart
   - Model file validation

3. **Create Configuration Schema**
   - Define JSON schema for settings
   - Create settings.json template
   - Validate on load and update

4. **Add Metrics Collection**
   - Prometheus metrics export
   - Inference time tracking
   - Model performance metrics
   - System resource usage

5. **Integration Testing**
   - Test model swapping
   - Verify hot-reload functionality
   - Ensure no audio interruption

## Next Immediate Steps
1. Create SettingsManager class with JSON handling
2. Define configuration schema
3. Implement ModelManager with hot-reload
4. Add Prometheus metrics
5. Create integration tests
6. Deploy and test on Raspberry Pi

## Success Criteria for M2
- ✅ Can load custom TFLite/ONNX models from /models directory
- ✅ Configuration changes trigger model reload without container restart
- ✅ Settings persist across restarts
- ✅ Metrics available at /metrics endpoint
- ✅ No audio processing interruption during reload

## Technical Notes
- Current deployment shows TensorFlow dependency issues on ARM64
- May need to handle tflite-runtime installation separately
- Consider using model file checksums for change detection
- Ensure thread safety during model swapping