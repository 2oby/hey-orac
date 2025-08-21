# Current Focus: Integration Verified - Ready for Enhancements

## ✅ Integration Status (Verified August 21, 2025)
The Hey ORAC → ORAC STT integration is **CONFIRMED WORKING** with 2+ weeks of stable operation!

## 🔍 Current Status
- ✅ Wake word detection → STT pipeline fully operational
- ✅ Audio streaming to ORAC STT successful (tested today)
- ✅ Transcriptions received and logged correctly
- ✅ Complete end-to-end integration stable for 2+ weeks
- ✅ Both containers healthy with excellent uptime

## 📝 Integration Fixes Applied
1. **Always initialize STT components** - Removed global dependency
2. **Use per-model webhook URLs** - STT triggers based on webhook_url
3. **Dynamic URL support** - STT client accepts webhook URLs
4. **Fixed JSON serialization** - Convert numpy float32 to Python types
5. **Always create speech_recorder** - Even if initial health check fails
6. **Fixed health check timing** - Moved after STT client initialization

## 🚀 Ready for Next Steps

With the core integration working, the system is ready for:
- Audio quality improvements (compression, AGC)
- Performance optimization
- Additional wake word models
- Enhanced monitoring and metrics
- Production deployment considerations

## Next Steps

### 1. **Add Audio Preprocessing** (Priority: HIGH)
Implement audio processing to reduce clipping and improve quality:
- **Dynamic Range Compression** - Reduce volume peaks that cause clipping
- **Automatic Gain Control (AGC)** - Normalize audio levels
- **Low-pass filtering** - Remove high-frequency noise
- **Peak limiting** - Prevent values exceeding valid range

### 2. **Debug ORAC STT Integration** (Priority: HIGH)
Since whisper-cli works but ORAC STT doesn't:
- Check whisper.cpp Python bindings in ORAC STT
- Add detailed logging to trace where transcription is lost
- Verify audio format compatibility
- Test with different whisper models

### 3. **Audio Quality Testing**
- Save preprocessed audio for comparison
- Test with various speaking volumes
- Verify no clipping at hardware level
- Monitor RMS levels throughout pipeline

## Technical Details

### Audio Pipeline Issues Found:
1. **Float32 normalization** - Fixed ✅
2. **Pre-roll audio mixing** - Fixed ✅
3. **Clipping from loud input** - Needs compression ⚠️
4. **ORAC STT whisper.cpp binding** - Returns empty text ❌

### Proposed Audio Processing Chain:
```
Microphone → Ring Buffer → Wake Detection
                ↓
         Speech Recording → Pre-processing → STT
                              (compression,
                               AGC, limiting)
```

---

# Previous Focus: Multi-Phase Implementation Plan for Outstanding Requirements

## 📋 Overview
This document outlines a phased approach to implement the outstanding requirements for the Hey ORAC wake-word service. Each phase builds incrementally with small, testable changes that can be committed, deployed, and verified before proceeding.

---

## Phase 1: GUI Bug Fixes & Configuration Validation (High Priority)

### 1.1 Fix Multi-Trigger Checkbox State Bug (#4)
**Files**: `src/hey_orac/web/routes.py`, `src/hey_orac/web/templates/settings.html`
- Fix checkbox binding to reflect actual `multi_trigger` state from config
- Ensure checkbox updates properly save to settings.json
- Test: Toggle checkbox and verify settings.json updates correctly
- Commit: "Fix multi-trigger checkbox state synchronization"

### 1.2 Validate Cooldown Slider Range
**Files**: `src/hey_orac/web/templates/settings.html`, `src/hey_orac/web/routes.py`
- Add HTML5 validation: min="0" max="5" step="0.1" default="2"
- Add server-side validation in routes.py
- Update tooltips/labels to show "0-5 seconds"
- Test: Try invalid values, verify defaults work
- Commit: "Add cooldown slider validation (0-5s range)"

---

## Phase 2: Audio Capture & Pre-roll Buffer

### 2.1 Implement Ring Buffer for Pre-roll
**Files**: `src/hey_orac/audio/ring_buffer.py` (enhance existing)
- Extend RingBuffer to maintain 1 second of audio history
- Add method `get_pre_roll()` to extract last 1s of audio
- Test with unit tests using synthetic audio
- Commit: "Add pre-roll capability to RingBuffer"

### 2.2 Integrate Pre-roll with Detection
**Files**: `src/hey_orac/wake_word_detection.py`
- Modify detection loop to continuously feed RingBuffer
- On wake detection, retrieve pre-roll audio
- Log pre-roll retrieval for verification
- Test: Verify 1s of audio precedes wake event
- Commit: "Integrate pre-roll buffer with wake detection"

---

## Phase 3: Endpointing Implementation

### 3.1 Create Endpointing Module
**Files**: `src/hey_orac/audio/endpointing.py` (enhance existing)
- Implement silence detection (RMS < threshold for 300ms)
- Add 400ms grace period logic
- Add 15s failsafe timeout
- Unit tests with various audio patterns
- Commit: "Implement audio endpointing logic"

### 3.2 Integrate Endpointing with Detection
**Files**: `src/hey_orac/wake_word_detection.py`
- After wake detection, start recording with endpointing
- Combine pre-roll + active speech + trailing silence
- Log endpointing decisions (silence detected, timeout, etc.)
- Test: Speak with pauses, verify correct endpoint
- Commit: "Integrate endpointing with wake detection"

---

## Phase 4: HTTP Streaming Transport

### 4.1 Create Stream Transport Module
**Files**: `src/hey_orac/transport/http_stream.py` (new)
- Implement HTTP POST to configurable endpoint_url
- Format: 16kHz, 16-bit mono WAV
- Add connection pooling with requests.Session
- Basic error handling and logging
- Unit tests with mock server
- Commit: "Add HTTP stream transport module"

### 4.2 Add Reliability Features
**Files**: `src/hey_orac/transport/http_stream.py`
- Implement exponential backoff (1s, 2s, 4s... max 30s)
- Add circuit breaker (fail after N attempts)
- Add retry queue for failed streams
- Unit tests for failure scenarios
- Commit: "Add reliability features to stream transport"

### 4.3 Integrate Streaming with Detection
**Files**: `src/hey_orac/wake_word_detection.py`, `src/hey_orac/config/manager.py`
- Add endpoint_url to model config
- On detection + endpointing, stream audio
- Make streaming async (don't block detection)
- Test: Monitor server logs, verify audio received
- Commit: "Integrate HTTP streaming with wake detection"

---

## Phase 5: Threading & Permissions Hardening

### 5.1 Finalize Manager Queue/Events
**Files**: `src/hey_orac/models/manager.py`, `src/hey_orac/wake_word_detection.py`
- Review and fix any race conditions
- Handle microphone disconnect gracefully
- Ensure config reload is thread-safe
- Test: Unplug/replug mic during operation
- Commit: "Harden threading and event handling"

### 5.2 Docker Non-Root Permissions
**Files**: `Dockerfile`, `docker-compose.yml`
- Verify appuser has correct permissions
- Ensure OpenWakeWord cache directory is writable
- Test model download as non-root user
- Add volume mount for model cache if needed
- Commit: "Fix Docker non-root user permissions"

---

## Phase 6: Observability

### 6.1 Prometheus Metrics
**Files**: `src/hey_orac/metrics/collector.py` (enhance existing)
- Add counters: wake_detections_total, stream_success_total, stream_failures_total
- Add histograms: audio_rms, inference_time_seconds
- Expose on metrics_port (8000)
- Test: curl metrics endpoint, verify data
- Commit: "Add Prometheus metrics for observability"

### 6.2 Structured Logging
**Files**: `src/hey_orac/wake_word_detection.py`, logging config
- Add JSON formatter for structured logs
- Include labels: app="hey-orac", component, level
- Configure for Loki ingestion
- Test: Verify JSON output format
- Commit: "Add structured logging for Loki"

---

## Phase 7: Testing & Automation

### 7.1 Golden WAV Test Fixtures
**Files**: `tests/fixtures/audio/` (new), `tests/unit/test_endpointing.py`
- Create test WAVs: wake words, silence, noise
- Unit tests for endpointing with fixtures
- Unit tests for streaming with fixtures
- Commit: "Add golden WAV test fixtures"

### 7.2 Stress Test Suite
**Files**: `tests/stress/test_detection_load.py` (new)
- Generate background noise corpus
- Simulate 50 wakes/hour
- Monitor memory, CPU, detection accuracy
- Commit: "Add stress test suite"

### 7.3 Template Corruption Tests
**Files**: `tests/unit/test_settings_manager.py`
- Test missing template file
- Test corrupted JSON in template
- Test invalid schema in template
- Verify fallback to defaults
- Commit: "Add settings template corruption tests"

### 7.4 Docker Deploy Helper
**Files**: `scripts/deploy_helper.sh` (new)
- Check ALSA device permissions
- Suggest udev rules if needed
- Verify audio group membership
- Test microphone access
- Commit: "Add Docker deploy helper script"

---

## Phase 8: Documentation

### 8.1 Update README
**Files**: `README.md`
- Document streaming setup (endpoint_url config)
- Document new settings keys
- Add troubleshooting section
- Commit: "Update README with streaming docs"

### 8.2 User Guide
**Files**: `docs/user_guide.md` (new)
- Complete setup walkthrough
- Configuration reference
- Troubleshooting guide
- Performance tuning tips
- Commit: "Add comprehensive user guide"

---

## Implementation Notes

1. **Each phase should be tested independently** before moving to the next
2. **Use the deploy_and_test.sh script** after each commit
3. **Monitor logs carefully** during testing for any issues
4. **Update devlog.md** after completing each phase
5. **Keep changes small and focused** - one feature per commit

## Current Status
- Ready to begin Phase 1: GUI Bug Fixes
- All phases designed to be incremental and testable
- Total estimated commits: ~25-30 small, focused changes