# Current Focus: Multi-Phase Implementation Plan for Outstanding Requirements

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