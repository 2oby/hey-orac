# Technical Requirements Document  
**Project Name:** Hey ORAC  
**Version:** 1.0  
**Date:** 2025-07-17  
**Owner:** [Product/Tech Toby Weston]  

---

## Overview

**Hey ORAC** is a Raspberry Pi-based, Dockerized wake-word detection system using the [OpenWakeWord](https://github.com/dscripka/openWakeWord) framework with custom model support. It provides a real-time web interface for model selection, configuration, and RMS monitoring. When a wake word is detected, the system captures audio input, then streams or sends it to a configurable API endpoint.  

This project is designed to be extensible, robust, and usable with multiple wake-word models concurrently.

---

## Functional Requirements

### 1. Wake Word Detection

- Compatible with `onnx` and `tflite` models.
- Custom models and OpenWakeWord defaults (e.g., "hey jarvis", "hey orac") supported.
- Multiple models can be loaded and active concurrently.
- Models are stored locally and loaded from a directory defined in the config.
- Each model has a unique name and individual configuration options.

> ✅ **Decision:** Follow OpenWakeWord model loading standards for compatibility.  
> 💡 **Suggestion:** Default to ONNX for future flexibility unless performance testing proves otherwise.

---

### 2. Audio Pipeline

- Continuous (streaming) microphone input via PyAudio or equivalent backend.
- After wake-word detection:
  - Minimum 2 seconds of audio is captured.
  - Additional time added until silence is detected using RMS threshold (voice endpointing).
- Captured audio is streamed to a user-defined API endpoint (URL set per model).

> 💡 **Suggestion:** Use a circular buffer for recent audio and trigger post-detection streaming with timeout + RMS-based silence detection.

---

### 3. Web Interface

- Web GUI served via Flask (same process as API).
- Allows live interaction with:
  - Model selection (multi-select)
  - Threshold and sensitivity (per-model)
  - API endpoint URL
- All parameters saved **only** after pressing a "Save" button.
- Save button appears floating (styled to match GUI) when changes are made.

#### GUI Feedback Features

- RMS volume level displayed at 5 Hz (5x/sec).
- Model activation flashes red for ~1 second when detected.

> 💡 **Suggestion:** Use JavaScript polling to fetch RMS via REST API; consider WebSockets in future for lower-latency UI updates.

---

### 4. Settings & Configuration

- Settings are stored in a persistent, readable/writable JSON file (e.g., `/config/settings.json`).
- File supports per-model settings:
  - `threshold`
  - `sensitivity`
  - `enabled` flag
  - `endpoint_url`
- All GUI changes cached until "Save" is pressed.
- On save:
  - JSON config is written.
  - Wake-word engine reloads **all models and settings** to prevent partial/inconsistent state.
- If config is missing, defaults are recreated by a `SettingsManager` component.

> ✅ **Decision:** Reload all models on save to ensure consistency.  
> 💡 **Suggestion:** Use file locking or atomic writes to prevent race conditions.

---

### 5. Inter-Process / Inter-Thread Communication

- Flask (web + API) and wake-word listener will run in separate threads or processes.
- RMS values and wake events must be communicated efficiently:
  - RMS updated at 5Hz
  - Wake-event flag used to trigger GUI flash
- Shared memory or thread-safe constructs (e.g., `multiprocessing.Value`, `Queue`) will be used.

> 💡 **Suggestion:** Use `multiprocessing.Manager` or `threading.Lock` guarded shared dicts for simplicity.

---

### 6. Wake Word Handling (Draft)

- Wake word detection triggers:
  - Audio streaming to model-specific API URL
  - Red flash of active model(s) in web GUI
- Audio stream duration is dynamic based on voice endpointing.
- Web UI notification method (polling vs WebSocket) is **undecided**.

> ❓**Open Question:** Should detection trigger a WebSocket push or is polling sufficient?

---

### 7. Deployment / Containerization

- Dockerized application designed for Raspberry Pi.
- Uses host audio (likely via PyAudio backend).
- Supports persistent config via Docker volume mount.
- Flask backend, model inference loop, and frontend UI are served from the **same container**.

> 💡 **Suggestion:** Run in privileged mode with proper `--device /dev/snd` configuration for audio.

---

## Non-Functional Requirements

- Low-latency wake word detection (< 500ms trigger delay).
- Lightweight footprint for Raspberry Pi.
- Secure config handling (readable/writable by app, not world-writable).
- Robust startup behavior (auto-create config if missing).
- Minimal blocking operations in audio loop or web server.

---

## Technology Summary

| Component         | Technology                             |
|------------------|-----------------------------------------|
| Wake Detection   | [OpenWakeWord](https://github.com/dscripka/openWakeWord) |
| Audio Input      | PyAudio (via ALSA or PulseAudio)        |
| Web Backend      | Flask                                   |
| Web Frontend     | Custom HTML/JS GUI (provided)           |
| Data Sharing     | `multiprocessing` / shared memory       |
| Config Persistence | JSON config file + Docker volume      |
| Containerization | Docker (Raspberry Pi compatible)        |

---

## Open Issues / Decisions

| Topic                        | Status     | Notes                                                  |
|-----------------------------|------------|--------------------------------------------------------|
| Web UI update method        | ❓ Open     | Polling vs WebSocket                                   |
| Audio endpoint streaming    | 🔄 Placeholder | Implementation deferred                               |
| Audio backend (final choice)| 🔍 Investigate | Confirm PyAudio + ALSA or consider alternatives       |
| RMS Sharing method          | 🔍 Investigate | Queue vs shared memory (pending benchmark)            |

---

## Appendices

### Example Config Schema (`settings.json`)

```json
{
  "models": {
    "hey_orac": {
      "enabled": true,
      "threshold": 0.5,
      "sensitivity": 0.7,
      "endpoint_url": "http://example.com/api/orac"
    },
    "hey_jarvis": {
      "enabled": false,
      "threshold": 0.4,
      "sensitivity": 0.6,
      "endpoint_url": "http://example.com/api/jarvis"
    }
  }
}
