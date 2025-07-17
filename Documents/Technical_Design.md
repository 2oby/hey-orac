## Technical Design Specification – **Hey ORAC Wake-Word Module**

Version 1.0 – 17 July 2025

---

### 1  Purpose & Scope

This document translates the high-level requirements for **Hey ORAC** into a build-ready technical design.
It covers only the Raspberry Pi container that

* continuously monitors a microphone,
* detects one or more wake-words with **OpenWakeWord**,
* captures and ships the subsequent utterance to a configurable endpoint (Jetson Orin Nano 8 GB in v1), and
* exposes a real-time web UI and REST/WebSocket API for configuration and monitoring.

---

### 2  System Goals & Quality Attributes

| Goal              | Target                                                                                   |
| ----------------- | ---------------------------------------------------------------------------------------- |
| **Latency**       | ≤ 500 ms from wake phrase end to “wake detected” event                                   |
| **Throughput**    | ≥ 5 × RMS updates s-¹; audio capture ≤ 20 ms buffering jitter                            |
| **Footprint**     | < 25 % CPU and < 250 MB RAM on Pi 4 (2 GB) with two ONNX models                          |
| **Robustness**    | Auto-recovery on mic loss, corrupted config, or model reload failures                    |
| **Extensibility** | Plug-in model support; clean separation of UI ↔ engine ↔ transport                       |
| **Security**      | TLS option for streaming endpoint; config file chmod 600; CSRF-free token for GUI writes |

---

### 3  High-Level Architecture

```
+---------------------------------------------------------------+
|  Docker Container (Alpine + Python 3.11)                      |
|                                                               |
|  ┌───────────────┐   shared »multiprocessing.Manager« state   |
|  │  Wake Core    │◄────────────────────────────┐              |
|  │  (Thread)     │                            ┌┴────────────┐ |
|  │  • mic input  │                            │   Flask +   │ |
|  │  • ring buf   │→ Queue(AudioFrames) ──────►│ REST / WS   │ |  Static assets
|  │  • WW infer   │   Queue(RMS)               │  backend    │◄───────────────┐
|  │  • endpointing│   Queue(Events) ──────────►│ (Thread)    │ |               │
|  └───────────────┘                            └──────────────┘ |   Nginx (prod)│
|    ▲                                               ▲          |               │
|    | Docker volume: /models  /config/settings.json │          └───────────────┘
+----|-----------------------------------------------|---------------------------+
     |                                               |
     └── ▶ HTTP POST /stream  (PCM 16 k 16-bit mono) ─┘     (to Orin-Nano)
```

* **Wake Core Thread** – low-latency loop (PyAudio callback) that fills a lock-free ring buffer, computes RMS, and feeds OpenWakeWord sessions.
* **Flask/SocketIO Thread** – synchronous Flask-SocketIO server running under gevent (WSGI) that reads the shared queues every 200 ms, pushes RMS and events over a WebSocket, and exposes REST endpoints for settings CRUD and health-check.
* **Settings Manager** – singleton wrapper around `settings.json`, performing atomic writes (`tempfile + os.replace`) and publish-subscribe notifications to trigger a full model reload.

---

### 4  Component Design

#### 4.1 Audio Capture & Buffer

* **Library:** PyAudio (ALSA).
* **Format:** 16 kHz, 16-bit, mono, little-endian.
* **Ring Buffer:** `array.array('h', size = 16 kHz × 10 s)`; pointer updated in O(1).
* **Endpointing:**

  * Wake detected ⇒ copy last 1 s pre-roll.
  * Continue writing until trailing silence > 300 ms (`RMS < SIL_THRESH`) with 400 ms grace.
  * Upper limit failsafe 15 s.
* **Export:** Buffer slices → `BytesIO` → background coroutine performing streamed `multipart/form-data` POST.

#### 4.2 Wake-Word Inference

* **Runner:** One OpenWakeWord `WakeWordModel` instance per enabled model.
* **Concurrency:** Single thread, sequential inference per frame (7 ms/model @ Pi 4), but batched in groups of *N* frames to amortise PyTorch overhead.
* **Hot-Reload:** On settings change, stop loop, dispose sessions, reload models from `/models`. Fail-fast with rollback to previous set on error.

#### 4.3 Model Manager

```python
class ModelManager:
    def __init__(self, cfg_path: Path):
        self.models: dict[str, WakeWordModel] = {}
    def reload(self, cfg: Settings):
        # diff current vs new, unload, load, adjust thresholds
```

* Maintains mapping `name → model`.
* Pops `enabled=False` models to free RAM.
* Validates ONNX and TFLite files via magic numbers before load.

#### 4.4 Settings Manager

* Reads JSON schema (jsonschema validation).
* Exposes thread-safe `get()` / `update()` / `save()` functions.
* Atomic write: save to `settings.json.tmp` then `os.replace`.

#### 4.5 Web/API Layer

* **Framework:** Flask 3.0 + Flask-SocketIO.
* **Endpoints:**

  * `GET /api/v1/settings` – returns full config.
  * `PUT /api/v1/settings` – accepts full JSON; on success emits `config_changed` WS event.
  * `GET /api/v1/rms` – (legacy) JSON of latest RMS.
  * `WS /ws` – bi-directional; server pushes `rms` (5 Hz) and `wake` events.
* **Static GUI:** Parcel-built SPA (Vue 3). WebSocket reconnect loop; diff-aware form with floating “Save” CTA.

#### 4.6 Inter-Thread Communication

| Data               | Producer    | Consumer     | Mechanism                                      |
| ------------------ | ----------- | ------------ | ---------------------------------------------- |
| RMS float          | Wake thread | Flask thread | `multiprocessing.Value('f')`                   |
| Wake events (dict) | Wake thread | Flask thread | `Queue(maxsize=64)`                            |
| Config updates     | Flask       | Wake         | `multiprocessing.Event` + shared dict snapshot |

All primitives created from a single `multiprocessing.Manager()` to work even if we later move to multiprocess isolation.

#### 4.7 Audio Transport to Orin Nano

* **Protocol v1:** HTTP POST `audio/wav` (16-bit PCM, little-endian) with `X-Model-Name` header.
* **Future v2:** gRPC bidirectional stream for chunked low-latency transfer; protobuf spec TBD.
* **Retries:** Exponential back-off up to 3 ×; circuit-break if offline > 60 s (skip streaming to avoid queue build-up).
* **Security:** Optional bearer token; TLS via environment variable `STREAM_TLS=1`.

#### 4.8 Logging & Metrics

* `structlog` JSON logs (ISO timestamps).
* Prometheus exporter on `/metrics` (RMS, inference time, queue length, audio duration).
* Loki / Grafana dashboards via labels `app=hey-orac`.

---

### 5  Project Scaffolding

```
hey-orac/
├─ docker/
│   ├─ Dockerfile              # multi-stage: build → runtime
│   └─ entrypoint.sh
├─ src/
│   └─ hey_orac/
│       ├─ __init__.py
│       ├─ cli.py              # `python -m hey_orac run`
│       ├─ config/
│       │   └─ manager.py
│       ├─ audio/
│       │   ├─ microphone.py
│       │   ├─ ring_buffer.py
│       │   └─ endpointing.py
│       ├─ models/
│       │   ├─ manager.py
│       │   └─ wake_detector.py
│       ├─ transport/
│       │   └─ streamer.py
│       ├─ web/
│       │   ├─ app.py
│       │   ├─ routes.py
│       │   └─ socketio.py
│       └─ utils/
│           └─ logging.py
├─ ui/                         # built SPA artefacts (copied into image)
├─ models/
│   ├─ openwakeword/           # default shipped models
│   └─ custom/                 # user-added
├─ config/
│   └─ settings.json           # volume-mounted at runtime
├─ tests/
│   ├─ unit/
│   └─ integration/
├─ scripts/
│   ├─ build_image.sh
│   └─ run_local.sh
├─ pyproject.toml
└─ README.md
```

---

### 6  Implementation Plan & Milestones

| #       | Milestone                                                                        | Target Week | Success Criteria                                      |
| ------- | -------------------------------------------------------------------------------- | ----------- | ----------------------------------------------------- |
| **M0**  | **Project bootstrap** – repo, CI (GitHub Actions), dev-container                 | W30 ’25     | `pytest` runs; image builds                           |
| **M1**  | **Baseline wake detection** – mic input, ring buffer, built-in “hey jarvis” ONNX | W31         | ≥ 90 % recall on test clip                            |
| **M2**  | **Custom model loading** – config schema, hot-reload, metrics                    | W33         | Swap to custom TFLite model without restart           |
| **M3**  | **Audio endpointing + streamer** – silence detection, 1 s pre-roll               | W34         | Correct utterance boundaries in 20 sample files       |
| **M4**  | **Web API + minimal GUI** – show RMS bar, toggle enable/disable, save to JSON    | W35         | Changes survive restart; RMS updates 5 Hz             |
| **M5**  | **WebSocket notifications** – event push, flashing UI; deprecate RMS polling     | W36         | < 150 ms GUI flash after wake                         |
| **M6**  | **Thread comm & resilience** – Manager queues, error injection tests             | W37         | Engine survives forced config reload & mic disconnect |
| **M7**  | **Docker hardening & deploy script** – ALSA permissions, volume mounts           | W38         | `docker run` one-liner on clean Pi                    |
| **M8**  | **Integration with Orin Nano STT** – network tests, TLS option                   | W39         | End-to-end “Hey ORAC → text” demo                     |
| **M9**  | **Performance & soak testing** – 24 h run, metrics dashboards                    | W40         | < 25 % CPU, no memory leak                            |
| **M10** | **Docs & Release 1.0**                                                           | W41         | README, architecture.md, version tag                  |

---

### 7  Testing Strategy

* **Unit tests** with pytest for ring buffer, endpointing logic, config manager.
* **Golden-clip regression suite** – WAV files with and without wake-phrases; assert TP/FP/FN counts.
* **Stress test script** that plays background noise + 50 wake events/hour.
* **Contract tests** for HTTP streaming (mock server on Orin side).
* **End-to-end CI job** running headless via `alsa-dummy` and xvfb for GUI load.

---

### 8  Deployment & Operations

1. **Build**

   ```bash
   ./scripts/build_image.sh --tag ghcr.io/acme/hey-orac:1.0.0
   ```
2. **Run on Pi**

   ```bash
   docker run -d --name hey-orac \
     --device /dev/snd \
     -v $(pwd)/config:/config \
     -v $(pwd)/models:/models \
     -p 8000:8000 \
     ghcr.io/acme/hey-orac:1.0.0
   ```
3. **Update** – pull new tag, `docker compose up -d`.
4. **Health** – `/api/v1/health` returns JSON; Prometheus scrape `/metrics`.
5. **Backup** – Volume-mounted `config/` and `models/custom/` included in system backup.

---

### 9  Risks & Mitigations

| Risk                            | Impact          | Mitigation                                                                   |
| ------------------------------- | --------------- | ---------------------------------------------------------------------------- |
| High CPU under multi-model load | Latency spike   | Limit concurrent models, benchmark ONNX vs TFLite, consider quantised models |
| Docker audio access failures    | Module unusable | Provide helper script that sets `--privileged` and ALSA device mapping       |
| Corrupted config on power loss  | Start-up crash  | Atomic writes + schema validation + auto-backup `settings.json.bak`          |
| Network loss to Orin            | Audio backlog   | Streamer uses in-memory queue with TTL, drops oldest when full               |
| UI race during save/reload      | Stale settings  | Lock reload sequence, send `reload_complete` WS ACK to UI                    |

---

### 10  Open Questions for Product Owner

1. **Transport spec** – Do we need authentication beyond bearer token (e.g., mutual TLS) for the POST to Orin Nano?
2. **Audio format** – Is PCM 16 kHz 16-bit mono acceptable to the STT component, or should we send FLAC/OPUS?
3. **Web UI branding** – Are there style guides or assets (logos, colours) we must embed in the GUI?
4. **Wake-event debouncing** – Should we rate-limit consecutive detections of the *same* model within N seconds?

Please advise on the above; otherwise the design can proceed as specified.

---

*Prepared by*: **\[Your Name]** – Lead Engineer
*Reviewers*: Firmware, Audio DSP, Front-end, DevOps leads
