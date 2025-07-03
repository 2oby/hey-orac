# ORAC Voice‑Control Architecture & Phase‑1 Plan

> **Goal:** Hands‑free Home‑Assistant control with < 0.2 s latency and < 1.5 GB RAM, using a Raspberry Pi for wake‑word capture and a Jetson Orin Nano for heavy inference.

---

## 1. Target Architecture (All Phases)

### 1.1 Micro‑services & Hosts

* **Edge Capture – `pi‑wakeword‑service` (Raspberry Pi)**
  • Continuous audio capture from USB mic
  • Porcupine detects the wake‑word **“ORAC”**
  • 0–2 s configurable pre‑buffer
  • On trigger → streams a WAV clip to the Jetson API

* **Speech‑to‑Text – `jetson‑stt‑service` (Jetson Orin Nano)**
  • Receives audio via HTTP POST or WebSocket
  • Runs `whisper.cpp` with Distil‑Whisper‑Tiny (4‑bit)
  • Returns a JSON transcript

* **Intent/NLP – `jetson‑llm‑service` (Jetson)**
  • Qwen3 0.6 B (gguf) under `llama.cpp`
  • Injects dynamic JSON grammar
  • Produces structured intents

* **Automation Bridge – `jetson‑ha‑bridge` (Jetson)**
  • Maps intents → Home Assistant REST API
  • Caches synonyms & mappings in YAML

* **API & Web UI – `jetson‑orac‑api` (Jetson)**
  • FastAPI docs/UI
  • Endpoints: `/speech`, `/speech_stream`, `/config`, `/status`

*(Optional later)* **Logging & Metrics – `orac‑loki‑grafana`**

### 1.2 Message Flow (happy path)

```
USB Mic → pi-wakeword-service (RPi)
             │ (wake word + pre-buffer)
             │──► HTTP POST /speech  (Jetson)
jetson-stt-service ─► jetson-llm-service ─► jetson-ha-bridge ─► Home Assistant
                      ▲                                 │
                      └──────────── jetson-orac-api ◄──┘
```

---

## 2. Locked Design Decisions for Phase 1

| Category         | Decision                                                    |
| ---------------- | ----------------------------------------------------------- |
| Wake‑word engine | Porcupine, single keyword **“ORAC”** (personal‑use licence) |
| Pre‑buffer       | 1 s ring‑buffer (0–2 s configurable)                        |
| Audio transport  | HTTP POST `/speech` with 16 kHz mono WAV (≤ 4 s)            |
| Config reload    | Bind‑mounted YAML; container restart on change              |
| Concurrency      | Single request end‑to‑end (queue later if needed)           |
| Security         | LAN‑only; no auth/SSL for Phase 1                           |
| CI test assets   | Synthetic WAV clips (< 100 kB) kept in repo                 |

---

## 3. Phase‑1 Deliverables – *`pi‑wakeword‑streamer`*

### 3.1 Repository Layout

```text
pi-wakeword-streamer/
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── src/
│   ├── main.py           # capture → Porcupine → streamer
│   ├── audio_buffer.py   # ring buffer util
│   └── config.yaml       # default settings
├── tests/
│   ├── sample_orac.wav
│   └── test_wakeword.py
└── README.md
```

### 3.2 Dockerfile Highlights (conceptual)

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y portaudio19-dev \
    && pip install pvporcupine pyaudio soundfile requests
COPY docker/entrypoint.sh /app/
ENTRYPOINT ["/app/entrypoint.sh"]
```

Run command:

```bash
docker run --device /dev/snd --network host \
  -v $(pwd)/config.yaml:/app/config.yaml pi-wakeword
```

### 3.3 Core Logic (pseudocode)

```python
load_yaml("config.yaml")
init_ring_buffer(preroll_seconds)
with PyAudio(open_device(mic_index, 16k, mono)) as stream:
    porcupine = pvporcupine.create(keyword_path, sensitivity)
    while True:
        frame = stream.read(512)          # ~32 ms
        ring_buffer.append(frame)
        if porcupine.process(frame):      # wake‑word!
            clip = ring_buffer.get_preroll() + stream.read(postroll_bytes)
            wav_bytes = pcm_to_wav(clip, 16_000)
            requests.post(JETSON_ENDPOINT, files={"audio": wav_bytes})
```

Latency on Pi: < 10 ms for capture + HTTP POST.

### 3.4 `config.yaml` (example)

```yaml
mic_index: 0
sample_rate: 16000
wake_word_path: /models/porcupine/orac.ppn
sensitivity: 0.6
preroll_seconds: 1.0
postroll_seconds: 2.0
jetson_endpoint: http://jetson-orin:8000/speech
```

### 3.5 Tests

* **Unit**  Run Porcupine on `sample_orac.wav`, assert detection offset < 0.5 s.
* **Integration**  Mock Jetson endpoint (Flask) ⇒ expect HTTP 200 when posting audio.

### 3.6 Documentation TODOs

* USB‑mic setup on RPi (`arecord -l`).
* Building & running the container.
* Generating the Porcupine keyword (Picovoice portal).

---

## 4. Road‑map at a Glance

1. **Phase 1 – Wake‑word & Streamer (this repo)**
   *Success:* WAV clip reaches Jetson in < 150 ms.
2. **Phase 2 – Jetson STT + LLM + HA bridge**
   *Success:* Spoken command toggles Home‑Assistant entity; end‑to‑end latency < 200 ms.
3. **Phase 3 – Security, "undo", advanced intents, metrics**
   *Success:* Auth token required; voice "undo" implemented; Grafana dashboard live.

---

## 5. Immediate Next Steps

1. Scaffold `pi‑wakeword‑streamer` repo and commit skeleton.
2. Generate the **ORAC** Porcupine model and store it in `/models/porcupine/`.
3. Prototype `main.py`; verify detection + pre‑buffer locally.
4. Containerise and test on the Pi with USB mic.
5. Mock Jetson endpoint; measure end‑to‑end latency & CPU usage.
6. Iterate until Phase 1 success criteria met.

---

*Document generated 2025‑07‑03*
