# ORAC STT Service API Reference

## Overview

The ORAC Speech-to-Text (STT) Service provides high-performance audio transcription using Whisper models optimized for NVIDIA Orin Nano. The service accepts audio files and returns transcribed text with metadata.

**Base URL**: `http://orin3:7272` or `http://192.168.8.191:7272`

## Authentication

Currently, no authentication is required. Future versions will support mTLS for secure communication.

## Endpoints

### 1. Transcribe Audio

**Endpoint**: `POST /stt/v1/stream`

Transcribes audio files to text.

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Parameters:
  - `file` (required): Audio file (WAV format, 16kHz, 16-bit, mono)
  - `language` (optional): Language code (e.g., "en", "es", "fr")
  - `task` (optional): "transcribe" (default) or "translate" (to English)

**Response**: `200 OK`
```json
{
  "text": "The transcribed text from the audio",
  "confidence": 0.95,
  "language": "en",
  "duration": 3.5,
  "processing_time": 0.125
}
```

**Error Responses**:
- `400 Bad Request`: Invalid audio format or duration exceeds 15 seconds
- `500 Internal Server Error`: Transcription processing failed

**Example**:
```bash
curl -X POST http://orin3:7272/stt/v1/stream \
  -F "file=@audio.wav" \
  -F "language=en"
```

### 2. Health Check

**Endpoint**: `GET /health`

Returns service health status.

**Response**: `200 OK`
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-01-23T10:30:00Z"
}
```

### 3. STT Health Status

**Endpoint**: `GET /stt/v1/health`

Returns STT-specific health information including model status.

**Response**: `200 OK`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "whisper-tiny",
  "backend": "whisper.cpp",
  "device": "cuda"
}
```

### 4. Preload Model

**Endpoint**: `POST /stt/v1/preload`

Preloads the model for faster first transcription. Useful during service initialization.

**Response**: `200 OK`
```json
{
  "status": "success",
  "message": "Model loaded in 2.34s"
}
```

### 5. Metrics

**Endpoint**: `GET /metrics`

Returns Prometheus-compatible metrics for monitoring.

## Audio Requirements

- **Format**: WAV (PCM)
- **Sample Rate**: 16 kHz (16000 Hz)
- **Bit Depth**: 16-bit
- **Channels**: Mono (1 channel)
- **Maximum Duration**: 15 seconds

Audio files not meeting these requirements will be rejected with a `400 Bad Request` error.

## Performance Characteristics

- **Latency**: <500ms for 15s audio (with GPU)
- **Throughput**: ~50 requests/second (model-dependent)
- **GPU Memory**: 1-2GB (varies by model size)
- **Models Available**: tiny, base, small, medium

## Integration Example

```python
import requests

# Transcribe an audio file
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://orin3:7272/stt/v1/stream",
        files={"file": f},
        data={"language": "en"}
    )
    
result = response.json()
print(f"Transcription: {result['text']}")
print(f"Confidence: {result['confidence']}")
print(f"Processing time: {result['processing_time']}s")
```

## Error Handling

All errors return a JSON response with a `detail` field:

```json
{
  "detail": "Audio must be mono (1 channel), got 2 channels"
}
```

## Rate Limiting

No rate limiting is currently implemented. The service processes requests as fast as the hardware allows.

## Future Enhancements

- WebSocket endpoint for real-time streaming transcription
- Batch processing endpoint for multiple files
- Support for additional audio formats (MP3, FLAC, OGG)
- Configurable confidence thresholds
- Word-level timestamps