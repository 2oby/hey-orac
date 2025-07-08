# ORAC Voice Assistant System Design

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RASPBERRY PI 5                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       Hey_Orac Container                                │ │
│  │                                                                         │ │
│  │  USB Mic → Volume Check (RMS) → Wake Word → Stream to Orin             │ │
│  │                    (OpenWakeWord)                                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    Home Assistant Container                             │ │
│  │                                                                         │ │
│  │  REST API ← JSON Commands ← Network ← Orin                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                Network Stream
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             ORIN NANO                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       Orac STT Container                                │ │
│  │                                                                         │ │
│  │  Audio Stream → STT Processing → Text → Orac Container                 │ │
│  │              (faster-whisper)                                           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         Orac Container                                  │ │
│  │                                                                         │ │
│  │  Text → NLP/LLM → JSON Commands → Home Assistant                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Container Details

### 1. Hey_Orac Container (Raspberry Pi 5)

**Purpose:** Continuous audio monitoring, wake word detection, and audio streaming

**Key Libraries:**
- `pyaudio` - USB microphone capture
- `openwakeword` - Wake word detection
- `numpy` - Audio processing and RMS calculation
- `socket` or `websockets` - Network streaming to Orin

## OpenWakeWord Implementation Guide

### Understanding OpenWakeWord

OpenWakeWord is designed to run **multiple wake word models simultaneously**. When you initialize it, all pre-trained models (alexa, hey_mycroft, hey_jarvis, timer, weather) run in parallel on every audio chunk. Each model independently produces a confidence score (0.0 to 1.0) for whether its specific wake word was detected.

**Key Concepts:**
- **Frame-based processing**: Processes audio in 80ms chunks (1280 samples at 16kHz)
- **Parallel detection**: All loaded models analyze the same audio simultaneously
- **Confidence scoring**: Each model returns a score between 0 and 1
- **Threshold detection**: Scores above 0.5 (default) indicate wake word detection

### Complete OpenWakeWord Implementation

```python
#!/usr/bin/env python3
"""
Hey_Orac Container - OpenWakeWord Implementation
Continuous wake word detection with audio streaming to Orin
"""

import pyaudio
import numpy as np
import time
import logging
import socket
import threading
from collections import deque
from typing import Dict, List

# OpenWakeWord imports
import openwakeword
from openwakeword.model import Model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HeyOracDetector:
    def __init__(self, 
                 threshold: float = 0.5,
                 chunk_size: int = 1280,  # 80ms at 16kHz - CRITICAL for OpenWakeWord
                 custom_models: List[str] = None,
                 vad_threshold: float = 0.0,
                 enable_noise_suppression: bool = False,
                 orin_ip: str = "192.168.1.100",
                 orin_port: int = 8888):
        
        # OpenWakeWord configuration
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.vad_threshold = vad_threshold
        self.enable_noise_suppression = enable_noise_suppression
        self.custom_models = custom_models
        
        # Audio configuration - MUST match OpenWakeWord requirements
        self.FORMAT = pyaudio.paInt16  # 16-bit
        self.CHANNELS = 1              # Mono
        self.RATE = 16000             # 16kHz sample rate
        
        # Network configuration
        self.orin_ip = orin_ip
        self.orin_port = orin_port
        
        # Monitoring variables
        self.detection_history = deque(maxlen=100)
        self.audio_level_history = deque(maxlen=50)
        self.last_detection_time = 0
        self.total_frames_processed = 0
        
        # Audio stream objects
        self.audio = None
        self.mic_stream = None
        self.model = None
        
        self._setup_openwakeword()
        self._setup_audio()
    
    def _setup_openwakeword(self):
        """Initialize OpenWakeWord model with proper configuration"""
        try:
            logger.info("Downloading/loading OpenWakeWord models...")
            
            # Download models if not present (one-time operation)
            openwakeword.utils.download_models()
            
            # Configure model initialization
            model_kwargs = {
                'vad_threshold': self.vad_threshold,
                'enable_speex_noise_suppression': self.enable_noise_suppression
            }
            
            # Load specific models or all available models
            if self.custom_models:
                model_kwargs['wakeword_models'] = self.custom_models
                logger.info(f"Loading specific models: {self.custom_models}")
            else:
                logger.info("Loading all available pre-trained models")
            
            # Initialize the model
            self.model = Model(**model_kwargs)
            
            # Log loaded models for verification
            if hasattr(self.model, 'models'):
                loaded_models = list(self.model.models.keys())
                logger.info(f"Successfully loaded models: {loaded_models}")
            else:
                logger.info("Model loaded successfully (legacy format)")
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenWakeWord: {e}")
            raise
    
    def _setup_audio(self):
        """Initialize PyAudio with proper USB microphone configuration"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # Log available audio devices for debugging
            logger.info("Available audio input devices:")
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    logger.info(f"  {i}: {info['name']} (inputs: {info['maxInputChannels']})")
            
            # Open microphone stream with OpenWakeWord requirements
            self.mic_stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=None  # Use default device
            )
            
            logger.info(f"Audio stream opened: {self.RATE}Hz, {self.chunk_size} samples per chunk")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")
            if self.audio:
                self.audio.terminate()
            raise
    
    def _calculate_audio_level(self, audio_data: np.ndarray) -> float:
        """Calculate RMS audio level for volume monitoring"""
        return np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
    
    def _check_wake_word_detections(self, predictions: Dict[str, float]) -> List[str]:
        """
        Check all loaded models for wake word detections above threshold
        
        OpenWakeWord runs ALL models simultaneously! This function:
        1. Examines predictions from ALL loaded models
        2. Finds models with confidence scores above threshold
        3. Returns list of detected wake word names
        
        Args:
            predictions: Dict where keys are model names (e.g., 'alexa', 'hey_mycroft') 
                        and values are confidence scores (0.0 to 1.0)
        
        Returns:
            List of detected model names (empty if no detections)
        """
        # This list comprehension finds ALL models with scores above threshold
        # Example: if predictions = {'alexa': 0.02, 'hey_mycroft': 0.75, 'hey_jarvis': 0.15}
        # and threshold = 0.5, this returns ['hey_mycroft']
        detected_models = [name for name, score in predictions.items() if score > self.threshold]
        
        return detected_models
    
    def _log_detection_stats(self, predictions: Dict[str, float]):
        """
        Log detailed statistics about wake word detections
        
        This function monitors the system and logs when wake words are detected.
        It processes predictions from ALL loaded OpenWakeWord models simultaneously.
        """
        current_time = time.time()
        
        # Calculate average audio level for monitoring
        if len(self.audio_level_history) > 0:
            avg_audio_level = np.mean(self.audio_level_history)
        else:
            avg_audio_level = 0
        
        # *** CRITICAL SECTION: Multi-model detection check ***
        # Check ALL wake word models for detections above threshold
        detected_models = self._check_wake_word_detections(predictions)
        
        # Log successful detections
        if detected_models:
            time_since_last = current_time - self.last_detection_time
            self.last_detection_time = current_time
            
            logger.info(f"🎯 WAKE WORD DETECTED! Models: {detected_models}")
            logger.info(f"   Time since last detection: {time_since_last:.1f}s")
            logger.info(f"   All model scores: {predictions}")
            
            # Trigger audio streaming to Orin
            self._start_audio_streaming()
        
        # Periodic status update every 50 frames (~4 seconds)
        if self.total_frames_processed % 50 == 0:
            max_score = max(predictions.values()) if predictions else 0
            max_model = max(predictions.keys(), key=lambda k: predictions[k]) if predictions else "none"
            
            logger.info(f"Status: Frame {self.total_frames_processed}, "
                       f"Audio level: {avg_audio_level:.0f}, "
                       f"Best score: {max_score:.3f} ({max_model})")
    
    def detect_continuously(self):
        """
        Main wake word detection loop
        
        This is the heart of the system:
        1. Continuously reads audio from USB microphone
        2. Feeds audio to ALL OpenWakeWord models simultaneously
        3. Checks if any model detected its wake word above threshold
        4. Triggers audio streaming when wake word detected
        """
        logger.info("🎤 Starting Hey_Orac wake word detection...")
        logger.info(f"Threshold: {self.threshold}, Chunk size: {self.chunk_size}")
        logger.info("Listening for wake words: Say 'Alexa', 'Hey Mycroft', 'Hey Jarvis', etc.")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                # *** STEP 1: Audio Capture ***
                # Read exactly chunk_size samples (1280 = 80ms at 16kHz)
                try:
                    audio_data = self.mic_stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                except Exception as e:
                    logger.error(f"Error reading audio: {e}")
                    continue
                
                # *** STEP 2: Audio Level Monitoring ***
                audio_level = self._calculate_audio_level(audio_np)
                self.audio_level_history.append(audio_level)
                
                # Skip processing if audio too quiet (optimization)
                if audio_level < 100:  # Adjustable silence threshold
                    continue
                
                # *** STEP 3: OpenWakeWord Detection ***
                try:
                    # This runs ALL loaded models on the audio chunk!
                    prediction = self.model.predict(audio_np)
                    self.total_frames_processed += 1
                    
                    # *** STEP 4: Extract Predictions from ALL Models ***
                    predictions = {}
                    if hasattr(self.model, 'prediction_buffer'):
                        # Each model maintains a buffer of recent predictions
                        for model_name, scores in self.model.prediction_buffer.items():
                            if len(scores) > 0:
                                # Get the most recent score (latest prediction for this audio chunk)
                                # scores[-1] means "last item in the list" (newest prediction)
                                predictions[model_name] = scores[-1]
                    
                    # *** STEP 5: Store and Analyze Results ***
                    self.detection_history.append({
                        'timestamp': time.time(),
                        'predictions': predictions.copy(),
                        'audio_level': audio_level
                    })
                    
                    # Check for wake word detections and log statistics
                    self._log_detection_stats(predictions)
                    
                except Exception as e:
                    logger.error(f"Error during OpenWakeWord prediction: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Detection stopped by user")
        finally:
            self.cleanup()
    
    def _start_audio_streaming(self):
        """Start streaming audio to Orin Nano after wake word detection"""
        logger.info("Starting audio stream to Orin...")
        
        # Start streaming in separate thread to avoid blocking detection
        stream_thread = threading.Thread(target=self._stream_audio_to_orin)
        stream_thread.daemon = True
        stream_thread.start()
    
    def _stream_audio_to_orin(self):
        """Stream audio to Orin Nano with grace period handling"""
        try:
            # Connect to Orin
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.orin_ip, self.orin_port))
            logger.info(f"Connected to Orin at {self.orin_ip}:{self.orin_port}")
            
            grace_period_active = False
            grace_start_time = None
            grace_period_duration = 1.0  # seconds
            
            while True:
                # Read audio chunk
                audio_data = self.mic_stream.read(self.chunk_size, exception_on_overflow=False)
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                audio_level = self._calculate_audio_level(audio_np)
                
                if audio_level > 100:  # Above silence threshold
                    # Send audio chunk to Orin
                    sock.send(audio_data)
                    grace_period_active = False
                    
                else:
                    # Below threshold - start/continue grace period
                    if not grace_period_active:
                        grace_period_active = True
                        grace_start_time = time.time()
                    
                    # Continue sending during grace period
                    if (time.time() - grace_start_time) < grace_period_duration:
                        sock.send(audio_data)
                    else:
                        # Grace period expired - end streaming
                        sock.send(b'END_OF_SPEECH')
                        logger.info("Audio streaming completed")
                        break
                        
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
        finally:
            sock.close()
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
        if self.audio:
            self.audio.terminate()
        logger.info("Audio resources cleaned up")

def main():
    # Configuration
    detector = HeyOracDetector(
        threshold=0.5,                    # Wake word confidence threshold
        chunk_size=1280,                  # 80ms chunks (required for OpenWakeWord)
        custom_models=None,               # Use all available models
        vad_threshold=0.0,                # Voice activity detection (0 = disabled)
        enable_noise_suppression=False,   # Speex noise suppression (Linux only)
        orin_ip="192.168.1.100",         # Orin Nano IP address
        orin_port=8888                    # Orin listening port
    )
    
    # Start detection
    detector.detect_continuously()

if __name__ == "__main__":
    main()
```

### Docker Configuration for OpenWakeWord

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

# Install system dependencies for audio and OpenWakeWord
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    libportaudio2 \
    libsndfile1 \
    libspeexdsp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    openwakeword \
    pyaudio \
    numpy

WORKDIR /app
COPY hey_orac_detector.py .

# Create directory for OpenWakeWord models
RUN mkdir -p /app/models

ENV PYTHONUNBUFFERED=1
CMD ["python", "hey_orac_detector.py"]
```

### Common OpenWakeWord Troubleshooting

#### Issue: Always getting 0 predictions

**Debugging Steps:**

1. **Check audio format**: OpenWakeWord requires 16-bit, 16kHz, mono audio
2. **Verify chunk size**: Must be 1280 samples (80ms at 16kHz)
3. **Test audio levels**: Ensure microphone is working and audio level > 100
4. **Check model loading**: Verify `prediction_buffer` exists and contains data
5. **Docker audio permissions**: Ensure USB mic access in container

**Debug Script:**
```python
# Add this to your detection loop for debugging
if predictions:
    print(f"DEBUG - Raw predictions: {predictions}")
    print(f"DEBUG - Audio level: {audio_level}")
    print(f"DEBUG - Max score: {max(predictions.values())}")
    print(f"DEBUG - Models loaded: {list(predictions.keys())}")
else:
    print("DEBUG - No predictions received!")
```

#### Issue: Docker audio access

**Solution:**
```bash
# Run with proper audio device access
docker run -it --rm \
  --device /dev/snd \
  -v $(pwd)/models:/app/models \
  hey-orac-container
```

### 2. Home Assistant Container (Raspberry Pi 5)

**Purpose:** Smart home automation platform with REST API

**Configuration:**
- Standard Home Assistant installation
- REST API enabled
- Accessible at `http://pi_ip:8123/api/`

### 3. Orac STT Container (Orin Nano)

**Purpose:** Receive audio streams and perform speech-to-text conversion

**Key Libraries:**
- `faster-whisper` - GPU-accelerated STT
- `socket` or `websockets` - Network communication
- `numpy` - Audio processing

### 4. Orac Container (Orin Nano)

**Purpose:** Convert natural language commands to structured JSON for Home Assistant

**Key Libraries:**
- `transformers` - For NLP/LLM processing
- `requests` - HTTP client for Home Assistant API
- `json` - JSON processing

## Network Communication

**Between Containers:**
- **Pi → Orin**: TCP sockets for audio streaming
- **Orin → Pi**: HTTP requests to Home Assistant REST API
- **Internal Orin**: Shared queue or IPC between STT and Orac containers

## Docker Compose Configuration

**Pi docker-compose.yml:**
```yaml
version: '3.8'
services:
  hey_orac:
    build: ./hey_orac
    privileged: true
    devices:
      - /dev/snd:/dev/snd
    volumes:
      - ./models:/app/models
    environment:
      - ORIN_IP=192.168.1.100
      - ORIN_PORT=8888
      - PYTHONUNBUFFERED=1
  
  homeassistant:
    container_name: homeassistant
    image: homeassistant/home-assistant:stable
    volumes:
      - ./homeassistant:/config
    ports:
      - "8123:8123"
    restart: unless-stopped
```

**Orin docker-compose.yml:**
```yaml
version: '3.8'
services:
  orac_stt:
    build: ./orac_stt
    ports:
      - "8888:8888"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  orac:
    build: ./orac
    depends_on:
      - orac_stt
    environment:
      - HOME_ASSISTANT_URL=http://pi_ip:8123/api
      - HOME_ASSISTANT_TOKEN=your_token
```

## Example Command Flow

**User says:** "Hey Orac, turn on the bedroom lights"

1. **Hey_Orac Container**: OpenWakeWord detects wake word in audio stream
2. **Hey_Orac Container**: Streams "turn on the bedroom lights" to Orin via TCP
3. **Orac STT Container**: Transcribes audio to text using faster-whisper
4. **Orac Container**: Converts text to Home Assistant JSON command
5. **Orac Container**: Sends HTTP request to Home Assistant REST API
6. **Home Assistant**: Executes command and controls the bedroom lights

## Key OpenWakeWord Implementation Notes

- **Multiple Models**: OpenWakeWord runs ALL loaded models simultaneously on every audio chunk
- **Frame Processing**: Processes audio in 80ms chunks (1280 samples at 16kHz)
- **Confidence Scoring**: Each model returns scores 0.0-1.0, threshold typically 0.5
- **Buffer Management**: Uses `prediction_buffer` to store recent predictions per model
- **Docker Considerations**: Requires proper audio device mounting and permissions# Architecture Update 