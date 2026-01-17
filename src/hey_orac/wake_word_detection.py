#!/usr/bin/env python3
"""
OpenWakeWord script for detecting wake words from a USB microphone on a Raspberry Pi in a Docker container.
Uses AudioManager for robust audio device handling.
"""

import sys
import os
import argparse
import time
import wave
import json
import signal
import queue
from datetime import datetime
# Use environment variable for unbuffered output instead
os.environ['PYTHONUNBUFFERED'] = '1'

import openwakeword
import logging
import numpy as np
import requests
import threading
from multiprocessing import Manager, Queue
from hey_orac.audio.utils import AudioManager  # Import the AudioManager class
from hey_orac.audio.microphone import AudioCapture  # Import AudioCapture for preprocessing
from hey_orac.audio.ring_buffer import RingBuffer  # Import RingBuffer for pre-roll
from hey_orac.audio.speech_recorder import SpeechRecorder  # Import SpeechRecorder
from hey_orac.audio.endpointing import EndpointConfig  # Import EndpointConfig
from hey_orac.audio.preprocessor import AudioPreprocessorConfig  # Import preprocessor config
from hey_orac.audio.audio_reader_thread import AudioReaderThread  # Import audio reader thread
from hey_orac.transport.stt_client import STTClient  # Import STT client
from hey_orac.config.manager import SettingsManager  # Import the SettingsManager
from hey_orac.web.app import create_app, socketio
from hey_orac.web.routes import init_routes
from hey_orac.web.broadcaster import WebSocketBroadcaster
from hey_orac.heartbeat_sender import HeartbeatSender  # Import heartbeat sender
from hey_orac import constants  # Import constants
from hey_orac.audio.conversion import convert_to_openwakeword_format  # Import audio conversion

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)

# Silence noisy Socket.IO loggers
logging.getLogger('engineio.server').setLevel(logging.WARNING)
logging.getLogger('socketio.server').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='OpenWakeWord test script')
    parser.add_argument('-record_test', '-rt', action='store_true', 
                       help='Record 10 seconds of audio for testing')
    parser.add_argument('-test_pipeline', '-tp', action='store_true',
                       help='Test pipeline with recorded audio file')
    parser.add_argument('-audio_file', default=None,
                       help='Audio file to use for testing (default: auto-generate timestamp)')
    parser.add_argument('--input-wav', '--input_wav', dest='input_wav', default=None,
                       help='Use WAV file as input instead of microphone for live detection')
    return parser.parse_args()

def generate_timestamp_filename():
    """Generate filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"/app/recordings/wake_word_test_{timestamp}.wav"

class WavFileStream:
    """A class that mimics audio stream interface but reads from WAV file."""
    def __init__(self, filename, chunk_size=constants.CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.filename = filename
        self.audio_data = None
        self.position = 0
        self.channels = 1
        self._load_wav_file()
    
    def _load_wav_file(self):
        """Load WAV file and prepare audio data."""
        logger.info(f"📂 Loading WAV file: {self.filename}")
        try:
            with wave.open(self.filename, 'rb') as wf:
                self.channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                
                logger.info(f"   Audio format: {self.channels} channels, {sample_width} bytes/sample, {framerate} Hz")
                
                if framerate != constants.SAMPLE_RATE:
                    logger.warning(f"⚠️  Sample rate is {framerate} Hz, expected {constants.SAMPLE_RATE} Hz")
                
                # Read all frames
                frames = wf.readframes(wf.getnframes())
                
                # Convert to numpy array (keep as int16 for now)
                self.audio_data = np.frombuffer(frames, dtype=np.int16)
                
                duration = len(self.audio_data) / (framerate * self.channels)
                logger.info(f"✅ Loaded {duration:.2f} seconds of audio data")
                
        except Exception as e:
            logger.error(f"❌ Error loading WAV file: {e}")
            raise
    
    def read(self, chunk_size, exception_on_overflow=False):
        """Read a chunk of audio data, mimicking stream.read() interface."""
        if self.position >= len(self.audio_data):
            # Loop back to beginning when we reach the end
            logger.info("🔄 Reached end of WAV file, looping back to beginning")
            self.position = 0
        
        # Calculate how many samples to read
        samples_to_read = chunk_size
        if self.channels == 2:
            samples_to_read = chunk_size * 2  # Need twice as many samples for stereo
        
        # Get the chunk
        end_pos = min(self.position + samples_to_read, len(self.audio_data))
        chunk = self.audio_data[self.position:end_pos]
        self.position = end_pos
        
        # If chunk is smaller than requested (at end of file), pad with zeros
        if len(chunk) < samples_to_read:
            chunk = np.pad(chunk, (0, samples_to_read - len(chunk)), 'constant')
        
        # Convert back to bytes to match stream interface
        return chunk.tobytes()
    
    def stop_stream(self):
        """Compatibility method - does nothing for WAV file."""
        pass
    
    def close(self):
        """Compatibility method - does nothing for WAV file."""
        pass

def record_test_audio(audio_manager, usb_mic, model, settings_manager, filename='test_recording.wav'):
    """Record 10 seconds of test audio with countdown and metadata generation."""
    logger.info("🎤 RECORDING MODE: Recording 10 seconds of test audio...")
    
    # Metadata collection
    metadata = {
        "recording_info": {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "duration_seconds": constants.TEST_RECORDING_DURATION_SECONDS,
            "sample_rate": constants.SAMPLE_RATE,
            "channels": constants.CHANNELS_STEREO,
            "chunk_size": constants.CHUNK_SIZE,
            "microphone_info": {
                "name": usb_mic.name,
                "index": usb_mic.index
            }
        },
        "rms_data": [],
        "detection_results": {
            "wake_words_detected": [],
            "total_detections": 0,
            "confidence_scores": []
        }
    }
    
    # Start audio stream for recording using config values
    with settings_manager.get_config() as config:
        audio_config = config.audio
    
    stream = audio_manager.start_stream(
        device_index=usb_mic.index if audio_config.device_index is None else audio_config.device_index,
        sample_rate=audio_config.sample_rate,
        channels=audio_config.channels,
        chunk_size=audio_config.chunk_size
    )
    
    if not stream:
        logger.error("Failed to start audio stream for recording")
        return False, None
    
    # Countdown
    logger.info("📡 Starting countdown...")
    for i in range(constants.TEST_RECORDING_COUNTDOWN_SECONDS, 0, -1):
        logger.info(f"   {i}...")
        time.sleep(1)
    
    logger.info("🔴 RECORDING...")
    sys.stdout.flush()
    
    # Record audio data and process in real-time
    frames = []
    chunks_per_second = constants.SAMPLE_RATE // constants.CHUNK_SIZE  # ~12.5 chunks per second
    total_chunks = chunks_per_second * constants.TEST_RECORDING_DURATION_SECONDS  # 10 seconds
    
    for i in range(total_chunks):
        try:
            data = stream.read(constants.CHUNK_SIZE, exception_on_overflow=False)
            if data:
                frames.append(data)

                # Convert to audio data for real-time processing
                audio_data = convert_to_openwakeword_format(data)
                
                # Calculate RMS
                rms = np.sqrt(np.mean(audio_data**2))
                timestamp = i / chunks_per_second
                
                # Store RMS data
                metadata["rms_data"].append({
                    "timestamp": timestamp,
                    "rms": float(rms)
                })
                
                # Process through model for real-time detection
                prediction = model.predict(audio_data)
                
                # Find highest confidence
                max_confidence = 0.0
                best_model = None
                
                for wakeword, score in prediction.items():
                    if score > max_confidence:
                        max_confidence = score
                        best_model = wakeword
                
                # Store confidence scores
                confidence_entry = {
                    "timestamp": timestamp,
                    "scores": {k: float(v) for k, v in prediction.items()},
                    "best_model": best_model,
                    "max_confidence": float(max_confidence)
                }
                metadata["detection_results"]["confidence_scores"].append(confidence_entry)
                
                # Check for wake word detection
                detection_threshold = constants.DETECTION_THRESHOLD_DEFAULT
                if max_confidence >= detection_threshold:
                    detection = {
                        "timestamp": timestamp,
                        "wake_word": best_model,
                        "confidence": float(max_confidence),
                        "all_scores": {k: float(v) for k, v in prediction.items()}
                    }
                    metadata["detection_results"]["wake_words_detected"].append(detection)
                    logger.info(f"🎯 DETECTED during recording at {timestamp:.2f}s: {best_model} = {max_confidence:.6f}")
                
                # Progress indicator
                if i % (chunks_per_second * constants.TEST_RECORDING_STATUS_INTERVAL_SECONDS) == 0:  # Every 2 seconds
                    seconds_elapsed = i // chunks_per_second
                    logger.info(f"   Recording... {seconds_elapsed}/{constants.TEST_RECORDING_DURATION_SECONDS} seconds (RMS: {rms:.4f})")
                
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            break
    
    # Stop recording
    stream.stop_stream()
    stream.close()
    
    # Update metadata with final results
    metadata["detection_results"]["total_detections"] = len(metadata["detection_results"]["wake_words_detected"])
    
    # Save to WAV file
    logger.info(f"💾 Saving recording to {filename}...")
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(constants.CHANNELS_STEREO)  # Stereo
            wf.setsampwidth(constants.SAMPLE_WIDTH_BYTES)  # 16-bit
            wf.setframerate(constants.SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
        
        logger.info(f"✅ Recording saved successfully to {filename}")
        
        # Save metadata file
        metadata_filename = filename.replace('.wav', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📊 Metadata saved to {metadata_filename}")
        
        # Log summary
        total_detections = metadata["detection_results"]["total_detections"]
        avg_rms = np.mean([entry["rms"] for entry in metadata["rms_data"]])

        logger.info(f"📈 Recording Summary:")
        logger.info(f"   Duration: {constants.TEST_RECORDING_DURATION_SECONDS} seconds")
        logger.info(f"   Average RMS: {avg_rms:.4f}")
        logger.info(f"   Wake words detected: {total_detections}")
        
        if total_detections > 0:
            logger.info(f"   Detections:")
            for detection in metadata["detection_results"]["wake_words_detected"]:
                logger.info(f"     {detection['timestamp']:.2f}s: {detection['wake_word']} (confidence: {detection['confidence']:.6f})")
        else:
            logger.info(f"   No wake words detected during recording")
        
        return True, metadata
        
    except Exception as e:
        logger.error(f"❌ Error saving recording: {e}")
        return False, None

def load_test_audio(filename):
    """Load recorded audio file and return as audio data for pipeline."""
    logger.info(f"📂 Loading test audio from {filename}...")
    
    try:
        with wave.open(filename, 'rb') as wf:
            # Verify audio format
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            
            logger.info(f"   Audio format: {channels} channels, {sample_width} bytes/sample, {framerate} Hz")

            if framerate != constants.SAMPLE_RATE:
                logger.warning(f"⚠️  Sample rate is {framerate} Hz, expected {constants.SAMPLE_RATE} Hz")
            
            # Read all frames
            frames = wf.readframes(wf.getnframes())

            # Convert to audio data using centralized conversion function
            audio_data = convert_to_openwakeword_format(frames, channels=channels)

            if channels == 2:
                logger.info(f"   Converted stereo to mono: {len(np.frombuffer(frames, dtype=np.int16))} -> {len(audio_data)} samples")
            else:
                logger.info(f"   Mono audio: {len(audio_data)} samples")

            duration = len(audio_data) / constants.SAMPLE_RATE
            logger.info(f"✅ Loaded {duration:.2f} seconds of audio data")
            
            return audio_data
            
    except Exception as e:
        logger.error(f"❌ Error loading audio file: {e}")
        return None

def test_pipeline_with_audio(model, audio_data):
    """Test the pipeline with recorded audio, processing in chunks like live audio."""
    logger.info("🧪 TESTING PIPELINE with recorded audio...")

    chunk_size = constants.CHUNK_SIZE
    total_chunks = len(audio_data) // chunk_size
    
    logger.info(f"   Processing {total_chunks} chunks of {chunk_size} samples each")
    
    detected_words = []
    
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        # Extract chunk (same size as live audio)
        chunk = audio_data[start_idx:end_idx]
        
        # Calculate RMS for this chunk
        rms = np.sqrt(np.mean(chunk**2))
        
        # Process through model (exactly like live pipeline)
        prediction = model.predict(chunk)
        
        # Find highest confidence
        max_confidence = 0.0
        best_model = None
        
        for wakeword, score in prediction.items():
            if score > max_confidence:
                max_confidence = score
                best_model = wakeword
        
        # Log RMS and confidence every 25 chunks (~2 seconds)
        if i % constants.PIPELINE_TEST_LOG_INTERVAL_CHUNKS == 0:
            timestamp = (i * chunk_size) / constants.SAMPLE_RATE
            logger.info(f"   📊 Time: {timestamp:.2f}s, RMS: {rms:.4f}, Best: {best_model} = {max_confidence:.6f}")

        # Detection logic (very low threshold for custom model testing)
        detection_threshold = constants.PIPELINE_TEST_DETECTION_THRESHOLD
        if max_confidence >= detection_threshold:
            timestamp = (i * chunk_size) / constants.SAMPLE_RATE
            logger.info(f"🎯 WAKE WORD DETECTED at {timestamp:.2f}s! Confidence: {max_confidence:.6f} - Source: {best_model}")
            detected_words.append((timestamp, best_model, max_confidence))

        # Also log moderate confidence like live pipeline
        elif max_confidence > constants.MODERATE_CONFIDENCE_THRESHOLD:
            timestamp = (i * chunk_size) / constants.SAMPLE_RATE
            logger.info(f"🔍 Moderate confidence at {timestamp:.2f}s: {best_model} = {max_confidence:.6f}")
    
    # Summary
    logger.info(f"🏁 Pipeline test completed. Detected {len(detected_words)} wake words:")
    for timestamp, word, confidence in detected_words:
        logger.info(f"   {timestamp:.2f}s: {word} (confidence: {confidence:.6f})")
    
    return detected_words

# Download pre-trained OpenWakeWord models if not already present
# This ensures models like "alexa", "hey jarvis", etc., are available
openwakeword.utils.download_models()

def check_all_stt_health(active_model_configs, stt_client, stt_config):
    """
    Check health status of all configured webhook URLs.

    Args:
        active_model_configs: Dict of active model configurations
        stt_client: STTClient instance for health checks
        stt_config: STT configuration from settings

    Returns:
        str: 'connected' if all healthy, 'partial' if some healthy, 'disconnected' if none healthy
    """
    webhook_urls = []
    for name, cfg in active_model_configs.items():
        webhook_url = get_effective_webhook_url(cfg, stt_config)
        if webhook_url:
            webhook_urls.append((name, webhook_url))

    if not webhook_urls:
        logger.debug("No webhook URLs configured")
        return 'disconnected'

    healthy_count = 0
    total_count = len(webhook_urls)

    for name, webhook_url in webhook_urls:
        try:
            if stt_client.health_check(webhook_url=webhook_url):
                healthy_count += 1
                logger.debug(f"✅ STT healthy for model '{name}' at {webhook_url}")
            else:
                logger.debug(f"❌ STT unhealthy for model '{name}' at {webhook_url}")
        except Exception as e:
            logger.debug(f"❌ STT health check failed for model '{name}': {e}")

    if healthy_count == total_count:
        return 'connected'
    elif healthy_count > 0:
        return 'partial'
    else:
        return 'disconnected'

def setup_audio_input(args, audio_config, audio_manager=None, usb_mic=None):
    """
    Initialize audio input source (WAV file or microphone).

    Args:
        args: Command line arguments (contains input_wav flag)
        audio_config: Audio configuration from SettingsManager
        audio_manager: AudioManager instance (only for microphone mode)
        usb_mic: USB microphone info (only for microphone mode)

    Returns:
        stream: Audio stream object (WavFileStream or PyAudio stream)

    Raises:
        RuntimeError: If audio initialization fails
    """
    stream = None

    if args.input_wav:
        # Use WAV file as input
        logger.info(f"🎵 Using WAV file as input: {args.input_wav}")
        if not os.path.exists(args.input_wav):
            logger.error(f"❌ WAV file not found: {args.input_wav}")
            raise RuntimeError(f"WAV file not found: {args.input_wav}")
        stream = WavFileStream(args.input_wav)
    else:
        # Start audio stream with parameters from configuration
        stream = audio_manager.start_stream(
            device_index=usb_mic.index if audio_config.device_index is None else audio_config.device_index,
            sample_rate=audio_config.sample_rate,
            channels=audio_config.channels,
            chunk_size=audio_config.chunk_size
        )
        if not stream:
            logger.error("Failed to start audio stream. Exiting.")
            raise RuntimeError("Failed to start audio stream")

    # Test audio stream
    logger.info("🧪 Testing audio stream...")
    sys.stdout.flush()
    try:
        test_data = stream.read(constants.CHUNK_SIZE, exception_on_overflow=False)
        logger.info(f"✅ Audio stream test successful, read {len(test_data)} bytes")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"❌ Audio stream test failed: {e}")
        sys.stdout.flush()
        raise

    return stream

def setup_wake_word_models(models_config, system_config, settings_manager, shared_data):
    """
    Initialize OpenWakeWord models from configuration.

    Args:
        models_config: List of model configurations from SettingsManager
        system_config: System configuration (contains vad_threshold)
        settings_manager: SettingsManager instance for updating configs
        shared_data: Shared data dict for web GUI

    Returns:
        tuple: (model, active_model_configs, model_name_mapping, enabled_models)
            - model: OpenWakeWord Model instance
            - active_model_configs: Dict mapping config names to ModelConfig objects
            - model_name_mapping: Dict mapping OpenWakeWord keys to config names
            - enabled_models: List of enabled ModelConfig objects

    Raises:
        ValueError: If no valid models found
        Exception: If model creation or testing fails
    """
    try:
        # Get enabled models from configuration
        enabled_models = [model for model in models_config if model.enabled]
        if not enabled_models:
            logger.warning("⚠️  No enabled models found in configuration")
            # Auto-enable the first model if none are enabled
            if models_config:
                first_model = models_config[0]
                logger.info(f"Auto-enabling first available model: {first_model.name}")
                settings_manager.update_model_config(first_model.name, enabled=True)
                settings_manager.save()
                # Refresh config
                models_config = settings_manager.get_models_config()
                enabled_models = [model for model in models_config if model.enabled]
            else:
                raise ValueError("No models available in configuration")

        # Load ALL enabled models
        model_paths = []
        active_model_configs = {}
        model_name_mapping = {}  # Maps OpenWakeWord prediction keys to config names

        for model_cfg in enabled_models:
            if os.path.exists(model_cfg.path):
                logger.info(f"✅ Loading model: {model_cfg.name} from {model_cfg.path}")
                model_paths.append(model_cfg.path)
                active_model_configs[model_cfg.name] = model_cfg

                # Create mapping for OpenWakeWord prediction key to config name
                base_name = os.path.basename(model_cfg.path).replace('.tflite', '').replace('.onnx', '')
                model_name_mapping[base_name] = model_cfg.name
                logger.debug(f"   Model name mapping: '{base_name}' -> '{model_cfg.name}'")
            else:
                logger.error(f"❌ Model file NOT found: {model_cfg.path}")

        if not model_paths:
            raise ValueError("No valid model files found")

        logger.info(f"Creating OpenWakeWord with {len(model_paths)} models: {list(active_model_configs.keys())}")

        model = openwakeword.Model(
            wakeword_models=model_paths,
            vad_threshold=system_config.vad_threshold,
            enable_speex_noise_suppression=False
        )

        # Update shared data with loaded models info
        shared_data['loaded_models'] = list(active_model_configs.keys())
        shared_data['models_config'] = {name: {
            'enabled': True,
            'threshold': cfg.threshold,
            'webhook_url': cfg.webhook_url
        } for name, cfg in active_model_configs.items()}

        logger.info("OpenWakeWord model initialized")

        # Check what models are actually loaded
        if hasattr(model, 'models'):
            logger.info(f"Loaded models: {list(model.models.keys()) if model.models else 'None'}")
        logger.info(f"Prediction buffer keys: {list(model.prediction_buffer.keys())}")

        # Enhanced model verification - test with dummy audio
        logger.info("🔍 Testing model with dummy audio to verify initialization...")
        test_audio = np.zeros(constants.CHUNK_SIZE, dtype=np.float32)
        try:
            test_predictions = model.predict(test_audio)
            logger.info(f"✅ Model test successful - prediction type: {type(test_predictions)}")
            logger.info(f"   Test prediction content: {test_predictions}")
            logger.info(f"   Test prediction keys: {list(test_predictions.keys())}")

            # Check prediction_buffer after first prediction
            if hasattr(model, 'prediction_buffer'):
                logger.info(f"✅ Prediction buffer populated after test prediction")
                logger.info(f"   Prediction buffer keys: {list(model.prediction_buffer.keys())}")
                for key, scores in model.prediction_buffer.items():
                    latest_score = scores[-1] if scores else 'N/A'
                    logger.info(f"     Model '{key}': {len(scores)} scores, latest: {latest_score}")
            else:
                logger.warning("⚠️ Prediction buffer not available after test prediction")

        except Exception as e:
            logger.error(f"❌ Error testing model after creation: {e}")
            raise

        # Force log flush
        sys.stdout.flush()

        return model, active_model_configs, model_name_mapping, enabled_models

    except Exception as e:
        print(f"ERROR: Model creation failed: {e}", flush=True)
        raise

def get_effective_webhook_url(model_config, stt_config):
    """
    Construct the effective webhook URL for a model using the configuration hierarchy.

    Args:
        model_config: ModelConfig instance
        stt_config: STT configuration from settings

    Returns:
        str: Constructed webhook URL or None if no URL configured
    """
    # Use model's explicit webhook_url if set (legacy support)
    if model_config.webhook_url:
        return model_config.webhook_url

    # Check if we have per-model base URL or use global default
    base_url = model_config.base_url if model_config.base_url else stt_config.default_base_url

    # Get stream path: model override → global config → fallback
    stream_path = model_config.stream_path if model_config.stream_path else (stt_config.stream_path or "/stt/v1/stream")

    # Construct full webhook URL
    if base_url:
        webhook_url = f"{base_url}{stream_path}"
        logger.debug(f"Constructed webhook URL: {webhook_url} (base_url={base_url}, stream_path={stream_path})")
        return webhook_url

    return None

def setup_heartbeat_sender(enabled_models, stt_config):
    """
    Initialize and start heartbeat sender for ORAC STT integration.

    Args:
        enabled_models: List of enabled ModelConfig objects
        stt_config: STT configuration from settings

    Returns:
        HeartbeatSender: Started heartbeat sender instance
    """
    # Initialize heartbeat sender for ORAC STT integration
    logger.info("💓 Initializing heartbeat sender for ORAC STT...")

    # Get heartbeat path from config (fallback to default)
    heartbeat_path = stt_config.heartbeat_path or "/stt/v1/heartbeat"

    # Initialize with default base URL and heartbeat path
    heartbeat_sender = HeartbeatSender(
        orac_stt_url=stt_config.default_base_url,
        heartbeat_path=heartbeat_path
    )

    # Register all enabled models with the heartbeat sender
    for model_cfg in enabled_models:
        # Construct webhook URL for this model
        webhook_url = get_effective_webhook_url(model_cfg, stt_config)

        # Use the actual topic from the model configuration
        heartbeat_sender.register_model(
            name=model_cfg.name,
            topic=model_cfg.topic,  # Use the configured topic (e.g., "general", "Alexa__5")
            wake_word=model_cfg.name,  # Use the full model name as wake word
            enabled=model_cfg.enabled,
            webhook_url=webhook_url  # Pass constructed webhook URL
        )
        logger.info(f"   Registered model '{model_cfg.name}' with topic '{model_cfg.topic}' (webhook: {webhook_url})")

    # Start heartbeat sender
    heartbeat_sender.start()
    logger.info("✅ Heartbeat sender started")

    return heartbeat_sender

def setup_web_server(settings_manager, shared_data, event_queue):
    """
    Initialize and start web server with WebSocket broadcaster.

    Args:
        settings_manager: SettingsManager instance for routes
        shared_data: Shared data dict for web GUI
        event_queue: Queue for detection events

    Returns:
        WebSocketBroadcaster: Started broadcaster instance
    """
    # Initialize web server in a separate thread
    logger.info("🌐 Starting web server on port 7171...")

    # Create Flask app FIRST
    app = create_app()

    # Then initialize routes with shared resources
    init_routes(settings_manager, shared_data, event_queue)

    # Create and start WebSocket broadcaster
    broadcaster = WebSocketBroadcaster(socketio, shared_data, event_queue)
    broadcaster.start()

    # Start Flask-SocketIO server in a separate thread
    web_thread = threading.Thread(
        target=socketio.run,
        args=(app,),
        kwargs={'host': '0.0.0.0', 'port': 7171, 'debug': False, 'allow_unsafe_werkzeug': True}
    )
    web_thread.daemon = True
    web_thread.start()

    logger.info("✅ Web server started successfully")

    # Update shared data
    shared_data['status'] = 'Connected'
    shared_data['is_active'] = True
    shared_data['status_changed'] = True

    return broadcaster

def setup_stt_components(stt_config, audio_config, active_model_configs, shared_data):
    """
    Initialize STT components (ring buffer, client, speech recorder).

    Args:
        stt_config: STT configuration from SettingsManager
        audio_config: Audio configuration from SettingsManager
        active_model_configs: Dict of active model configurations
        shared_data: Shared data dict for web GUI

    Returns:
        tuple: (ring_buffer, speech_recorder, stt_client, stt_health_status)
            - ring_buffer: RingBuffer instance for pre-roll audio
            - speech_recorder: SpeechRecorder instance
            - stt_client: STTClient instance
            - stt_health_status: Initial health status string
    """
    # Always initialize STT components
    logger.info("🎙️ Initializing STT components...")
    logger.debug(f"STT Configuration: base_url={stt_config.base_url}, timeout={stt_config.timeout}s")
    logger.debug(f"STT Endpoint settings: pre_roll={stt_config.pre_roll_duration}s, silence_threshold={stt_config.silence_threshold}, silence_duration={stt_config.silence_duration}s")

    # Initialize ring buffer for pre-roll audio
    ring_buffer = RingBuffer(
        capacity_seconds=constants.RING_BUFFER_SECONDS,  # Keep 10 seconds of audio history
        sample_rate=audio_config.sample_rate
    )
    logger.debug(f"RingBuffer initialized with capacity={constants.RING_BUFFER_SECONDS}s, sample_rate={audio_config.sample_rate}Hz")

    # Initialize STT client (HTTP mode)
    stt_client = STTClient(
        base_url=stt_config.base_url,
        timeout=stt_config.timeout
    )

    # Initialize WebSocket streaming client if enabled
    streaming_client = None
    use_streaming = False
    fallback_to_http = True

    if stt_config.streaming and stt_config.streaming.enabled:
        from .transport.stt_client import WebSocketSTTClient
        streaming_client = WebSocketSTTClient(
            base_url=stt_config.default_base_url,
            ws_path=stt_config.streaming.ws_path,
            timeout=stt_config.timeout
        )
        use_streaming = True
        fallback_to_http = stt_config.streaming.fallback_to_http
        logger.info(f"🌊 WebSocket streaming enabled: {stt_config.default_base_url}{stt_config.streaming.ws_path}")
    else:
        logger.info("📦 Using HTTP bulk mode for STT")

    # Check STT service health
    logger.debug(f"Checking STT service health at {stt_config.base_url}/stt/v1/health")
    if stt_client.health_check():
        logger.info("✅ STT service is healthy")
        logger.debug("STT health check passed, service is ready for transcription")
    else:
        logger.warning("⚠️ STT service is not healthy at startup, but will proceed with per-model webhook URLs")

    # Always initialize speech recorder (for per-model webhook URLs)
    endpoint_config = EndpointConfig(
        silence_threshold=stt_config.silence_threshold,
        silence_duration=stt_config.silence_duration,
        grace_period=stt_config.grace_period,
        max_duration=stt_config.max_recording_duration,
        pre_roll=stt_config.pre_roll_duration
    )

    speech_recorder = SpeechRecorder(
        ring_buffer=ring_buffer,
        stt_client=stt_client,
        endpoint_config=endpoint_config,
        streaming_client=streaming_client,
        use_streaming=use_streaming,
        fallback_to_http=fallback_to_http
    )

    logger.info("✅ STT components initialized successfully")

    # Perform health checks for models with webhook URLs
    logger.info("🏥 Performing per-model STT health checks...")
    for name, cfg in active_model_configs.items():
        webhook_url = get_effective_webhook_url(cfg, stt_config)
        if webhook_url:
            logger.debug(f"Checking STT health for model '{name}' at {webhook_url}")
            if stt_client.health_check(webhook_url=webhook_url):
                logger.info(f"✅ STT healthy for model '{name}'")
            else:
                logger.warning(f"⚠️ STT unhealthy for model '{name}' at {webhook_url}")

    # Update initial STT health status
    stt_health_status = check_all_stt_health(active_model_configs, stt_client, stt_config)
    shared_data['stt_health'] = stt_health_status
    shared_data['status_changed'] = True
    logger.info(f"🏥 Initial STT health status: {stt_health_status}")

    return ring_buffer, speech_recorder, stt_client, stt_health_status

def reload_models_on_config_change(settings_manager, heartbeat_sender, stt_client, shared_data):
    """
    Reload OpenWakeWord models when configuration changes.

    This function is called when the web GUI signals a configuration change.
    It creates a new Model instance with the updated configuration and replaces
    the old model. This allows dynamic model loading without restarting the app.

    Args:
        settings_manager: SettingsManager instance to get updated config
        heartbeat_sender: HeartbeatSender to update with new models
        stt_client: STTClient for health checks
        shared_data: Shared data dict to update with new model info

    Returns:
        tuple: (success, model, active_model_configs, model_name_mapping, enabled_models)
            - success: bool - True if reload succeeded
            - model: OpenWakeWord Model instance (new if success, None if failed)
            - active_model_configs: Dict mapping config names to ModelConfig objects
            - model_name_mapping: Dict mapping OpenWakeWord keys to config names
            - enabled_models: List of enabled ModelConfig objects
    """
    logger.info("🔄 Reloading models due to configuration change...")

    try:
        # Get current enabled models from configuration
        with settings_manager.get_config() as current_config:
            models_config = current_config.models
            system_config = current_config.system
            stt_config = current_config.stt

        enabled_models = [model for model in models_config if model.enabled]
        if not enabled_models:
            logger.warning("⚠️  No enabled models found after config change")
            return False, None, {}, {}, []

        # Build new model paths and configs
        new_model_paths = []
        new_active_configs = {}
        new_name_mapping = {}

        for model_cfg in enabled_models:
            if os.path.exists(model_cfg.path):
                logger.info(f"✅ Loading model: {model_cfg.name} from {model_cfg.path}")
                new_model_paths.append(model_cfg.path)
                new_active_configs[model_cfg.name] = model_cfg

                # Create mapping for OpenWakeWord prediction key to config name
                base_name = os.path.basename(model_cfg.path).replace('.tflite', '').replace('.onnx', '')
                new_name_mapping[base_name] = model_cfg.name
                logger.debug(f"   Model name mapping: '{base_name}' -> '{model_cfg.name}'")
            else:
                logger.error(f"❌ Model file NOT found: {model_cfg.path}")

        if not new_model_paths:
            logger.error("No valid model files found after config change")
            return False, None, {}, {}, []

        # Create new model instance
        logger.info(f"Creating new OpenWakeWord instance with {len(new_model_paths)} models: {list(new_active_configs.keys())}")
        new_model = openwakeword.Model(
            wakeword_models=new_model_paths,
            vad_threshold=system_config.vad_threshold,
            enable_speex_noise_suppression=False
        )

        # Test the new model
        test_audio = np.zeros(constants.CHUNK_SIZE, dtype=np.float32)
        test_predictions = new_model.predict(test_audio)
        logger.info(f"✅ New model test successful - predictions: {list(test_predictions.keys())}")

        # Update shared data
        shared_data['loaded_models'] = list(new_active_configs.keys())
        shared_data['models_config'] = {name: {
            'enabled': True,
            'threshold': cfg.threshold,
            'webhook_url': get_effective_webhook_url(cfg, stt_config)
        } for name, cfg in new_active_configs.items()}

        # Update heartbeat sender with new models
        # Clear existing models and re-register
        heartbeat_sender._models.clear()
        for model_cfg in enabled_models:
            # Construct webhook URL for this model
            webhook_url = get_effective_webhook_url(model_cfg, stt_config)

            heartbeat_sender.register_model(
                name=model_cfg.name,
                topic=model_cfg.topic,
                wake_word=model_cfg.name,
                enabled=model_cfg.enabled,
                webhook_url=webhook_url
            )
        logger.info(f"✅ Updated heartbeat sender with {len(enabled_models)} models")

        # Perform health checks for reloaded models with webhook URLs
        if stt_client:
            logger.info("🏥 Performing per-model STT health checks for reloaded models...")
            for name, cfg in new_active_configs.items():
                webhook_url = get_effective_webhook_url(cfg, stt_config)
                if webhook_url:
                    logger.debug(f"Checking STT health for model '{name}' at {webhook_url}")
                    if stt_client.health_check(webhook_url=webhook_url):
                        logger.info(f"✅ STT healthy for model '{name}'")
                    else:
                        logger.warning(f"⚠️ STT unhealthy for model '{name}' at {webhook_url}")

        # Update STT health status after reload
        stt_health_status = check_all_stt_health(new_active_configs, stt_client, stt_config)
        shared_data['stt_health'] = stt_health_status
        shared_data['status_changed'] = True
        logger.info(f"🏥 STT health after reload: {stt_health_status}")

        logger.info("✅ Models reloaded successfully")
        return True, new_model, new_active_configs, new_name_mapping, enabled_models

    except Exception as e:
        logger.error(f"❌ Error reloading models: {e}")
        return False, None, {}, {}, []

def run_detection_loop(
    model,
    active_model_configs,
    model_name_mapping,
    audio_reader,
    main_consumer_queue,
    settings_manager,
    shared_data,
    ring_buffer,
    speech_recorder,
    stt_client,
    heartbeat_sender,
    reload_models_func,
    event_queue
):
    """
    Main wake word detection loop.

    This is the core detection loop that:
    1. Reads audio from the audio reader thread
    2. Converts audio to appropriate format
    3. Feeds audio through OpenWakeWord model
    4. Detects wake words above threshold
    5. Triggers webhooks and STT recording
    6. Monitors system health (config changes, STT health, thread health)

    The loop runs until interrupted or an error occurs that requires restart.

    Args:
        model: OpenWakeWord Model instance
        active_model_configs: Dict of active model configurations
        model_name_mapping: Dict mapping OpenWakeWord keys to config names
        audio_reader: AudioReaderThread instance providing audio data
        main_consumer_queue: Queue to read audio chunks from
        settings_manager: SettingsManager for config checks
        shared_data: Shared data dict for web GUI updates
        ring_buffer: RingBuffer for STT pre-roll audio
        speech_recorder: SpeechRecorder for STT integration
        stt_client: STTClient for health checks
        heartbeat_sender: HeartbeatSender for model activation tracking
        reload_models_func: Function to call when config changes detected
        event_queue: Queue for detection events to web GUI

    Returns:
        int: Exit code (0 for normal, 1 for error requiring restart)
    """
    # Continuously listen to the audio stream and detect wake words
    logger.info("🎤 Starting wake word detection loop with multi-consumer audio distribution...")
    sys.stdout.flush()
    chunk_count = 0
    last_config_check = time.time()
    last_health_check = time.time()
    last_thread_check = time.time()
    CONFIG_CHECK_INTERVAL = constants.CONFIG_CHECK_INTERVAL_SECONDS
    HEALTH_CHECK_INTERVAL = constants.HEALTH_CHECK_INTERVAL_SECONDS
    THREAD_CHECK_INTERVAL = constants.THREAD_CHECK_INTERVAL_SECONDS

    # Variables for detecting stuck RMS
    last_rms = None
    stuck_rms_count = 0
    max_stuck_count = constants.MAX_STUCK_RMS_COUNT

    while True:
        try:
            # Check for configuration changes
            current_time = time.time()
            if current_time - last_config_check >= CONFIG_CHECK_INTERVAL:
                if shared_data.get('config_changed', False):
                    logger.info("📢 Configuration change detected")
                    if reload_models_func():
                        shared_data['config_changed'] = False
                        logger.info("✅ Configuration change applied")
                    else:
                        logger.error("❌ Failed to apply configuration change")
                last_config_check = current_time

            # Check STT health periodically
            if current_time - last_health_check >= HEALTH_CHECK_INTERVAL:
                if stt_client:
                    # Get current stt_config for health checks
                    with settings_manager.get_config() as config:
                        stt_config = config.stt
                    stt_health_status = check_all_stt_health(active_model_configs, stt_client, stt_config)
                    if shared_data.get('stt_health') != stt_health_status:
                        shared_data['stt_health'] = stt_health_status
                        shared_data['status_changed'] = True
                        logger.info(f"🏥 STT health status changed to: {stt_health_status}")
                    else:
                        logger.debug(f"🏥 Periodic STT health check: {stt_health_status}")
                last_health_check = current_time

            # Check audio thread health periodically
            if current_time - last_thread_check >= THREAD_CHECK_INTERVAL:
                if not audio_reader.is_healthy():
                    logger.warning("⚠️ Audio thread unhealthy, attempting restart...")
                    stats = audio_reader.get_stats()
                    logger.info(f"Thread stats before restart: {stats}")

                    if audio_reader.restart():
                        logger.info("✅ Audio thread restarted successfully")
                    else:
                        logger.error("❌ Failed to restart audio thread - exiting")
                        return 1  # Exit code for restart
                last_thread_check = current_time

            # Get audio data from consumer queue with timeout
            try:
                data = main_consumer_queue.get(timeout=constants.QUEUE_TIMEOUT_SECONDS)
            except queue.Empty:
                data = None

            if data is None:
                logger.warning("No audio data received from queue (timeout)")
                # Check if thread is still alive
                if not audio_reader.is_healthy():
                    logger.error("Audio thread appears dead, attempting restart...")
                    if not audio_reader.restart():
                        logger.error("Failed to restart audio thread - exiting")
                        return 1  # Exit code for restart
                continue

            if len(data) == 0:
                logger.warning("Empty audio data from queue")
                continue

            # Convert bytes to audio data using centralized conversion function
            audio_data = convert_to_openwakeword_format(data)

            # Calculate RMS for web GUI display
            rms = np.sqrt(np.mean(audio_data**2))
            shared_data['rms'] = float(rms)

            # Check for stuck RMS values (indicates frozen audio thread)
            if last_rms is not None and abs(rms - last_rms) < constants.STUCK_RMS_THRESHOLD:
                stuck_rms_count += 1
                if stuck_rms_count >= max_stuck_count:
                    logger.error(f"RMS stuck at {rms} for {stuck_rms_count} iterations - audio thread frozen")
                    logger.error("Forcing exit to trigger container restart")
                    return 1  # Exit code for restart
            else:
                stuck_rms_count = 0

            last_rms = rms

            # Update listening state
            shared_data['is_listening'] = True

            # Feed audio to ring buffer if STT is enabled
            if ring_buffer is not None:
                # Convert to int16 for ring buffer storage
                audio_int16 = audio_data.astype(np.int16)
                ring_buffer.write(audio_int16)

            # Log every 100 chunks to show we're processing audio
            chunk_count += 1
            if chunk_count % constants.AUDIO_LOG_INTERVAL_CHUNKS == 0:
                audio_volume = np.abs(audio_data).mean()
                logger.info(f"📊 Processed {chunk_count} audio chunks")
                logger.info(f"   Audio data shape: {audio_data.shape}, volume: {audio_volume:.4f}, RMS: {rms:.4f}")
                logger.info(f"   Raw data size: {len(data)} bytes, samples: {len(audio_data)}")
                if len(audio_data) > constants.CHUNK_SIZE:
                    logger.info(f"   ✅ Stereo→Mono conversion active")

            # Pass the audio data to the model for wake word prediction
            prediction = model.predict(audio_data)

            # Log ALL confidence scores after each processed chunk
            if chunk_count % constants.AUDIO_LOG_INTERVAL_CHUNKS == 0:
                all_scores = {word: f"{score:.6f}" for word, score in prediction.items()}
                logger.debug(f"🎯 All confidence scores: {all_scores}")
                # Also log what models are in active_model_configs for debugging
                if chunk_count == 100:
                    logger.debug(f"Active model configs: {list(active_model_configs.keys())}")
                    logger.debug(f"Model paths: {[cfg.path for cfg in active_model_configs.values()]}")
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            continue

        # Get system config for multi-trigger setting and STT config
        with settings_manager.get_config() as config:
            multi_trigger_enabled = config.system.multi_trigger
            stt_config = config.stt

        if multi_trigger_enabled:
            # MULTI-TRIGGER MODE: Check each model independently
            triggered_models = []

            for wakeword, score in prediction.items():
                # Map OpenWakeWord model name to our config name
                config_name = None
                if wakeword in model_name_mapping:
                    config_name = model_name_mapping[wakeword]
                elif wakeword in active_model_configs:
                    config_name = wakeword

                if config_name and config_name in active_model_configs:
                    model_config = active_model_configs[config_name]
                    detection_threshold = model_config.threshold

                    if score >= detection_threshold:
                        triggered_models.append({
                            'wakeword': wakeword,
                            'config_name': config_name,
                            'confidence': score,
                            'threshold': detection_threshold,
                            'model_config': model_config
                        })

            # Process each triggered model
            for trigger_info in triggered_models:
                logger.info(f"🎯 WAKE WORD DETECTED (MULTI-TRIGGER)! Confidence: {trigger_info['confidence']:.6f} (threshold: {trigger_info['threshold']:.6f}) - Source: {trigger_info['wakeword']}")
                logger.info(f"   All model scores: {[f'{k}: {v:.6f}' for k, v in prediction.items()]}")

                # Create detection event for each triggered model
                detection_event = {
                    'type': 'detection',
                    'model': trigger_info['config_name'],
                    'confidence': float(trigger_info['confidence']),
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'threshold': trigger_info['threshold'],
                    'multi_trigger': True
                }

                # Update shared data with latest detection
                shared_data['last_detection'] = detection_event

                # Add to event queue
                try:
                    event_queue.put_nowait(detection_event)
                except queue.Full:
                    logger.debug("Event queue full, skipping detection event")
                except Exception as e:
                    logger.debug(f"Failed to add detection event to queue: {e}")

                # Record activation in heartbeat sender
                heartbeat_sender.record_activation(trigger_info['config_name'])

                # Construct effective webhook URL
                webhook_url = get_effective_webhook_url(trigger_info['model_config'], stt_config)

                # Call webhook if configured
                if webhook_url:
                    try:
                        # Prepare webhook payload
                        webhook_data = {
                            "wake_word": trigger_info['wakeword'],
                            "confidence": trigger_info['confidence'],
                            "threshold": trigger_info['threshold'],
                            "timestamp": time.time(),
                            "model_name": trigger_info['config_name'],
                            "all_scores": prediction,
                            "multi_trigger": True
                        }

                        # Make webhook call
                        logger.info(f"📞 Calling webhook (multi-trigger): {webhook_url}")
                        response = requests.post(
                            webhook_url,
                            json=webhook_data,
                            timeout=constants.WEBHOOK_TIMEOUT_SECONDS
                        )

                        if response.status_code == 200:
                            logger.info(f"✅ Webhook call successful (multi-trigger)")
                        else:
                            logger.warning(f"⚠️ Webhook returned status code: {response.status_code}")

                    except requests.exceptions.Timeout:
                        logger.error("❌ Webhook call timed out (multi-trigger)")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"❌ Webhook call failed (multi-trigger): {e}")
                    except Exception as e:
                        logger.error(f"❌ Unexpected error during webhook call (multi-trigger): {e}")

                # Trigger STT recording if enabled
                if (speech_recorder is not None and
                    trigger_info['model_config'].stt_enabled and
                    not speech_recorder.is_busy()):
                    # Get STT language from config
                    with settings_manager.get_config() as config:
                        stt_language = config.stt.language

                    # Start recording in background thread
                    speech_recorder.start_recording(
                        audio_stream=audio_reader,
                        wake_word=trigger_info['config_name'],
                        confidence=trigger_info['confidence'],
                        language=stt_language,
                        webhook_url=webhook_url,
                        topic=trigger_info['model_config'].topic
                    )

        else:
            # SINGLE-TRIGGER MODE: Original "winner takes all" behavior
            max_confidence = 0.0
            best_model = None

            # Find the highest confidence score
            for wakeword, score in prediction.items():
                if score > max_confidence:
                    max_confidence = score
                    best_model = wakeword

            # Get threshold from the active model configuration
            config_name = None

            # First check the model name mapping
            if best_model in model_name_mapping:
                config_name = model_name_mapping[best_model]
            # Then try direct match
            elif best_model in active_model_configs:
                config_name = best_model
            else:
                # Log unmapped model for debugging
                if best_model is not None:
                    logger.warning(f"Could not map prediction key '{best_model}' to config name")
                    logger.debug(f"Available mappings: {model_name_mapping}")
                    logger.debug(f"Active configs: {list(active_model_configs.keys())}")

            if config_name and config_name in active_model_configs:
                model_config = active_model_configs[config_name]
                detection_threshold = model_config.threshold
            else:
                # Fallback threshold if model not found in config
                detection_threshold = constants.DETECTION_THRESHOLD_DEFAULT
                if best_model is not None:
                    logger.warning(f"Model '{best_model}' not found in active configs, using default threshold")

            if max_confidence >= detection_threshold:
                logger.info(f"🎯 WAKE WORD DETECTED! Confidence: {max_confidence:.6f} (threshold: {detection_threshold:.6f}) - Source: {best_model}")
                logger.info(f"   All model scores: {[f'{k}: {v:.6f}' for k, v in prediction.items()]}")
                logger.debug(f"Detection details: model={best_model}, config_name={config_name}, stt_enabled={active_model_configs[config_name].stt_enabled if config_name and config_name in active_model_configs else False}")

                # Add detection event to queue for web GUI
                detection_event = {
                    'type': 'detection',
                    'model': config_name or best_model,
                    'confidence': float(max_confidence),
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'threshold': detection_threshold,
                    'multi_trigger': False
                }

                # Update shared data
                shared_data['last_detection'] = detection_event

                # Add to event queue
                try:
                    event_queue.put_nowait(detection_event)
                except queue.Full:
                    logger.debug("Event queue full, skipping detection event")
                except Exception as e:
                    logger.debug(f"Failed to add detection event to queue: {e}")

                # Record activation in heartbeat sender
                heartbeat_sender.record_activation(config_name or best_model)

                # Construct effective webhook URL
                webhook_url = get_effective_webhook_url(active_model_configs[config_name], stt_config) if config_name else None

                # Call webhook if configured
                if webhook_url:
                    try:
                        # Prepare webhook payload
                        webhook_data = {
                            "wake_word": best_model,
                            "confidence": float(max_confidence),
                            "threshold": float(detection_threshold),
                            "timestamp": time.time(),
                            "model_name": config_name,
                            "all_scores": {k: float(v) for k, v in prediction.items()},
                            "multi_trigger": False
                        }

                        # Make webhook call
                        logger.info(f"📞 Calling webhook: {webhook_url}")
                        response = requests.post(
                            webhook_url,
                            json=webhook_data,
                            timeout=constants.WEBHOOK_TIMEOUT_SECONDS
                        )

                        if response.status_code == 200:
                            logger.info(f"✅ Webhook call successful")
                        else:
                            logger.warning(f"⚠️ Webhook returned status code: {response.status_code}")

                    except requests.exceptions.Timeout:
                        logger.error("❌ Webhook call timed out")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"❌ Webhook call failed: {e}")
                    except Exception as e:
                        logger.error(f"❌ Unexpected error during webhook call: {e}")

                # Trigger STT recording based on webhook URL presence
                if (speech_recorder is not None and
                    webhook_url and
                    not speech_recorder.is_busy()):
                    logger.info(f"🎤 Triggering STT recording for wake word '{config_name}' (webhook URL: {webhook_url})")
                    logger.debug(f"STT recording conditions met: speech_recorder={speech_recorder is not None}, config_name={config_name}, webhook_url={webhook_url}, is_busy={speech_recorder.is_busy() if speech_recorder else 'N/A'}")

                    # Get STT language from config
                    with settings_manager.get_config() as config:
                        stt_language = config.stt.language

                    logger.debug(f"Starting recording with language={stt_language}")
                    # Start recording in background thread
                    speech_recorder.start_recording(
                        audio_stream=audio_reader,
                        wake_word=config_name,
                        confidence=max_confidence,
                        language=stt_language,
                        webhook_url=webhook_url,
                        topic=active_model_configs[config_name].topic
                    )
                else:
                    logger.debug(f"STT recording NOT triggered. Conditions: speech_recorder={speech_recorder is not None}, config_name={config_name}, webhook_url={webhook_url}, is_busy={speech_recorder.is_busy() if speech_recorder else 'N/A'}")
            else:
                # Enhanced debugging - log more frequent confidence updates
                if chunk_count % constants.MODERATE_CONFIDENCE_LOG_INTERVAL_CHUNKS == 0:
                    logger.debug(f"🎯 Best confidence: {max_confidence:.6f} from '{best_model}' (threshold: {detection_threshold:.6f})")
                    logger.debug(f"   All scores: {[f'{k}: {v:.6f}' for k, v in prediction.items()]}")

                # Also check for moderate confidence levels for debugging
                if max_confidence > constants.MODERATE_CONFIDENCE_THRESHOLD:
                    logger.info(f"🔍 Moderate confidence detected: {best_model} = {max_confidence:.6f}")
                elif max_confidence > constants.WEAK_SIGNAL_THRESHOLD:
                    logger.debug(f"🔍 Weak signal: {best_model} = {max_confidence:.6f}")
                elif max_confidence > constants.VERY_WEAK_SIGNAL_THRESHOLD:
                    logger.debug(f"🔍 Very weak signal: {best_model} = {max_confidence:.6f}")

    # If we exit the loop normally (shouldn't happen), return 0
    return 0

def main():
    """Main function to run the wake word detection system."""
    # Log git commit if available
    try:
        with open('/app/git_commit.txt', 'r') as f:
            commit = f.read().strip()
            logger.info(f"🔧 Running code from git commit: {commit}")
    except (FileNotFoundError, IOError, OSError) as e:
        logger.debug(f"Git commit info not available: {e}")
        logger.info("🔧 Git commit info not available")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize shared data structures for web GUI
    manager = Manager()
    shared_data = manager.dict({
        'rms': 0.0,
        'is_listening': False,
        'is_active': False,
        'status': 'Starting...',
        'last_detection': None,
        'recent_detections': [],
        'config_changed': False,
        'status_changed': True,
        'stt_health': 'disconnected'
    })
    event_queue = Queue(maxsize=constants.EVENT_QUEUE_MAXSIZE)
    
    # Initialize SettingsManager (triggers auto-discovery)
    logger.info("🔧 Initializing SettingsManager...")
    settings_manager = SettingsManager()
    
    # Get configurations
    with settings_manager.get_config() as config:
        models_config = config.models
        audio_config = config.audio
        system_config = config.system
    
    # Configure logging level from config
    logging.getLogger().setLevel(getattr(logging, system_config.log_level, logging.INFO))
    logger.info(f"✅ SettingsManager initialized with {len(models_config)} models")

    # Try to explicitly load specific models
    logger.info("Attempting to load pre-trained models...")

    try:
        # Handle test pipeline mode FIRST - don't need audio manager for testing
        if args.test_pipeline:
            # Use default filename if not provided
            if args.audio_file is None:
                logger.error("❌ No audio file specified for testing. Use -audio_file parameter.")
                return
            
            # Load the recorded audio
            audio_data = load_test_audio(args.audio_file)
            if audio_data is None:
                logger.error("❌ Failed to load test audio. Exiting.")
                return
            
            # Get enabled models from configuration
            enabled_models = [model for model in models_config if model.enabled]
            if not enabled_models:
                logger.warning("⚠️ No enabled models found in configuration, using all available models")
                enabled_models = models_config
            
            custom_models = [model.path for model in enabled_models]
            
            all_detected_words = []
            
            for i, custom_model_path in enumerate(custom_models):
                logger.info(f"🔧 Testing custom model {i+1}/{len(custom_models)}: {custom_model_path}")
                
                # Check if model file exists
                if os.path.exists(custom_model_path):
                    logger.info(f"✅ Model file found at: {custom_model_path}")
                else:
                    logger.error(f"❌ Model file NOT found at: {custom_model_path}")
                    continue
                
                # Load model without class mapping to see basename detection
                logger.info("Creating Model for pipeline testing...")
                try:
                    model = openwakeword.Model(
                        wakeword_models=[custom_model_path],
                        vad_threshold=system_config.vad_threshold,
                        enable_speex_noise_suppression=False
                    )
                    
                    # Run pipeline test for this model
                    logger.info(f"🧪 TESTING {os.path.basename(custom_model_path)} with recorded audio...")
                    detected_words = test_pipeline_with_audio(model, audio_data)
                    
                    if detected_words:
                        logger.info(f"✅ Model {os.path.basename(custom_model_path)} detected {len(detected_words)} wake words.")
                        all_detected_words.extend([(os.path.basename(custom_model_path), *word) for word in detected_words])
                    else:
                        logger.info(f"ℹ️  Model {os.path.basename(custom_model_path)} detected no wake words.")
                    
                except Exception as e:
                    logger.error(f"❌ Error testing model {custom_model_path}: {e}")
                
                logger.info("=" * 60)
            
            detected_words = all_detected_words
            
            if detected_words:
                logger.info(f"✅ Pipeline test completed. Found {len(detected_words)} wake word detections.")
            else:
                logger.info("ℹ️  Pipeline test completed. No wake words detected.")
            
            return

        # Initialize audio source - either WAV file or microphone
        audio_manager = None
        usb_mic = None

        if not args.input_wav:
            # Initialize AudioManager for audio device handling
            audio_manager = AudioManager()
            logger.info("AudioManager initialized")

            # Find the USB microphone
            usb_mic = audio_manager.find_usb_microphone()
            if not usb_mic:
                logger.error("No USB microphone found. Exiting.")
                raise RuntimeError("No USB microphone detected")

            logger.info(f"Using USB microphone: {usb_mic.name} (index {usb_mic.index})")

        # Handle recording mode (special mode that exits early)
        if args.record_test:
            # Generate timestamp filename if not provided
            if args.audio_file is None:
                audio_filename = generate_timestamp_filename()
            else:
                audio_filename = args.audio_file

            # Initialize model for recording (needed for real-time detection)
            logger.info("Creating Model for recording with real-time detection...")
            model = openwakeword.Model(
                wakeword_models=['hey_jarvis', 'alexa', 'hey_mycroft'],
                vad_threshold=system_config.vad_threshold
            )

            success, metadata = record_test_audio(audio_manager, usb_mic, model, settings_manager, audio_filename)
            if success:
                logger.info("✅ Recording completed successfully. Exiting.")
            else:
                logger.error("❌ Recording failed. Exiting.")
            return

        # Setup audio input
        stream = setup_audio_input(args, audio_config, audio_manager, usb_mic)

        # Initialize the OpenWakeWord model with enabled models from configuration
        model, active_model_configs, model_name_mapping, enabled_models = setup_wake_word_models(
            models_config, system_config, settings_manager, shared_data
        )

        # Get STT configuration
        with settings_manager.get_config() as config:
            stt_config = config.stt

        # Initialize heartbeat sender for ORAC STT integration
        heartbeat_sender = setup_heartbeat_sender(enabled_models, stt_config)

        # Initialize web server in a separate thread
        broadcaster = setup_web_server(settings_manager, shared_data, event_queue)

        # Initialize STT components
        ring_buffer, speech_recorder, stt_client, stt_health_status = setup_stt_components(
            stt_config, audio_config, active_model_configs, shared_data
        )
        
        # Function to reload models when configuration changes
        def reload_models():
            nonlocal model, active_model_configs, model_name_mapping, enabled_models
            success, new_model, new_configs, new_mapping, new_enabled = reload_models_on_config_change(
                settings_manager, heartbeat_sender, stt_client, shared_data
            )
            if success:
                # Clean up old model before replacing
                old_model = model
                model = new_model
                active_model_configs = new_configs
                model_name_mapping = new_mapping
                enabled_models = new_enabled
                del old_model
            return success
        
        # Initialize audio reader thread for non-blocking audio capture
        audio_reader = AudioReaderThread(stream, chunk_size=constants.CHUNK_SIZE, queue_maxsize=constants.AUDIO_READER_QUEUE_MAXSIZE)
        if not audio_reader.start():
            logger.error("Failed to start audio reader thread")
            raise RuntimeError("Failed to start audio reader thread")
        
        # Register main loop as a consumer
        main_consumer_queue = audio_reader.register_consumer("main_loop")
        logger.info("✅ Main loop registered as audio consumer")

        # Run the detection loop (pass event_queue directly instead of storing in shared_data)
        exit_code = run_detection_loop(
            model=model,
            active_model_configs=active_model_configs,
            model_name_mapping=model_name_mapping,
            audio_reader=audio_reader,
            main_consumer_queue=main_consumer_queue,
            settings_manager=settings_manager,
            shared_data=shared_data,
            ring_buffer=ring_buffer,
            speech_recorder=speech_recorder,
            stt_client=stt_client,
            heartbeat_sender=heartbeat_sender,
            reload_models_func=reload_models,
            event_queue=event_queue
        )

        # If detection loop exits with error code, exit with that code
        if exit_code != 0:
            sys.exit(exit_code)

    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        logger.info("Stopping audio stream and terminating AudioManager...")
        
        # Unregister main loop consumer and stop audio reader thread
        if 'audio_reader' in locals():
            audio_reader.unregister_consumer("main_loop")
            logger.info("Main loop unregistered as audio consumer")
            audio_reader.stop()
            logger.info("Audio reader thread stopped")
        
        # Stop heartbeat sender
        if 'heartbeat_sender' in locals():
            heartbeat_sender.stop()
            logger.info("Heartbeat sender stopped")
        
        # Stop speech recorder if active
        if speech_recorder:
            speech_recorder.stop()
        
        # Close STT client
        if stt_client:
            stt_client.close()
        
        if stream:
            stream.stop_stream()
            stream.close()
        if audio_manager:
            audio_manager.close()  # Explicitly clean up AudioManager

    except Exception as e:
        # Log any other errors and clean up
        import traceback
        logger.error(f"Error during execution: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Unregister main loop consumer and stop audio reader thread
        if 'audio_reader' in locals():
            audio_reader.unregister_consumer("main_loop")
            logger.info("Main loop unregistered as audio consumer")
            audio_reader.stop()
            logger.info("Audio reader thread stopped")
        
        # Stop heartbeat sender
        if 'heartbeat_sender' in locals():
            heartbeat_sender.stop()
            logger.info("Heartbeat sender stopped")
        
        # Stop speech recorder if active
        if speech_recorder:
            speech_recorder.stop()
        
        # Close STT client
        if stt_client:
            stt_client.close()
        
        if stream:
            stream.stop_stream()
            stream.close()
        if audio_manager:
            audio_manager.close()

if __name__ == "__main__":
    main()