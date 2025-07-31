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
from hey_orac.audio.preprocessing_manager import PreprocessingManager  # Import PreprocessingManager
from hey_orac.transport.stt_client import STTClient  # Import STT client
from hey_orac.config.manager import SettingsManager  # Import the SettingsManager
from hey_orac.web.app import create_app, socketio
from hey_orac.web.routes import init_routes
from hey_orac.web.broadcaster import WebSocketBroadcaster

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
    def __init__(self, filename, chunk_size=1280):
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
                
                if framerate != 16000:
                    logger.warning(f"⚠️  Sample rate is {framerate} Hz, expected 16000 Hz")
                
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


def read_audio_chunk(args, stream, audio_capture, chunk_size=1280):
    """
    Read audio chunk from appropriate source.
    
    Args:
        args: Command line arguments
        stream: PyAudio stream or WavFileStream
        audio_capture: AudioCapture instance (may be None)
        chunk_size: Number of samples to read
        
    Returns:
        tuple: (raw_data, source_type) or (None, None) on error
    """
    try:
        if args.input_wav:
            # WAV file reading
            data = stream.read(chunk_size, exception_on_overflow=False)
            return data, 'wav_file'
        elif audio_capture and audio_capture.is_active():
            # Preprocessed audio from AudioCapture
            chunk = audio_capture.get_audio_chunk()
            if chunk is not None and len(chunk) == chunk_size:
                # Convert float32 back to int16 bytes for compatibility
                audio_int16 = (chunk * 32768.0).astype(np.int16)
                return audio_int16.tobytes(), 'preprocessed'
            else:
                return None, None
        else:
            # Raw stream reading (current approach)
            data = stream.read(chunk_size, exception_on_overflow=False)
            return data, 'microphone'
    except Exception as e:
        logger.error(f"Error reading audio chunk: {e}")
        return None, None


def process_audio_data(raw_data, source_type='microphone', wav_channels=None):
    """
    Process raw audio data to format expected by OpenWakeWord.
    
    Args:
        raw_data: Raw audio bytes
        source_type: 'microphone', 'wav_file', or 'preprocessed'
        wav_channels: Number of channels for WAV files
        
    Returns:
        numpy array of float32 audio data or None on error
    """
    if raw_data is None or len(raw_data) == 0:
        return None
        
    try:
        # Convert bytes to numpy array
        audio_array = np.frombuffer(raw_data, dtype=np.int16)
        
        # Handle channel conversion based on source
        if source_type == 'wav_file' and wav_channels == 2 and len(audio_array) > 1280:
            # Stereo WAV file - convert to mono
            stereo_data = audio_array.reshape(-1, 2)
            audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
        elif source_type == 'preprocessed':
            # Already processed, just convert
            audio_data = audio_array.astype(np.float32)
        else:
            # Microphone input - check if stereo
            if len(audio_array) > 1280:  # Stereo data
                stereo_data = audio_array.reshape(-1, 2)
                audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
            else:
                # Already mono
                audio_data = audio_array.astype(np.float32)
                
        return audio_data
    except Exception as e:
        logger.error(f"Error processing audio data: {e}")
        return None


def record_test_audio(audio_manager, usb_mic, model, settings_manager, filename='test_recording.wav'):
    """Record 10 seconds of test audio with countdown and metadata generation."""
    logger.info("🎤 RECORDING MODE: Recording 10 seconds of test audio...")
    
    # Metadata collection
    metadata = {
        "recording_info": {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "duration_seconds": 10,
            "sample_rate": 16000,
            "channels": 2,
            "chunk_size": 1280,
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
    for i in range(5, 0, -1):
        logger.info(f"   {i}...")
        time.sleep(1)
    
    logger.info("🔴 RECORDING...")
    sys.stdout.flush()
    
    # Record audio data and process in real-time
    frames = []
    chunks_per_second = 16000 // 1280  # ~12.5 chunks per second
    total_chunks = chunks_per_second * 10  # 10 seconds
    
    for i in range(total_chunks):
        try:
            data = stream.read(1280, exception_on_overflow=False)
            if data:
                frames.append(data)
                
                # Convert to audio data for real-time processing
                audio_array = np.frombuffer(data, dtype=np.int16)
                if len(audio_array) > 1280:  # Stereo
                    stereo_data = audio_array.reshape(-1, 2)
                    audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
                else:
                    audio_data = audio_array.astype(np.float32)
                
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
                detection_threshold = 0.3
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
                if i % (chunks_per_second * 2) == 0:  # Every 2 seconds
                    seconds_elapsed = i // chunks_per_second
                    logger.info(f"   Recording... {seconds_elapsed}/10 seconds (RMS: {rms:.4f})")
                
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
            wf.setnchannels(2)  # Stereo
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
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
        logger.info(f"   Duration: 10 seconds")
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
            
            if framerate != 16000:
                logger.warning(f"⚠️  Sample rate is {framerate} Hz, expected 16000 Hz")
            
            # Read all frames
            frames = wf.readframes(wf.getnframes())
            
            # Convert to numpy array
            audio_array = np.frombuffer(frames, dtype=np.int16)
            
            # Convert stereo to mono (same as live pipeline)
            if channels == 2:
                stereo_data = audio_array.reshape(-1, 2)
                audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
                logger.info(f"   Converted stereo to mono: {len(audio_array)} -> {len(audio_data)} samples")
            else:
                audio_data = audio_array.astype(np.float32)
                logger.info(f"   Mono audio: {len(audio_data)} samples")
                
            duration = len(audio_data) / 16000
            logger.info(f"✅ Loaded {duration:.2f} seconds of audio data")
            
            return audio_data
            
    except Exception as e:
        logger.error(f"❌ Error loading audio file: {e}")
        return None

def test_pipeline_with_audio(model, audio_data):
    """Test the pipeline with recorded audio, processing in chunks like live audio."""
    logger.info("🧪 TESTING PIPELINE with recorded audio...")
    
    chunk_size = 1280
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
        if i % 25 == 0:
            timestamp = (i * chunk_size) / 16000
            logger.info(f"   📊 Time: {timestamp:.2f}s, RMS: {rms:.4f}, Best: {best_model} = {max_confidence:.6f}")
        
        # Detection logic (very low threshold for custom model testing)
        detection_threshold = 0.05
        if max_confidence >= detection_threshold:
            timestamp = (i * chunk_size) / 16000
            logger.info(f"🎯 WAKE WORD DETECTED at {timestamp:.2f}s! Confidence: {max_confidence:.6f} - Source: {best_model}")
            detected_words.append((timestamp, best_model, max_confidence))
            
        # Also log moderate confidence like live pipeline
        elif max_confidence > 0.1:
            timestamp = (i * chunk_size) / 16000
            logger.info(f"🔍 Moderate confidence at {timestamp:.2f}s: {best_model} = {max_confidence:.6f}")
    
    # Summary
    logger.info(f"🏁 Pipeline test completed. Detected {len(detected_words)} wake words:")
    for timestamp, word, confidence in detected_words:
        logger.info(f"   {timestamp:.2f}s: {word} (confidence: {confidence:.6f})")
    
    return detected_words

# Download pre-trained OpenWakeWord models if not already present
# This ensures models like "alexa", "hey jarvis", etc., are available
openwakeword.utils.download_models()

def main():
    """Main function to run the wake word detection system."""
    # Log git commit if available
    try:
        with open('/app/git_commit.txt', 'r') as f:
            commit = f.read().strip()
            logger.info(f"🔧 Running code from git commit: {commit}")
    except:
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
        'status_changed': True
    })
    event_queue = Queue(maxsize=100)
    
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
    
    # Initialize variables that might be accessed in exception handlers
    speech_recorder = None
    stt_client = None
    stream = None
    audio_manager = None
    preprocessing_active = False
    preprocessing_manager = None

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
        usb_mic = None
        
        if args.input_wav:
            # Use WAV file as input
            logger.info(f"🎵 Using WAV file as input: {args.input_wav}")
            if not os.path.exists(args.input_wav):
                logger.error(f"❌ WAV file not found: {args.input_wav}")
                return
            stream = WavFileStream(args.input_wav)
        else:
            # Initialize AudioManager for audio device handling
            audio_manager = AudioManager()
            logger.info("AudioManager initialized")

            # Find the USB microphone
            usb_mic = audio_manager.find_usb_microphone()
            if not usb_mic:
                logger.error("No USB microphone found. Exiting.")
                raise RuntimeError("No USB microphone detected")

            logger.info(f"Using USB microphone: {usb_mic.name} (index {usb_mic.index})")
        
        # Handle recording mode
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

        # Initialize preprocessing manager
        preprocessing_manager = PreprocessingManager(settings_manager, logger)
        
        # Start audio stream if using microphone (skip if using WAV file)
        if not args.input_wav:
            # Try to initialize preprocessing first (it will create its own stream)
            preprocessing_active = preprocessing_manager.initialize(
                usb_mic=usb_mic, 
                stream=None,  # Don't pass a stream - let preprocessing create its own
                audio_config={
                    'channels': audio_config.channels,
                    'sample_rate': audio_config.sample_rate,
                    'chunk_size': audio_config.chunk_size,
                    'preprocessing': audio_config.preprocessing.__dict__ if audio_config.preprocessing else {}
                }
            )
            
            # If preprocessing is not active, create a regular stream
            if not preprocessing_active:
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
            else:
                # Preprocessing is active, no need for separate stream
                stream = None
            
            if preprocessing_active:
                logger.info("✅ Audio preprocessing is active")
            else:
                logger.info("ℹ️  Audio preprocessing is disabled or unavailable")

        # Initialize the OpenWakeWord model with enabled models from configuration
        print("DEBUG: About to create Model()", flush=True)
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
            
            print("DEBUG: Model created successfully", flush=True)
            logger.info("OpenWakeWord model initialized")
            
            # Check what models are actually loaded
            if hasattr(model, 'models'):
                logger.info(f"Loaded models: {list(model.models.keys()) if model.models else 'None'}")
            logger.info(f"Prediction buffer keys: {list(model.prediction_buffer.keys())}")
            
            # Enhanced model verification - test with dummy audio like old working code
            logger.info("🔍 Testing model with dummy audio to verify initialization...")
            test_audio = np.zeros(1280, dtype=np.float32)
            try:
                test_predictions = model.predict(test_audio)
                logger.info(f"✅ Model test successful - prediction type: {type(test_predictions)}")
                logger.info(f"   Test prediction content: {test_predictions}")
                logger.info(f"   Test prediction keys: {list(test_predictions.keys())}")
                
                # Check prediction_buffer after first prediction (like old working code)
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
            
            print("DEBUG: After model initialized log", flush=True)
        except Exception as e:
            print(f"ERROR: Model creation failed: {e}", flush=True)
            raise
        
        # Force log flush
        sys.stdout.flush()
        print("DEBUG: After sys.stdout.flush()", flush=True)

        # Test audio stream first
        print("DEBUG: About to test audio stream", flush=True)
        logger.info("🧪 Testing audio stream...")
        print("DEBUG: After audio stream test log", flush=True)
        sys.stdout.flush()
        try:
            if preprocessing_active:
                # Test preprocessing manager instead of direct stream
                test_data = preprocessing_manager.get_audio_chunk(1280)
                if test_data is not None:
                    logger.info(f"✅ Audio preprocessing test successful, read {len(test_data) * 2} bytes")
                else:
                    logger.warning("⚠️ Audio preprocessing returned no data in test")
            else:
                test_data = stream.read(1280, exception_on_overflow=False)
                logger.info(f"✅ Audio stream test successful, read {len(test_data)} bytes")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"❌ Audio stream test failed: {e}")
            sys.stdout.flush()
            raise

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
        
        # Initialize STT components if enabled
        ring_buffer = None
        
        with settings_manager.get_config() as config:
            stt_config = config.stt
        
        # Always initialize STT components
        logger.info("🎙️ Initializing STT components...")
        logger.debug(f"STT Configuration: base_url={stt_config.base_url}, timeout={stt_config.timeout}s")
        logger.debug(f"STT Endpoint settings: pre_roll={stt_config.pre_roll_duration}s, silence_threshold={stt_config.silence_threshold}, silence_duration={stt_config.silence_duration}s")
        
        # Initialize ring buffer for pre-roll audio
        ring_buffer = RingBuffer(
            capacity_seconds=10.0,  # Keep 10 seconds of audio history
            sample_rate=audio_config.sample_rate
        )
        logger.debug(f"RingBuffer initialized with capacity={10.0}s, sample_rate={audio_config.sample_rate}Hz")
        
        # Initialize STT client
        stt_client = STTClient(
            base_url=stt_config.base_url,
            timeout=stt_config.timeout
        )
        
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
            endpoint_config=endpoint_config
        )
        
        logger.info("✅ STT components initialized successfully")
        
        # Perform health checks for models with webhook URLs
        if stt_client:
            logger.info("🏥 Performing per-model STT health checks...")
            for name, cfg in active_model_configs.items():
                if cfg.webhook_url:
                    logger.debug(f"Checking STT health for model '{name}' at {cfg.webhook_url}")
                    if stt_client.health_check(webhook_url=cfg.webhook_url):
                        logger.info(f"✅ STT healthy for model '{name}'")
                    else:
                        logger.warning(f"⚠️ STT unhealthy for model '{name}' at {cfg.webhook_url}")
        
        # Function to reload models when configuration changes
        def reload_models():
            nonlocal model, active_model_configs, model_name_mapping
            logger.info("🔄 Reloading models due to configuration change...")
            
            try:
                # Get current enabled models from configuration
                with settings_manager.get_config() as current_config:
                    models_config = current_config.models
                    system_config = current_config.system
                
                enabled_models = [model for model in models_config if model.enabled]
                if not enabled_models:
                    logger.warning("⚠️  No enabled models found after config change")
                    return False
                
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
                    return False
                
                # Create new model instance
                logger.info(f"Creating new OpenWakeWord instance with {len(new_model_paths)} models: {list(new_active_configs.keys())}")
                new_model = openwakeword.Model(
                    wakeword_models=new_model_paths,
                    vad_threshold=system_config.vad_threshold,
                    enable_speex_noise_suppression=False
                )
                
                # Test the new model
                test_audio = np.zeros(1280, dtype=np.float32)
                test_predictions = new_model.predict(test_audio)
                logger.info(f"✅ New model test successful - predictions: {list(test_predictions.keys())}")
                
                # Replace old model and configs
                old_model = model
                model = new_model
                active_model_configs = new_active_configs
                model_name_mapping = new_name_mapping
                
                # Update shared data
                shared_data['loaded_models'] = list(active_model_configs.keys())
                shared_data['models_config'] = {name: {
                    'enabled': True,
                    'threshold': cfg.threshold,
                    'webhook_url': cfg.webhook_url
                } for name, cfg in active_model_configs.items()}
                
                # Perform health checks for reloaded models with webhook URLs
                if stt_client:
                    logger.info("🏥 Performing per-model STT health checks for reloaded models...")
                    for name, cfg in active_model_configs.items():
                        if cfg.webhook_url:
                            logger.debug(f"Checking STT health for model '{name}' at {cfg.webhook_url}")
                            if stt_client.health_check(webhook_url=cfg.webhook_url):
                                logger.info(f"✅ STT healthy for model '{name}'")
                            else:
                                logger.warning(f"⚠️ STT unhealthy for model '{name}' at {cfg.webhook_url}")
                
                # Clean up old model (helps with memory)
                del old_model
                
                logger.info("✅ Models reloaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error reloading models: {e}")
                return False
        
        # Continuously listen to the audio stream and detect wake words
        logger.info("🎤 Starting wake word detection loop...")
        sys.stdout.flush()
        chunk_count = 0
        last_config_check = time.time()
        CONFIG_CHECK_INTERVAL = 1.0  # Check for config changes every second
        
        # Debouncing: track last detection time per model
        last_detection_times = {}
        DETECTION_COOLDOWN = 2.0  # Minimum seconds between detections for same model
        
        while True:
            try:
                # Check for configuration changes
                current_time = time.time()
                if current_time - last_config_check >= CONFIG_CHECK_INTERVAL:
                    if shared_data.get('config_changed', False):
                        logger.info("📢 Configuration change detected")
                        if reload_models():
                            shared_data['config_changed'] = False
                            logger.info("✅ Configuration change applied")
                        else:
                            logger.error("❌ Failed to apply configuration change")
                    last_config_check = current_time
                
                # Read audio chunk using our extracted function
                data, source_type = read_audio_chunk(
                    args, 
                    stream, 
                    preprocessing_manager.audio_capture if preprocessing_manager.is_preprocessing_active() else None,
                    chunk_size=1280
                )
                
                if data is None:
                    logger.warning("No audio data read from stream")
                    continue

                # Process audio data using our extracted function
                wav_channels = stream.channels if args.input_wav and hasattr(stream, 'channels') else None
                audio_data = process_audio_data(data, source_type, wav_channels)
                
                if audio_data is None:
                    logger.warning("Failed to process audio data")
                    continue

                # Calculate RMS for web GUI display
                rms = np.sqrt(np.mean(audio_data**2))
                shared_data['rms'] = float(rms)
                
                # Update listening state
                shared_data['is_listening'] = True
                
                # Feed audio to ring buffer if STT is enabled
                if ring_buffer is not None:
                    # Convert to int16 for ring buffer storage
                    audio_int16 = audio_data.astype(np.int16)
                    ring_buffer.write(audio_int16)
                
                # Log every 100 chunks to show we're processing audio
                chunk_count += 1
                if chunk_count % 100 == 0:
                    audio_volume = np.abs(audio_data).mean()
                    logger.info(f"📊 Processed {chunk_count} audio chunks")
                    logger.info(f"   Audio data shape: {audio_data.shape}, volume: {audio_volume:.4f}, RMS: {rms:.4f}")
                    logger.info(f"   Source: {source_type}, Raw data size: {len(data)} bytes")
                    if preprocessing_manager.is_preprocessing_active():
                        logger.info(f"   ✅ Audio preprocessing active (AGC, compression, limiting)")
                    elif source_type == 'microphone' and len(data) > 2560:
                        logger.info(f"   ✅ Stereo→Mono conversion active")

                # Pass the audio data to the model for wake word prediction
                prediction = model.predict(audio_data)
                
                # Log ALL confidence scores after each processed chunk
                if chunk_count % 100 == 0:
                    all_scores = {word: f"{score:.6f}" for word, score in prediction.items()}
                    logger.debug(f"🎯 All confidence scores: {all_scores}")
                    # Also log what models are in active_model_configs for debugging
                    if chunk_count == 100:
                        logger.debug(f"Active model configs: {list(active_model_configs.keys())}")
                        logger.debug(f"Model paths: {[cfg.path for cfg in active_model_configs.values()]}")
            except Exception as e:
                logger.error(f"Error processing audio data: {e}")
                continue

            # Get system config for multi-trigger setting
            with settings_manager.get_config() as config:
                multi_trigger_enabled = config.system.multi_trigger
            
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
                    # Check debounce cooldown
                    model_name = trigger_info['config_name']
                    current_time = time.time()
                    if model_name in last_detection_times:
                        time_since_last = current_time - last_detection_times[model_name]
                        if time_since_last < DETECTION_COOLDOWN:
                            logger.debug(f"Debouncing {model_name}: {time_since_last:.1f}s since last detection (cooldown: {DETECTION_COOLDOWN}s)")
                            continue
                    
                    # Update last detection time
                    last_detection_times[model_name] = current_time
                    
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
                    except:
                        pass  # Queue full, skip
                    
                    # Call webhook if configured
                    if trigger_info['model_config'].webhook_url:
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
                            logger.info(f"📞 Calling webhook (multi-trigger): {trigger_info['model_config'].webhook_url}")
                            response = requests.post(
                                trigger_info['model_config'].webhook_url,
                                json=webhook_data,
                                timeout=5  # 5 second timeout
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
                            audio_stream=stream,
                            wake_word=trigger_info['config_name'],
                            confidence=trigger_info['confidence'],
                            language=stt_language
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
                # Map OpenWakeWord model name to our config name
                config_name = None
                
                # First check the model name mapping
                if best_model in model_name_mapping:
                    config_name = model_name_mapping[best_model]
                # Then try direct match (in case config name matches prediction key)
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
                    detection_threshold = 0.3
                    if best_model is not None:
                        logger.warning(f"Model '{best_model}' not found in active configs, using default threshold")
                
                if max_confidence >= detection_threshold:
                    # Check debounce cooldown
                    current_time = time.time()
                    if config_name and config_name in last_detection_times:
                        time_since_last = current_time - last_detection_times[config_name]
                        if time_since_last < DETECTION_COOLDOWN:
                            logger.debug(f"Debouncing {config_name}: {time_since_last:.1f}s since last detection (cooldown: {DETECTION_COOLDOWN}s)")
                            continue  # Skip to next audio chunk
                    
                    # Update last detection time
                    if config_name:
                        last_detection_times[config_name] = current_time
                    
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
                    
                    # Add to event queue if not full
                    try:
                        event_queue.put_nowait(detection_event)
                    except:
                        pass  # Queue full, skip
                    
                    # Call webhook if configured
                    if config_name and active_model_configs[config_name].webhook_url:
                        try:
                            # Prepare webhook payload (convert numpy types to Python types)
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
                            logger.info(f"📞 Calling webhook: {active_model_configs[config_name].webhook_url}")
                            response = requests.post(
                                active_model_configs[config_name].webhook_url,
                                json=webhook_data,
                                timeout=5  # 5 second timeout
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
                    model_has_webhook = config_name and active_model_configs[config_name].webhook_url
                    
                    if (speech_recorder is not None and 
                        model_has_webhook and
                        not speech_recorder.is_busy()):
                        logger.info(f"🎤 Triggering STT recording for wake word '{config_name}' (webhook URL: {active_model_configs[config_name].webhook_url})")
                        logger.debug(f"STT recording conditions met: speech_recorder={speech_recorder is not None}, config_name={config_name}, webhook_url={active_model_configs[config_name].webhook_url}, is_busy={speech_recorder.is_busy() if speech_recorder else 'N/A'}")
                        
                        # Get STT language from config
                        with settings_manager.get_config() as config:
                            stt_language = config.stt.language
                        
                        logger.debug(f"Starting recording with language={stt_language}")
                        # Start recording in background thread with webhook URL
                        speech_recorder.start_recording(
                            audio_stream=stream,
                            wake_word=config_name,
                            confidence=max_confidence,
                            language=stt_language,
                            webhook_url=active_model_configs[config_name].webhook_url
                        )
                    else:
                        logger.debug(f"STT recording NOT triggered. Conditions: speech_recorder={speech_recorder is not None}, config_name={config_name}, webhook_url={active_model_configs[config_name].webhook_url if config_name and config_name in active_model_configs else None}, is_busy={speech_recorder.is_busy() if speech_recorder else 'N/A'}")
                else:
                    # Enhanced debugging - log more frequent confidence updates
                    if chunk_count % 50 == 0:  # Every 50 chunks instead of 100
                        logger.debug(f"🎯 Best confidence: {max_confidence:.6f} from '{best_model}' (threshold: {detection_threshold:.6f})")
                        logger.debug(f"   All scores: {[f'{k}: {v:.6f}' for k, v in prediction.items()]}")
                    
                    # Also check for moderate confidence levels for debugging
                    if max_confidence > 0.1:
                        logger.info(f"🔍 Moderate confidence detected: {best_model} = {max_confidence:.6f}")
                    elif max_confidence > 0.05:
                        logger.debug(f"🔍 Weak signal: {best_model} = {max_confidence:.6f}")
                    elif max_confidence > 0.01:
                        logger.debug(f"🔍 Very weak signal: {best_model} = {max_confidence:.6f}")

    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        logger.info("Stopping audio stream and terminating AudioManager...")
        
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
            audio_manager.__del__()  # Explicitly clean up AudioManager

    except Exception as e:
        # Log any other errors and clean up
        import traceback
        logger.error(f"Error during execution: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
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
            audio_manager.__del__()

if __name__ == "__main__":
    main()