#!/usr/bin/env python3
"""
OpenWakeWord script for detecting wake words from a USB microphone on a Raspberry Pi in a Docker container.
Uses AudioManager for robust audio device handling.
"""

import sys
import os
# Use environment variable for unbuffered output instead
os.environ['PYTHONUNBUFFERED'] = '1'

import openwakeword
from openwakeword.model import Model
import logging
import numpy as np
from audio_utils import AudioManager  # Import the AudioManager class

# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# Download pre-trained OpenWakeWord models if not already present
# This ensures models like "alexa", "hey jarvis", etc., are available
openwakeword.utils.download_models()

# Try to explicitly load specific models
logger.info("Attempting to load pre-trained models...")

try:
    # Initialize AudioManager for audio device handling
    audio_manager = AudioManager()
    logger.info("AudioManager initialized")

    # Find the USB microphone
    usb_mic = audio_manager.find_usb_microphone()
    if not usb_mic:
        logger.error("No USB microphone found. Exiting.")
        raise RuntimeError("No USB microphone detected")

    logger.info(f"Using USB microphone: {usb_mic.name} (index {usb_mic.index})")

    # Start audio stream with parameters suitable for OpenWakeWord
    # - Sample rate: 16000 Hz (required by OpenWakeWord)  
    # - Channels: 2 (stereo) to match microphone capabilities, then convert to mono
    # - Chunk size: 1280 samples (80 ms at 16000 Hz) for optimal efficiency
    # Note: Microphone has 2 input channels, so read as stereo then process to mono
    stream = audio_manager.start_stream(
        device_index=usb_mic.index,
        sample_rate=16000,
        channels=2,  # Read as stereo to match microphone capabilities
        chunk_size=1280
    )
    if not stream:
        logger.error("Failed to start audio stream. Exiting.")
        raise RuntimeError("Failed to start audio stream")

    # Initialize the OpenWakeWord model, loading all pre-trained models
    print("DEBUG: About to create Model()", flush=True)
    try:
        # Use default model loading - leave wakeword_models empty to load all pre-trained models
        logger.info("Creating Model with default settings to load all pre-trained models...")
        model = Model(inference_framework='tflite')  # Explicitly use tflite
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
        test_data = stream.read(1280, exception_on_overflow=False)
        logger.info(f"✅ Audio stream test successful, read {len(test_data)} bytes")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"❌ Audio stream test failed: {e}")
        sys.stdout.flush()
        raise

    # Continuously listen to the audio stream and detect wake words
    logger.info("🎤 Starting wake word detection loop...")
    sys.stdout.flush()
    chunk_count = 0
    while True:
        try:
            # Read one chunk of audio data (1280 samples)
            data = stream.read(1280, exception_on_overflow=False)
            if data is None or len(data) == 0:
                logger.warning("No audio data read from stream")
                continue

            # Convert bytes to numpy array - now handling stereo input
            audio_array = np.frombuffer(data, dtype=np.int16)
            
            # Convert stereo to mono by averaging the channels (like old working code)
            if len(audio_array) > 1280:  # If we got stereo data (2560 samples for stereo vs 1280 for mono)
                # Reshape to separate left and right channels, then average
                stereo_data = audio_array.reshape(-1, 2)
                # CRITICAL FIX: OpenWakeWord expects raw int16 values as float32, NOT normalized!
                audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
                logger.debug(f"Converted stereo ({len(audio_array)} samples) to mono ({len(audio_data)} samples)")
            else:
                # Already mono - CRITICAL FIX: no normalization!
                audio_data = audio_array.astype(np.float32)

            # Log every 100 chunks to show we're processing audio
            chunk_count += 1
            if chunk_count % 100 == 0:
                audio_volume = np.abs(audio_data).mean()
                logger.info(f"📊 Processed {chunk_count} audio chunks")
                logger.info(f"   Audio data shape: {audio_data.shape}, volume: {audio_volume:.4f}")
                logger.info(f"   Raw data size: {len(data)} bytes, samples: {len(audio_array)}")
                if len(audio_array) > 1280:
                    logger.info(f"   ✅ Stereo→Mono conversion active")

            # Pass the audio data to the model for wake word prediction
            prediction = model.predict(audio_data)
            
            # Log ALL confidence scores after each processed chunk
            if chunk_count % 100 == 0:
                all_scores = {word: f"{score:.6f}" for word, score in prediction.items()}
                logger.debug(f"🎯 All confidence scores: {all_scores}")
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            continue

        # Enhanced wake word detection logic like old working code
        max_confidence = 0.0
        best_model = None
        
        # Find the highest confidence score
        for wakeword, score in prediction.items():
            if score > max_confidence:
                max_confidence = score
                best_model = wakeword
        
        # Check against proper threshold (like old working code)
        detection_threshold = 0.5
        if max_confidence >= detection_threshold:
            logger.info(f"🎯 WAKE WORD DETECTED! Confidence: {max_confidence:.6f} (threshold: {detection_threshold:.6f}) - Source: {best_model}")
            logger.info(f"   All model scores: {[f'{k}: {v:.6f}' for k, v in prediction.items()]}")
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
    if stream:
        stream.stop_stream()
        stream.close()
    audio_manager.__del__()  # Explicitly clean up AudioManager

except Exception as e:
    # Log any other errors and clean up
    logger.error(f"Error during execution: {e}")
    if stream:
        stream.stop_stream()
        stream.close()
    audio_manager.__del__()