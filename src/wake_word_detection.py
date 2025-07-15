#!/usr/bin/env python3
"""
OpenWakeWord script for detecting wake words from a USB microphone on a Raspberry Pi in a Docker container.
Uses AudioManager for robust audio device handling.
"""

import sys
import os
# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 0)

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
    # - Channels: 1 (mono) for compatibility
    # - Chunk size: 1280 samples (80 ms at 16000 Hz) for optimal efficiency
    stream = audio_manager.start_stream(
        device_index=usb_mic.index,
        sample_rate=16000,
        channels=1,
        chunk_size=1280
    )
    if not stream:
        logger.error("Failed to start audio stream. Exiting.")
        raise RuntimeError("Failed to start audio stream")

    # Initialize the OpenWakeWord model, loading all pre-trained models
    model = Model()
    logger.info("OpenWakeWord model initialized")
    
    # Force log flush
    import sys
    sys.stdout.flush()

    # Test audio stream first
    logger.info("🧪 Testing audio stream...")
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

            # Convert bytes to numpy array for OpenWakeWord
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0

            # Log every 100 chunks to show we're processing audio
            chunk_count += 1
            if chunk_count % 100 == 0:
                logger.info(f"📊 Processed {chunk_count} audio chunks, latest volume: {np.abs(audio_data).mean():.4f}")

            # Pass the audio data to the model for wake word prediction
            prediction = model.predict(audio_data)
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            continue

        # Check predictions for each pre-trained model
        for wakeword, score in prediction.items():
            # If the score exceeds 0.3, consider it a wake word detection (lowered for testing)
            if score > 0.3:
                logger.info(f"🚨 Wake word '{wakeword}' detected with score {score}")
            # Log any score above 0.1 for debugging
            elif score > 0.1:
                logger.debug(f"🔍 Weak signal for '{wakeword}': {score:.3f}")

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