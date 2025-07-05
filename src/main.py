#!/usr/bin/env python3
"""
Hey Orac - Wake-word Detection Service
Phase 1a of the ORAC Voice-Control Architecture
"""

import argparse
import logging
import sys
import yaml
import time
import numpy as np
from pathlib import Path
from audio_utils import AudioManager
from wake_word_interface import WakeWordDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def test_audio_file(audio_path: str, config: dict) -> None:
    """Test wake-word detection with audio file."""
    logger.info(f"Testing wake-word detection with {audio_path}")
    # TODO: Implement audio file testing
    logger.info("Audio file testing not yet implemented")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Hey Orac Wake-word Detection Service")
    parser.add_argument(
        "--config", 
        default="/app/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-audio",
        help="Test wake-word detection with audio file"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices"
    )
    parser.add_argument(
        "--test-recording",
        action="store_true",
        help="Test recording from USB microphone"
    )
    parser.add_argument(
        "--test-wakeword",
        action="store_true",
        help="Test wake-word detection with live audio"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.list_devices:
        logger.info("Listing audio devices...")
        audio_manager = AudioManager()
        devices = audio_manager.list_input_devices()
        
        print("\n=== Available Audio Input Devices ===")
        for device in devices:
            print(f"Device {device.index}: {device.name}")
            print(f"  - Max Input Channels: {device.max_input_channels}")
            print(f"  - Default Sample Rate: {device.default_sample_rate}")
            print(f"  - Host API: {device.host_api}")
            print(f"  - USB Device: {'Yes' if device.is_usb else 'No'}")
            print()
        
        # Find USB microphone
        usb_device = audio_manager.find_usb_microphone()
        if usb_device:
            print(f"✅ Found USB microphone: {usb_device.name} (index {usb_device.index})")
        else:
            print("⚠️  No USB microphone found")
        
        return
    
    if args.test_audio:
        test_audio_file(args.test_audio, config)
        return
    
    if args.test_recording:
        logger.info("Testing USB microphone recording...")
        audio_manager = AudioManager()
        
        # Debug: Check AudioManager type and attributes
        logger.info(f"AudioManager type: {type(audio_manager)}")
        logger.info(f"AudioManager module: {audio_manager.__class__.__module__}")
        logger.info(f"AudioManager attributes: {[attr for attr in dir(audio_manager) if not attr.startswith('_')]}")
        
        # Find USB microphone
        usb_device = audio_manager.find_usb_microphone()
        if not usb_device:
            logger.error("❌ No USB microphone found for recording test")
            return
        
        # Test recording for 3 seconds
        output_file = "test_usb_recording.wav"
        logger.info(f"Recording 3 seconds from {usb_device.name}...")
        
        if audio_manager.record_to_file(
            usb_device.index, 
            duration=3.0, 
            output_file=output_file,
            sample_rate=16000,
            channels=1
        ):
            logger.info(f"✅ Recording successful! Saved to {output_file}")
        else:
            logger.error("❌ Recording failed")
        
        return
    
    if args.test_wakeword:
        logger.info("Testing wake-word detection with live audio...")
        
        # Initialize wake word detector
        wake_detector = WakeWordDetector()
        if not wake_detector.initialize(config['wake_word']):
            logger.error("❌ Failed to initialize wake word detector")
            return
        
        # Initialize audio manager
        audio_manager = AudioManager()
        
        # Debug: Check AudioManager type and attributes
        logger.info(f"AudioManager type: {type(audio_manager)}")
        logger.info(f"AudioManager module: {audio_manager.__class__.__module__}")
        logger.info(f"AudioManager attributes: {[attr for attr in dir(audio_manager) if not attr.startswith('_')]}")
        
        usb_device = audio_manager.find_usb_microphone()
        if not usb_device:
            logger.error("❌ No USB microphone found")
            return
        
        logger.info(f"🎤 Listening for '{wake_detector.get_wake_word_name()}' on {usb_device.name}")
        logger.info("Press Ctrl+C to stop...")
        
        # Simple test mode to prevent system crashes
        logger.info("🔧 Using simple test mode to prevent system crashes...")
        
        try:
            for i in range(5):  # Test 5 samples
                logger.info(f"🎙️ Recording 2-second sample...")
                
                # Record audio sample
                if not audio_manager.record_to_file(
                    usb_device.index,
                    2.0,
                    "/tmp/test_audio.wav",
                    sample_rate=wake_detector.get_sample_rate(),
                    channels=1
                ):
                    logger.error("❌ Failed to record audio sample")
                    continue
                
                # Load and process audio in chunks
                import soundfile as sf
                import numpy as np
                
                try:
                    # Load the recorded audio
                    audio_data, sample_rate = sf.read("/tmp/test_audio.wav")
                    
                    # Convert to mono if stereo
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data[:, 0]
                    
                    # Convert to int16 if needed
                    if audio_data.dtype != np.int16:
                        audio_data = (audio_data * 32767).astype(np.int16)
                    
                    # Process audio in 1280-sample chunks
                    frame_length = wake_detector.get_frame_length()
                    detected = False
                    
                    for start in range(0, len(audio_data) - frame_length, frame_length // 2):  # 50% overlap
                        chunk = audio_data[start:start + frame_length]
                        if len(chunk) == frame_length:  # Only process complete frames
                            if wake_detector.process_audio(chunk):
                                logger.info("🎉 Wake word detected!")
                                detected = True
                                break
                    
                    if not detected:
                        logger.info("👂 No wake word detected in this sample")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing audio: {e}")
                    
                time.sleep(1)  # Wait between samples
                
        except KeyboardInterrupt:
            logger.info("🛑 Test stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in test mode: {e}")
        finally:
            # Clean up
            try:
                import os
                if os.path.exists("/tmp/test_audio.wav"):
                    os.remove("/tmp/test_audio.wav")
                logger.info("✅ Test mode cleanup completed")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        return
    
    # Main service loop
    logger.info("Starting Hey Orac wake-word detection service...")
    logger.info(f"Configuration: {config}")
    
    # TODO: Implement main service loop
    logger.info("Main service loop not yet implemented")
    
    try:
        # Placeholder for main loop - with sleep to prevent CPU runaway
        logger.info("Main service loop not yet implemented - sleeping to prevent CPU runaway")
        while True:
            time.sleep(1)  # Sleep 1 second to prevent 100% CPU usage
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 