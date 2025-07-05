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
                    
                    # Debug: Check audio quality
                    rms_level = np.sqrt(np.mean(audio_data**2))
                    max_level = np.max(np.abs(audio_data))
                    min_level = np.min(audio_data)
                    mean_level = np.mean(audio_data)
                    
                    logger.info(f"🎵 Audio Quality Check:")
                    logger.info(f"   RMS Level: {rms_level:.6f}")
                    logger.info(f"   Max Level: {max_level:.6f}")
                    logger.info(f"   Min Level: {min_level:.6f}")
                    logger.info(f"   Mean Level: {mean_level:.6f}")
                    logger.info(f"   Sample Rate: {sample_rate}")
                    logger.info(f"   Duration: {len(audio_data)/sample_rate:.2f}s")
                    logger.info(f"   Samples: {len(audio_data)}")
                    
                    # Check if audio is too quiet
                    if rms_level < 0.001:
                        logger.warning("⚠️ Audio is very quiet - check microphone volume!")
                    elif rms_level < 0.01:
                        logger.warning("⚠️ Audio is quiet - consider speaking louder")
                    else:
                        logger.info("✅ Audio levels look good")
                    
                    # Save a copy for manual inspection
                    sf.write("/tmp/debug_audio.wav", audio_data, sample_rate)
                    logger.info("💾 Saved debug audio to /tmp/debug_audio.wav")
                    
                    # Process audio in 1280-sample chunks
                    chunk_size = wake_detector.get_frame_length()
                    detected = False
                    
                    for i in range(0, len(audio_data) - chunk_size + 1, chunk_size):
                        chunk = audio_data[i:i + chunk_size]
                        
                        # Convert to int16 format for OpenWakeWord
                        chunk_int16 = (chunk * 32767).astype(np.int16)
                        
                        if wake_detector.process_audio(chunk_int16):
                            detected = True
                            logger.info(f"🎯 Wake word detected in chunk {i//chunk_size + 1}!")
                            break
                    
                    if not detected:
                        logger.info("👂 No wake word detected in this sample")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing audio: {e}")
                    continue
                
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