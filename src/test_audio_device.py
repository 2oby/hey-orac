#!/usr/bin/env python3
"""
Test script to verify audio device access in isolation
This script stops the main service and tests the USB microphone properly
"""

import logging
import subprocess
import time
import sys
import os
from audio_utils import AudioManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def stop_main_service():
    """Stop the main service if it's running."""
    logger.info("🛑 Stopping main service to free audio device...")
    
    try:
        # Kill any running main.py processes
        result = subprocess.run(['pkill', '-f', 'python.*main.py'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("✅ Stopped running main service")
        else:
            logger.info("ℹ️ No running main service found")
        
        # Wait for device to be released
        time.sleep(3)
        return True
        
    except Exception as e:
        logger.error(f"❌ Could not stop main service: {e}")
        return False

def test_audio_device():
    """Test audio device access after stopping main service."""
    logger.info("🎤 Testing audio device access...")
    
    # Initialize audio manager
    audio_manager = AudioManager()
    
    # List devices
    devices = audio_manager.list_input_devices()
    logger.info(f"📊 Found {len(devices)} input devices")
    
    if not devices:
        logger.error("❌ No input devices found!")
        return False
    
    # Find USB microphone
    usb_device = audio_manager.find_usb_microphone()
    if not usb_device:
        logger.error("❌ No USB microphone found!")
        return False
    
    logger.info(f"✅ Found USB microphone: {usb_device.name}")
    
    # Test device access
    logger.info("🧪 Testing device access...")
    try:
        test_stream = audio_manager.start_stream(
            device_index=usb_device.index,
            sample_rate=16000,
            channels=1,
            chunk_size=512
        )
        
        if test_stream:
            logger.info("✅ Successfully opened audio stream!")
            
            # Test reading audio data
            try:
                audio_data = test_stream.read(512, exception_on_overflow=False)
                logger.info(f"✅ Successfully read audio data: {len(audio_data)} bytes")
            except Exception as e:
                logger.error(f"❌ Error reading audio data: {e}")
                test_stream.stop_stream()
                test_stream.close()
                return False
            
            # Close stream
            test_stream.stop_stream()
            test_stream.close()
            logger.info("✅ Successfully closed audio stream")
            return True
        else:
            logger.error("❌ Failed to open audio stream")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing device access: {e}")
        return False

def test_recording():
    """Test recording functionality."""
    logger.info("🎙️ Testing recording functionality...")
    
    audio_manager = AudioManager()
    usb_device = audio_manager.find_usb_microphone()
    
    if not usb_device:
        logger.error("❌ No USB microphone found for recording test")
        return False
    
    # Test recording
    output_file = "/tmp/test_recording.wav"
    logger.info(f"Recording 3 seconds to {output_file}...")
    
    if audio_manager.record_to_file(
        usb_device.index,
        duration=3.0,
        output_file=output_file,
        sample_rate=16000,
        channels=1
    ):
        logger.info("✅ Recording successful!")
        
        # Check file size
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            logger.info(f"📁 Recorded file size: {file_size} bytes")
            
            # Clean up
            os.remove(output_file)
            logger.info("🧹 Cleaned up test file")
            return True
        else:
            logger.error("❌ Recording file not found")
            return False
    else:
        logger.error("❌ Recording failed")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting isolated audio device test")
    logger.info("=" * 50)
    
    # Stop main service
    if not stop_main_service():
        logger.error("❌ Failed to stop main service")
        return False
    
    # Test device access
    if not test_audio_device():
        logger.error("❌ Audio device test failed")
        return False
    
    # Test recording
    if not test_recording():
        logger.error("❌ Recording test failed")
        return False
    
    logger.info("🎉 All audio device tests passed!")
    logger.info("✅ USB microphone is working correctly")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 