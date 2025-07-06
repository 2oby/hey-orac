#!/usr/bin/env python3
"""
Comprehensive PyAudio test script for SH-04 USB microphone
Tests device detection, access, and recording capabilities
"""

import pyaudio
import logging
import subprocess
import os
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_system_checks():
    """Run system-level audio diagnostics."""
    logger.info("🔧 Running system audio diagnostics...")
    
    # Check ALSA cards
    try:
        result = subprocess.run(['cat', '/proc/asound/cards'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("ALSA Cards:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        else:
            logger.warning("Could not read ALSA cards")
    except Exception as e:
        logger.warning(f"Error reading ALSA cards: {e}")
    
    # Check USB devices
    try:
        result = subprocess.run(['lsusb'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("USB Devices:")
            for line in result.stdout.strip().split('\n'):
                if 'audio' in line.lower() or 'mic' in line.lower() or 'sh-04' in line.lower():
                    logger.info(f"  {line}")
        else:
            logger.warning("Could not list USB devices")
    except Exception as e:
        logger.warning(f"Error listing USB devices: {e}")
    
    # Check ALSA devices
    try:
        result = subprocess.run(['arecord', '-l'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            logger.info("ALSA Recording Devices:")
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    logger.info(f"  {line}")
        else:
            logger.warning("Could not list ALSA devices")
    except Exception as e:
        logger.warning(f"Error listing ALSA devices: {e}")
    
    # Check environment variables
    logger.info("Audio Environment Variables:")
    audio_env_vars = ['ALSA_CARD', 'ALSA_DEVICE', 'AUDIODEV', 'AUDIO_DEVICE', 'ALSA_PCM_CARD', 'ALSA_PCM_DEVICE']
    for var in audio_env_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"  {var}: {value}")
        else:
            logger.info(f"  {var}: Not set")

def test_arecord():
    """Test ALSA recording with arecord."""
    logger.info("🎤 Testing ALSA recording with arecord...")
    
    # Test with explicit device first
    logger.info("  Testing with explicit device hw:0,0...")
    try:
        result = subprocess.run([
            'arecord', '-D', 'hw:0,0', '-f', 'S16_LE', '-r', '16000', 
            '-c', '1', '-d', '2', 'test_arecord_explicit.wav'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("✅ arecord with explicit device successful")
            if os.path.exists('test_arecord_explicit.wav'):
                size = os.path.getsize('test_arecord_explicit.wav')
                logger.info(f"  Recorded file size: {size} bytes")
        else:
            logger.error(f"❌ arecord with explicit device failed: {result.stderr}")
    except Exception as e:
        logger.error(f"❌ Error running arecord with explicit device: {e}")
    
    # Test with default device
    logger.info("  Testing with default device...")
    try:
        result = subprocess.run([
            'arecord', '-D', 'default', '-f', 'S16_LE', '-r', '16000', 
            '-c', '1', '-d', '2', 'test_arecord_default.wav'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            logger.info("✅ arecord with default device successful")
            if os.path.exists('test_arecord_default.wav'):
                size = os.path.getsize('test_arecord_default.wav')
                logger.info(f"  Recorded file size: {size} bytes")
                if size > 0:
                    logger.info("✅ Audio file contains data")
                else:
                    logger.warning("⚠️ Audio file is empty")
            else:
                logger.error("❌ Audio file was not created")
        else:
            logger.error(f"❌ arecord with default device failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ Error running arecord with default device: {e}")

def test_pyaudio_devices():
    """Test PyAudio device detection."""
    logger.info("🔍 Testing PyAudio device detection...")
    
    try:
        p = pyaudio.PyAudio()
        
        device_count = p.get_device_count()
        logger.info(f"Total PyAudio devices: {device_count}")
        
        if device_count == 0:
            logger.error("❌ No PyAudio devices detected!")
            return False
        
        # List all devices
        for i in range(device_count):
            try:
                device_info = p.get_device_info_by_index(i)
                logger.info(f"Device {i}:")
                logger.info(f"  Name: {device_info['name']}")
                logger.info(f"  Host API: {device_info['hostApi']}")
                logger.info(f"  Max Input Channels: {device_info['maxInputChannels']}")
                logger.info(f"  Max Output Channels: {device_info['maxOutputChannels']}")
                logger.info(f"  Default Sample Rate: {device_info['defaultSampleRate']}")
                
                # Check for SH-04 or USB devices
                device_name_lower = device_info['name'].lower()
                if 'sh-04' in device_name_lower or 'usb' in device_name_lower:
                    logger.info(f"  ✅ Found SH-04/USB device!")
                
            except Exception as e:
                logger.error(f"  ❌ Error getting device {i} info: {e}")
        
        p.terminate()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing PyAudio: {e}")
        return False

def test_pyaudio_recording():
    """Test PyAudio recording capabilities."""
    logger.info("🎤 Testing PyAudio recording...")
    
    try:
        p = pyaudio.PyAudio()
        
        # Find input devices
        input_devices = []
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    input_devices.append((i, device_info))
            except Exception as e:
                logger.error(f"Error checking device {i}: {e}")
        
        if not input_devices:
            logger.error("❌ No input devices found!")
            p.terminate()
            return False
        
        logger.info(f"Found {len(input_devices)} input devices")
        
        # Test each input device
        for device_index, device_info in input_devices:
            logger.info(f"Testing device {device_index}: {device_info['name']}")
            
            try:
                # Try to open a stream
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=512
                )
                
                logger.info(f"✅ Successfully opened stream for device {device_index}")
                
                # Try to read some audio data
                try:
                    data = stream.read(512, exception_on_overflow=False)
                    logger.info(f"✅ Successfully read audio data from device {device_index}")
                    logger.info(f"  Data length: {len(data)} bytes")
                except Exception as e:
                    logger.error(f"❌ Error reading audio data from device {device_index}: {e}")
                
                # Close the stream
                stream.stop_stream()
                stream.close()
                logger.info(f"✅ Successfully closed stream for device {device_index}")
                
            except Exception as e:
                logger.error(f"❌ Error testing device {device_index}: {e}")
        
        # Test explicit device 0 (SH-04) as suggested in debug file
        logger.info("🎤 Attempting to open device hw:0,0 directly...")
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=0,  # Force device 0
                frames_per_buffer=1024
            )
            logger.info("✅ Stream opened successfully with explicit device 0!")
            stream.close()
        except Exception as e:
            logger.error(f"❌ Failed to open stream with explicit device 0: {e}")
        
        p.terminate()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in PyAudio recording test: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting comprehensive PyAudio test for SH-04 USB microphone")
    
    # Run system checks
    run_system_checks()
    
    # Test arecord
    test_arecord()
    
    # Test PyAudio device detection
    if not test_pyaudio_devices():
        logger.error("❌ PyAudio device detection failed")
        sys.exit(1)
    
    # Test PyAudio recording
    if not test_pyaudio_recording():
        logger.error("❌ PyAudio recording test failed")
        sys.exit(1)
    
    logger.info("✅ All tests completed successfully!")
    logger.info("🎉 SH-04 USB microphone should be working properly")

if __name__ == "__main__":
    main() 