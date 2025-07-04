#!/usr/bin/env python3
"""
TEMPORARY DEVELOPMENT FILE - DELETE AFTER USB RECORDING IS WORKING
Test USB microphone recording functionality
"""

import pyaudio
import wave
import sys
import time
from pathlib import Path

def list_audio_devices():
    """List all available audio input devices."""
    p = pyaudio.PyAudio()
    
    print("=== Available Audio Input Devices ===")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:  # Input device
            print(f"Device {i}: {device_info['name']}")
            print(f"  - Max Input Channels: {device_info['maxInputChannels']}")
            print(f"  - Default Sample Rate: {device_info['defaultSampleRate']}")
            print(f"  - Host API: {device_info['hostApi']}")
            print()
    
    p.terminate()

def test_device_recording(device_index, duration=3, sample_rate=16000, channels=1):
    """Test recording from a specific device."""
    p = pyaudio.PyAudio()
    
    try:
        # Get device info
        device_info = p.get_device_info_by_index(device_index)
        print(f"Testing device {device_index}: {device_info['name']}")
        
        # Open stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=512
        )
        
        print(f"Recording for {duration} seconds...")
        frames = []
        
        for i in range(0, int(sample_rate / 512 * duration)):
            data = stream.read(512)
            frames.append(data)
            if i % 10 == 0:  # Progress indicator
                print(f"Recording... {i * 512 / sample_rate:.1f}s")
        
        print("Recording complete!")
        
        # Close stream
        stream.stop_stream()
        stream.close()
        
        # Save to file
        output_filename = f"test_recording_device_{device_index}.wav"
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
        
        print(f"Saved recording to: {output_filename}")
        return True
        
    except Exception as e:
        print(f"Error recording from device {device_index}: {e}")
        return False
    finally:
        p.terminate()

def test_usb_microphone():
    """Test specifically for USB microphone."""
    print("=== USB Microphone Test ===")
    
    # List all devices first
    list_audio_devices()
    
    # Try to find USB device
    p = pyaudio.PyAudio()
    usb_devices = []
    
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if (device_info['maxInputChannels'] > 0 and 
            'usb' in device_info['name'].lower()):
            usb_devices.append(i)
    
    p.terminate()
    
    if usb_devices:
        print(f"Found {len(usb_devices)} USB input device(s): {usb_devices}")
        for device_index in usb_devices:
            test_device_recording(device_index)
    else:
        print("No USB input devices found. Testing device 0 (default)...")
        test_device_recording(0)

def main():
    """Main test function."""
    print("=== TEMPORARY USB AUDIO TEST ===")
    print("This file should be deleted once basic recording is working!")
    print()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_audio_devices()
        elif sys.argv[1] == "test":
            device_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            test_device_recording(device_index)
        elif sys.argv[1] == "usb":
            test_usb_microphone()
    else:
        print("Usage:")
        print("  python dev_audio_test.py list    - List all audio devices")
        print("  python dev_audio_test.py test [device_index] - Test recording from device")
        print("  python dev_audio_test.py usb     - Test USB microphone specifically")

if __name__ == "__main__":
    main() 