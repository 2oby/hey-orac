#!/usr/bin/env python3
"""
Minimal PyAudio test script to debug PortAudio initialization issues
"""

import sys
import os

def test_pyaudio_initialization():
    """Test PyAudio initialization and device detection."""
    print("🔍 Testing PyAudio Initialization")
    print("=" * 50)
    
    # Check environment
    print(f"Python version: {sys.version}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        import pyaudio
        print(f"✅ PyAudio imported successfully")
        print(f"📦 PyAudio version: {pyaudio.__version__}")
        print(f"📁 PyAudio file: {pyaudio.__file__}")
        
        # Test PyAudio initialization
        print("\n🔧 Testing PyAudio initialization...")
        p = pyaudio.PyAudio()
        print("✅ PyAudio initialized successfully!")
        
        # Test host APIs
        print(f"\n🎵 Host APIs available: {p.get_host_api_count()}")
        for i in range(p.get_host_api_count()):
            try:
                api_info = p.get_host_api_info_by_index(i)
                print(f"  API {i}: {api_info['name']} (id: {api_info['type']})")
                print(f"    Default input device: {api_info.get('defaultInputDevice', 'None')}")
                print(f"    Default output device: {api_info.get('defaultOutputDevice', 'None')}")
                print(f"    Device count: {api_info['deviceCount']}")
            except Exception as e:
                print(f"  API {i}: Error getting info - {e}")
        
        # Test device count
        device_count = p.get_device_count()
        print(f"\n📊 Total device count: {device_count}")
        
        if device_count > 0:
            print("\n📋 Available devices:")
            for i in range(device_count):
                try:
                    device_info = p.get_device_info_by_index(i)
                    print(f"  Device {i}: {device_info['name']}")
                    print(f"    Max Input Channels: {device_info['maxInputChannels']}")
                    print(f"    Max Output Channels: {device_info['maxOutputChannels']}")
                    print(f"    Default Sample Rate: {device_info['defaultSampleRate']}")
                    print(f"    Host API: {device_info['hostApi']}")
                    
                    # Check if it's a USB device
                    device_name_lower = device_info['name'].lower()
                    is_usb = any(keyword in device_name_lower for keyword in [
                        'usb', 'sh-04', 'mv', 'blue', 'wireless', 'bluetooth'
                    ])
                    print(f"    USB Device: {'✅ Yes' if is_usb else '❌ No'}")
                    print()
                except Exception as e:
                    print(f"  Device {i}: Error getting info - {e}")
        else:
            print("❌ No devices detected!")
            
        # Try to force ALSA usage
        print("\n🔧 Testing ALSA-specific device access...")
        try:
            # Try to open a stream with ALSA-specific parameters
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                input_device_index=None,  # Use default
                frames_per_buffer=1024
            )
            print("✅ Successfully opened audio stream!")
            stream.close()
        except Exception as e:
            print(f"❌ Failed to open audio stream: {e}")
            
        # Cleanup
        p.terminate()
        print("✅ PyAudio terminated successfully")
        
    except ImportError as e:
        print(f"❌ Failed to import PyAudio: {e}")
        return False
    except Exception as e:
        print(f"❌ PyAudio initialization failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False
    
    return True

def test_alsa_directly():
    """Test ALSA access directly."""
    print("\n🔍 Testing ALSA Access Directly")
    print("=" * 50)
    
    try:
        import subprocess
        
        # Test ALSA cards
        result = subprocess.run(['cat', '/proc/asound/cards'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ ALSA cards accessible:")
            print(result.stdout)
        else:
            print("❌ Cannot access ALSA cards")
            
        # Test arecord
        result = subprocess.run(['arecord', '-l'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ arecord working:")
            print(result.stdout)
        else:
            print("❌ arecord failed:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ ALSA test failed: {e}")

if __name__ == "__main__":
    print("🧪 Minimal PyAudio Test Script")
    print("=" * 50)
    
    # Test ALSA first
    test_alsa_directly()
    
    # Test PyAudio
    success = test_pyaudio_initialization()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 PyAudio test completed successfully!")
    else:
        print("❌ PyAudio test failed!")
    
    sys.exit(0 if success else 1) 