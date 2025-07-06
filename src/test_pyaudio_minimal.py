#!/usr/bin/env python3
"""
Enhanced PyAudio test script to debug PortAudio initialization issues
Provides detailed diagnostics for ALSA backend and device detection
"""

import sys
import os
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

def test_pyaudio_initialization():
    """Test PyAudio initialization and device detection with enhanced diagnostics."""
    print("🔍 Testing PyAudio Initialization with Enhanced Diagnostics")
    print("=" * 60)
    
    # Check environment
    print(f"Python version: {sys.version}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    print(f"ALSA_CARD: {os.environ.get('ALSA_CARD', 'Not set')}")
    print(f"ALSA_DEVICE: {os.environ.get('ALSA_DEVICE', 'Not set')}")
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
        
        # Enhanced host API diagnostics
        host_api_count = p.get_host_api_count()
        print(f"\n🎵 Host APIs available: {host_api_count}")
        
        alsa_api_index = None
        for i in range(host_api_count):
            try:
                api_info = p.get_host_api_info_by_index(i)
                print(f"  API {i}: {api_info['name']} (id: {api_info['type']})")
                print(f"    Default input device: {api_info.get('defaultInputDevice', 'None')}")
                print(f"    Default output device: {api_info.get('defaultOutputDevice', 'None')}")
                print(f"    Device count: {api_info['deviceCount']}")
                
                # Find ALSA API
                if api_info['name'] == 'ALSA':
                    alsa_api_index = i
                    print(f"    ✅ Found ALSA API at index {i}")
                    
                    # List devices for ALSA API specifically
                    print(f"    🔍 ALSA API devices:")
                    for j in range(api_info['deviceCount']):
                        try:
                            dev_info = p.get_device_info_by_host_api_device_index(i, j)
                            print(f"      Device {j}: {dev_info['name']}")
                            print(f"        Inputs: {dev_info['maxInputChannels']}")
                            print(f"        Outputs: {dev_info['maxOutputChannels']}")
                            print(f"        Sample Rate: {dev_info['defaultSampleRate']}")
                        except Exception as e:
                            print(f"      Device {j}: Error getting info - {e}")
                            
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
            
        # Enhanced ALSA stream testing
        print("\n🔧 Testing ALSA-specific stream access...")
        if alsa_api_index is not None:
            print(f"🎯 Using ALSA API at index {alsa_api_index}")
            
            # Try multiple approaches to open a stream
            stream_attempts = [
                {
                    'name': 'Default device with ALSA API',
                    'params': {
                        'format': pyaudio.paInt16,
                        'channels': 1,
                        'rate': 16000,
                        'input': True,
                        'host_api_index': alsa_api_index
                    }
                },
                {
                    'name': 'Device 0 with ALSA API',
                    'params': {
                        'format': pyaudio.paInt16,
                        'channels': 1,
                        'rate': 16000,
                        'input': True,
                        'input_device_index': 0,
                        'host_api_index': alsa_api_index
                    }
                },
                {
                    'name': 'Default device without API specification',
                    'params': {
                        'format': pyaudio.paInt16,
                        'channels': 1,
                        'rate': 16000,
                        'input': True
                    }
                }
            ]
            
            for attempt in stream_attempts:
                print(f"\n🔍 Attempt: {attempt['name']}")
                try:
                    stream = p.open(**attempt['params'])
                    print(f"✅ Successfully opened audio stream!")
                    print(f"   Stream active: {stream.is_active()}")
                    stream.close()
                    break
                except Exception as e:
                    print(f"❌ Failed to open audio stream: {e}")
                    print(f"   Error type: {type(e).__name__}")
        else:
            print("❌ ALSA API not found!")
            
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
    """Test ALSA access directly with enhanced diagnostics."""
    print("\n🔍 Testing ALSA Access Directly")
    print("=" * 50)
    
    try:
        import subprocess
        
        # Test ALSA cards
        print("📋 Checking ALSA cards...")
        result = subprocess.run(['cat', '/proc/asound/cards'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ ALSA cards accessible:")
            print(result.stdout)
        else:
            print("❌ Cannot access ALSA cards")
            
        # Test arecord
        print("\n🎤 Testing arecord...")
        result = subprocess.run(['arecord', '-l'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ arecord working:")
            print(result.stdout)
        else:
            print("❌ arecord failed:")
            print(result.stderr)
            
        # Test aplay
        print("\n🔊 Testing aplay...")
        result = subprocess.run(['aplay', '-l'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ aplay working:")
            print(result.stdout)
        else:
            print("❌ aplay failed:")
            print(result.stderr)
            
        # Test ALSA version
        print("\n📦 Checking ALSA version...")
        result = subprocess.run(['cat', '/proc/asound/version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ ALSA version:")
            print(result.stdout)
        else:
            print("❌ Cannot access ALSA version")
            
        # Test USB devices
        print("\n🔌 Checking USB devices...")
        result = subprocess.run(['lsusb'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ USB devices:")
            print(result.stdout)
        else:
            print("❌ Cannot list USB devices")
            
    except Exception as e:
        print(f"❌ ALSA test failed: {e}")

def test_audio_devices():
    """Test audio device permissions and access."""
    print("\n🔍 Testing Audio Device Access")
    print("=" * 50)
    
    try:
        import subprocess
        
        # Check /dev/snd contents
        print("📁 Checking /dev/snd directory...")
        result = subprocess.run(['ls', '-la', '/dev/snd/'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ /dev/snd contents:")
            print(result.stdout)
        else:
            print("❌ Cannot access /dev/snd")
            
        # Check audio group membership
        print("\n👥 Checking audio group membership...")
        result = subprocess.run(['groups'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Current user groups:")
            print(result.stdout)
        else:
            print("❌ Cannot check groups")
            
        # Check current user
        print("\n👤 Checking current user...")
        result = subprocess.run(['whoami'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Current user: {result.stdout.strip()}")
        else:
            print("❌ Cannot check current user")
            
    except Exception as e:
        print(f"❌ Audio device test failed: {e}")

if __name__ == "__main__":
    print("🧪 Enhanced PyAudio Test Script")
    print("=" * 60)
    
    # Test ALSA first
    test_alsa_directly()
    
    # Test audio device access
    test_audio_devices()
    
    # Test PyAudio
    success = test_pyaudio_initialization()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 PyAudio test completed successfully!")
    else:
        print("❌ PyAudio test failed!")
    
    sys.exit(0 if success else 1) 