#!/usr/bin/env python3
"""
Test script to verify PyAudio ALSA support and capabilities
"""

import pyaudio
import sys
import os

def test_pyaudio_alsa():
    """Test PyAudio ALSA support and capabilities."""
    print("🔍 PyAudio ALSA Support Test")
    print("=" * 40)
    
    try:
        # Initialize PyAudio
        print("📋 Initializing PyAudio...")
        p = pyaudio.PyAudio()
        
        # Check PyAudio version
        print(f"📦 PyAudio version: {pyaudio.__version__}")
        
        # Check device count
        device_count = p.get_device_count()
        print(f"🎵 Total devices detected: {device_count}")
        
        if device_count == 0:
            print("❌ No devices detected!")
            print("🔧 This indicates PyAudio may not have ALSA support")
            
            # Check if we can access ALSA directly
            print("\n🔍 Checking ALSA access...")
            try:
                import subprocess
                result = subprocess.run(['python3', '-c', 'import pyaudio; p = pyaudio.PyAudio(); print(f"Devices: {p.get_device_count()}")'], 
                                      capture_output=True, text=True, timeout=10)
                print(f"   Subprocess test: {result.stdout.strip()}")
                if result.stderr:
                    print(f"   Subprocess errors: {result.stderr.strip()}")
            except Exception as e:
                print(f"   Subprocess test failed: {e}")
            
            return False
        
        # List all devices
        print("\n📋 Device Details:")
        for i in range(device_count):
            try:
                device_info = p.get_device_info_by_index(i)
                print(f"\n🎤 Device {i}: {device_info['name']}")
                print(f"   Host API: {device_info['hostApi']}")
                print(f"   Max Input Channels: {device_info['maxInputChannels']}")
                print(f"   Max Output Channels: {device_info['maxOutputChannels']}")
                print(f"   Default Sample Rate: {device_info['defaultSampleRate']}")
                
                # Check if it's an input device
                if device_info['maxInputChannels'] > 0:
                    print(f"   ✅ Input device")
                else:
                    print(f"   ❌ No input channels")
                    
            except Exception as e:
                print(f"   ❌ Error getting device {i} info: {e}")
        
        # Check default devices
        print("\n🎯 Default Devices:")
        try:
            default_input = p.get_default_input_device_info()
            print(f"   Default Input: {default_input['name']} (index {default_input['index']})")
        except Exception as e:
            print(f"   ❌ No default input device: {e}")
        
        try:
            default_output = p.get_default_output_device_info()
            print(f"   Default Output: {default_output['name']} (index {default_output['index']})")
        except Exception as e:
            print(f"   ❌ No default output device: {e}")
        
        # Test opening a stream
        print("\n🧪 Testing Stream Opening:")
        input_devices = [i for i in range(device_count) 
                        if p.get_device_info_by_index(i)['maxInputChannels'] > 0]
        
        if input_devices:
            test_device = input_devices[0]
            print(f"   Testing with device {test_device}...")
            
            try:
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    input_device_index=test_device,
                    frames_per_buffer=512
                )
                print(f"   ✅ Successfully opened stream on device {test_device}")
                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"   ❌ Failed to open stream on device {test_device}: {e}")
        else:
            print("   ❌ No input devices available for testing")
        
        # Cleanup
        p.terminate()
        print("\n✅ PyAudio test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ PyAudio test failed: {e}")
        return False

def check_alsa_environment():
    """Check ALSA environment and configuration."""
    print("\n🔧 ALSA Environment Check")
    print("=" * 30)
    
    # Check environment variables
    alsa_vars = ['ALSA_CARD', 'ALSA_DEVICE', 'AUDIODEV', 'AUDIO_DEVICE']
    for var in alsa_vars:
        value = os.environ.get(var)
        print(f"   {var}: {value if value else 'Not set'}")
    
    # Check ALSA configuration files
    alsa_configs = ['/etc/asound.conf', '/etc/asoundrc', '~/.asoundrc', '/app/.asoundrc']
    for config_path in alsa_configs:
        expanded_path = os.path.expanduser(config_path)
        if os.path.exists(expanded_path):
            print(f"   Found ALSA config: {expanded_path}")
        else:
            print(f"   No ALSA config: {expanded_path}")

if __name__ == "__main__":
    print("🚀 PyAudio ALSA Diagnostic Tool")
    print("=" * 50)
    
    check_alsa_environment()
    success = test_pyaudio_alsa()
    
    if success:
        print("\n🎉 PyAudio appears to be working correctly!")
        sys.exit(0)
    else:
        print("\n❌ PyAudio has issues - check the diagnostics above")
        sys.exit(1) 