Step-by-Step Fix:
1. First, Stop the Application Using the Device
bash# Stop the main container to release the device
ssh pi 'cd ~/hey-orac && docker-compose stop hey-orac'

# Or if you need to keep the container running, just kill the audio process
ssh pi 'cd ~/hey-orac && docker-compose exec -T hey-orac pkill -f "python.*audio" || true'
2. Fix ALSA Configuration
Create a simplified .asoundrc that matches your card number:
bashssh pi 'cd ~/hey-orac && cat > .asoundrc << EOF
# Use card 0 (which is your SH-04)
pcm.!default {
    type asym
    playback.pcm {
        type plug
        slave.pcm "hw:0,0"
    }
    capture.pcm {
        type plug
        slave.pcm "hw:0,0"
    }
}

ctl.!default {
    type hw
    card 0
}

# Disable all the problematic plugins
pcm.dmix { type plug; slave.pcm "null"; }
pcm.dsnoop { type plug; slave.pcm "null"; }
pcm.rear { type plug; slave.pcm "null"; }
pcm.center_lfe { type plug; slave.pcm "null"; }
pcm.side { type plug; slave.pcm "null"; }
pcm.hdmi { type plug; slave.pcm "null"; }
pcm.modem { type plug; slave.pcm "null"; }
pcm.phoneline { type plug; slave.pcm "null"; }
EOF'

# Copy to container
ssh pi 'cd ~/hey-orac && docker-compose exec -T hey-orac bash -c "cp /home/pi/hey-orac/.asoundrc /home/appuser/.asoundrc && chown appuser:appuser /home/appuser/.asoundrc"'
3. Update Environment Variables in docker-compose.yml
yamlversion: '3.8'
services:
  hey-orac:
    image: your_image_name
    devices:
      - /dev/snd:/dev/snd
      - /dev/bus/usb:/dev/bus/usb
    volumes:
      - /home/pi/hey-orac/.asoundrc:/home/appuser/.asoundrc:ro
    user: "1000:29"
    environment:
      - ALSA_CARD=0  # Changed from SH04 to 0
      - ALSA_PCM_CARD=0
      - AUDIODEV=hw:0,0
      - PYTHONUNBUFFERED=1
4. Test ALSA First
bash# Test recording with explicit device
ssh pi 'cd ~/hey-orac && docker-compose exec -T hey-orac arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 -d 3 test.wav'

# Test with default device
ssh pi 'cd ~/hey-orac && docker-compose exec -T hey-orac arecord -D default -f S16_LE -r 16000 -c 1 -d 3 test_default.wav'
5. Fix PyAudio Detection
Create a test script that forces PyAudio to rescan devices:
python# test_pyaudio_fixed.py
import os
import pyaudio
import warnings

# Suppress ALSA warnings
warnings.filterwarnings("ignore")
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"

# Force ALSA to use our device
os.environ['ALSA_CARD'] = '0'
os.environ['AUDIODEV'] = 'hw:0,0'

try:
    p = pyaudio.PyAudio()
    
    # Get device count
    device_count = p.get_device_count()
    print(f"🎵 Total devices detected: {device_count}")
    
    # List all devices
    for i in range(device_count):
        info = p.get_device_info_by_index(i)
        print(f"\nDevice {i}: {info['name']}")
        print(f"  Channels: {info['maxInputChannels']} in, {info['maxOutputChannels']} out")
        print(f"  Sample Rate: {info['defaultSampleRate']}")
    
    # Try to open the device directly
    print("\n🎤 Attempting to open device hw:0,0 directly...")
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=0,  # Force device 0
            frames_per_buffer=1024
        )
        print("✅ Stream opened successfully!")
        stream.close()
    except Exception as e:
        print(f"❌ Failed to open stream: {e}")
    
    p.terminate()
    
except Exception as e:
    print(f"❌ PyAudio initialization failed: {e}")