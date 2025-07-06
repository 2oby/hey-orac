Plan to Debug and Fix PyAudio Initialization in Docker on Raspberry Pi
This plan addresses the issue where PyAudio fails to initialize PortAudio in a Docker container on a Raspberry Pi, despite ALSA correctly detecting the SH-04 USB microphone. The goal is to enable PyAudio to initialize and access the microphone for wake-word detection, leveraging the provided audio_utils.py diagnostics and ALSA configurations (asound.conf, .asoundrc).
Step 1: Verify Host ALSA and Microphone Functionality
Objective: Confirm ALSA and the SH-04 microphone work on the Raspberry Pi host to rule out hardware issues.

Check USB device detection:

Run: lsusb
Look for: SH-04 or "audio"/"mic" in the output (e.g., Bus 001 Device 003: ID xxxx:xxxx SH-04 USB Audio).
If missing: Ensure the microphone is connected. Use a powered USB hub if power is insufficient. Check: dmesg | grep -i usb.


Verify ALSA devices:

Run: arecord -l
Look for: card 0: SH04 [SH-04 USB Audio], device 0: USB Audio.
Note the card (SH04 or 0) and device (0).
If missing: Update ALSA: sudo apt update && sudo apt install alsa-utils. Load module: sudo modprobe snd_usb_audio.


Test recording:

Run: arecord -D hw:SH04,0 -f S16_LE -r 16000 -c 1 test.wav
Play back: aplay test.wav
If fails: Check USB power, connections, or ALSA module: lsmod | grep snd_usb_audio.



Step 2: Validate Docker Container Setup
Objective: Ensure the container has access to ALSA devices and necessary permissions.

Pass ALSA and USB devices:

Use --device to map devices:docker run --device=/dev/snd:/dev/snd -v /dev/bus/usb:/dev/bus/usb ...


If fails: Try privileged mode (cautiously): docker run --privileged ....


Mount ALSA configurations:

Mount asound.conf and .asoundrc:docker run -v /etc/asound.conf:/etc/asound.conf -v ~/.asoundrc:/root/.asoundrc ...


Verify files are in the container: cat /etc/asound.conf and cat /root/.asoundrc.


Check ALSA in container:

Run: cat /proc/asound/cards
Look for: card 0: SH04 [SH-04 USB Audio].
Run: ls /dev/snd/
Look for: pcmC0D0c (capture device).
If missing: Confirm --device flags and host ALSA setup.


Ensure audio permissions:

Add container user to audio group:usermod -a -G audio <container-user>


Or run as host user with audio group:docker run -u $(id -u):$(getent group audio | cut -d: -f3) ...





Step 3: Fix ALSA Configuration
Objective: Ensure ALSA configurations are correct and conflict-free.

Resolve .asoundrc conflicts:

The .asoundrc has duplicate pcm.!default and ctl.!default definitions. Only the last (card SH04) is used.
Fix: Use a single configuration:# ~/.asoundrc
pcm.!default {
    type hw
    card SH04
    device 0
}

ctl.!default {
    type hw
    card SH04
}

pcm.usb_mic {
    type hw
    card SH04
    device 0
}

ctl.usb_mic {
    type hw
    card SH04
}


If asound.conf is used system-wide, remove .asoundrc to avoid conflicts or ensure they match.


Test ALSA in container:

Run: arecord -D usb_mic -f S16_LE -r 16000 -c 1 test.wav
If fails: Check audio_utils.py logs (Step 4).



Step 4: Debug PyAudio Initialization
Objective: Identify why PyAudio fails to initialize PortAudio.

Enable detailed logging in audio_utils.py:

Modify:logging.basicConfig(level=logging.DEBUG)




Run diagnostics:

Execute: python3 audio_utils.py
Check logs from _run_system_audio_diagnostics and list_input_devices:
ALSA version: cat /proc/asound/version
ALSA cards: cat /proc/asound/cards (should list SH-04)
USB devices: lsusb (should show SH-04)
ALSA devices: arecord -l (should list SH-04)
PyAudio devices: Look for OSError: [Errno -10000] PortAudio not initialized or get_device_count() == 0.




Test PyAudio directly:

Run a minimal script in the container:import pyaudio
try:
    p = pyaudio.PyAudio()
    print(f"Device count: {p.get_device_count()}")
    for i in range(p.get_device_count()):
        print(p.get_device_info_by_index(i))
    p.terminate()
except Exception as e:
    print(f"Error: {e}")


Expected error: OSError: [Errno -10000] PortAudio not initialized.


Common causes and fixes:

Missing ALSA libraries:
Ensure libasound2 is installed:RUN apt-get update && apt-get install -y libasound2 alsa-utils




PyAudio not compiled with ALSA support:
Reinstall PyAudio with ALSA development headers:RUN apt-get update && apt-get install -y libasound-dev
RUN pip install --force-reinstall pyaudio




PortAudio initialization failure:
PortAudio may fail due to container namespaces or missing ALSA plugins. Install additional ALSA plugins:RUN apt-get install -y libasound2-plugins




Environment variables:
Set LD_LIBRARY_PATH to include ALSA libraries:export LD_LIBRARY_PATH=/usr/lib/arm-linux-gnueabihf:$LD_LIBRARY_PATH


Add to Dockerfile:ENV LD_LIBRARY_PATH=/usr/lib/arm-linux-gnueabihf:$LD_LIBRARY_PATH







Step 5: Verify Docker Image Dependencies
Objective: Ensure all required dependencies are present.

Update Dockerfile:
RUN apt-get update && apt-get install -y \
    libasound2 \
    libasound2-dev \
    libasound2-plugins \
    alsa-utils \
    libsndfile1 \
    python3-pyaudio \
    usbutils
RUN pip install --force-reinstall pyaudio numpy
ENV LD_LIBRARY_PATH=/usr/lib/arm-linux-gnueabihf:$LD_LIBRARY_PATH


Verify Python dependencies:

Check: pip show pyaudio numpy
Ensure PyAudio version supports ALSA (e.g., 0.2.11 or later).



Step 6: Address Raspberry Pi-Specific Issues
Objective: Handle Raspberry Pi hardware and ALSA nuances.

Check USB power:

Use a powered USB hub if SH-04 draws too much power.
Check: dmesg | grep -i usb for power errors.


Verify ALSA module:

Run: lsmod | grep snd_usb_audio
If missing: sudo modprobe snd_usb_audio


Reduce resource usage:

If CPU is strained (top), lower sample rate in audio_utils.py:audio_manager.start_recording(device_index, sample_rate=8000, channels=1)





Step 7: Test Outside Docker
Objective: Confirm PyAudio works on the host to isolate Docker issues.

Run minimal PyAudio script on the host:import pyaudio
p = pyaudio.PyAudio()
print(f"Device count: {p.get_device_count()}")
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))
p.terminate()


If works: Issue is Docker-specific (revisit Steps 2 and 4).
If fails: Install dependencies on host:sudo apt install libasound2 libasound2-dev libasound2-plugins python3-pyaudio
pip3 install pyaudio



Step 8: Example Docker Run Command
Objective: Provide a complete command to run the container.
docker run --device=/dev/snd:/dev/snd \
           -v /dev/bus/usb:/dev/bus/usb \
           -v /etc/asound.conf:/etc/asound.conf \
           -v ~/.asoundrc:/root/.asoundrc \
           -u $(id -u):$(getent group audio | cut -d: -f3) \
           -e LD_LIBRARY_PATH=/usr/lib/arm-linux-gnueabihf:$LD_LIBRARY_PATH \
           your_image_name

Step 9: Collect Additional Information
Objective: Gather data if the issue persists.

Host outputs:
lsusb
arecord -l
cat /proc/asound/cards
ldd $(which arecord) (to check ALSA library paths)


Container outputs:
audio_utils.py logs with DEBUG level.
Output of minimal PyAudio script (Step 4.3).
ldd $(python3 -c "import pyaudio; print(pyaudio.__file__)") to check PyAudio’s linked libraries.


Docker setup:
Share docker run command or docker-compose.yml.
Share Dockerfile.


Error messages:
Full OSError stack trace from PyAudio.



Step 10: Iterate and Resolve

Analyze collected data to identify root cause (e.g., missing libraries, PortAudio misconfiguration, container isolation).
Apply fixes from Steps 2–6.
Retest with minimal PyAudio script and audio_utils.py until p.get_device_count() returns > 0 and SH-04 is accessible.














# Plan to Debug and Fix USB Microphone Access in Docker on Raspberry Pi

## 🎯 **CURRENT STATUS & PROGRESS**

### ✅ **What We've Successfully Fixed:**

1. **ALSA Configuration Issues** - FIXED ✅
   - **Problem**: `.asoundrc` had duplicate `pcm.!default` and `ctl.!default` sections causing parsing errors
   - **Solution**: Cleaned up `.asoundrc` to remove duplicates
   - **Result**: No more ALSA parsing errors, clean configuration

2. **Docker Audio Device Access** - WORKING ✅
   - **Problem**: Container couldn't access host audio devices
   - **Solution**: Added proper device mounts in `docker-compose.yml`:
     ```yaml
     volumes:
       - /dev/snd:/dev/snd:rw
       - /dev/bus/usb:/dev/bus/usb
     devices:
       - /dev/snd:/dev/snd
       - /dev/bus/usb:/dev/bus/usb
     group_add:
       - audio
     ```
   - **Result**: Audio devices are accessible in container

3. **ALSA Library Access** - WORKING ✅
   - **Problem**: Container missing ALSA shared libraries
   - **Solution**: Added ALSA library mounts:
     ```yaml
     volumes:
       - /usr/lib/aarch64-linux-gnu/libasound.so.2:/usr/lib/aarch64-linux-gnu/libasound.so.2:ro
       - /usr/lib/aarch64-linux-gnu/libasound.so.2.0.0:/usr/lib/aarch64-linux-gnu/libasound.so.2.0.0:ro
     environment:
       - LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu
     ```
   - **Result**: ALSA libraries are accessible

### 🔍 **What We've Discovered:**

1. **Host Audio System** - WORKING ✅
   - USB microphone (SH-04) detected: `card 0: SH04 [SH-04]`
   - ALSA devices working: `arecord -l` shows microphone
   - Host recording works (confirmed in DEV_LOG.txt)

2. **Container ALSA Access** - WORKING ✅
   - Audio devices mounted: `/dev/snd/` shows `pcmC0D0c` (capture device)
   - ALSA cards visible: `cat /proc/asound/cards` shows SH-04
   - ALSA tools working: `arecord -l` works in container

3. **PyAudio Issue** - STILL BROKEN ❌
   - **Problem**: PyAudio detects 0 devices despite ALSA working
   - **Error**: `OSError: [Errno -10000] PortAudio not initialized`
   - **Attempted Solutions**:
     - ✅ Fixed ALSA configuration
     - ✅ Added ALSA library mounts
     - ✅ Tried explicit compilation flags
     - 🔄 **Current**: Trying system PyAudio package instead of pip

### 🎯 **Current Focus: PyAudio Device Detection**

The core issue is that PyAudio cannot initialize PortAudio properly in the Docker container, even though:
- ✅ ALSA system works perfectly
- ✅ Audio devices are accessible
- ✅ ALSA tools (`arecord`, `cat /proc/asound/cards`) work
- ❌ PyAudio detects 0 devices

## 📋 **SYSTEMATIC DEBUGGING STEPS COMPLETED**

### **Step 1: Verify USB Microphone on Host** - ✅ COMPLETED
- ✅ USB device detected: `card 0: SH04 [SH-04]`
- ✅ ALSA devices working: `arecord -l` shows the microphone
- ✅ Host recording works (from DEV_LOG.txt)

### **Step 2: Configure Docker Container for Audio Access** - ✅ COMPLETED
- ✅ Audio devices mounted: `/dev/snd:/dev/snd`
- ✅ USB devices mounted: `/dev/bus/usb:/dev/bus/usb`
- ✅ ALSA config mounted: `.asoundrc`
- ✅ Audio group added
- ✅ ALSA libraries mounted

### **Step 3: Fix ALSA Configuration** - ✅ COMPLETED
- ✅ Fixed duplicate sections in `.asoundrc`
- ✅ Clean configuration now in place

### **Step 4: Debug with audio_utils.py** - 🔄 IN PROGRESS
- ✅ ALSA system diagnostics working
- ✅ Audio device listing working
- ❌ PyAudio device detection failing

## 🚀 **NEXT STEPS TO TRY**

### **Step 5: Fix PyAudio Installation** - 🔄 CURRENT
**Current Approach**: Use system PyAudio package instead of pip
```dockerfile
# Remove PyAudio from pip and use system package instead
RUN /app/venv/bin/pip uninstall -y pyaudio || true

# Copy system PyAudio to virtual environment
RUN cp /usr/lib/python3/dist-packages/pyaudio* /app/venv/lib/python3.12/site-packages/ 2>/dev/null || true
```

**Alternative Approaches to Try**:
1. **Use system Python instead of virtual environment**
2. **Install PyAudio with different compilation flags**
3. **Use a different audio library (sounddevice, etc.)**
4. **Test with a simpler PyAudio test script**

### **Step 6: Test Basic Audio Recording** - ⏳ PENDING
Once PyAudio works, test basic recording functionality.

### **Step 7: Test Wake Word Detection** - ⏳ PENDING
Test wake word detection with live audio.

### **Step 8: Test Continuous Streaming** - ⏳ PENDING
Test the complete continuous streaming wake word detection pipeline.

## 🔧 **TECHNICAL INSIGHTS**

### **Key Findings**:
1. **ALSA vs PyAudio**: ALSA works perfectly, PyAudio doesn't
2. **Container Isolation**: Audio devices accessible, but PyAudio can't initialize
3. **Permission Issues**: Not the problem (audio group working)
4. **Library Dependencies**: ALSA libraries accessible, but PyAudio still fails

### **Docker Audio Best Practices Discovered**:
1. **Device Mounting**: Essential for audio access
2. **Library Mounting**: Required for ALSA functionality
3. **Group Permissions**: Audio group needed
4. **PyAudio Installation**: System packages often work better than pip in containers

## 📊 **SUCCESS METRICS**

### **Current Progress**: 75% Complete
- ✅ Infrastructure: 100%
- ✅ Docker Setup: 100%
- ✅ ALSA Configuration: 100%
- 🔄 PyAudio Integration: 25%
- ⏳ Audio Recording: 0%
- ⏳ Wake Word Detection: 0%

### **Next Milestone**: PyAudio Device Detection
**Goal**: PyAudio should detect the SH-04 microphone in the container
**Success Criteria**: `python -c "import pyaudio; p = pyaudio.PyAudio(); print(p.get_device_count())"` returns > 0

---

*This document tracks our systematic approach to fixing USB microphone access in Docker on Raspberry Pi. Each step builds on the previous, ensuring we understand and resolve each issue before moving to the next.*
