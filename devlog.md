# OpenWakeWord Test Development Log

## 2025-07-15 18:30 - Initial Project Setup
- Created project structure for minimal OpenWakeWord test on Raspberry Pi
- Set up requirements.txt with OpenWakeWord==0.6.0 and audio dependencies
- Created Dockerfile for Raspberry Pi with ALSA audio support
- Created docker-compose.yml with audio device access and privileged mode
- Updated deploy_and_test.sh script for WakeWordTest project

## 2025-07-15 18:35 - Git Repository Setup
- Initialized git repository in WakeWordTest directory
- Created wake-word-test branch in hey-orac repository
- Successfully pushed initial implementation to GitHub
- Fixed SSH authentication issues and used stored credentials

## 2025-07-15 18:45 - First Deployment Attempt
- Successfully deployed to Raspberry Pi via deploy script
- Container built successfully and detected USB microphone (SH-04)
- Identified NumPy compatibility issue with OpenWakeWord/tflite-runtime
- Fixed NumPy version constraint to <2.0.0,>=1.21.0

## 2025-07-15 18:50 - Successful Container Launch
- Container now starts successfully with correct dependencies
- USB microphone properly detected: SH-04: USB Audio (hw:2,0)
- Audio stream creation successful with 16kHz, mono audio
- OpenWakeWord models loaded: alexa, hey_mycroft, hey_jarvis, hey_rhasspy, timer, weather

## 2025-07-15 18:55 - Detection Loop Issue Identified
- Container running but wake word detection loop not starting
- Script hangs after "OpenWakeWord model initialized" message
- Added debug logging and lowered detection threshold to 0.3 for testing
- Attempted various fixes for logging buffering issues

## 2025-07-15 19:00 - Debugging Efforts
- Confirmed Model() creation works fine when tested directly (0.23s)
- Issue appears to be script hanging after model initialization
- Tried unbuffered output, forced log flushing, and stream testing
- Current status: Container healthy, audio accessible, but main detection loop not executing

## Current State
- ✅ Docker container building and running
- ✅ USB microphone detection and audio stream creation
- ✅ OpenWakeWord model loading (6 models available)
- ❌ Main wake word detection loop not executing (hangs after model init)
- 🔍 Next steps: Investigate why script execution stops after model initialization