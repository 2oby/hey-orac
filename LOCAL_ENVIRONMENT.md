# Local Environment Configuration

## Current Implementation Note
**IMPORTANT**: The system is currently using `main_new.py` (minimal implementation) instead of `main.py`. 
The startup script runs `main_new.py` which integrates:
- `audio_pipeline_new.py` - Pure audio processing with RMS monitoring
- `wake_word_monitor_new.py` - Configuration-driven model management
- `web_backend.py` - REST API for settings and model management

This provides a modular architecture with proper separation of concerns.

## Deploy and Test Script Usage
The main deployment script automates the entire deployment process to the Raspberry Pi:


# Deploy with a descriptive commit message
./scripts/deploy_and_test.sh "Your descriptive commit message here"
```

### What the Script Does
1. **Commits and pushes** your changes to GitHub
2. **SSH to Pi** and pulls the latest code
3. **Builds and starts** Docker containers
4. **Runs automated tests** on the Pi
5. **Reports results** back to your development machine

### Example Usage
```bash
# Deploy a new feature
./scripts/deploy_and_test.sh "Add OpenWakeWord engine support"


## SSH Configuration

### Pi SSH Alias
The Raspberry Pi is configured with an SSH alias in `~/.ssh/config`:

```
Host pi
  HostName 192.168.8.99
  User 2oby
  IdentityFile ~/.ssh/id_ed25519
  ServerAliveInterval 60
```

### Connection Test
```bash
ssh pi "whoami && pwd && hostname"
# Output: 2oby, /home/2oby, niederpi
```

## Pi Environment Details

### System Information
- **Hostname**: niederpi
- **User**: 2oby
- **IP Address**: 192.168.8.99
- **Home Directory**: /home/2oby

### Docker Environment
- **Docker Version**: 20.10.24+dfsg1, build 297e128
- **Docker Compose Version**: 1.29.2, build unknown

### Quick Commands
```bash
# SSH to Pi
ssh pi

e.g.
# Check Docker containers
ssh pi "docker ps -a"


## Project Deployment Path
The hey-orac project will be deployed to:
```
/home/2oby/hey-orac/
```

## Audio Hardware

### USB Microphone
- **Device**: SH-04 USB Audio
- **Card Index**: 0
- **ALSA Name**: SH-04: USB Audio (hw:0,0)
- **Status**: Detected and working
- **Notes**: MV SH-04 at usb-xhci-hcd.1-2, full speed
- **Volume**: Currently set to 95% (3780/3996) with 14.75dB gain
- **Test Results**: Audio levels vary - sometimes very quiet (RMS: 0.000177), sometimes good (RMS: 0.033690)


## Notes
- This file contains local environment details and should NOT be committed to git
- Update this file as environment changes occur
- Keep sensitive information like IP addresses and credentials here instead of in public documentation 




------ MOST IMPORTANT UNDERSTAND SSH PI and DEPLOY AND TEST


## Project Overview
This is Phase 1a of the ORAC Voice-Control Architecture - a wake-word detection and audio streaming service designed to run on a Raspberry Pi and stream audio clips to a Jetson Orin Nano for processing.

## Repository Structure
```
Hey_Orac/
├── docker/                 # Docker configuration files
│   └── Dockerfile         # Container definition
├── src/                   # Application source code
│   ├── main.py           # Main application logic
│   ├── audio_buffer.py   # Ring buffer utility
│   ├── audio_pipeline.py # Audio processing pipeline
│   ├── wake_word_engines/ # Wake word detection engines
│   │   ├── openwakeword_engine.py
│   │   ├── porcupine_engine.py
│   │   └── test_engine.py
│   └── config.yaml       # Configuration file
├── third_party/           # Third-party models and libraries
│   ├── openwakeword/     # OpenWakeWord models
│   │   └── custom_models/ # Custom wake word models
│   │       ├── Hey_computer.onnx
│   │       ├── Hey_computer.tflite
│   │       ├── Hay--compUta_v_lrg.onnx
│   │       ├── Hay--compUta_v_lrg.tflite
│   │       ├── hey-CompUter_lrg.onnx
│   │       └── hey-CompUter_lrg.tflite
│   └── porcupine/        # Porcupine models
│       └── custom_models/ # Custom Porcupine models
│           └── Hey-Uruk_en_raspberry-pi_v3_0_0.ppn
├── tests/                 # Test files and data
│   └── test_wakeword.py  # Test automation
├── scripts/               # Deployment and utility scripts
│   ├── deploy_and_test.sh # Main deployment script
│   ├── debug_audio.sh    # Audio debugging script
│   └── monitor_detections.sh # Detection monitoring
├── docs/                  # Documentation
│   ├── CRITICAL_PATH.txt # Development roadmap
│   ├── DEV_LOG.txt       # Progress tracking
│   └── ProjectOutline.md # Project overview
├── web/                   # Web interface
│   ├── index.html        # Web dashboard
│   ├── css/              # Stylesheets
│   └── js/               # JavaScript files
├── assets/                # Static assets
│   ├── audio/            # Audio files
│   └── images/           # Images and icons
├── logs/                  # Application logs
├── docker-compose.yml     # Service orchestration
└── README.md             # Project overview



### Development Machine
- Mac
- Python 3.12+
- Docker and Docker Compose
- Git
- SSH key pair for Pi access
- Picovoice account (for Porcupine model generation)

### Raspberry Pi
- Raspberry Pi 4 (recommended) or Pi 3B+
- USB microphone
- Docker and Docker Compose
- SSH access enabled
- Internet connection for package installation



