# pi-wakeword-streamer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)

> **Phase 1a of the ORAC Voice-Control Architecture** - A high-performance wake-word detection and audio streaming service designed for Raspberry Pi deployment.

## 🎯 Overview

`pi-wakeword-streamer` is a lightweight, containerized service that continuously monitors audio input for the wake-word e.g. "Hey ORAC" or "Hey Computer" and streams detected audio clips to a remote processing endpoint. Built for ultra-low latency (<150ms) and designed to run efficiently on Raspberry Pi hardware.

### Key Features

- **Ultra-fast wake-word detection** using Porcupine engine
- **Configurable audio buffering** with pre/post-roll capture
- **Docker containerization** for easy deployment
- **Automated testing framework** with sample audio validation
- **Real-time performance monitoring** and latency tracking
- **SSH-based deployment automation** with `deploy_and_test.sh`

### Target Performance

- ⚡ **Wake-word detection**: < 10ms latency
- 🚀 **Audio streaming**: < 150ms end-to-end
- 💾 **Memory usage**: < 200MB
- 🔋 **CPU usage**: < 30% on Pi 4
- 🎯 **Detection accuracy**: > 95%
- 🚫 **False positives**: < 1%

## 🏗️ Architecture

```
USB Microphone → Raspberry Pi (wake-word detection)
                      ↓
                Audio Buffer (1s pre-roll)
                      ↓
                HTTP POST to Jetson Orin Nano
                      ↓
                Remote Speech Processing
```

## 🚀 Quick Start

### Prerequisites

- **Raspberry Pi 4** (recommended) or Pi 3B+
- **USB microphone** with 16kHz mono support
- **Docker & Docker Compose** installed on Pi
- **SSH access** to Pi from development machine
- **Picovoice account** for wake-word model generation

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/2oby/hey-orac.git
cd hey-orac
   ```

2. **Generate Porcupine wake-word model**
   - Create account at [Picovoice Console](https://console.picovoice.ai/)
   - Generate custom wake-word "ORAC"
   - Download `.ppn` file to `models/porcupine/orac.ppn`

3. **Configure your setup**
   ```bash
   # Edit configuration
   nano src/config.yaml
   ```

4. **Deploy to Raspberry Pi**
   ```bash
   # Make deployment script executable
   chmod +x scripts/deploy_and_test.sh
   
   # Deploy and test
   ./scripts/deploy_and_test.sh
   ```

### Configuration

Key settings in `src/config.yaml`:

```yaml
# Audio settings
mic_index: 0                    # USB microphone device index
sample_rate: 16000             # Audio sample rate
sensitivity: 0.6               # Wake-word detection sensitivity

# Buffer settings
preroll_seconds: 1.0           # Audio captured before wake-word
postroll_seconds: 2.0          # Audio captured after wake-word

# Network settings
jetson_endpoint: http://jetson-orin:8000/speech
```

## 🧪 Testing

### Local Development
```bash
# Run unit tests
python -m pytest tests/ -v

# Test with sample audio
python src/main.py --test-audio tests/sample_orac.wav

# Build and test Docker container
docker build -t pi-wakeword-streamer .
docker run --rm pi-wakeword-streamer --help
```

### Pi Testing
```bash
# SSH to Pi and run tests
ssh pi@your-pi-ip "cd hey-orac && python -m pytest tests/ -v"

# Test with real microphone
ssh pi@your-pi-ip "cd hey-orac && docker-compose up"
# Then speak "ORAC" into the microphone
```

## 📁 Project Structure

```
hey-orac/
├── docker/                 # Docker configuration
│   ├── Dockerfile         # Container definition
│   ├── docker-compose.yml # Service orchestration
│   └── entrypoint.sh      # Startup script
├── src/                   # Application source
│   ├── main.py           # Main application logic
│   ├── audio_buffer.py   # Ring buffer utility
│   └── config.yaml       # Configuration
├── tests/                 # Test files
│   ├── sample_orac.wav   # Test audio
│   └── test_wakeword.py  # Test automation
├── models/                # ML models
│   └── porcupine/        # Wake-word models
├── scripts/               # Deployment scripts
│   └── deploy_and_test.sh # Main deployment script
├── docs/                  # Documentation
│   ├── CRITICAL_PATH.txt # Development roadmap
│   ├── DEV_LOG.txt       # Progress tracking
│   └── INSTRUCTIONS.txt  # Detailed setup guide
└── README.md             # This file
```

## 🔧 Development

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- SSH key authentication with Pi
- Picovoice account

### Development Workflow
1. **Make changes** in development environment
2. **Test locally** with unit tests and mock audio
3. **Commit and push** to GitHub
4. **Deploy to Pi** using `deploy_and_test.sh`
5. **Verify functionality** on Pi with real microphone
6. **Update DEV_LOG.txt** with progress

See [INSTRUCTIONS.txt](docs/INSTRUCTIONS.txt) for detailed development setup and troubleshooting.

## 🛠️ Troubleshooting

### Common Issues

**Microphone not detected**
```bash
# List audio devices on Pi
arecord -l
# Update mic_index in config.yaml
```

**Wake-word not detecting**
- Check sensitivity setting (0.5-0.8 recommended)
- Verify microphone: `arecord -D hw:1,0 -f S16_LE -r 16000 -c 1 test.wav`
- Test with sample audio file

**Docker permission denied**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

For more detailed troubleshooting, see [INSTRUCTIONS.txt](docs/INSTRUCTIONS.txt).

## 📊 Performance Monitoring

Monitor system performance on Pi:
```bash
# CPU usage
htop

# Memory usage
free -h

# Audio device status
arecord -l

# Container logs
docker-compose logs -f
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

This is Phase 1a of the **ORAC Voice-Control Architecture**. Future phases include:

- **Phase 1b**: Jetson STT service implementation
- **Phase 2**: LLM intent processing and Home Assistant bridge
- **Phase 3**: Security, undo functionality, and advanced features

## 📚 Documentation

- **[INSTRUCTIONS.txt](docs/INSTRUCTIONS.txt)** - Comprehensive setup and deployment guide
- **[CRITICAL_PATH.txt](docs/CRITICAL_PATH.txt)** - Development roadmap and milestones
- **[DEV_LOG.txt](docs/DEV_LOG.txt)** - Progress tracking and issue resolution

## 🆘 Support

- 📖 Check [INSTRUCTIONS.txt](docs/INSTRUCTIONS.txt) for detailed setup procedures
- 🐛 Create [GitHub Issues](https://github.com/2oby/hey-orac/issues) for bugs
- 💡 Review [DEV_LOG.txt](docs/DEV_LOG.txt) for known issues and solutions

---

**Built with ❤️ for the ORAC Voice-Control Architecture** 