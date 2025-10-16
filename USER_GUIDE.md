# Hey ORAC User Guide

**Version**: 1.0
**Last Updated**: 2025-10-16
**Target Audience**: End Users, System Administrators

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Using the Web Interface](#using-the-web-interface)
4. [Configuration Guide](#configuration-guide)
5. [Managing Wake Word Models](#managing-wake-word-models)
6. [Monitoring & Status](#monitoring--status)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Configuration](#advanced-configuration)
9. [FAQ](#faq)

---

## Introduction

### What is Hey ORAC?

Hey ORAC is a voice-activated wake word detection system that runs on a Raspberry Pi. It continuously listens for custom wake words (like "Hey ORAC" or "Hey Jarvis"), captures your voice command after detection, and sends it to a configured API endpoint for processing.

### Key Features

**Multiple Wake Words**
- Support for multiple wake word models simultaneously
- Enable/disable individual models on the fly
- Works with both ONNX and TensorFlow Lite models
- Compatible with OpenWakeWord default and custom models

**Real-Time Monitoring**
- Live audio level display (updates 5 times per second)
- Visual feedback when wake words are detected
- System status indicators (audio, STT connection, listening state)
- WebSocket-based updates for minimal delay

**Easy Configuration**
- Web-based interface accessible from any device on your network
- Per-model settings for threshold and detection behavior
- Custom API endpoint URLs for each wake word
- All settings persist through restarts
- Hot configuration reload (no restart required)

**Smart Audio Capture**
- Automatically captures at least 2 seconds after wake word detection
- Continues recording until you stop speaking (silence detection)
- Includes 1 second of audio before the wake word (pre-roll)
- Maximum 15-second recording as a safety limit

**Speech-to-Text Integration**
- Optional integration with ORAC STT service
- Automatic transcription of captured voice commands
- Per-model STT enable/disable
- Health monitoring for STT endpoints

### System Requirements

**Hardware:**
- Raspberry Pi 4 (2GB RAM minimum, 4GB recommended)
- USB microphone or compatible audio input device
- MicroSD card (8GB minimum, 16GB+ recommended)
- Power supply for Raspberry Pi (5V/3A recommended)
- Network connection (WiFi or Ethernet)

**Network:**
- Access to configured API endpoints
- Port 7171 available for web interface
- (Optional) Access to ORAC STT service

**Software:**
- Docker installed on Raspberry Pi
- Raspberry Pi OS (Bullseye or later)

---

## Getting Started

### Quick Start

If Hey ORAC is already installed on your Raspberry Pi:

1. **Find your Raspberry Pi's IP address:**
   ```bash
   # On the Pi itself:
   hostname -I
   ```

2. **Open web browser:**
   - Navigate to `http://[your-pi-ip]:7171`
   - Example: `http://192.168.1.100:7171`

3. **You should see:**
   - Audio level meter (RMS bar)
   - List of available wake word models
   - System status indicators

4. **Test wake word detection:**
   - Say one of the enabled wake words clearly
   - Watch for red flash on the model card
   - Check logs if needed

### Installation (For Administrators)

If setting up Hey ORAC for the first time:

**Step 1: SSH to Raspberry Pi**
```bash
ssh pi@your-pi-ip
```

**Step 2: Clone Repository**
```bash
cd ~
git clone https://github.com/your-org/hey-orac.git
cd hey-orac
```

**Step 3: Configure Settings**
```bash
# Copy template configuration
cp config/settings.json.template config/settings.json

# Edit if needed
nano config/settings.json
```

**Step 4: Start Docker Container**
```bash
docker-compose up -d
```

**Step 5: Check Status**
```bash
docker-compose ps
docker-compose logs -f hey-orac
```

**Step 6: Access Web Interface**
- Open browser to `http://[pi-ip]:7171`

---

## Using the Web Interface

### Dashboard Overview

The Hey ORAC web interface provides real-time monitoring and configuration:

```
┌─────────────────────────────────────────────┐
│           Hey ORAC Dashboard                │
├─────────────────────────────────────────────┤
│  Audio Level: [████████░░░░░░░░░░] 0.0234  │ ← RMS Meter (live)
│  Status: ● Connected  ● Listening  ● STT OK│
├─────────────────────────────────────────────┤
│  ┌─────────────────────────────────┐        │
│  │ hey_orac              [Enabled] │        │ ← Model Card
│  │ Threshold: 0.5      ──────●──── │        │
│  │ Webhook: http://orac-stt:8080  │        │
│  │ [Configure]                     │        │
│  └─────────────────────────────────┘        │
│  ┌─────────────────────────────────┐        │
│  │ hey_jarvis           [Disabled] │        │
│  │ Threshold: 0.4      ──────●──── │        │
│  │ Webhook: (not configured)       │        │
│  │ [Configure]                     │        │
│  └─────────────────────────────────┘        │
│                                             │
│  [Global Settings]                          │
└─────────────────────────────────────────────┘
```

### UI Components

**Audio Level Meter**
- Real-time audio level visualization
- Updates 5 times per second
- Helps verify microphone is working
- Green bars indicate audio activity

**Status Indicators**
- **Connected** (blue): WebSocket connection to server
- **Listening** (green): Actively processing audio
- **STT** (green/orange/red): Speech-to-text service health
  - Green: All endpoints healthy
  - Orange: Some endpoints healthy
  - Red: No endpoints healthy

**Model Cards**
- One card per wake word model
- Enable/disable toggle
- Threshold slider
- Webhook URL field
- Visual detection feedback (red flash on detection)

**Global Settings**
- Multi-trigger mode toggle
- VAD threshold slider
- Cooldown period setting
- RMS filter threshold

---

## Configuration Guide

### Per-Model Settings

Each wake word model has individual settings:

#### Enable/Disable Toggle
- **Purpose**: Activate or deactivate the model
- **Effect**: Disabled models are unloaded from memory
- **Use Case**: Temporarily disable unused wake words

#### Threshold Slider (0.0 - 1.0)
- **Purpose**: Set detection confidence required
- **Lower values (0.3-0.4)**: More sensitive, may have false positives
- **Medium values (0.5)**: Balanced (default)
- **Higher values (0.6-0.8)**: Less sensitive, fewer false positives
- **Tip**: Start at 0.5 and adjust based on experience

#### Webhook URL
- **Purpose**: API endpoint to receive captured audio
- **Format**: `http://hostname:port/path`
- **Example**: `http://orac-stt:8080/stt/v1/stream`
- **Required**: Must be set for model to send audio
- **Per-Model**: Each model can have different endpoint

#### STT Enabled (if configured)
- **Purpose**: Enable speech-to-text transcription
- **Effect**: Captured audio sent to STT service
- **Requires**: Valid webhook URL and STT service running

### Global Settings

#### Multi-Trigger Mode
- **Default**: Off (single-trigger mode)
- **Single-Trigger**: Only highest-confidence model triggers
- **Multi-Trigger**: All models above threshold trigger simultaneously
- **Use Case**: Enable if you want multiple wake words active at once

#### VAD Threshold (0.0 - 1.0)
- **Purpose**: Voice Activity Detection sensitivity
- **Default**: 0.5
- **Lower**: More sensitive to quiet sounds
- **Higher**: Only responds to louder audio
- **Note**: This is a global OpenWakeWord parameter

#### Cooldown Period (0-5 seconds)
- **Purpose**: Minimum time between consecutive detections
- **Default**: 2.0 seconds
- **Effect**: Prevents rapid repeated triggers
- **Use Case**: Increase if getting duplicate detections

#### RMS Filter (0-100)
- **Purpose**: Minimum audio level threshold
- **Default**: 50
- **Effect**: Filters out background noise
- **Tip**: Adjust based on your environment

### Saving Changes

**Important**: Changes are not applied immediately!

1. Make any adjustments to settings
2. A "Save" button appears at the top
3. Click "Save" to apply changes
4. System reloads configuration (takes 1-2 seconds)
5. "Save" button disappears when complete

**Hot Reload**: Configuration changes are applied without restarting the container.

### Configuration File

Settings are stored in `/config/settings.json` inside the container.

**On Host System**: `~/hey-orac/config/settings.json`

**Manual Editing** (advanced users):
```bash
# Stop container
docker-compose down

# Edit configuration
nano ~/hey-orac/config/settings.json

# Start container
docker-compose up -d
```

**Validation**: Invalid JSON will prevent container startup. Check logs:
```bash
docker-compose logs hey-orac
```

---

## Managing Wake Word Models

### Available Models

Hey ORAC includes several default OpenWakeWord models:

- **hey_orac** - Custom trained model
- **hey_jarvis** - Popular voice assistant wake word
- **alexa** - Amazon Alexa wake word
- **hey_mycroft** - Mycroft AI wake word
- **ok_nabu** - Alternative wake word

### Enabling/Disabling Models

1. **Find model card** in web interface
2. **Toggle switch** to enable or disable
3. **Click "Save"** to apply changes
4. **Verify** model loads successfully in logs

**Memory Management**: Disabled models are unloaded to conserve RAM. Only enable models you plan to use.

### Model File Formats

**Supported Formats:**
- **ONNX** (`.onnx`) - Open Neural Network Exchange format
- **TFLite** (`.tflite`) - TensorFlow Lite format

**Performance**: ONNX typically performs better on Raspberry Pi 4.

### Adding Custom Models

**Step 1: Train or Obtain Model**
- Use OpenWakeWord training tools
- Obtain pre-trained compatible model
- Ensure model is ONNX or TFLite format

**Step 2: Copy to Pi**
```bash
# From your computer:
scp my_custom_model.onnx pi@pi-ip:~/hey-orac/models/custom/
```

**Step 3: Add to Configuration**
```bash
# SSH to Pi
ssh pi@pi-ip

# Edit settings
nano ~/hey-orac/config/settings.json

# Add new model entry:
{
  "name": "my_custom",
  "enabled": true,
  "threshold": 0.5,
  "path": "/models/custom/my_custom_model.onnx",
  "topic": "wake/my_custom",
  "webhook_url": "http://your-endpoint:8080/api",
  "stt_enabled": true
}
```

**Step 4: Restart Container**
```bash
docker-compose restart hey-orac
```

**Step 5: Verify**
- Check web interface shows new model
- Test by speaking your custom wake word
- Monitor logs for detection events

### Troubleshooting Model Loading

**Model Not Appearing:**
- Check file path is correct
- Verify file exists in models directory
- Check JSON syntax in settings.json
- Review logs for errors

**Model Not Detecting:**
- Lower threshold value (try 0.3-0.4)
- Increase microphone volume
- Test with different pronunciations
- Check audio level meter shows activity

**High CPU Usage:**
- Limit enabled models to 2-3 maximum
- Consider using quantized models
- Disable unused models
- Check model file size

---

## Monitoring & Status

### Real-Time Monitoring

**Audio Level Meter:**
- Shows live microphone input level
- Updates 5 times per second
- Normal speech: ~0.01-0.10 RMS
- Loud speech: ~0.10-0.30 RMS
- Background noise: <0.01 RMS

**Detection Events:**
- Model card flashes red on detection
- Duration: ~1 second
- Indicates wake word was detected
- Audio is being captured and sent

**System Status:**
- **Connected**: WebSocket connection active
- **Listening**: Audio processing active
- **STT Health**: Speech-to-text service status

### Checking Logs

**From Web Interface:**
- (Future feature: logs viewer)

**From Command Line:**
```bash
# View recent logs
docker-compose logs --tail=50 hey-orac

# Follow logs in real-time
docker-compose logs -f hey-orac

# Search for errors
docker-compose logs hey-orac | grep "❌"

# Search for detections
docker-compose logs hey-orac | grep "🎯"
```

### Understanding Log Messages

**Common Emoji Indicators:**
- 🎤 Audio operations
- 🎯 Wake word detected
- ✅ Success
- ❌ Error
- 📊 Statistics
- 🔄 Reload/restart
- 🏥 Health check
- 📞 Network call

**Key Log Patterns:**

**Successful Detection:**
```
🎯 WAKE WORD DETECTED! Confidence: 0.8234 (threshold: 0.5) - Source: hey_orac
✅ Webhook call successful
🎤 Starting STT recording for wake word 'hey_orac'
```

**Audio Processing:**
```
📊 Processed 100 audio chunks
   Audio data shape: (1280,), volume: 0.0142, RMS: 0.0089
```

**Configuration Change:**
```
🔄 Reloading models due to configuration change...
✅ Models reloaded successfully
```

**Error:**
```
❌ Webhook call failed: Connection timeout
⚠️ STT unhealthy for model 'hey_orac'
```

### Health Checks

**Automatic Monitoring:**
- Audio thread health checked every 5 seconds
- STT endpoint health checked every 30 seconds
- Configuration validated on load

**Manual Health Check:**
```bash
curl http://pi-ip:7171/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "is_listening": true,
  "stt_health": "connected",
  "loaded_models": ["hey_orac", "hey_jarvis"]
}
```

---

## Troubleshooting

### Audio Issues

**No Audio Input Detected**

**Symptoms**: RMS meter shows 0.0000, no audio activity

**Solutions:**
1. Check microphone is connected:
   ```bash
   ssh pi@pi-ip "arecord -l"
   ```
2. Verify Docker has audio device access:
   ```bash
   docker exec hey-orac arecord -l
   ```
3. Test microphone outside Docker:
   ```bash
   arecord -d 5 test.wav
   aplay test.wav
   ```
4. Check Docker compose includes device mapping:
   ```yaml
   devices:
     - /dev/snd:/dev/snd
   ```
5. Restart container:
   ```bash
   docker-compose restart hey-orac
   ```

**Audio Level Too Low**

**Symptoms**: RMS <0.001, detection not working

**Solutions:**
1. Increase microphone volume:
   ```bash
   alsamixer
   # Press F6 to select USB device
   # Increase volume with up arrow
   ```
2. Position microphone closer (1-3 feet)
3. Lower RMS filter threshold in settings
4. Test in quieter environment
5. Try different USB microphone

**Audio Level Too High / Distorted**

**Symptoms**: RMS constantly >0.30, clipping

**Solutions:**
1. Reduce microphone gain
2. Position microphone farther away
3. Adjust microphone input level:
   ```bash
   alsamixer
   ```
4. Increase RMS filter threshold

### Detection Issues

**Wake Word Not Detecting**

**Symptoms**: Speaking wake word, no detection event

**Checklist:**
- [ ] Model is enabled (check web UI)
- [ ] Audio level shows activity (RMS >0.01)
- [ ] Threshold not too high (try 0.3-0.4)
- [ ] Pronunciation matches training data
- [ ] Background noise not too high
- [ ] Microphone working (test with recording)

**Solutions:**
1. Lower threshold to 0.3:
   - Open web interface
   - Move threshold slider left
   - Click "Save"
   - Test again

2. Check logs for confidence scores:
   ```bash
   docker-compose logs -f hey-orac | grep "🎯"
   ```

3. Verify model loaded:
   ```bash
   docker-compose logs hey-orac | grep "✅ Loading model"
   ```

4. Test with different pronunciation
5. Increase VAD threshold if too sensitive to noise

**Too Many False Positives**

**Symptoms**: Detecting wake word when not spoken

**Solutions:**
1. Increase threshold (try 0.6-0.7)
2. Increase cooldown period (3-5 seconds)
3. Increase VAD threshold
4. Reduce background noise
5. Adjust RMS filter threshold

**Multiple Detections for Single Wake Word**

**Symptoms**: Same wake word triggers multiple times

**Solutions:**
1. Increase cooldown period:
   - Open Global Settings
   - Set cooldown to 3-5 seconds
   - Click "Save"

2. Increase threshold slightly
3. Disable multi-trigger mode if enabled

### Web Interface Issues

**Cannot Access Web Interface**

**Symptoms**: Browser shows "Connection refused" or timeout

**Solutions:**
1. Verify container is running:
   ```bash
   docker ps | grep hey-orac
   ```

2. Check port 7171 is exposed:
   ```bash
   docker port hey-orac
   ```

3. Verify network connectivity:
   ```bash
   ping pi-ip
   ```

4. Check firewall rules:
   ```bash
   sudo ufw status
   # If blocked, allow port 7171:
   sudo ufw allow 7171
   ```

5. Restart container:
   ```bash
   docker-compose restart hey-orac
   ```

**WebSocket Disconnecting**

**Symptoms**: "Connected" indicator flickering, status not updating

**Solutions:**
1. Check network stability
2. Review logs for errors:
   ```bash
   docker-compose logs hey-orac | grep WebSocket
   ```
3. Refresh browser page
4. Clear browser cache
5. Try different browser

**Changes Not Saving**

**Symptoms**: Click "Save" but settings revert

**Solutions:**
1. Check logs for errors:
   ```bash
   docker-compose logs hey-orac | grep "❌"
   ```
2. Verify configuration file is writable:
   ```bash
   ls -l ~/hey-orac/config/settings.json
   ```
3. Validate JSON syntax:
   ```bash
   cat ~/hey-orac/config/settings.json | jq .
   ```
4. Check disk space:
   ```bash
   df -h
   ```

### STT Integration Issues

**STT Health Shows "Disconnected"**

**Symptoms**: Red STT indicator in status bar

**Solutions:**
1. Verify STT service is running:
   ```bash
   curl http://orac-stt:8080/health
   ```

2. Check webhook URL is correct in settings
3. Test network connectivity to STT service
4. Review logs for STT errors:
   ```bash
   docker-compose logs hey-orac | grep STT
   ```

**Audio Not Reaching STT Service**

**Symptoms**: Detection works but no transcription

**Checklist:**
- [ ] Webhook URL is configured
- [ ] STT enabled for model
- [ ] STT service is reachable
- [ ] STT health shows "connected"

**Solutions:**
1. Test webhook manually:
   ```bash
   curl -X POST http://your-stt-url/stt \
     -H "Content-Type: audio/wav" \
     --data-binary @test.wav
   ```

2. Check STT service logs
3. Verify audio format compatibility
4. Test with different wake word model

### Container Issues

**Container Fails to Start**

**Symptoms**: `docker ps` doesn't show hey-orac

**Solutions:**
1. Check logs:
   ```bash
   docker-compose logs hey-orac
   ```

2. Verify configuration file:
   ```bash
   cat ~/hey-orac/config/settings.json | jq .
   ```

3. Check Docker disk space:
   ```bash
   docker system df
   ```

4. Rebuild container:
   ```bash
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

**High CPU Usage**

**Symptoms**: Pi running hot, slow performance

**Solutions:**
1. Limit enabled models (max 2-3)
2. Check for runaway processes:
   ```bash
   docker stats hey-orac
   ```
3. Review logs for errors/loops
4. Consider using quantized models
5. Restart container

**High Memory Usage**

**Symptoms**: Out of memory errors

**Solutions:**
1. Disable unused models
2. Check memory usage:
   ```bash
   docker stats hey-orac
   ```
3. Clear model cache
4. Restart container
5. Upgrade Pi to 4GB+ RAM

---

## Advanced Configuration

### Environment Variables

Set in `docker-compose.yml`:

```yaml
environment:
  - LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
  - STREAM_TLS=0            # 1 to enable TLS for endpoints
  - PROMETHEUS_ENABLED=0    # 1 to enable metrics
```

### Custom Model Paths

Models can be placed in custom directories:

```json
{
  "name": "my_model",
  "path": "/models/custom/subdirectory/model.onnx"
}
```

### Multiple STT Endpoints

Each model can have its own STT endpoint:

```json
{
  "hey_orac": {
    "webhook_url": "http://orac-stt-1:8080/stt"
  },
  "hey_jarvis": {
    "webhook_url": "http://orac-stt-2:8080/stt"
  }
}
```

### Backup & Restore

**Backup Configuration:**
```bash
# Backup settings
cp ~/hey-orac/config/settings.json ~/settings.backup.json

# Backup custom models
tar -czf ~/hey-orac-models.tar.gz ~/hey-orac/models/custom/
```

**Restore Configuration:**
```bash
# Restore settings
cp ~/settings.backup.json ~/hey-orac/config/settings.json

# Restore models
tar -xzf ~/hey-orac-models.tar.gz -C ~/

# Restart container
docker-compose restart hey-orac
```

### Docker Compose Customization

Example `docker-compose.override.yml`:

```yaml
version: '3.8'
services:
  hey-orac:
    restart: unless-stopped
    environment:
      - LOG_LEVEL=DEBUG
    volumes:
      - ./logs:/logs
```

---

## FAQ

**Q: How many wake words can I run simultaneously?**
A: Typically 2-3 models on Raspberry Pi 4 (2GB). More models increase CPU usage and latency.

**Q: Can I train my own wake word models?**
A: Yes! Use OpenWakeWord training tools: https://github.com/dscripka/openWakeWord

**Q: Does this work offline?**
A: Wake word detection works offline. STT integration requires network access to STT service.

**Q: What audio format is sent to the endpoint?**
A: WAV format, 16kHz, 16-bit, mono, PCM encoding.

**Q: How much audio is captured?**
A: Minimum 2 seconds, continues until silence detected, maximum 15 seconds.

**Q: Can I use this with Alexa/Google Assistant?**
A: Not directly. Hey ORAC sends audio to your own configured endpoints.

**Q: What's the detection latency?**
A: Typically 200-500ms from wake word end to detection event.

**Q: Does it record continuously?**
A: It listens continuously but only captures/saves audio after wake word detection.

**Q: Can I use a Raspberry Pi 3?**
A: Possible but not recommended. Performance will be significantly slower.

**Q: What USB microphones are compatible?**
A: Most USB microphones work. Tested with Blue Snowball iCE, SH-04 USB Audio.

---

## Getting Help

**Check Logs First:**
```bash
docker-compose logs --tail=100 hey-orac
```

**Common Error Patterns:**
- 🎤 ❌ Audio issues
- 🎯 ❌ Detection issues
- 📞 ❌ Network/webhook issues
- ⚠️ Configuration warnings

**Resources:**
- Developer Guide: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- STT API Reference: [STT_API_REFERENCE.md](STT_API_REFERENCE.md)
- GitHub Issues: [Report a problem](https://github.com/your-org/hey-orac/issues)

**Community:**
- Discord: (link if available)
- Forum: (link if available)

---

**Last Updated**: 2025-10-16
**Questions?** Open an issue on GitHub
