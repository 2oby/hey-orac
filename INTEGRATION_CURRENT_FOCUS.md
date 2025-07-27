# Hey_Orac Integration Current Focus

**Date**: July 27, 2025  
**Status**: Wake word detection working, STT integration needs configuration fixes  
**Priority**: HIGH - Core integration functionality partially working

## Current Issue Summary

### ✅ What's Working
- **Hey_Orac wake word detection**: Working perfectly (99.7-99.8% confidence)
- **Wake word models**: Successfully detecting "computer" wake word
- **ORAC STT service**: Running and healthy (whisper-tiny on CUDA)
- **Network connectivity**: Both services can communicate
- **Service health**: Both containers operational
- **Enhanced logging**: Comprehensive debug logging implemented

### ❌ What's Not Working
- **STT triggering**: Wake words detected but audio not sent to STT service
- **Configuration issue**: `stt_enabled` is per-model but should be global
- **Model configuration**: Some models missing `stt_enabled` field entirely

### 🔍 Evidence from Testing

**Wake word detection logs showed:**
```
🎯 WAKE WORD DETECTED! Confidence: 0.984698 (threshold: 0.300000) - Source: computer_v2
🎯 WAKE WORD DETECTED! Confidence: 0.997383 (threshold: 0.300000) - Source: computer_v2
🎯 WAKE WORD DETECTED! Confidence: 0.980022 (threshold: 0.300000) - Source: computer_v2
🎯 WAKE WORD DETECTED! Confidence: 0.997883 (threshold: 0.300000) - Source: computer_v2
```

**However:**
- No "Triggering STT recording" logs
- No "Starting speech recording" logs
- No audio sent to ORAC STT service
- No transcriptions logged

## Root Cause Analysis

### 1. Configuration Design Issue
The current implementation expects `stt_enabled` to be a per-model setting, but the actual requirement is:
- STT should be **globally enabled** in the configuration
- Each model should have its own **webhook URL** configured via GUI
- When ANY wake word is detected, audio should be sent to that model's configured URL

### 2. Missing Configuration Fields
The `computer_v2` model configuration lacks the `stt_enabled` field:
```json
{
  "name": "computer_v2",
  "path": "/app/models/openwakeword/computer_v2.tflite",
  "framework": "tflite",
  "enabled": true,
  "threshold": 0.3,
  "webhook_url": "",
  "priority": 6
  // Missing: "stt_enabled": true
}
```

### 3. Configuration Mismatch
The code expects STT to be configured per-model, but the business logic should be:
1. Global STT enable/disable setting
2. Per-model webhook URLs (configured via GUI)
3. Audio sent to the webhook URL when that specific model triggers

## Proposed Solution

### Short-term Fix (Immediate)
1. Add `stt_enabled: true` to all active wake word models
2. Configure the webhook URL to point to ORAC STT service

### Long-term Fix (Proper Implementation)
1. Refactor STT configuration to be global
2. Use model's `webhook_url` field for STT endpoint
3. Remove per-model `stt_enabled` field
4. Update GUI to allow webhook URL configuration per model

## Configuration Structure Issues

### Current (Problematic):
```json
{
  "stt": {
    "enabled": true,              // Global setting
    "base_url": "http://...",     // Global STT URL
    "enable_per_model": true      // Confusing flag
  },
  "models": [
    {
      "name": "model_name",
      "stt_enabled": true,        // Per-model (inconsistent)
      "webhook_url": ""           // Should be used for STT
    }
  ]
}
```

### Proposed (Clear):
```json
{
  "stt": {
    "enabled": true,              // Global on/off
    "default_url": "http://...",  // Default if model has no webhook
    "timeout": 30,
    // ... other STT settings
  },
  "models": [
    {
      "name": "model_name",
      "webhook_url": "http://192.168.8.191:7272/stt/v1/stream"  // Model-specific endpoint
    }
  ]
}
```

## Next Steps

1. **Immediate**: Add `stt_enabled: true` to computer_v2 model configuration
2. **Test**: Verify audio is sent to STT after wake word detection
3. **Refactor**: Update code to use webhook_url instead of global STT URL
4. **Document**: Update configuration documentation to clarify the design

## Success Criteria

After fixes, we should see in logs:
1. 🎯 Wake word detected
2. 🎤 STT recording triggered
3. 📤 Audio sent to webhook URL (or default STT URL)
4. 📝 TRANSCRIPTION LOG with actual text
5. Complete flow: "computer" → [speech] → transcribed text in logs

## Technical Details

### Enhanced Logging Added
- `wake_word_detection.py`: STT initialization, detection details, recording triggers
- `speech_recorder.py`: Recording lifecycle, audio capture, endpoint detection
- `stt_client.py`: HTTP requests, responses, transcription results

### STT Service Details
- Endpoint: `http://192.168.8.191:7272/stt/v1/stream`
- Model: whisper-tiny
- Backend: whisper.cpp
- Device: CUDA
- Status: Healthy and ready

---

**Status**: Ready to implement immediate fix and test end-to-end flow  
**Owner**: Configuration update needed, then retest