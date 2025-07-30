# Hey_Orac Integration Current Focus

**Date**: July 30, 2025  
**Status**: STT integration code fixed, but audio still not streaming to ORAC STT  
**Priority**: HIGH - Core integration functionality implemented but not working end-to-end

## Current Issue Summary

### ✅ What's Working
- **Hey_Orac wake word detection**: Working perfectly (99.7-99.8% confidence)
- **Wake word models**: Successfully detecting "computer" wake word
- **ORAC STT service**: Running and healthy (whisper-tiny on CUDA)
- **Network connectivity**: Both services can communicate
- **Service health**: Both containers operational
- **Enhanced logging**: Comprehensive debug logging implemented

### ✅ Recent Fixes Implemented (July 30)
1. **Always initialize STT components** - Removed global `stt.enabled` dependency
2. **Use per-model webhook URLs** - STT triggers based on webhook_url presence
3. **Dynamic URL support** - STT client accepts webhook URLs per request
4. **Per-model health checks** - Check each model's STT endpoint on startup

### ❌ What's Still Not Working
- **Audio not streaming**: Wake words detected but audio still not sent to ORAC STT
- **Missing webhook URLs**: Models don't have webhook_url configured
- **No transcriptions arriving**: ORAC STT not receiving any audio streams

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

## Root Cause Analysis (UPDATED)

### Previous Issues (NOW FIXED ✅)
1. ~~STT components only initialized if globally enabled~~ → **FIXED**: Always initialized
2. ~~Code checked `stt_enabled` per model~~ → **FIXED**: Now checks `webhook_url` presence
3. ~~STT client couldn't use dynamic URLs~~ → **FIXED**: Accepts webhook_url parameter

### Current Issue
The `computer_v2` model configuration has an **empty webhook_url**:
```json
{
  "name": "computer_v2",
  "path": "/app/models/openwakeword/computer_v2.tflite",
  "framework": "tflite",
  "enabled": true,
  "threshold": 0.3,
  "webhook_url": "",  // ← EMPTY! This prevents STT triggering
  "priority": 6
}
```

**The fix is simple**: Set the webhook_url to the ORAC STT endpoint!

## Proposed Solution

### Immediate Fix Required
Set the webhook_url for computer_v2 model:
```json
"webhook_url": "http://192.168.8.191:7272"
```

### What We've Already Fixed ✅
1. ~~Refactor STT configuration to be global~~ → **DONE**: Always initialized
2. ~~Use model's `webhook_url` field for STT endpoint~~ → **DONE**: Code updated
3. ~~Remove per-model `stt_enabled` field~~ → **DONE**: Now uses webhook_url
4. ~~Add dynamic URL support~~ → **DONE**: STT client accepts webhook URLs
5. ~~Add per-model health checks~~ → **DONE**: Checks on startup/reload

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

1. **Immediate**: Update computer_v2 model configuration to set webhook_url to "http://192.168.8.191:7272"
2. **Test**: Verify audio is sent to STT after wake word detection
3. **Monitor**: Check ORAC STT logs for incoming transcription requests
4. **Verify**: Confirm transcriptions appear in Hey ORAC logs

## Success Criteria

After fixes, we should see in logs:
1. 🎯 Wake word detected
2. 🎤 STT recording triggered
3. 📤 Audio sent to webhook URL (or default STT URL)
4. 📝 TRANSCRIPTION LOG with actual text
5. Complete flow: "computer" → [speech] → transcribed text in logs

## Technical Details

### Code Changes Implemented (July 30)
1. **wake_word_detection.py**:
   - Line 721: Removed `if stt_config.enabled` check - always initialize STT
   - Line 1129: Check `webhook_url` instead of `stt_enabled`
   - Line 1148: Pass webhook_url to speech_recorder
   - Lines 622-630, 827-835: Added per-model health checks

2. **stt_client.py**:
   - Line 70: Added `webhook_url` parameter to `transcribe()`
   - Lines 102-108: Use webhook_url if provided
   - Line 157: Added `webhook_url` parameter to `health_check()`

3. **speech_recorder.py**:
   - Line 53: Added `webhook_url` parameter to `start_recording()`
   - Line 183: Pass webhook_url to STT client

### STT Service Details
- Endpoint: `http://192.168.8.191:7272/stt/v1/stream`
- Model: whisper-tiny
- Backend: whisper.cpp
- Device: CUDA
- Status: Healthy and ready

---

**Status**: Code implementation complete. Configuration update needed to set webhook URLs  
**Action Required**: Update model configurations with webhook URLs via GUI or config file