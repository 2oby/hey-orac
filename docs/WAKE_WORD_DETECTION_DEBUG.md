# Wake Word Detection Debug Guide

## Current Status
**Issue**: No wake word detection events are being generated despite having:
- ✅ Audio pipeline working (RMS values being generated)
- ✅ Settings manager working (model discovery and configuration)
- ✅ Web interface working (real-time updates)
- ✅ Wake word monitor initialized

## What's Missing
The `wake_word_monitor_new.py` has the detection infrastructure implemented but **no actual detection events are being logged**. This suggests either:
1. Audio callback isn't reaching the detection logic
2. Models aren't being loaded properly
3. Audio format mismatch between pipeline and detectors

## Next Steps for Developer

### 1. Verify Audio Callback Flow
Check if `main_new.py` is calling the wake word callback:
```python
# In main_new.py - verify this callback is being called
def wake_word_callback(audio_data, chunk_count, rms_level, avg_volume):
    # Add debug logging here
    logger.info(f"🎯 Audio callback called - RMS: {rms_level}")
    self.wake_word_monitor.process_audio(audio_data)
```

### 2. Check Model Loading
Verify models are actually being loaded in `wake_word_monitor_new.py`:
```python
# Add debug logging in _load_active_models()
logger.info(f"🔧 Loading active models: {active_models}")
for model_name in active_models:
    detector = self._load_single_model(model_name)
    if detector:
        logger.info(f"✅ Loaded model: {model_name}")
    else:
        logger.error(f"❌ Failed to load model: {model_name}")
```

### 3. Compare with Working Implementation
**Old Working Code**: `src/monitor_custom_model.py`
- This file contains the working wake word detection logic
- Compare how it processes audio vs. the new implementation
- Check audio format handling differences

### 4. Test Audio Processing
Add debug logging in `process_audio()` method:
```python
def process_audio(self, audio_data: np.ndarray) -> bool:
    logger.debug(f"🎵 Processing audio chunk: shape={audio_data.shape}, dtype={audio_data.dtype}")
    # ... existing code
```

## Expected Behavior
When working correctly, you should see logs like:
```
🎯 Audio callback called - RMS: 0.0234
🎵 Processing audio chunk: shape=(1280,), dtype=int16
✅ Loaded model: Hay--compUta_v_lrg
🎯 WAKE WORD DETECTED by model 'Hay--compUta_v_lrg'!
```

## Quick Test
1. Deploy with debug logging enabled
2. Speak near the microphone
3. Check logs for audio callback and detection events
4. If no audio callback logs, the issue is in `main_new.py`
5. If audio callback logs but no detection, the issue is in model loading or audio processing

## Reference Files
- **Current Implementation**: `src/wake_word_monitor_new.py`
- **Working Reference**: `src/monitor_custom_model.py`
- **Audio Pipeline**: `src/audio_pipeline_new.py`
- **Main Application**: `src/main_new.py` 