# Refactor and Integrate Audio Preprocessing - Detailed Plan

## Overview
This document provides a detailed step-by-step plan to integrate the newly implemented audio preprocessing modules into the main `wake_word_detection.py` file. The goal is to enable AGC, compression, and limiting to improve audio quality for STT while maintaining backward compatibility.

## Prerequisites
- ✅ Audio preprocessing modules implemented (`audio/preprocessor.py`)
- ✅ AudioCapture updated to support preprocessing (`audio/microphone.py`)
- ✅ Ring buffer supports float32 audio (`audio/ring_buffer.py`)
- ✅ Configuration management updated (`config/manager.py`)
- ✅ Settings include preprocessing configuration

## Integration Plan

### Phase 1: Create Integration Layer (Low Risk)

#### Step 1.1: Create Preprocessing Manager
**File**: `src/hey_orac/audio/preprocessing_manager.py` (new)
**Purpose**: Centralized management of audio preprocessing with fallback support

```python
class PreprocessingManager:
    """Manages audio preprocessing with graceful fallback"""
    
    def __init__(self, settings_manager, logger):
        self.audio_capture = None
        self.fallback_mode = False
        self.stream = None
        self.ring_buffer = None
        
    def initialize(self, usb_mic, stream, audio_config):
        """Initialize preprocessing if available"""
        # Try to initialize AudioCapture with preprocessing
        # Fall back to stream if it fails
        
    def get_audio_chunk(self):
        """Get audio chunk with or without preprocessing"""
        # Return preprocessed audio if available
        # Fall back to raw stream reading
```

**Testing**: Unit test the manager in isolation

#### Step 1.2: Add Feature Flag
**File**: `config/settings.json.template`
**Change**: Add feature flag for gradual rollout
```json
"system": {
    "enable_audio_preprocessing": false,  // Feature flag
    ...
}
```

### Phase 2: Refactor Audio Pipeline (Medium Risk)

#### Step 2.1: Extract Audio Reading Logic
**File**: `src/hey_orac/wake_word_detection.py`
**Current**: Audio reading is inline in the main loop (lines ~882-923)
**Action**: Extract to a dedicated function

```python
def read_audio_chunk(args, stream, audio_capture, chunk_size=1280):
    """Read audio chunk from appropriate source"""
    if args.input_wav:
        # WAV file reading logic
    elif audio_capture and audio_capture.is_active():
        # Preprocessed audio from AudioCapture
    else:
        # Raw stream reading (current approach)
```

#### Step 2.2: Extract Audio Processing Logic
**Current**: Stereo-to-mono conversion and normalization inline
**Action**: Create dedicated function

```python
def process_audio_data(raw_data, source_type='microphone'):
    """Process raw audio data to format expected by OpenWakeWord"""
    # Handle stereo to mono
    # Handle format conversion
    # Return processed audio
```

### Phase 3: Integrate AudioCapture (Medium Risk)

#### Step 3.1: Conditional AudioCapture Initialization
**Location**: After USB microphone detection (~line 537)
**Action**: Add conditional initialization

```python
# Initialize audio capture with preprocessing if enabled
audio_capture = None
if system_config.get('enable_audio_preprocessing', False):
    try:
        audio_capture = initialize_audio_capture(audio_config, usb_mic)
        if audio_capture:
            # Use AudioCapture's ring buffer
            ring_buffer = audio_capture.ring_buffer
            logger.info("✅ Audio preprocessing enabled")
    except Exception as e:
        logger.warning(f"Failed to initialize preprocessing: {e}")
        audio_capture = None

# Fall back to original approach if needed
if not audio_capture:
    # Original stream initialization
    stream = audio_manager.start_stream(...)
    # Original ring buffer initialization
    ring_buffer = RingBuffer(...)
```

#### Step 3.2: Update Main Detection Loop
**Location**: Main while loop (~line 868)
**Action**: Use the new functions

```python
while True:
    try:
        # Read audio from appropriate source
        audio_data = read_audio_chunk(args, stream, audio_capture)
        if audio_data is None:
            continue
            
        # Update metrics
        update_audio_metrics(shared_data, audio_capture, audio_data, args)
        
        # Rest of the detection logic remains the same
        prediction = model.predict(audio_data)
        ...
```

### Phase 4: Update Speech Recording (High Risk)

#### Step 4.1: Update Speech Recorder Initialization
**Location**: Speech recorder creation (~line 757)
**Action**: Pass appropriate stream reference

```python
# Determine which stream to use for recording
recording_stream = None
if args.input_wav:
    recording_stream = stream  # WAV file stream
elif audio_capture and audio_capture.stream:
    recording_stream = audio_capture.stream  # PyAudio stream from AudioCapture
else:
    recording_stream = stream  # Original stream

speech_recorder = SpeechRecorder(
    ring_buffer=ring_buffer,
    stt_client=stt_client,
    endpoint_config=endpoint_config,
    recording_stream=recording_stream  # Pass stream reference
)
```

#### Step 4.2: Update Recording Logic
**File**: `src/hey_orac/audio/speech_recorder.py`
**Action**: Ensure compatibility with both stream types

### Phase 5: Testing and Validation

#### Step 5.1: Create Test Suite
**File**: `tests/test_audio_preprocessing_integration.py` (new)
```python
def test_preprocessing_disabled():
    """Verify system works with preprocessing disabled"""
    
def test_preprocessing_enabled():
    """Verify preprocessing improves audio quality"""
    
def test_fallback_on_error():
    """Verify graceful fallback if preprocessing fails"""
    
def test_audio_metrics():
    """Verify audio metrics are properly reported"""
```

#### Step 5.2: Manual Testing Checklist
- [ ] Test with preprocessing disabled (current behavior)
- [ ] Test with preprocessing enabled
- [ ] Test WAV file input still works
- [ ] Test recording mode still works
- [ ] Test STT integration
- [ ] Test web interface audio meters
- [ ] Test with various microphone distances
- [ ] Test with loud speech (verify no clipping)
- [ ] Monitor CPU usage with preprocessing

### Phase 6: Gradual Rollout

#### Step 6.1: Deploy with Feature Flag Off
1. Deploy code with preprocessing available but disabled
2. Verify no regression in current functionality
3. Monitor logs for any issues

#### Step 6.2: Enable for Testing
1. Enable preprocessing on test device
2. Compare STT accuracy with/without preprocessing
3. Collect audio quality metrics

#### Step 6.3: Production Rollout
1. Enable preprocessing in production
2. Monitor audio quality improvements
3. Fine-tune parameters based on real usage

### Phase 7: Cleanup and Documentation

#### Step 7.1: Remove Old Code
Once preprocessing is stable:
- Remove fallback code paths
- Simplify audio reading logic
- Update documentation

#### Step 7.2: Update Documentation
- Update README with preprocessing information
- Document configuration options
- Add troubleshooting guide

## Implementation Timeline

### Week 1: Low Risk Changes
- Day 1-2: Create PreprocessingManager and feature flag
- Day 3-4: Extract audio reading/processing functions
- Day 5: Write unit tests

### Week 2: Integration
- Day 1-2: Integrate AudioCapture conditionally
- Day 3-4: Update main detection loop
- Day 5: Update speech recording

### Week 3: Testing and Rollout
- Day 1-2: Comprehensive testing
- Day 3: Deploy with flag off
- Day 4-5: Gradual enable and monitor

## Risk Mitigation

1. **Feature Flag**: Allows instant rollback
2. **Fallback Logic**: System works even if preprocessing fails
3. **Extensive Logging**: Easy debugging
4. **Gradual Rollout**: Catch issues early
5. **Backward Compatibility**: WAV input and recording mode preserved

## Success Criteria

1. **No Regression**: Existing functionality works identically
2. **Audio Quality**: Measurable improvement in STT accuracy
3. **No Clipping**: Peak limiting prevents distortion
4. **Consistent Levels**: AGC normalizes volume
5. **Performance**: Minimal CPU impact (<5% increase)

## Configuration for Testing

```json
{
  "system": {
    "enable_audio_preprocessing": true
  },
  "audio": {
    "preprocessing": {
      "enable_agc": true,
      "agc_target_level": 0.3,
      "agc_max_gain": 10.0,
      "enable_compression": true,
      "compression_ratio": 4.0,
      "enable_limiter": true,
      "limiter_threshold": 0.95
    }
  }
}
```

## Rollback Plan

If issues arise:
1. Set `enable_audio_preprocessing` to `false`
2. Restart service
3. System reverts to original behavior
4. Debug offline with saved audio samples

## Conclusion

This plan provides a safe, gradual approach to integrating audio preprocessing while maintaining system stability. The use of feature flags and fallback mechanisms ensures we can deploy with confidence and roll back if needed.