# Implementing Non-Blocking Callback-Based Audio Capture for HeyORAC

## Problem Statement
The current HeyORAC implementation uses blocking `stream.read()` calls that can freeze indefinitely, causing the entire audio processing thread to hang. When this happens, the container appears healthy but stops processing audio, requiring manual restart.

## Solution: PyAudio Callback Mode
Implement callback-based audio capture using PyAudio's callback mechanism, which runs in PortAudio's managed thread and prevents blocking.

## Implementation Requirements

### 1. Add Callback Stream Method to AudioManager
In `/Users/2oby/pCloud Box/Projects/ORAC/Hey_Orac/src/hey_orac/audio/utils.py`, add a new method after the existing `start_stream` method:

```python
def start_callback_stream(self, device_index: int, callback, sample_rate: int = 16000,
                        channels: int = 1, chunk_size: int = 512):
    """Start audio stream with callback for non-blocking audio capture.
    
    The callback function should have the signature:
    callback(in_data, frame_count, time_info, status) -> (out_data, flag)
    """
    # Implementation that creates stream with stream_callback=callback parameter
```

### 2. Create AudioCallbackProcessor Class
Add a new class before the `main()` function in `wake_word_detection.py` that handles audio processing in the callback:

Key requirements:
- Must handle audio buffering (accumulate data until we have 1280 samples)
- Process audio in separate thread to avoid blocking the callback
- Maintain all existing functionality (RMS calculation, wake word detection, etc.)
- Handle thread safety with locks
- Return `(None, pyaudio.paContinue)` from callback to keep stream active

### 3. Modify Main Loop
The main loop should:
1. Check for `use_callback_mode` flag (initially set to `True`)
2. If callback mode:
   - Create `AudioCallbackProcessor` instance with all required parameters
   - Start stream with `audio_manager.start_callback_stream()`
   - Main thread enters simple loop that monitors stream health
3. If not callback mode:
   - Use existing blocking implementation

### 4. Critical Implementation Details

#### Audio Data Flow
1. Callback receives audio data in chunks (size depends on hardware)
2. Buffer accumulates data until we have exactly 1280 samples (2560 bytes for int16)
3. Process complete chunks in separate thread
4. Continue accumulating partial data for next chunk

#### Thread Safety
- Use `threading.Lock()` for processing to prevent concurrent execution
- Callback must never block - always return immediately
- Heavy processing happens in separate thread spawned from callback

#### Maintain Existing Functionality
All these features must continue working:
- Wake word detection with configurable models
- Multi-trigger mode support  
- STT recording trigger
- Webhook calls
- RMS monitoring and stuck detection
- Ring buffer feeding
- Web GUI updates
- Heartbeat sending
- Configuration hot-reloading

## Testing Checklist
After implementation:
- [ ] Container starts without errors
- [ ] Audio processing shows in logs (e.g., "Processed 100 audio chunks")
- [ ] RMS values change (not stuck)
- [ ] Wake word detection still works
- [ ] No "Audio stream read timed out" errors
- [ ] Monitor for 72 hours without freezing

## Expected Behavior
- Audio callback runs continuously in PortAudio's thread
- Main thread remains responsive for monitoring
- If callback crashes, PortAudio handles cleanup
- No more indefinite blocking on `stream.read()`

## Files to Modify
1. `/Users/2oby/pCloud Box/Projects/ORAC/Hey_Orac/src/hey_orac/audio/utils.py` - Add callback stream method
2. `/Users/2oby/pCloud Box/Projects/ORAC/Hey_Orac/src/hey_orac/wake_word_detection.py` - Add callback processor and modify main loop

## Important Notes
- The callback processor class needs access to many objects (model, settings_manager, speech_recorder, etc.)
- All indentation must be correct - Python is strict about this
- Test with `use_callback_mode = False` first to ensure nothing breaks
- Then enable callback mode and test thoroughly