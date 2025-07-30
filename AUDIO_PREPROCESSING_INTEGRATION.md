# Audio Preprocessing Integration Plan

Due to the complexity of the wake_word_detection.py file and the need to maintain compatibility with existing functionality (WAV file input, recording mode, etc.), I'll create a focused integration approach.

## Key Integration Points

1. **Conditional Audio Capture**:
   - Only use AudioCapture with preprocessing when using microphone input
   - Keep existing stream-based approach for WAV file input
   - Maintain compatibility with recording mode

2. **Modified Detection Loop**:
   - When using microphone: Get preprocessed audio from AudioCapture
   - When using WAV file: Use existing stream reading logic
   - Convert preprocessed float32 back to raw int16 range for OpenWakeWord

3. **Ring Buffer Integration**:
   - AudioCapture already manages the ring buffer internally
   - Access ring buffer through audio_capture.ring_buffer
   - Pre-roll audio is already preprocessed

## Implementation Steps

1. Add conditional AudioCapture initialization after microphone selection
2. Modify the main detection loop to handle both audio sources
3. Update speech recording to use the appropriate stream
4. Add audio metrics reporting to shared data

## Benefits

- Cleaner separation of concerns
- Minimal changes to existing code
- Easy to enable/disable preprocessing
- Maintains all existing functionality