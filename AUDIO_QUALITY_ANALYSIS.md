# Hey Orac Audio Quality Analysis & Recommendations

## Executive Summary

After analyzing the Hey Orac wake-word detection system, I've identified several areas where audio quality can be improved before sending to the ORAC STT service. The main issues stem from audio clipping at high volumes and the lack of preprocessing to normalize and enhance audio quality.

## Current Audio Pipeline

```
Microphone → PyAudio Stream → Ring Buffer → Wake Detection → Speech Recording → STT
    (16-bit)     (int16)        (int16)                      (float32 norm)    (16kHz mono)
```

## Key Findings

### 1. Audio Clipping Issues
- **Problem**: The current system directly passes raw 16-bit audio through the pipeline without any preprocessing
- **Impact**: Loud speech can cause clipping, resulting in distorted audio and poor STT accuracy
- **Location**: Audio is captured raw in `microphone.py:186` and only normalized to float32 during recording

### 2. Missing Audio Preprocessing
The system lacks essential audio preprocessing stages:
- No dynamic range compression
- No automatic gain control (AGC)
- No peak limiting
- No noise reduction

### 3. Normalization Timing
- Audio is normalized from int16 to float32 only at the final stage (`speech_recorder.py:116`)
- This is too late to prevent clipping that occurs during capture

### 4. Pre-roll Audio Quality
- Pre-roll audio from the ring buffer is mixed with actively recorded speech
- No quality enhancement is applied to pre-roll segments

## Recommendations

### 1. Implement Audio Preprocessing Pipeline

Add a preprocessing module between microphone capture and ring buffer:

```python
# New audio/preprocessor.py
class AudioPreprocessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.agc_target_level = 0.3  # Target RMS level
        self.compression_ratio = 4.0  # 4:1 compression
        self.limiter_threshold = 0.95  # Prevent peaks above 95%
        
    def process(self, audio_chunk):
        # 1. Apply AGC to normalize levels
        audio = self._apply_agc(audio_chunk)
        
        # 2. Apply compression to reduce dynamic range
        audio = self._apply_compression(audio)
        
        # 3. Apply peak limiting
        audio = self._apply_limiter(audio)
        
        # 4. Optional: Apply noise gate/reduction
        audio = self._reduce_noise(audio)
        
        return audio
```

### 2. Early Normalization

Move float32 normalization earlier in the pipeline:
- Convert int16 to float32 immediately after capture
- Process all audio in float32 to prevent integer overflow
- Only convert back to int16 when saving to files

### 3. Implement Configurable Audio Enhancement

Add audio enhancement settings to the configuration:
```json
{
  "audio": {
    "preprocessing": {
      "enable_agc": true,
      "agc_target_level": 0.3,
      "enable_compression": true,
      "compression_ratio": 4.0,
      "enable_limiter": true,
      "limiter_threshold": 0.95,
      "enable_noise_reduction": false,
      "noise_reduction_strength": 0.5
    }
  }
}
```

### 4. Add Real-time Audio Monitoring

Implement audio quality metrics:
- Peak level monitoring
- RMS level tracking
- Clipping detection counter
- Signal-to-noise ratio estimation

### 5. Specific Implementation Steps

1. **Create `audio/preprocessor.py`** with the preprocessing pipeline
2. **Modify `microphone.py`** to:
   - Convert audio to float32 immediately after capture
   - Apply preprocessing before writing to ring buffer
   - Track audio quality metrics

3. **Update `speech_recorder.py`** to:
   - Remove redundant normalization (already float32)
   - Apply final touch-up processing if needed

4. **Add audio quality monitoring** to the web interface:
   - Real-time level meters
   - Clipping indicators
   - Audio quality score

### 6. Testing Strategy

1. **Generate test audio** with various characteristics:
   - Quiet speech (test AGC boost)
   - Loud speech (test compression/limiting)
   - Mixed levels (test dynamic range handling)

2. **Measure improvements**:
   - STT accuracy before/after preprocessing
   - Clipping occurrence rate
   - Audio quality scores

## Implementation Priority

1. **High Priority**: 
   - AGC implementation (fixes most volume issues)
   - Peak limiting (prevents clipping)
   - Early float32 normalization

2. **Medium Priority**:
   - Dynamic range compression
   - Audio quality monitoring
   - Configuration options

3. **Low Priority**:
   - Noise reduction
   - Advanced filtering options

## Expected Benefits

- **Improved STT accuracy**: Consistent audio levels and reduced distortion
- **Better user experience**: No need to adjust speaking volume
- **Reduced failures**: Fewer STT errors due to audio quality issues
- **Enhanced debugging**: Audio quality metrics for troubleshooting

## Next Steps

1. Implement basic AGC and peak limiting
2. Test with various audio inputs
3. Measure STT accuracy improvements
4. Fine-tune parameters based on results
5. Add configuration options for different environments

This approach will significantly improve the audio quality sent to the ORAC STT service, resulting in better transcription accuracy and a more robust wake-word detection system.