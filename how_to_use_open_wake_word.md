```python
# OPENWAKEWORD AUDIO STREAM MONITORING - CORRECTED IMPLEMENTATION
# 1. MODEL INITIALIZATION
# When a custom model is loaded, we create the OpenWakeWord model like this:
custom_model_path = "third_party/openwakeword/custom_models/Hay--compUta_v_lrg.onnx"
model_name = "Hay--compUta_v_lrg"  # Extracted from filename

# ISSUE #1: VAD threshold conflict - we were using both OpenWakeWord's VAD and our own RMS filtering
# SOLUTION: Disable OpenWakeWord's VAD since we're doing our own audio filtering
self.model = openwakeword.Model(
    wakeword_model_paths=[custom_model_path],
    class_mapping_dicts=[{0: model_name}],  # Maps class 0 to our model name
    vad_threshold=0.0,  # CHANGED: Disabled to prevent conflict with our RMS filtering
    enable_speex_noise_suppression=False
)

# NEW: Add debug mode to help troubleshoot detection issues
self.debug_mode = True  # Set to False in production
self.detection_history = []  # Track recent predictions for pattern analysis

# 2. AUDIO PROCESSING CALLBACK
# The audio pipeline calls this function for each audio chunk:
def wake_word_callback(audio_data, chunk_count, rms_level, avg_volume):
    # audio_data is numpy array (int16) from microphone
    # chunk_count is the sequential chunk number
    # rms_level is the current RMS volume level
    # avg_volume is the average volume over time
    
    # Process through wake word monitor
    detection_result = self.wake_word_monitor.process_audio(audio_data)
    
    if detection_result:
        logger.info(f"🎯 WAKE WORD DETECTED! Chunk: {chunk_count}, RMS: {rms_level:.4f}")

# 3. WAKE WORD MONITOR PROCESSING
# The wake word monitor processes each audio chunk:
def process_audio(self, audio_data: np.ndarray) -> bool:
    # ISSUE #2: Audio normalization was missing
    # OpenWakeWord expects float32 audio normalized to [-1, 1] range
    # SOLUTION: Properly normalize int16 audio to float32 [-1, 1]
    if audio_data.dtype == np.int16:
        # Convert int16 [-32768, 32767] to float32 [-1.0, 1.0]
        audio_chunk = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype != np.float32:
        # If not int16 or float32, convert to float32
        audio_chunk = audio_data.astype(np.float32)
    else:
        # Already float32, use as-is
        audio_chunk = audio_data
    
    # Get predictions from OpenWakeWord model
    predictions = self.model.predict(audio_chunk)
    
    # ISSUE #3: We were only checking for our specific model name, which might not match
    # SOLUTION: Log all predictions to understand what the model is actually returning
    if self.debug_mode:
        # Log all model predictions to see what names are being used
        logger.debug(f"All predictions: {predictions}")
        # Track prediction history for pattern analysis
        self.detection_history.append({
            'timestamp': time.time(),
            'predictions': predictions.copy()
        })
        # Keep only last 50 predictions
        if len(self.detection_history) > 50:
            self.detection_history.pop(0)
    
    # ISSUE #4: Model name might not match what we expect
    # SOLUTION: Check all predictions, not just our expected name
    detected = False
    detection_model = None
    detection_confidence = 0.0
    
    for model_name, confidence in predictions.items():
        if confidence > self.threshold:
            detected = True
            detection_model = model_name
            detection_confidence = confidence
            break
    
    # ISSUE #5: Single-chunk detection might be too sensitive or not sensitive enough
    # SOLUTION: Implement a simple sliding window check for more robust detection
    if detected:
        # Check if we've had consistent detections in recent chunks
        if self._check_detection_consistency(detection_model):
            logger.info(f"🎯 WAKE WORD DETECTED by model '{detection_model}'! "
                       f"Confidence: {detection_confidence:.3f}")
            # Clear history after successful detection to prevent multiple triggers
            self.detection_history.clear()
            return True
    
    return False

# NEW: Helper method to check detection consistency
def _check_detection_consistency(self, model_name: str, window_size: int = 3, min_detections: int = 2) -> bool:
    """
    Check if we've had consistent detections in recent chunks to reduce false positives
    
    Args:
        model_name: The model that triggered detection
        window_size: Number of recent chunks to check
        min_detections: Minimum detections needed in the window
    
    Returns:
        True if detection is consistent enough
    """
    if len(self.detection_history) < window_size:
        # Not enough history yet, allow detection
        return True
    
    # Check last N predictions
    recent_detections = 0
    for entry in self.detection_history[-window_size:]:
        if entry['predictions'].get(model_name, 0.0) > self.threshold:
            recent_detections += 1
    
    return recent_detections >= min_detections

# 4. AUDIO PIPELINE INTEGRATION
# The audio pipeline continuously streams audio and calls the callback:
def run(self):
    # ISSUE #6: Ensure audio format matches OpenWakeWord expectations
    # OpenWakeWord requires: 16kHz sample rate, mono channel, 16-bit PCM
    # SOLUTION: Validate audio configuration at startup
    assert self.sample_rate == 16000, f"OpenWakeWord requires 16kHz, got {self.sample_rate}Hz"
    assert self.channels == 1, f"OpenWakeWord requires mono audio, got {self.channels} channels"
    
    while self.running:
        # Capture audio chunk from microphone
        # ISSUE #7: Chunk size should match model expectations
        # Default is 1280 samples (80ms at 16kHz), but verify this matches your model
        audio_data = self.audio_stream.read(self.chunk_size)
        
        # Calculate RMS level
        rms_level = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
        
        # Update average volume
        self.volume_history.append(rms_level)
        if len(self.volume_history) > 100:
            self.volume_history.pop(0)
        avg_volume = np.mean(self.volume_history)
        
        # Check if audio should be passed to wake word detection
        # This is our own pre-filtering to save CPU when there's silence
        should_pass_audio = rms_level > self.rms_filter_threshold
        
        if should_pass_audio and self.wake_word_callback:
            # Call the wake word callback with audio data
            self.wake_word_callback(audio_data, self.chunk_count, rms_level, avg_volume)
        elif self.debug_mode and self.chunk_count % 100 == 0:
            # Log periodic status in debug mode
            logger.debug(f"Audio level below threshold. RMS: {rms_level:.4f}, "
                        f"Threshold: {self.rms_filter_threshold:.4f}")
        
        self.chunk_count += 1

# 5. COMPLETE FLOW SUMMARY WITH CORRECTIONS
# 1. Audio pipeline captures 1280-sample chunks from USB microphone (16kHz, mono, int16)
# 2. RMS pre-filtering saves CPU by skipping silence
# 3. Each chunk is properly normalized from int16 to float32 [-1, 1] range
# 4. OpenWakeWord model.predict() returns confidence scores for all loaded models
# 5. We check ALL model predictions, not just our expected name
# 6. Detection consistency is verified across multiple chunks to reduce false positives
# 7. Debug logging helps identify model behavior and tuning needs
# 8. Audio pipeline continues streaming and monitoring

# ADDITIONAL DEBUGGING RECOMMENDATIONS:
# 1. Run with debug_mode=True initially to see actual prediction values
# 2. Adjust threshold based on observed confidence scores
# 3. Test with different chunk sizes if detection is inconsistent
# 4. Verify your custom model was trained with 16kHz audio
# 5. Consider implementing post-detection cooldown to prevent repeated triggers
```