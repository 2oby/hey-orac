Here's how we're calling OpenWakeWord to monitor an audio stream in a single comment:
```python
# OPENWAKEWORD AUDIO STREAM MONITORING - COMPLETE FLOW
# 1. MODEL INITIALIZATION
# When a custom model is loaded, we create the OpenWakeWord model like this:
custom_model_path = "third_party/openwakeword/custom_models/Hay--compUta_v_lrg.onnx"
model_name = "Hay--compUta_v_lrg"  # Extracted from filename
self.model = openwakeword.Model(
    wakeword_model_paths=[custom_model_path],
    class_mapping_dicts=[{0: model_name}],  # Maps class 0 to our model name
    vad_threshold=0.5,
    enable_speex_noise_suppression=False
)
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
        logger.info(f"�� WAKE WORD DETECTED! Chunk: {chunk_count}, RMS: {rms_level:.4f}")
# 3. WAKE WORD MONITOR PROCESSING
# The wake word monitor processes each audio chunk:
def process_audio(self, audio_data: np.ndarray) -> bool:
    # Convert audio to float32 if needed
    if audio_data.dtype != np.float32:
        audio_chunk = audio_data.astype(np.float32)

    # Get predictions from OpenWakeWord model
    predictions = self.model.predict(audio_chunk)

    # Check specifically for our custom model confidence
    model_confidence = predictions.get(self.wake_word_name, 0.0)

    if model_confidence > self.threshold:
        logger.info(f"�� WAKE WORD DETECTED by model '{self.wake_word_name}'! Confidence: {model_confidence:.3f}")
        return True

    return False
# 4. AUDIO PIPELINE INTEGRATION
# The audio pipeline continuously streams audio and calls the callback:
def run(self):
    while self.running:
        # Capture audio chunk from microphone
        audio_data = self.audio_stream.read(self.chunk_size)

        # Calculate RMS level
        rms_level = np.sqrt(np.mean(audio_data.astype(np.float32)**2))

        # Check if audio should be passed to wake word detection
        should_pass_audio = rms_level > self.rms_filter_threshold

        if should_pass_audio and self.wake_word_callback:
            # Call the wake word callback with audio data
            self.wake_word_callback(audio_data, self.chunk_count, rms_level, avg_volume)

        self.chunk_count += 1
# 5. COMPLETE FLOW SUMMARY
# 1. Audio pipeline captures 1280-sample chunks from USB microphone
# 2. Each chunk is converted to float32 and passed to OpenWakeWord model
# 3. OpenWakeWord model.predict() returns confidence scores for all models
# 4. We check if our custom model confidence exceeds threshold (0.3)
# 5. If detected, we log the detection and return True
# 6. Audio pipeline continues streaming and monitoring
```
This shows the complete flow from audio capture through OpenWakeWord processing to detection logging.