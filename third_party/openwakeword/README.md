# OpenWakeWord Custom Models

This directory contains custom OpenWakeWord models for the Hey Orac project.

## Custom Models

Place your custom trained models in the `custom_models/` subdirectory:

- `hey_orac.onnx` - Custom "hey orac" wake-word model (ONNX format)
- `hey_orac.tflite` - Custom "hey orac" wake-word model (TensorFlow Lite format)

## Model Requirements

### ONNX Models
- Input shape: Should match OpenWakeWord's expected input format
- Output: Probability scores for wake-word detection
- Sample rate: 16kHz
- Frame length: 512 samples (32ms at 16kHz)

### TFLite Models
- Compatible with TensorFlow Lite runtime
- Same input/output requirements as ONNX models

## Usage

To use a custom model:

1. Place your model file in `custom_models/`
2. Update `src/config.yaml`:
   ```yaml
   wake_word:
     engine: "openwakeword"
     keyword: "hey_orac"
     custom_model_path: "/app/third_party/openwakeword/custom_models/hey_orac.onnx"
   ```

3. The OpenWakeWord engine will automatically load your custom model

## Training

For training custom models, refer to the OpenWakeWord documentation:
https://github.com/dscripka/openWakeWord

## Model Performance

- **False Positive Rate**: Target < 1 per hour
- **False Negative Rate**: Target < 5%
- **Latency**: Target < 100ms inference time 