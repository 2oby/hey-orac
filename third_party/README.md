# Third Party Models

This directory contains external and custom wake-word detection models that are not part of the main codebase.

## Structure

- `openwakeword/` - Custom OpenWakeWord models and configurations
  - `custom_models/` - Custom trained ONNX and TFLite models
- `porcupine/` - Custom Porcupine wake-word models and configurations
  - `custom_models/` - Custom trained PPN models

## Usage

These models are loaded by the respective wake-word engines in `src/wake_word_engines/`.

## Adding Custom Models

1. Place your model files in the appropriate subdirectory
2. Update the configuration in `src/config.yaml` to point to your model
3. Update the engine implementation if needed to support the new model format

## Model Formats

- **ONNX**: Used by OpenWakeWord engine
- **TFLite**: TensorFlow Lite models (alternative format)
- **PPN**: Porcupine model format

## License

Ensure you have proper licensing for any third-party models you include here. 