# Porcupine Custom Models

This directory contains custom Porcupine wake-word models for the Hey Orac project.

## Custom Models

Place your custom trained models in the `custom_models/` subdirectory:

- `hey_orac.ppn` - Custom "hey orac" wake-word model (Porcupine format)
- `orac.ppn` - Alternative "orac" wake-word model

## Model Requirements

### Porcupine Models (.ppn)
- Format: Porcupine proprietary format
- Sample rate: 16kHz
- Frame length: 512 samples (32ms at 16kHz)
- Language: English (or as specified during training)

## Usage

To use a custom Porcupine model:

1. Place your model file in `custom_models/`
2. Update `src/config.yaml`:
   ```yaml
   wake_word:
     engine: "porcupine"
     keyword: "hey_orac"
     model_path: "/app/third_party/porcupine/custom_models/hey_orac.ppn"
     access_key: "your_picovoice_access_key"
   ```

3. The Porcupine engine will load your custom model

## Training

For training custom models, use the Picovoice Console:
https://console.picovoice.ai/

## Model Performance

- **False Positive Rate**: Target < 1 per hour
- **False Negative Rate**: Target < 5%
- **Latency**: Target < 100ms inference time

## Licensing

Porcupine models require a valid Picovoice access key. Ensure you have proper licensing for any custom models. 