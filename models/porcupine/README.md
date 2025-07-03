# Porcupine Wake-word Models

This directory contains Porcupine wake-word models for the Hey Orac project.

## Required Files

- `orac.ppn` - The custom "ORAC" wake-word model

## How to Obtain the Model

1. Create an account at [Picovoice Console](https://console.picovoice.ai/)
2. Navigate to "Wake Word" section
3. Create a new custom wake-word with the keyword "ORAC"
4. Download the generated `.ppn` file
5. Place it in this directory as `orac.ppn`

## Model Specifications

- **Keyword**: ORAC
- **Language**: English
- **Platform**: Linux ARM64 (Raspberry Pi)
- **License**: Personal use (free tier)

## Testing

Once the model is placed here, you can test it with:

```bash
# Test locally
python src/main.py --test-audio tests/sample_orac.wav

# Test in Docker
docker-compose up
```

## Notes

- The model file is excluded from git by default (see .gitignore)
- For production use, consider purchasing a commercial license
- Keep the model file secure and don't share it publicly 