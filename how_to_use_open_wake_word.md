# OpenWakeWord Model Fix Guide

## Quick Diagnostic Test (5 minutes)

Run this standalone test first to identify the exact issue:

```python
# Save as test_model.py and run: python3 test_model.py
import openwakeword
import numpy as np

model_path = "third_party/openwakeword/custom_models/Hay--compUta_v_lrg.onnx"

# Try loading the model
try:
    model = openwakeword.Model(wakeword_model_paths=[model_path])
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Loading failed: {e}")
    exit(1)

# Test with different inputs
test_inputs = {
    "silence": np.zeros(1280, dtype=np.float32),
    "noise": np.random.normal(0, 0.1, 1280).astype(np.float32),
    "sine": np.sin(2 * np.pi * 1000 * np.linspace(0, 0.08, 1280)).astype(np.float32)
}

for name, audio in test_inputs.items():
    pred = model.predict(audio)
    print(f"{name}: {pred}")

# Check if all predictions are identical (indicates broken model)
predictions = [model.predict(audio) for audio in test_inputs.values()]
if len(set(str(p) for p in predictions)) == 1:
    print("❌ BROKEN MODEL: All predictions identical!")
else:
    print("✅ Model working correctly")
```

## Root Issue
The model returns constant value `0.00083711743` for ALL inputs - it's not processing audio.

## Fix Priority Order

### 1. **Immediate Workaround** (2 minutes)
Lower threshold to bypass the issue temporarily:
```python
# In settings_manager.py, change threshold to:
"threshold": 0.0008  # Just below the constant value
```

### 2. **Try Alternative Loading** (5 minutes)
Replace model loading in `openwakeword_engine.py`:
```python
# Replace the current loading with these attempts in order:
# Attempt 1: Basic loading
self.model = openwakeword.Model(wakeword_model_paths=[custom_model_path])

# Attempt 2: With inference framework
self.model = openwakeword.Model(
    wakeword_model_paths=[custom_model_path],
    inference_framework="onnx"
)

# Attempt 3: Without class mapping
self.model = openwakeword.Model(
    wakeword_model_paths=[custom_model_path],
    vad_threshold=0.0
)
```

### 3. **Model Replacement** (10 minutes)
If above fails, the model file is corrupted. Options:

a) **Use pre-trained model**:
```python
# Comment out custom model loading and use:
self.model = openwakeword.Model()  # Loads all pre-trained models
```

b) **Download fresh model**:
- Check OpenWakeWord GitHub releases
- Look for community "Hey Computer" models
- Verify MD5 hash after download

### 4. **Version Update** (5 minutes)
```bash
pip install --upgrade openwakeword onnxruntime numpy
```

## Key Code Fixes Already Applied

1. **Audio normalization** (working correctly):
```python
audio_chunk = audio_chunk.astype(np.float32) / 32768.0
```

2. **Class mapping** (working correctly):
```python
class_mapping_dicts=[{0: self.wake_word_name}]
```

3. **Audio amplification** (working correctly):
```python
if raw_max < 1000:
    audio_chunk = audio_chunk * 10
```

## Validation Checklist
- [ ] Model file exists and >1MB
- [ ] Test script shows varying predictions
- [ ] Live audio produces varying confidence scores
- [ ] Detection triggers at reasonable threshold (0.3-0.5)

## If All Else Fails
The model file is incompatible with your OpenWakeWord version. Either:
1. Train a new model using OpenWakeWord's training pipeline
2. Use a different wake word with pre-trained models
3. Contact the model creator for a compatible version

**Time estimate**: 15-20 minutes total to identify and fix the issue.