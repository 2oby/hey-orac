# Current Focus: Code Cleanup Sprint - Simplified Plan

## 🎯 PRIORITY: Code Consolidation & Cleanup

### Goal
Clean up the codebase, consolidate working wake word detection with custom model support, and eliminate unnecessary files. GUI will handle model selection and threshold settings.

### Simplified Implementation Plan

#### Phase 1: Analysis ✅ (Completed)
- Analyzed all wake_word_detection variants
- Identified working custom model loading in test pipeline mode (lines 335-386)
- Confirmed Hay--compUta_v_lrg.tflite as the best performing custom model

#### Phase 2: Code Consolidation (IN PROGRESS)

1. **Port Custom Model Loading to Main Loop**
   - Extract working custom model loading from test pipeline mode
   - Integrate into main detection loop (around lines 441-443)
   - Hardcode Hay--compUta_v_lrg.tflite as the custom model
   - Keep detection threshold at 0.05 for now (GUI will handle this later)

2. **Add WAV Input Support**
   - Add `--input-wav <file>` command-line argument
   - When specified, feed WAV file chunks into main detection loop
   - Replace microphone stream with WAV file reader
   - Maintain same chunk processing logic

3. **Maintain Existing Functionality**
   - Keep recording mode (-rt) working
   - Keep test pipeline mode (-tp) for debugging
   - Ensure live microphone detection still works

#### Phase 3: File Cleanup

1. **Delete Conflicted File**
   - Remove `wake_word_detection_custom (conflicted).py`

2. **Rename Legacy Files with LEGACY_ prefix**
   - `wake_word_detection_enhanced.py` → `LEGACY_wake_word_detection_enhanced.py`
   - `wake_word_detection_custom.py` → `LEGACY_wake_word_detection_custom.py`
   - `test_custom_model.py` → `LEGACY_test_custom_model.py`
   - `test_m1.py` → `LEGACY_test_m1.py`
   - `test_tflite_integration.py` → `LEGACY_test_tflite_integration.py`

#### Phase 4: Testing & Documentation

1. **Test Consolidated Script**
   - Live microphone detection with custom model
   - WAV file input with `--input-wav` option
   - Verify recording mode still works
   - Ensure no regression in functionality

2. **Update Documentation**
   - Update CLAUDE.md with new `--input-wav` usage
   - Document that custom model is hardcoded for now
   - Note that GUI will handle model selection and thresholds

### Key Changes from Original Plan
- **NO** `--custom-model` switch (GUI will handle model selection)
- **NO** dynamic threshold adjustment in code (GUI will handle per-model thresholds)
- Hardcode Hay--compUta_v_lrg.tflite as the custom model
- Focus on adding WAV input support to main detection loop

### Current Status
Ready to start Phase 2 implementation - porting custom model loading to main detection loop.