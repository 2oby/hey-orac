# Wake Word Detection Fix Plan

## Problem: Very Low Confidence Scores (0.000001-0.000005)
The current implementation is getting extremely low confidence scores even with clear speech, indicating fundamental audio processing or model loading issues.

## Analysis of OLD WORKING FILES vs CURRENT IMPLEMENTATION

### Key Differences Found:
1. **Audio Normalization**: Current uses `/32767.0`, old working code uses `/32768.0`
2. **Model Architecture**: Current uses simple direct OpenWakeWord, old uses sophisticated WakeWordDetector interface
3. **Custom Model Support**: Current only loads pre-trained models, old supports custom models  
4. **Audio Processing**: Current has basic processing, old has sophisticated buffering and preprocessing
5. **Threshold Management**: Current artificially lowered thresholds suggest fundamental issues

---

## 5 POSSIBLE CAUSES & FIX PLANS

### CAUSE 1: Audio Normalization Error ✅ TESTED - MINIMAL IMPACT
**Issue**: Current code divides by 32767.0, but old working code divides by 32768.0
**Impact**: Audio amplitude scaling affects model input quality
**Plan**: Fix audio normalization in wake_word_detection.py  
**Status**: ✅ COMPLETED - Changed /32767.0 to /32768.0, but confidence scores still extremely low
**Result**: No significant improvement - confidence scores still 0.000001-0.000005

### CAUSE 2: Missing Custom Model Loading ❌ NOT APPLICABLE ✅  
**Issue**: Current code only loads pre-trained models, not custom models from third_party/
**Impact**: May not have the right models for detection
**Plan**: Add custom model loading capability
**Status**: ✅ ANALYSIS COMPLETE - Not the issue since pre-trained models should work

### CAUSE 3: Model Initialization Issues 🔄 IN PROGRESS
**Issue**: Current model initialization may not be loading models properly
**Impact**: Models may not be ready for prediction  
**Plan**: Improve model initialization with better error checking and logging
**Status**: 🔄 INVESTIGATING - Current model shows empty prediction buffer initially
**Analysis**: Old working code had more sophisticated model testing and verification

### CAUSE 4: Audio Stream Format Issues ❌ TODO
**Issue**: Audio stream parameters or data format may be incompatible with models
**Impact**: Poor quality audio input to models
**Plan**: Verify and fix audio stream configuration
**Status**: ⏳ PENDING

### CAUSE 5: Missing Audio Preprocessing ❌ TODO  
**Issue**: Current code lacks sophisticated audio preprocessing from old implementation
**Impact**: Raw audio may need additional processing for optimal model performance
**Plan**: Implement proper audio preprocessing pipeline  
**Status**: ⏳ PENDING

---

## PROGRESS LOG

### 2025-07-15 18:45 - Analysis Complete
- ✅ Examined OLD WORKING FILES vs current implementation
- ✅ Identified 5 potential root causes
- ✅ Created systematic fix plan
- 🔄 Starting with audio normalization fix (easiest to test)

### Next: Test each fix incrementally and monitor confidence score improvements