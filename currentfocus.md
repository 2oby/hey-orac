# Current Focus: 🧪 Test Multi-Trigger Implementation & Template-Based Configuration

## 🎯 Testing Requirements for Recent Changes

### **1. SettingsManager Template Support** (NEW - NEEDS TESTING)
- **Test Case**: Delete settings.json and verify it's created from template
- **Expected**: New settings.json should match settings.json.template exactly
- **Verify**: All default values (rms_filter: 50.0, cooldown: 2.0, vad_threshold: 0.5)
- **Edge Case**: Test with missing/corrupt template file (should fall back to hardcoded defaults)

### **2. Multi-Trigger Wake Word Detection** (NEEDS TESTING)
- **Test Case**: Enable multi_trigger in settings.json and load multiple models
- **Expected**: Multiple models can trigger simultaneously if they exceed thresholds
- **Verify**: Each model sends separate webhook/detection event
- **Compare**: Single-trigger mode should only trigger highest confidence model

### **3. Docker Non-Root User Permissions** (DEPLOYED - VERIFY)
- **Test Case**: Check file permissions of created settings.json
- **Expected**: Files created with appuser ownership, readable by host
- **Verify**: OpenWakeWord can download models to its resources directory

### **4. Multi-Trigger Checkbox State Issue** (NEW - NEEDS FIXING)
- **Problem**: Multi-trigger checkbox shows as unchecked in GUI but multiple model selection is enabled
- **Test Case**: Check checkbox state matches actual multi_trigger setting in config
- **Expected**: Checkbox should reflect true state of multi_trigger setting
- **Current Bug**: GUI shows unchecked but functionality works

---

## 🐛 EXISTING BUG: Model Switching - Detection Loop Not Reloading Models

## 🔴 CRITICAL BUG: Model Activation Changes Not Applied to Detection Engine

### **Bug Description:**
When users change the active model in the GUI, wake word detections still go to "Hay--compUta_v_lrg" regardless of which model is activated/deactivated. The GUI correctly updates and the configuration is saved, but the detection engine continues using the models loaded at startup.

### **Root Cause Analysis:**
1. **Models loaded once at startup**: OpenWakeWord model object created with initially enabled models
2. **No config change detection**: Detection loop never checks `shared_data['config_changed']` flag
3. **Stale model state**: `active_model_configs` dictionary never updated after startup
4. **Detection events use stale models**: Wake words trigger based on startup configuration

### **Evidence Found:**
- ✅ GUI correctly calls `/custom-models/{name}/activate` API
- ✅ API correctly updates config and sets `config_changed = True`
- ❌ Detection loop runs with original models loaded at startup
- ❌ Detection events use stale `active_model_configs` dictionary
- ❌ No mechanism to reload OpenWakeWord models during runtime

---

## 🛠️ Implementation Tasks

### **1. Add Config Change Detection** (HIGH PRIORITY)
- Modify detection loop in `wake_word_detection.py` to check `shared_data['config_changed']`
- Check periodically (e.g., every loop iteration or every N seconds)
- Trigger model reload when flag is True

### **2. Implement Model Reloading Function** (HIGH PRIORITY)
- Create function to reload OpenWakeWord models based on current config
- Update `active_model_configs` dictionary with current enabled models
- Properly dispose of old model object before creating new one
- Reset `config_changed` flag after successful reload

### **3. Fix Model Name Mapping** (MEDIUM PRIORITY)
- Ensure consistent model name mapping between OpenWakeWord and config
- Handle edge cases where model filename doesn't match config name
- Add logging to track model name resolution

### **4. Test End-to-End** (HIGH PRIORITY)
- Verify model activation/deactivation works correctly
- Ensure wake words trigger correct model names
- Test multiple model switches and edge cases
- Validate memory management during model reloading

---

## 📝 Code Changes Required

### **File: `src/hey_orac/wake_word_detection.py`**

1. **Add config change detection in main loop** (around line 706)
2. **Create `reload_models()` function** to:
   - Get current enabled models from settings
   - Rebuild model paths list
   - Create new OpenWakeWord model object
   - Update active_model_configs dictionary
3. **Fix model name mapping** in detection logic (lines 781-803)

### **Expected Behavior After Fix:**
- User toggles model in GUI
- API updates configuration file
- Detection loop detects config change
- Models are reloaded with new configuration
- Wake words trigger notifications with correct model names
- System continues running without restart

---

## 🎯 Success Criteria

1. **GUI model toggle** immediately affects which models are active
2. **Wake word detections** use the correct model name in events
3. **No false detections** from deactivated models
4. **Smooth transitions** without audio interruption during reload
5. **Memory stable** after multiple model switches