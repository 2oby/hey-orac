# Hey Orac Code Cleanup Project

**Status**: Planning Phase
**Created**: 2025-10-16
**Branch**: `code-cleanup` (to be created)

## Overview
This document tracks the systematic cleanup and refactoring of the Hey Orac codebase. The working application will be improved through small, testable iterations with git commits between each change to enable easy rollback.

---

## Cleanup Strategy

### Sprint Workflow
1. **Create branch** - Work on `code-cleanup` branch
2. **Before each mini-sprint** - Commit current state with descriptive message
3. **Make targeted changes** - Focus on one section at a time
4. **Deploy and test** - Use `./scripts/deploy_and_test.sh "Sprint X: description"` to:
   - Commit changes locally
   - Push to `code-cleanup` branch
   - Pull and deploy on the Pi
   - Test functionality automatically
5. **Verify on Pi** - Check logs to confirm wake word detection still works
6. **If broken** - Easy rollback with `git reset --hard HEAD^`
7. **When complete** - Merge back to master

**IMPORTANT**: Always use `deploy_and_test.sh` for testing. The application runs on the Pi, not locally.

### Testing Protocol After Each Sprint
- [ ] Application starts without errors
- [ ] Wake word detection triggers correctly
- [ ] Web interface accessible (if applicable for that sprint)
- [ ] Logs are clean and informative
- [ ] No new warnings or errors in logs

---

## Sprint Breakdown

### 🔴 SPRINT 1: Delete Redundant Files (HIGH PRIORITY)
**Goal**: Remove backup files and temporary utility scripts
**Risk**: Low - these files aren't imported
**Estimated Time**: 5 minutes

#### Files to Delete:
- [ ] `src/hey_orac/wake_word_detection_backup.py` (760 lines - old backup)
- [ ] `fix_indentation.py` (root)
- [ ] `fix_indent.py` (root)
- [ ] `fix_try_block.py` (root)
- [ ] `fix_all_indent.py` (root)

#### Steps:
1. Commit current state: `git commit -m "Pre-cleanup: Baseline before removing redundant files"`
2. Delete the 5 files listed above
3. Test: Run application, verify it starts correctly
4. Commit: `git commit -m "Cleanup: Remove backup and temporary utility files"`

**Rollback Command**: `git reset --hard HEAD^`

---

### 🔴 SPRINT 2: Remove Debug Print Statements (HIGH PRIORITY)
**Goal**: Replace print() debugging with proper logging
**Risk**: Low - just changing output method
**Estimated Time**: 10 minutes
**File**: `src/hey_orac/wake_word_detection.py`

#### Debug Print Statements to Remove:
Lines with `print("DEBUG:...", flush=True)`:
- [ ] Line 617: `print("DEBUG: About to create Model()", flush=True)`
- [ ] Line 672: `print("DEBUG: Model created successfully", flush=True)`
- [ ] Line 703: `print("DEBUG: After model initialized log", flush=True)`
- [ ] Line 710: `print("DEBUG: After sys.stdout.flush()", flush=True)`
- [ ] Line 732: `print("DEBUG: About to test audio stream", flush=True)`
- [ ] Line 734: `print("DEBUG: After audio stream test log", flush=True)`
- [ ] Line 598: `print("DEBUG: After model initialized log", flush=True)`
- [ ] Line 605: `print("DEBUG: After sys.stdout.flush()", flush=True)`
- [ ] Line 608: `print("DEBUG: About to test audio stream", flush=True)`
- [ ] Line 610: `print("DEBUG: After audio stream test log", flush=True)`

#### Steps:
1. Commit: `git commit -m "Pre-cleanup: Baseline before removing debug prints"`
2. Remove all `print("DEBUG:...)` statements (already have logger.info/debug equivalents nearby)
3. Test: Check logs still show expected information
4. Commit: `git commit -m "Cleanup: Remove debug print statements, use structured logging only"`

---

### 🔴 SPRINT 3: Fix Dangerous Resource Cleanup (HIGH PRIORITY)
**Goal**: Replace direct `__del__()` calls with proper cleanup
**Risk**: Medium - touching cleanup code
**Estimated Time**: 15 minutes
**Files**: `src/hey_orac/wake_word_detection.py`, `src/hey_orac/audio/utils.py`

#### Issues:
- Lines 1384, 1416: `audio_manager.__del__()` - Never call `__del__()` directly

#### Steps:
1. Commit: `git commit -m "Pre-cleanup: Baseline before fixing resource cleanup"`
2. Check if `AudioManager` has a proper `close()` method
3. If yes: Replace `__del__()` calls with `close()`
4. If no: Add `close()` method to `AudioManager` class
5. Update all cleanup locations to use `close()`
6. Test: Verify graceful shutdown on Ctrl+C
7. Commit: `git commit -m "Cleanup: Fix resource cleanup - use close() instead of __del__()"`

---

### 🔴 SPRINT 4: Fix Error Handling Anti-Patterns (HIGH PRIORITY)
**Goal**: Replace bare excepts with specific exception handling
**Risk**: Medium - changing error handling behavior
**Estimated Time**: 20 minutes
**File**: `src/hey_orac/wake_word_detection.py`

#### Bare Excepts to Fix:
- [ ] Lines 1015-1017: `except:` when reading from queue
- [ ] Lines 1155-1158: `except: pass` when putting to event queue
- [ ] Lines 1273-1276: `except: pass` when putting to event queue

#### Steps:
1. Commit: `git commit -m "Pre-cleanup: Baseline before fixing error handling"`
2. Replace bare `except:` with specific exceptions (e.g., `except queue.Empty:`)
3. Add logging for caught exceptions instead of silent `pass`
4. Consider: Should queue full be logged as warning?
5. Test: Verify app handles errors gracefully
6. Commit: `git commit -m "Cleanup: Fix error handling - catch specific exceptions and log failures"`

---

### 🟡 SPRINT 5: Resolve Dual Entry Points (MEDIUM PRIORITY)
**Goal**: Clarify which entry point is production
**Risk**: Medium - architectural decision
**Estimated Time**: 30 minutes
**Files**: `src/hey_orac/app.py`, `src/hey_orac/cli.py`, investigation needed

#### Investigation Questions:
- Which file does Docker container actually run?
- Is `app.py` (HeyOracApplication class) used in production?
- Is `cli.py` ever used?

#### Options:
**Option A**: If `wake_word_detection.py` is production:
- [ ] Delete `app.py` (196 lines)
- [ ] Delete or update `cli.py` to point to actual entry point
- [ ] Update README with correct usage

**Option B**: If `app.py` is production (or should be):
- [ ] Migrate web server, STT, heartbeat from `wake_word_detection.py` to `app.py`
- [ ] Rename `wake_word_detection.py` to `legacy_standalone.py` or delete
- [ ] Update Docker container to use proper entry point

#### Steps:
1. Commit: `git commit -m "Pre-cleanup: Baseline before resolving entry points"`
2. Check `Dockerfile` or `docker-compose.yml` for actual command
3. Decide on Option A or B based on actual usage
4. Implement chosen option
5. Test: Verify application starts via intended entry point
6. Commit: `git commit -m "Cleanup: Resolve dual entry points - [describe decision]"`

---

### 🟡 SPRINT 6: Extract Constants (MEDIUM PRIORITY)
**Goal**: Move magic numbers to named constants
**Risk**: Low - pure refactoring
**Estimated Time**: 20 minutes
**Files**: Create `src/hey_orac/constants.py`, update `wake_word_detection.py`

#### Magic Numbers to Extract:
```python
# Audio constants
CHUNK_SIZE = 1280  # samples per chunk
SAMPLE_RATE = 16000  # Hz
RING_BUFFER_SECONDS = 10.0

# Monitoring intervals
AUDIO_LOG_INTERVAL_CHUNKS = 100
MODERATE_CONFIDENCE_LOG_INTERVAL = 50
CONFIG_CHECK_INTERVAL_SECONDS = 1.0
HEALTH_CHECK_INTERVAL_SECONDS = 30.0
THREAD_CHECK_INTERVAL_SECONDS = 5.0

# Thread health
MAX_STUCK_RMS_COUNT = 10
STUCK_RMS_THRESHOLD = 0.0001

# Timeouts
QUEUE_TIMEOUT_SECONDS = 2.0
WEBHOOK_TIMEOUT_SECONDS = 5
THREAD_JOIN_TIMEOUT_SECONDS = 5

# Detection thresholds (examples - may be config-driven)
DETECTION_THRESHOLD_DEFAULT = 0.3
MODERATE_CONFIDENCE_THRESHOLD = 0.1
WEAK_SIGNAL_THRESHOLD = 0.05
VERY_WEAK_SIGNAL_THRESHOLD = 0.01
```

#### Steps:
1. Commit: `git commit -m "Pre-cleanup: Baseline before extracting constants"`
2. Create `src/hey_orac/constants.py` with above constants
3. Replace hardcoded numbers in `wake_word_detection.py` with constant references
4. Test: Verify behavior unchanged
5. Commit: `git commit -m "Cleanup: Extract magic numbers to constants module"`

---

### 🟡 SPRINT 7: Consolidate Audio Conversion Logic (MEDIUM PRIORITY)
**Goal**: Single stereo-to-mono conversion function
**Risk**: Low - pure refactoring
**Estimated Time**: 25 minutes
**File**: `src/hey_orac/audio/utils.py` (add function), update callers

#### Duplicate Logic Locations:
- `wake_word_detection.py:1033-1055` - Main loop
- `wake_word_detection.py:633-656` - Another location
- `wake_word_detection_preprocessing.py:109-121` - Helper module

#### Steps:
1. Commit: `git commit -m "Pre-cleanup: Baseline before consolidating audio conversion"`
2. Create function in `audio/utils.py`:
   ```python
   def convert_audio_to_mono_float32(
       audio_data: bytes,
       is_stereo: bool,
       chunk_size: int = 1280
   ) -> np.ndarray:
       """Convert raw audio bytes to mono float32 for OpenWakeWord."""
       ...
   ```
3. Replace all duplicate logic with calls to this function
4. Test: Verify audio processing works correctly
5. Commit: `git commit -m "Cleanup: Consolidate stereo-to-mono conversion logic"`

---

### 🟡 SPRINT 8: Refactor Massive Main Function - Part 1 (MEDIUM PRIORITY)
**Goal**: Extract setup functions from main()
**Risk**: High - major refactoring
**Estimated Time**: 45 minutes
**File**: `src/hey_orac/wake_word_detection.py`

#### Extract These Functions:
1. `setup_audio_input(args, audio_config, audio_manager, usb_mic)` → Returns stream
2. `setup_wake_word_models(models_config, system_config, settings_manager)` → Returns model, active_model_configs, model_name_mapping
3. `setup_heartbeat_sender(enabled_models)` → Returns heartbeat_sender
4. `setup_web_server(settings_manager, shared_data, event_queue)` → Returns broadcaster

#### Steps:
1. Commit: `git commit -m "Pre-cleanup: Baseline before refactoring main() part 1"`
2. Extract functions one at a time (test after each)
3. Keep main() as orchestrator calling these functions
4. Test thoroughly after each extraction
5. Commit: `git commit -m "Refactor: Extract setup functions from main() - Part 1"`

---

### 🟡 SPRINT 9: Refactor Massive Main Function - Part 2 (MEDIUM PRIORITY)
**Goal**: Extract STT and detection loop from main()
**Risk**: High - major refactoring
**Estimated Time**: 45 minutes
**File**: `src/hey_orac/wake_word_detection.py`

#### Extract These Functions:
1. `setup_stt_components(stt_config, audio_config, settings_manager)` → Returns ring_buffer, speech_recorder, stt_client
2. `run_detection_loop(...)` → Main loop logic, returns exit code

#### Steps:
1. Commit: `git commit -m "Pre-cleanup: Baseline before refactoring main() part 2"`
2. Extract STT setup function
3. Extract detection loop into separate function
4. Test: Full integration test
5. Commit: `git commit -m "Refactor: Extract STT and detection loop from main() - Part 2"`

---

### 🟢 SPRINT 10: Handle wake_word_detection_preprocessing.py (LOW PRIORITY)
**Goal**: Decide fate of preprocessing helper module
**Risk**: Low
**Estimated Time**: 20 minutes
**File**: `src/hey_orac/wake_word_detection_preprocessing.py`

#### Options:
- **Option A**: Fully integrate functions into main codebase if used
- **Option B**: Delete if unused/superseded
- **Option C**: Keep as utilities module if genuinely useful

#### Steps:
1. Commit: `git commit -m "Pre-cleanup: Baseline before handling preprocessing module"`
2. Check if functions are actually used anywhere
3. Make decision based on usage
4. Either integrate, delete, or document purpose
5. Test if changes made
6. Commit: `git commit -m "Cleanup: [Action taken] for preprocessing helper module"`

---

### 🟢 SPRINT 11: Standardize Configuration (LOW PRIORITY)
**Goal**: Fix hardcoded config in app.py
**Risk**: Low
**Estimated Time**: 15 minutes
**File**: `src/hey_orac/app.py`

#### Issue:
- Lines 65-79: `_load_config()` has hardcoded dict with "TODO: Implement config loading"

#### Steps:
1. Commit: `git commit -m "Pre-cleanup: Baseline before fixing config loading"`
2. Either: Implement JSON loading as TODO suggests
3. Or: Use SettingsManager like wake_word_detection.py does
4. Or: Remove if app.py is being deprecated (see Sprint 5)
5. Test: If keeping app.py, verify it loads config properly
6. Commit: `git commit -m "Cleanup: Implement proper config loading in app.py"`

---

### 🟢 SPRINT 12: Remove TODOs and Commented Code (LOW PRIORITY)
**Goal**: Clean up technical debt markers
**Risk**: Low
**Estimated Time**: 20 minutes
**Files**: `src/hey_orac/app.py` and others

#### TODOs to Address:
- [ ] `app.py:67` - "TODO: Implement config loading from JSON" (see Sprint 11)
- [ ] `app.py:151` - "TODO: Start speech capture" - Either implement or remove
- [ ] `app.py:183` - "TODO: Implement speech capture and streaming" - Either implement or remove

#### Steps:
1. Commit: `git commit -m "Pre-cleanup: Baseline before removing TODOs"`
2. For each TODO: Either implement, create GitHub issue, or delete
3. Remove any large commented-out code blocks
4. Test if any implementations added
5. Commit: `git commit -m "Cleanup: Address TODO comments and remove dead code"`

---

### 🟢 SPRINT 13: Standardize Naming Conventions (LOW PRIORITY)
**Goal**: Consistent naming across codebase
**Risk**: Low - mostly renaming
**Estimated Time**: 30 minutes
**Files**: Multiple

#### Naming Inconsistencies:
- `wake_word` vs `wakeword` vs `wake-word` - Pick one style
- `stt_client` vs `speech_recorder` - Clarify naming pattern
- `active_model_configs` vs `enabled_models` vs `models_config` - Standardize

#### Steps:
1. Commit: `git commit -m "Pre-cleanup: Baseline before naming standardization"`
2. Document naming conventions to follow
3. Systematically rename for consistency
4. Test: Verify no broken references
5. Commit: `git commit -m "Cleanup: Standardize naming conventions across codebase"`

---

### 🟢 SPRINT 14: Emoji Logging Standard (LOW PRIORITY)
**Goal**: Decide on emoji usage policy
**Risk**: Low - cosmetic
**Estimated Time**: 15 minutes
**Files**: All Python files

#### Current State:
- `wake_word_detection.py` uses emojis heavily (🎤, 🎯, ✅, ❌, 📊)
- Other modules don't use emojis

#### Decision Options:
- **Option A**: Keep emojis - they help visual scanning of logs
- **Option B**: Remove emojis - more professional/traditional
- **Option C**: Standardize - define emoji usage guide

#### Steps:
1. Commit: `git commit -m "Pre-cleanup: Baseline before emoji standardization"`
2. Decide on policy (suggest Option A with guide)
3. Document emoji meanings if keeping
4. Apply consistently across all modules
5. Commit: `git commit -m "Cleanup: Standardize emoji usage in logging"`

---

## Progress Tracking

### Completion Status
- [x] Sprint 1: Delete Redundant Files ✅ (Removed 953 lines of dead code)
- [x] Sprint 2: Remove Debug Prints ✅ (Removed 6 debug print statements)
- [ ] Sprint 3: Fix Resource Cleanup
- [ ] Sprint 4: Fix Error Handling
- [ ] Sprint 5: Resolve Dual Entry Points
- [ ] Sprint 6: Extract Constants
- [ ] Sprint 7: Consolidate Audio Conversion
- [ ] Sprint 8: Refactor Main - Part 1
- [ ] Sprint 9: Refactor Main - Part 2
- [ ] Sprint 10: Handle Preprocessing Module
- [ ] Sprint 11: Standardize Configuration
- [ ] Sprint 12: Remove TODOs
- [ ] Sprint 13: Standardize Naming
- [ ] Sprint 14: Emoji Logging Standard

### Statistics
- **Total Sprints**: 14
- **Completed**: 2
- **In Progress**: None
- **Blocked**: None

---

## Git Branch Strategy

### Branch Setup (COMPLETED ✅)
- [x] Created `code-cleanup` branch
- [x] Fixed `deploy_and_test.sh` to use current branch (not hardcoded master)
- [x] Committed CLEANUP.md roadmap

### Branch Commands
```bash
# Create cleanup branch (ALREADY DONE)
git checkout -b code-cleanup

# After each sprint commit
git add .
git commit -m "Sprint X: [description]"

# If something breaks
git reset --hard HEAD^  # Undo last commit
git reset --hard HEAD~3  # Undo last 3 commits

# Deploy to Pi while on code-cleanup branch
./scripts/deploy_and_test.sh "Sprint X: [description]"

# When all sprints complete
git checkout master
git merge code-cleanup
git branch -d code-cleanup
```

---

## Notes

### Why Small Sprints?
- Easy to understand changes
- Quick to test
- Simple to rollback
- Less merge conflict risk
- Progress visible

### What if We Find More Issues?
Add new sprints at the end with sprint number and priority indicator.

### When is Cleanup "Done"?
When all HIGH and MEDIUM priority sprints are complete and tested. LOW priority sprints are optional improvements.

---

## Final Integration Test Checklist

Before merging `code-cleanup` back to master:

- [ ] Application starts without errors
- [ ] Wake word detection works correctly
- [ ] Web interface accessible and functional
- [ ] STT integration working
- [ ] Heartbeat sending operational
- [ ] Model configuration changes work
- [ ] Health checks functioning
- [ ] Graceful shutdown on Ctrl+C
- [ ] No regression in functionality
- [ ] Code is more maintainable than before
- [ ] All tests pass (if tests exist)
- [ ] Documentation updated if needed

---

**Last Updated**: 2025-10-16
**Maintained By**: Development Team
