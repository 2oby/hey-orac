# Sprint 7: Consolidate Audio Conversion Logic

## Context from Previous Session

You are continuing the Hey Orac code cleanup project. **Sprints 1-6 are complete** and the application is working perfectly on the Raspberry Pi.

**📋 IMPORTANT**: This sprint is part of a 14-sprint cleanup plan documented in **`CLEANUP.md`**. Read that file first for the full context, testing protocol, and complete sprint breakdown. This document provides focused context for Sprint 7 specifically.

### Project Overview
Hey Orac is a wake word detection service running on a Raspberry Pi in a Docker container. It:
- Detects wake words using OpenWakeWord models
- Sends audio to ORAC STT service for speech-to-text processing
- Provides a web interface for monitoring (port 7171)
- Sends heartbeats to register wake word models with ORAC STT

### Current Branch Status
- **Working Branch**: `code-cleanup`
- **Completed Sprints**: 6/14 (all HIGH PRIORITY sprints done)
- **Sprint Progress Tracked In**: `CLEANUP.md`
- **Git Commits**: Clean history with descriptive messages

## Sprint 7 Goal: Consolidate Audio Conversion Logic

**Problem**: Audio conversion code (stereo→mono) is duplicated in multiple locations, violating the DRY (Don't Repeat Yourself) principle.

**CRITICAL FINDING**: There are **TWO DIFFERENT** conversion types needed:

1. **Un-normalized (for OpenWakeWord)**:
   - Converts int16 → float32 WITHOUT division
   - Used by wake word detection
   - **CRITICAL**: OpenWakeWord expects raw int16 values as float32, NOT normalized!
   - This bug was discovered and fixed previously - documented in devlog.md

2. **Normalized (for STT)**:
   - Converts int16 → float32 WITH division by 32768.0
   - Creates -1.0 to 1.0 range
   - Used by speech-to-text service

**Solution**: Create TWO utility functions in a new `src/hey_orac/audio/conversion.py` module.

## Audio Conversion Explained

### Stereo to Mono Conversion
When audio comes from a stereo microphone (2 channels: left and right), we need to convert it to mono (1 channel) by averaging the left and right channels:

```python
# Stereo data: [L1, R1, L2, R2, L3, R3, ...]
stereo_data = audio_array.reshape(-1, 2)  # [[L1, R1], [L2, R2], [L3, R3], ...]
mono_data = np.mean(stereo_data, axis=1)  # [(L1+R1)/2, (L2+R2)/2, (L3+R3)/2, ...]
```

### Data Type Conversion

**OpenWakeWord (Un-normalized)**:
```python
# int16 range: -32768 to 32767
# float32 range: -32768.0 to 32767.0 (same values, different type)
audio_data = mono_data.astype(np.float32)  # NO division!
```

**STT (Normalized)**:
```python
# int16 range: -32768 to 32767
# float32 range: -1.0 to 1.0 (normalized)
audio_data = mono_data.astype(np.float32) / 32768.0  # WITH division!
```

### Why This Matters

**CRITICAL**: The OpenWakeWord model was trained on un-normalized audio. If you normalize the audio (divide by 32768.0), the model will fail to detect wake words. This bug was discovered during development and is documented in devlog.md.

## Locations to Consolidate

### OpenWakeWord Locations (Un-normalized) - 3 locations in `wake_word_detection.py`

**Location 1: `record_test_audio()` function (lines 203-208)**
```python
audio_array = np.frombuffer(data, dtype=np.int16)
if len(audio_array) > constants.CHUNK_SIZE:  # Stereo
    stereo_data = audio_array.reshape(-1, 2)
    audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
else:
    audio_data = audio_array.astype(np.float32)
```
- **Purpose**: Records 10s test audio with real-time OpenWakeWord detection
- **Used by**: `-record_test` CLI flag
- **Replace with**: `convert_to_openwakeword_format(data)`

**Location 2: `load_test_audio()` function (lines 329-338)**
```python
audio_array = np.frombuffer(frames, dtype=np.int16)

if channels == 2:
    stereo_data = audio_array.reshape(-1, 2)
    audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
else:
    audio_data = audio_array.astype(np.float32)
```
- **Purpose**: Loads WAV file for offline pipeline testing
- **Used by**: `-test_pipeline` CLI flag
- **Replace with**: `convert_to_openwakeword_format(frames, channels=channels)`

**Location 3: Main detection loop (lines 1030-1051) - MOST CRITICAL**
```python
audio_array = np.frombuffer(data, dtype=np.int16)

if args.input_wav and hasattr(stream, 'channels'):
    if stream.channels == constants.CHANNELS_STEREO and len(audio_array) > constants.CHUNK_SIZE:
        stereo_data = audio_array.reshape(-1, 2)
        audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
    else:
        audio_data = audio_array.astype(np.float32)
else:
    if len(audio_array) > constants.CHUNK_SIZE:
        stereo_data = audio_array.reshape(-1, 2)
        # CRITICAL FIX: OpenWakeWord expects raw int16 values as float32, NOT normalized!
        audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
    else:
        # Already mono - CRITICAL FIX: no normalization!
        audio_data = audio_array.astype(np.float32)
```
- **Purpose**: **PRODUCTION** live wake word detection
- **MOST CRITICAL**: This is the core functionality
- **Replace with**: `convert_to_openwakeword_format(data)`

### STT Locations (Normalized) - NOT part of this sprint

These are already in separate modules and don't need consolidation:
- `src/hey_orac/audio/speech_recorder.py` (lines 152-161) - correctly normalized
- `src/hey_orac/audio/microphone.py` (lines 196-205) - correctly normalized

**Note**: We'll create the normalized conversion function for completeness, but we won't refactor these locations in this sprint.

## Implementation Steps

### Step 1: Create Conversion Module (10 minutes)

Create `src/hey_orac/audio/conversion.py`:

```python
"""
Audio format conversion utilities for Hey ORAC.

This module provides audio conversion functions for different components:
- OpenWakeWord: Un-normalized float32 (raw int16 values as float32)
- STT: Normalized float32 (-1.0 to 1.0 range)

CRITICAL: OpenWakeWord requires un-normalized audio! The model was trained on
raw int16 values converted to float32 WITHOUT normalization. Normalizing the
audio (dividing by 32768.0) will break wake word detection.

Historical context: This normalization bug was discovered during development
and caused wake word detection failures. See devlog.md for details.
"""

import numpy as np
from hey_orac import constants


def convert_to_openwakeword_format(
    audio_data: bytes,
    channels: int = None
) -> np.ndarray:
    """
    Convert audio bytes to OpenWakeWord-compatible float32 format.

    CRITICAL: This function does NOT normalize the audio. OpenWakeWord expects
    raw int16 values converted to float32 without division by 32768.0.

    Args:
        audio_data: Raw audio bytes from microphone or WAV file
        channels: Number of audio channels (1=mono, 2=stereo).
                 If None, auto-detect based on array length.

    Returns:
        Mono float32 numpy array suitable for OpenWakeWord

    Examples:
        # Auto-detect stereo/mono from chunk size
        >>> audio = convert_to_openwakeword_format(data)

        # Explicit channel count (useful for WAV files)
        >>> audio = convert_to_openwakeword_format(frames, channels=2)
    """
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Determine if stereo based on channels parameter or array length
    if channels is not None:
        is_stereo = (channels == constants.CHANNELS_STEREO)
    else:
        # Auto-detect: stereo data has 2x samples (left + right)
        is_stereo = len(audio_array) > constants.CHUNK_SIZE

    if is_stereo:
        # Convert stereo to mono by averaging left and right channels
        stereo_data = audio_array.reshape(-1, 2)
        audio_mono = np.mean(stereo_data, axis=1).astype(np.float32)
    else:
        # Already mono, just convert type
        audio_mono = audio_array.astype(np.float32)

    # CRITICAL FIX: NO normalization! OpenWakeWord needs raw int16→float32
    # DO NOT divide by 32768.0 here!
    return audio_mono


def convert_to_normalized_format(
    audio_data: bytes,
    channels: int = None
) -> np.ndarray:
    """
    Convert audio bytes to normalized float32 format for STT.

    This function normalizes the audio to the -1.0 to 1.0 range expected
    by speech-to-text services.

    Args:
        audio_data: Raw audio bytes from microphone or WAV file
        channels: Number of audio channels (1=mono, 2=stereo).
                 If None, auto-detect based on array length.

    Returns:
        Mono float32 numpy array normalized to -1.0 to 1.0 range
    """
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Determine if stereo
    if channels is not None:
        is_stereo = (channels == constants.CHANNELS_STEREO)
    else:
        is_stereo = len(audio_array) > constants.CHUNK_SIZE

    if is_stereo:
        # Convert stereo to mono
        stereo_data = audio_array.reshape(-1, 2)
        audio_mono = np.mean(stereo_data, axis=1).astype(np.float32)
    else:
        audio_mono = audio_array.astype(np.float32)

    # Normalize to -1.0 to 1.0 range
    return audio_mono / 32768.0
```

**Key features of this implementation**:
- Two separate functions for the two different use cases
- Extensive documentation explaining the CRITICAL no-normalization requirement
- Handles both auto-detection and explicit channel specification
- Preserves the "CRITICAL FIX" comment from the original code
- References devlog.md for historical context

### Step 2: Update wake_word_detection.py (15 minutes)

Add import at the top of the file:
```python
from hey_orac.audio.conversion import convert_to_openwakeword_format
```

**Replace Location 1** (`record_test_audio()` function, lines 203-208):
```python
# OLD CODE (delete these 6 lines):
audio_array = np.frombuffer(data, dtype=np.int16)
if len(audio_array) > constants.CHUNK_SIZE:  # Stereo
    stereo_data = audio_array.reshape(-1, 2)
    audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
else:
    audio_data = audio_array.astype(np.float32)

# NEW CODE (replace with this 1 line):
audio_data = convert_to_openwakeword_format(data)
```

**Replace Location 2** (`load_test_audio()` function, lines 329-338):
```python
# OLD CODE (delete these 7 lines):
audio_array = np.frombuffer(frames, dtype=np.int16)

if channels == 2:
    stereo_data = audio_array.reshape(-1, 2)
    audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
else:
    audio_data = audio_array.astype(np.float32)

# NEW CODE (replace with this 1 line):
audio_data = convert_to_openwakeword_format(frames, channels=channels)
```

**Replace Location 3** (Main detection loop, lines 1030-1051):
```python
# OLD CODE (delete these ~22 lines):
audio_array = np.frombuffer(data, dtype=np.int16)

if args.input_wav and hasattr(stream, 'channels'):
    if stream.channels == constants.CHANNELS_STEREO and len(audio_array) > constants.CHUNK_SIZE:
        stereo_data = audio_array.reshape(-1, 2)
        audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
    else:
        audio_data = audio_array.astype(np.float32)
else:
    if len(audio_array) > constants.CHUNK_SIZE:
        stereo_data = audio_array.reshape(-1, 2)
        # CRITICAL FIX: OpenWakeWord expects raw int16 values as float32, NOT normalized!
        audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
    else:
        # Already mono - CRITICAL FIX: no normalization!
        audio_data = audio_array.astype(np.float32)

# NEW CODE (replace with this 1 line):
audio_data = convert_to_openwakeword_format(data)
```

**Note**: The WAV file channel detection logic in the old Location 3 is redundant because `convert_to_openwakeword_format()` auto-detects stereo based on chunk size, which works for both microphone and WAV file input.

### Step 3: Deploy and Test (10 minutes)

**Commit and deploy**:
```bash
./scripts/deploy_and_test.sh "Sprint 7: Consolidate audio conversion - add conversion.py and update wake_word_detection.py"
```

**What to verify**:
1. Container builds and starts successfully
2. Wake word detection still works (say "hey computer" or your wake word)
3. Test recording works: `-record_test` flag
4. Pipeline testing works: `-test_pipeline` flag
5. STT integration works (audio sent to ORAC STT after wake word)
6. Web interface accessible on port 7171

**Check logs**:
```bash
# Watch real-time logs
ssh pi "cd ~/hey-orac && docker-compose logs -f hey-orac"

# Check last 50 lines
ssh pi "cd ~/hey-orac && docker-compose logs --tail=50 hey-orac"

# Check for errors
ssh pi "cd ~/hey-orac && docker-compose logs hey-orac | grep -i error"
```

**Test wake word detection**:
```bash
# Watch logs while saying wake word
ssh pi "cd ~/hey-orac && docker-compose logs -f hey-orac | grep -i 'wake word\\|detected\\|confidence'"
```

### Step 4: Update Documentation (5 minutes)

**Update CLEANUP.md**:
- Mark Sprint 7 as complete
- Update statistics (7/14 complete = 50%)
- Add completion notes

**Update devlog.md**:
Add entry at the bottom:
```markdown
## 2025-10-16 - Sprint 7: Consolidated Audio Conversion Logic

- Created `src/hey_orac/audio/conversion.py` with two functions:
  - `convert_to_openwakeword_format()` - Un-normalized for wake word detection
  - `convert_to_normalized_format()` - Normalized for STT
- Replaced 3 duplicate conversion locations in `wake_word_detection.py`:
  - Location 1: `record_test_audio()` function (lines 203-208)
  - Location 2: `load_test_audio()` function (lines 329-338)
  - Location 3: Main detection loop (lines 1030-1051) - CRITICAL production code
- Preserved CRITICAL FIX documentation: OpenWakeWord requires un-normalized audio
- Deployed and tested on Raspberry Pi - all functionality working
- Status: Sprint 7 complete (7/14 sprints = 50% complete)
```

**Commit documentation**:
```bash
git add CLEANUP.md devlog.md
git commit -m "Update CLEANUP.md and devlog.md: Sprint 7 complete (7/14)"
git push
```

## Expected Outcome

After Sprint 7:
- ✅ Created `src/hey_orac/audio/conversion.py` with 2 utility functions
- ✅ Removed ~35 lines of duplicate code from `wake_word_detection.py`
- ✅ Consolidated 3 OpenWakeWord conversion locations into 1 function call each
- ✅ Preserved CRITICAL FIX documentation about no normalization
- ✅ Application still works perfectly on Pi
- ✅ CLEANUP.md updated (7/14 sprints complete = 50%)
- ✅ devlog.md updated with sprint details
- ✅ Clean git history with descriptive commits

## Known Good State

The application currently works with these features:
- ✅ Wake word detection (OpenWakeWord)
- ✅ Audio processing (stereo→mono conversion)
- ✅ Multi-consumer audio distribution (AudioReaderThread)
- ✅ STT integration with ORAC STT service
- ✅ Web interface (Flask-SocketIO on port 7171)
- ✅ Heartbeat sender (registers models with ORAC STT)
- ✅ Configuration management (SettingsManager)
- ✅ Health checks (STT, audio thread, models)
- ✅ Constants extracted (42 named constants from Sprint 6)

**Don't break these!** Test thoroughly after changes.

## Deployment and Testing Process

**CRITICAL**: The application runs on the Raspberry Pi, **NOT locally**. Always use the deployment script.

### Primary Deployment Command
```bash
./scripts/deploy_and_test.sh "Sprint 7: description of changes"
```

**What `deploy_and_test.sh` does**:
1. ✅ Commits changes locally with your message
2. ✅ Pushes to current branch (`code-cleanup`) on GitHub
3. ✅ SSHs to Raspberry Pi (alias: `pi`)
4. ✅ Pulls latest code from GitHub on the Pi
5. ✅ Builds Docker container (with smart caching)
6. ✅ Starts container with `docker-compose up -d`
7. ✅ Runs health checks
8. ✅ Shows initial logs

### SSH Commands for Pi Monitoring

```bash
# View real-time logs (most useful for debugging)
ssh pi "cd ~/hey-orac && docker-compose logs -f hey-orac"

# View last 30 lines of logs
ssh pi "cd ~/hey-orac && docker-compose logs --tail=30 hey-orac"

# Check container status
ssh pi "cd ~/hey-orac && docker-compose ps"

# Check current git commit on Pi
ssh pi "cd ~/hey-orac && git log -1 --oneline"

# Restart container (without rebuild)
ssh pi "cd ~/hey-orac && docker-compose restart hey-orac"

# Stop container
ssh pi "cd ~/hey-orac && docker-compose down"

# Force full rebuild manually
ssh pi "cd ~/hey-orac && docker-compose down && docker-compose build --no-cache && docker-compose up -d"
```

## Deployment Troubleshooting

**If deployment fails**:
```bash
# Check what went wrong
ssh pi "cd ~/hey-orac && docker-compose logs --tail=100 hey-orac"

# Rollback locally
git reset --hard HEAD^

# Force deploy previous commit
./scripts/deploy_and_test.sh "Rollback: reverting Sprint 7 changes"
```

**If wake word detection stops working**:
- Check logs for errors
- Verify the conversion function is being called
- Ensure no normalization (division by 32768.0) was accidentally added
- Rollback and compare the audio data values before/after

## Important Notes

1. **Always use `deploy_and_test.sh`** - Never test locally, the app runs on Pi
2. **Check logs after deployment** - Use `ssh pi "cd ~/hey-orac && docker-compose logs --tail=30 hey-orac"`
3. **Test wake word detection** - Say the wake word and verify it's detected in logs
4. **Preserve CRITICAL FIX comments** - The no-normalization requirement is documented for a reason
5. **Update both CLEANUP.md and devlog.md** - After completing sprint
6. **Commit descriptively** - Use format: "Sprint X: brief description of changes"

## Historical Context

**Why this matters**: During development, there was a bug where OpenWakeWord detection failed because the audio was being normalized (divided by 32768.0). After debugging, we discovered that OpenWakeWord expects raw int16 values converted to float32 WITHOUT normalization.

This bug has been fixed multiple times because the documentation wasn't prominent enough. The new `conversion.py` module puts this critical information front and center in:
- Module docstring
- Function docstring
- Inline comments
- devlog.md entry

See `devlog.md` for the full historical context of this normalization bug discovery.

## Next Steps After Sprint 7

If you complete Sprint 7, the remaining **MEDIUM PRIORITY** sprints are:
- Sprint 8-9: Refactor Massive Main Function (break up 975-line main() function)
- Sprint 10: Fix Inconsistent Naming (standardize variable/function names)
- Sprint 11: Add Type Hints (improve code maintainability)
- Sprint 12: Extract Model Operations (separate concerns)
- Sprint 13: Extract Configuration (settings management)
- Sprint 14: Final Cleanup (remove TODO comments, fix remaining issues)

---

## Quick Command Reference

```bash
# Deploy changes
./scripts/deploy_and_test.sh "Sprint 7: consolidated audio conversion logic"

# Watch logs in real-time
ssh pi "cd ~/hey-orac && docker-compose logs -f hey-orac"

# Check last 30 log lines
ssh pi "cd ~/hey-orac && docker-compose logs --tail=30 hey-orac"

# Check container status
ssh pi "cd ~/hey-orac && docker-compose ps"

# Check deployed commit
ssh pi "cd ~/hey-orac && git log -1 --oneline"

# Rollback if broken
git reset --hard HEAD^
./scripts/deploy_and_test.sh "Rollback: reverting broken changes"
```

---

**Ready to start Sprint 7!** Create the conversion module, update the three locations, deploy, and test thoroughly.
