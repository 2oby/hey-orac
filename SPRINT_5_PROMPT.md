# Sprint 5: Resolve Dual Entry Points

## Context from Previous Session

You are continuing the Hey Orac code cleanup project. **Sprints 1-4 are complete** and the application is working perfectly on the Raspberry Pi.

**📋 IMPORTANT**: This sprint is part of a 14-sprint cleanup plan documented in **`CLEANUP.md`**. Read that file first for the full context, testing protocol, and complete sprint breakdown. This document provides focused context for Sprint 5 specifically.

### Project Overview
Hey Orac is a wake word detection service running on a Raspberry Pi in a Docker container. It:
- Detects wake words using OpenWakeWord models
- Sends audio to ORAC STT service for speech-to-text processing
- Provides a web interface for monitoring (port 7171)
- Sends heartbeats to register wake word models with ORAC STT

### Current Branch Status
- **Working Branch**: `code-cleanup`
- **Completed Sprints**: 4/14 (all HIGH PRIORITY sprints done)
- **Sprint Progress Tracked In**: `CLEANUP.md`
- **Git Commits**: Clean history with descriptive messages

### Important Files Context

#### Entry Point Files (SPRINT 5 FOCUS)
The codebase has **three potential entry points** and we need to determine which are actually used:

1. **`src/hey_orac/wake_word_detection.py`** (1413 lines)
   - **Main production entry point** (confirmed by Dockerfile)
   - Integrated system with:
     - Wake word detection loop
     - Web server (Flask-SocketIO on port 7171)
     - STT components (speech recorder, ring buffer)
     - Heartbeat sender for ORAC STT registration
     - Audio reader thread with multi-consumer audio distribution
     - Settings manager for configuration
   - Command: `python -m hey_orac.wake_word_detection` (in Docker)

2. **`src/hey_orac/app.py`** (196 lines)
   - Contains `HeyOracApplication` class
   - **Status**: Appears to be unused/deprecated
   - Has TODOs: "Implement config loading from JSON", "Start speech capture", "Implement speech capture and streaming"
   - Has hardcoded config (lines 65-79)
   - **Investigation needed**: Check if this is ever actually used

3. **`src/hey_orac/cli.py`** (unknown - not read yet)
   - **Investigation needed**: Check if this is ever actually used
   - May be a command-line interface wrapper

#### Key Supporting Files
- `docker-compose.yml` - Check this for the actual Docker command
- `Dockerfile` - Check this for the CMD or ENTRYPOINT
- `src/hey_orac/config/manager.py` - SettingsManager (used by wake_word_detection.py)
- `src/hey_orac/heartbeat_sender.py` - Heartbeat sender (used by wake_word_detection.py)
- `src/hey_orac/web/app.py` - Web server components (used by wake_word_detection.py)

### Deployment and Testing Process

**CRITICAL**: The application runs on the Raspberry Pi, **NOT locally**. Always use the deployment script.

#### Primary Deployment Command
```bash
./scripts/deploy_and_test.sh "Sprint X: description of changes"
```

**What `deploy_and_test.sh` does**:
1. ✅ Commits changes locally with your message
2. ✅ Pushes to current branch (`code-cleanup`) on GitHub
3. ✅ SSHs to Raspberry Pi (alias: `pi`)
4. ✅ Pulls latest code from GitHub on the Pi
5. ✅ Builds Docker container (with smart caching)
6. ✅ Starts container with `docker-compose up -d`
7. ✅ Runs health checks:
   - Audio device detection
   - PyAudio initialization
   - OpenWakeWord model loading
8. ✅ Shows initial logs

**Smart Build Detection**: The script detects what changed and:
- **Full rebuild** (`--no-cache`): If `requirements.txt`, `Dockerfile`, or first deployment
- **Incremental rebuild**: If Python source code or models changed
- **Cache only**: If only config files changed

#### SSH Commands for Pi Monitoring

The Pi is accessible via SSH with the alias `pi`:

```bash
# View real-time logs (most useful for debugging)
ssh pi "cd ~/hey-orac && docker-compose logs -f hey-orac"

# View last 30 lines of logs
ssh pi "cd ~/hey-orac && docker-compose logs --tail=30 hey-orac"

# Check container status
ssh pi "cd ~/hey-orac && docker-compose ps"

# Check current git commit on Pi
ssh pi "cd ~/hey-orac && git log -1 --oneline"

# Check current branch on Pi
ssh pi "cd ~/hey-orac && git branch --show-current"

# Restart container (without rebuild)
ssh pi "cd ~/hey-orac && docker-compose restart hey-orac"

# Stop container
ssh pi "cd ~/hey-orac && docker-compose down"

# Force full rebuild manually
ssh pi "cd ~/hey-orac && docker-compose down && docker-compose build --no-cache && docker-compose up -d"

# Check Pi system resources
ssh pi "free -h"  # Memory usage
ssh pi "df -h"    # Disk usage
```

#### Git Branch Management

```bash
# Current branch (local and remote)
git branch  # Shows you're on 'code-cleanup'

# If something breaks, rollback
git reset --hard HEAD^     # Undo last commit
git reset --hard HEAD~3    # Undo last 3 commits

# Check status
git status

# View recent commits
git log --oneline -10
```

### Sprint 5 Investigation Tasks

**Goal**: Determine which entry points are actually used and clean up unused code.

#### Step 1: Investigation Phase (15 minutes)

1. **Check Docker configuration**:
   ```bash
   # Read Dockerfile to see actual command
   cat Dockerfile | grep -A 5 "CMD\|ENTRYPOINT"

   # Read docker-compose.yml to see command override
   cat docker-compose.yml | grep -A 10 "hey-orac:"
   ```

2. **Check if app.py is imported anywhere**:
   ```bash
   # Search for imports of app.py
   grep -r "from hey_orac.app import\|from hey_orac import app\|import hey_orac.app" src/
   ```

3. **Check if cli.py is imported anywhere**:
   ```bash
   # Search for imports of cli.py
   grep -r "from hey_orac.cli import\|from hey_orac import cli\|import hey_orac.cli" src/
   ```

4. **Read cli.py to understand its purpose**:
   ```bash
   # Use the Read tool to examine cli.py
   ```

5. **Check if there are any tests that use these files**:
   ```bash
   # Search for test files
   find . -name "*test*.py" -o -name "test_*"
   ```

#### Step 2: Decision Making

Based on investigation, choose **ONE** of these options:

**Option A: Delete Unused Entry Points** (if app.py and cli.py are not used)
- Delete `src/hey_orac/app.py` (196 lines)
- Delete `src/hey_orac/cli.py` (if unused)
- Keep `wake_word_detection.py` as the single entry point
- Update README if it mentions wrong entry points

**Option B: Migrate to Proper Architecture** (if app.py should be the entry point)
- Migrate web server, STT, heartbeat from `wake_word_detection.py` to `app.py`
- Update `HeyOracApplication` class to use SettingsManager
- Implement the TODOs in app.py
- Update Dockerfile to use app.py
- Rename `wake_word_detection.py` to `standalone.py` or similar

**Option C: Keep All But Document** (if multiple entry points serve different purposes)
- Document in README when to use each entry point
- Fix the TODOs in app.py so it actually works
- Add command-line help explaining options

#### Step 3: Implementation (15 minutes)

**If choosing Option A (Delete Unused - most likely)**:

1. **Commit baseline**:
   ```bash
   git add -A
   git commit -m "Pre-Sprint 5: Baseline before resolving dual entry points"
   ```

2. **Delete unused files**:
   ```bash
   git rm src/hey_orac/app.py
   git rm src/hey_orac/cli.py  # if unused
   ```

3. **Deploy and test**:
   ```bash
   ./scripts/deploy_and_test.sh "Sprint 5: Remove unused entry points (app.py, cli.py)"
   ```

4. **Verify on Pi**:
   ```bash
   # Check logs - application should start normally
   ssh pi "cd ~/hey-orac && docker-compose logs --tail=50 hey-orac"

   # Verify wake word detection still works
   # (Say "hey computer" or your configured wake word)

   # Check web interface is accessible
   # http://<pi-ip>:7171
   ```

5. **Update CLEANUP.md**:
   - Mark Sprint 5 as complete
   - Update statistics (5/14 complete)
   - Commit the update

**If choosing Option B or C**: Follow similar pattern with appropriate implementation steps.

### Expected Outcome

After Sprint 5:
- ✅ Clear understanding of entry point architecture
- ✅ Removed unused code OR properly documented multiple entry points
- ✅ Application still works perfectly on Pi
- ✅ CLEANUP.md updated (5/14 sprints complete)
- ✅ Clean git history with descriptive commit

### Known Good State

The application currently works with these features:
- ✅ Wake word detection (OpenWakeWord)
- ✅ Audio processing (stereo→mono conversion)
- ✅ Multi-consumer audio distribution (AudioReaderThread)
- ✅ STT integration with ORAC STT service
- ✅ Web interface (Flask-SocketIO on port 7171)
- ✅ Heartbeat sender (registers models with ORAC STT)
- ✅ Configuration management (SettingsManager)
- ✅ Health checks (STT, audio thread, models)

**Don't break these!** Test thoroughly after changes.

### Deployment Troubleshooting

**If deployment fails**:
```bash
# Check what went wrong
ssh pi "cd ~/hey-orac && docker-compose logs --tail=100 hey-orac"

# Rollback locally
git reset --hard HEAD^

# Force deploy previous commit
./scripts/deploy_and_test.sh "Rollback: reverting Sprint 5 changes"
```

**If container won't start**:
```bash
# Check Docker status
ssh pi "cd ~/hey-orac && docker-compose ps"

# Check build logs
ssh pi "cd ~/hey-orac && docker-compose logs hey-orac"

# Try manual rebuild
ssh pi "cd ~/hey-orac && docker-compose down && docker-compose build --no-cache && docker-compose up -d"
```

### Files Changed in Previous Sprints

For context, here's what was changed in Sprints 1-4:

**Sprint 1** (Deleted files):
- ❌ `src/hey_orac/wake_word_detection_backup.py` (760 lines)
- ❌ `fix_indentation.py`, `fix_indent.py`, `fix_try_block.py`, `fix_all_indent.py`

**Sprint 2** (Modified):
- ✏️ `src/hey_orac/wake_word_detection.py`: Removed 6 debug print statements

**Sprint 3** (Modified):
- ✏️ `src/hey_orac/audio/utils.py`: Added `close()` method to AudioManager
- ✏️ `src/hey_orac/wake_word_detection.py`: Changed `audio_manager.__del__()` to `audio_manager.close()` (2 locations)

**Sprint 4** (Modified):
- ✏️ `src/hey_orac/wake_word_detection.py`:
  - Added `import queue`
  - Replaced `except:` with `except queue.Empty:` (line 1010)
  - Replaced `except: pass` with `except queue.Full:` + logging (lines 1151, 1270)

### Important Notes

1. **Always use `deploy_and_test.sh`** - Never test locally, the app runs on Pi
2. **Check logs after deployment** - Use `ssh pi "cd ~/hey-orac && docker-compose logs --tail=30 hey-orac"`
3. **Test wake word detection** - Say the wake word and verify it's detected in logs
4. **Update CLEANUP.md** - After completing sprint, update progress tracking
5. **Commit descriptively** - Use format: "Sprint X: brief description of changes"

### Next Steps After Sprint 5

If you complete Sprint 5, the remaining **MEDIUM PRIORITY** sprints are:
- Sprint 6: Extract Constants (magic numbers → named constants)
- Sprint 7: Consolidate Audio Conversion (DRY principle)
- Sprint 8-9: Refactor Massive Main Function (break up 975-line main() function)

You can tackle these one at a time or ask the user what they'd like to focus on next.

---

## Quick Command Reference

```bash
# Deploy changes
./scripts/deploy_and_test.sh "Sprint 5: description"

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

## Additional Resources

- **📋 CLEANUP.md** - Master cleanup plan with all 14 sprints, testing protocol, and completion status
- **🔧 CLAUDE.md** - Project instructions and development guidelines
- **📜 README.md** - Project overview and setup instructions (may need updating after this sprint)

---

**Ready to start Sprint 5!** Begin with the investigation phase to determine which entry points are actually used.

After completing Sprint 5, update `CLEANUP.md` to mark this sprint as complete (5/14) and update the statistics section.
