# Claude Usage Instructions for WakeWordTest Project

## Project Context
This is an OpenWakeWord test implementation for Raspberry Pi deployment. The project uses Docker containers and automated deployment scripts to test wake word detection on a Raspberry Pi.

## Key Commands

### Deploy and Test Script
```bash
./scripts/deploy_and_test.sh [commit_message]
```
- **Purpose**: Automatically commits changes, pushes to git, deploys to Raspberry Pi, and runs tests
- **Example**: `./scripts/deploy_and_test.sh "Fix NumPy compatibility issue"`
- **What it does**:
  1. Commits and pushes local changes to `wake-word-test` branch
  2. SSHs to Pi and pulls latest code
  3. Builds Docker container with `--no-cache` if needed
  4. Runs comprehensive tests (audio devices, model loading, etc.)
  5. Provides status and monitoring commands

### SSH Commands for Pi
```bash
ssh pi "<command>"
```
- **Purpose**: Execute commands directly on the Raspberry Pi
- **Examples**:
  ```bash
  ssh pi "cd ~/WakeWordTest && docker-compose logs --tail=20 wake-word-test"
  ssh pi "cd ~/WakeWordTest && docker-compose ps"
  ssh pi "cd ~/WakeWordTest && docker-compose restart wake-word-test"
  ssh pi "cd ~/WakeWordTest && docker-compose down"
  ```

### Monitoring Commands
```bash
# Check container status
ssh pi "cd ~/WakeWordTest && docker-compose ps"

# View logs (real-time)
ssh pi "cd ~/WakeWordTest && docker-compose logs -f wake-word-test"

# View recent logs
ssh pi "cd ~/WakeWordTest && docker-compose logs --tail=50 wake-word-test"

# Restart container
ssh pi "cd ~/WakeWordTest && docker-compose restart wake-word-test"

# Force rebuild (no cache)
ssh pi "cd ~/WakeWordTest && docker-compose down && docker-compose build --no-cache && docker-compose up -d"
```

## File Management Instructions

### Current Focus Tracking
- **Read**: `currentfocus.md` contains the current issue being investigated
- **Purpose**: Helps Claude understand what problem is being solved
- **Update**: Modify this file when switching to a new problem or making significant progress

### Development Log
- **File**: `devlog.md`
- **Purpose**: Chronological log of all development activities
- **Format**: Always append new entries at the bottom with timestamp
- **Template**:
  ```markdown
  ## YYYY-MM-DD HH:MM - Brief Description
  - Bullet points describing what was done
  - Key findings or issues discovered
  - Current status and next steps
  ```

### Project Files
- **Source**: `src/wake_word_detection.py` - Main detection script
- **Config**: `requirements.txt`, `Dockerfile`, `docker-compose.yml`
- **Deployment**: `scripts/deploy_and_test.sh`

## Workflow for Claude
1. **Before starting**: Read `currentfocus.md` to understand current problem
2. **During work**: Use deploy script and SSH commands to test changes
3. **After progress**: Update `devlog.md` with timestamp and progress made
4. **When switching focus**: Update `currentfocus.md` with new problem description

## Git Branch
- **Working branch**: `wake-word-test` 
- **Repository**: https://github.com/2oby/hey-orac/tree/wake-word-test
- **Deployment**: Changes pushed to this branch are automatically deployed by the script