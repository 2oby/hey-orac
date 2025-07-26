# Hey ORAC Deployment Strategy

## Overview

This document outlines the improved deployment strategy that balances build speed with ensuring important changes are properly compiled.

## Build Strategies

### 1. **Full Rebuild** (`--no-cache`)
**When used:**
- Dependencies changed (`requirements.txt`, `pyproject.toml`)
- Dockerfile modified
- First deployment (no previous commit)
- Manual override needed

**Build time:** ~10-15 minutes
**When to expect:** Infrequent, only when dependencies change

### 2. **Incremental Rebuild** (with cache busting)
**When used:**
- Python source code changed (`src/*.py`)
- Model files changed (`models/`)
- Configuration files changed

**Build time:** ~2-5 minutes
**When to expect:** Most common for code changes

### 3. **Cache-Only Build** (reuse all layers)
**When used:**
- Documentation changes
- Log files
- Non-critical files

**Build time:** ~30 seconds
**When to expect:** Minor changes, documentation updates

## Smart Change Detection

The deployment script automatically detects what changed and chooses the appropriate build strategy:

```bash
# Check what files changed since last deployment
git diff --name-only $LAST_COMMIT $CURRENT_COMMIT

# Build strategy decision tree:
if (requirements.txt || pyproject.toml || Dockerfile changed):
    → Full rebuild
elif (src/*.py || models/ changed):
    → Incremental rebuild
else:
    → Cache-only build
```

## Docker Layer Optimization

### Layer Caching Strategy
1. **System dependencies** (rarely change)
2. **Python dependencies** (change with requirements.txt)
3. **Source code** (changes frequently)
4. **Configuration** (changes occasionally)

### Build Context Optimization
- `.dockerignore` excludes unnecessary files
- Reduces build context size by ~80%
- Faster build context transfer to Docker daemon

## Usage

### Basic Deployment
```bash
./scripts/deploy_and_test.sh "Your commit message"
```

### Force Full Rebuild
```bash
# Edit requirements.txt or Dockerfile, then:
./scripts/deploy_and_test.sh "Force rebuild with new dependencies"
```

### Quick Code Changes
```bash
# Edit source code, then:
./scripts/deploy_and_test.sh "Fix bug in wake word detection"
```

## Monitoring

### Check Build Strategy Used
```bash
ssh pi "cd ~/hey-orac && cat .last_deploy_commit"
```

### View Build History
```bash
ssh pi "cd ~/hey-orac && docker images hey-orac_hey-orac"
```

### Monitor Resource Usage
```bash
ssh pi "df -h && docker system df"
```

## Best Practices

### 1. **Commit Strategy**
- Group related changes together
- Use descriptive commit messages
- Avoid mixing dependency changes with code changes

### 2. **Dependency Management**
- Pin versions in `requirements.txt`
- Update dependencies in separate commits
- Test dependency changes locally first

### 3. **Build Optimization**
- Keep Dockerfile changes minimal
- Use multi-stage builds effectively
- Leverage `.dockerignore` for large files

### 4. **Monitoring**
- Check build times regularly
- Monitor disk usage on Pi
- Review build logs for optimization opportunities

## Troubleshooting

### Build Takes Too Long
1. Check if full rebuild is necessary
2. Review `.dockerignore` for large files
3. Consider splitting large changes into smaller commits

### Build Fails
1. Check Docker disk space: `docker system df`
2. Clean up old images: `docker image prune -a`
3. Verify dependencies are compatible

### Container Won't Start
1. Check logs: `docker-compose logs hey-orac`
2. Verify audio device access
3. Check resource limits in docker-compose.yml

## Performance Metrics

### Expected Build Times (Raspberry Pi 4)
- **Full rebuild:** 10-15 minutes
- **Incremental rebuild:** 2-5 minutes
- **Cache-only build:** 30 seconds

### Disk Usage
- **Base image:** ~150MB
- **Dependencies:** ~500MB
- **Application:** ~50MB
- **Total:** ~700MB per build

### Memory Usage
- **Build process:** ~1GB peak
- **Runtime:** ~256MB-512MB
- **Available for Pi:** ~3GB total 