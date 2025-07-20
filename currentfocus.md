# Current Focus: Fix Docker Build Caching in deploy_and_test.sh

## 🚨 CRITICAL ISSUE: Container Running Old Code

### Problem Summary
- Container frequently runs old code after deployment
- Changes not reflected even after restart
- Need proper Docker build caching strategy

## 📋 Deploy Script Requirements

### MUST HAVE:
1. **Cache Dependencies**: Don't rebuild Python packages every time
2. **Rebuild on Code Changes**: Always rebuild when our source code changes
3. **Fast Deployment**: Minimize build time while ensuring fresh code

### Docker Build Strategy:
```dockerfile
# Good caching pattern:
COPY requirements.txt .
RUN pip install -r requirements.txt  # This layer cached if requirements unchanged
COPY src/ .  # This forces rebuild when code changes
```

### deploy_and_test.sh Improvements Needed:
1. **Smart Cache Invalidation**:
   - Use Docker build cache for dependencies
   - Force rebuild of application layer when code changes
   - Consider using `--cache-from` for better caching

2. **Verify Fresh Code**:
   - Add git commit hash to container
   - Log the commit hash on startup
   - Verify expected vs actual commit

3. **Restart Strategy**:
   - Stop container before pulling new code
   - Use `docker-compose up --force-recreate` when needed
   - Consider `docker-compose pull` before `up`

## 🔧 Proposed Solution

Update deploy_and_test.sh to:
```bash
# After git pull on Pi
docker-compose down
docker-compose build --build-arg CACHEBUST=$(date +%s) wake-word-test
docker-compose up -d
```

Or use a smarter approach:
```bash
# Only rebuild if source files changed
if git diff --name-only HEAD~1 | grep -E "(src/|requirements.txt)"; then
  docker-compose build wake-word-test
fi
docker-compose up -d --force-recreate wake-word-test
```

## 🎯 Success Criteria
✅ Code changes visible immediately after deployment
✅ Dependencies cached (no pip install if requirements unchanged)
✅ Build time under 2 minutes for code-only changes
✅ Container runs exact commit that was pushed

---

## Previous Issue: WebSocket Streaming Not Working

### Status: PARTIALLY FIXED
- Changed from eventlet to threading mode
- Removed eventlet from requirements.txt
- Added better client-side debugging

### Still TODO:
- Force Docker rebuild without eventlet
- Verify WebSocket messages reach client
- Test RMS streaming works continuously