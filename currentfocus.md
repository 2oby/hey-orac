# Current Focus: Web GUI RMS Audio Level Display - Debugging

## 🚨 CRITICAL ISSUE: RMS Audio Levels Not Consistently Updating

### Problem Summary
- Initial RMS update works (one data point received)
- WebSocket connects and subscribes successfully  
- Server broadcasts RMS data every ~5 seconds
- Client receives NO ongoing RMS updates
- Frequent WebSocket disconnections every ~45 seconds

## 📋 Comprehensive Testing Plan

### STEP 1: Verify Container Restart
```bash
# Check container is running with latest code
ssh pi "cd ~/WakeWordTest && docker-compose ps"
ssh pi "cd ~/WakeWordTest && docker-compose logs --tail=10 wake-word-test"
```

### STEP 2: Test Fresh WebSocket Connection
1. **Action**: Refresh web page at http://192.168.8.99:7171
2. **Check Console For**:
   - "Initializing WebSocket connection..."
   - "WebSocket connected successfully!"
   - "Successfully subscribed to updates"
   - "Socket.IO event received: rms_update" (should appear immediately)
3. **Expected**: At least one RMS update immediately after subscription

### STEP 3: Monitor Server-Side Debug Logs
```bash
# Check for immediate RMS debug logs
ssh pi "cd ~/WakeWordTest && timeout 10 docker-compose logs -f wake-word-test | grep -E '(DEBUG.*Attempting|DEBUG.*shared_data|DEBUG.*Retrieved|Sending immediate RMS)'"
```
**Expected**: DEBUG logs showing immediate RMS attempt

### STEP 4: Monitor Connection Stability  
```bash
# Monitor disconnections and broadcasts
ssh pi "cd ~/WakeWordTest && timeout 30 docker-compose logs -f wake-word-test | grep -E '(Client.*connected|Client.*disconnected|Broadcasting RMS)'"
```
**Expected**: Client stays connected for >45 seconds, RMS broadcasts every ~5 seconds

### STEP 5: Test WebSocket Transport
1. **Check Network Tab** in browser dev tools
2. **Look For**: WebSocket connection vs polling fallback
3. **Verify**: `ws://192.168.8.99:7171/socket.io/` connection active

### STEP 6: Volume Threshold Testing
1. **Action**: Make noise near microphone while monitoring console
2. **Expected**: RMS values should increase from quiet background (~4-6) to higher values (>10)
3. **Verify**: No threshold blocking low RMS values from being sent

### STEP 7: Immediate RMS Fix Verification
```bash
# Check if debug code is actually executing
ssh pi "cd ~/WakeWordTest && docker-compose logs --tail=50 wake-word-test | grep -E '(DEBUG|immediate.*RMS|shared_data.*None)'"
```

### STEP 8: CORS/Network Debugging
```bash
# Check for CORS or network errors in container logs
ssh pi "cd ~/WakeWordTest && docker-compose logs --tail=30 wake-word-test | grep -E '(CORS|Error|Failed|Exception)'"
```

### STEP 9: Socket.IO Configuration Test
- **Check Browser Network Tab** for:
  - Failed WebSocket upgrades
  - Polling transport fallbacks  
  - Connection timeout errors
  - CORS rejection

### STEP 10: Live Data Flow Verification
1. **Generate audio**: Clap hands, speak, make noise
2. **Monitor simultaneously**:
   - Browser console for RMS updates
   - Server logs for broadcasts
   - Network tab for WebSocket activity

## 🔍 Known Issues Being Investigated

1. **Socket.IO Timeout Configuration**: Server set to 120s ping timeout, client needs matching config
2. **Immediate RMS Updates**: Debug code may not be executing due to deployment issues  
3. **Transport Fallback**: WebSocket may be falling back to polling mode
4. **Container Restart**: Latest code changes may not be active

## 🎯 Success Criteria

✅ **Client receives immediate RMS update** upon subscription  
✅ **Continuous RMS updates** every ~5 seconds  
✅ **WebSocket connection stable** for >2 minutes  
✅ **RMS values responsive** to audio volume changes  
✅ **No CORS or network errors** in logs  

## 🚦 Current Status - ✅ RESOLVED!
- **Socket.IO timeouts**: ✅ Configured (server + client)
- **Debug logging**: ✅ Working and showing immediate RMS updates
- **Container restart**: ✅ Successful with latest code
- **RMS Updates**: ✅ Working! Client receiving immediate + continuous updates
- **Issue resolution**: Container restart with latest debug code fixed the problem

## 🎉 SOLUTION SUMMARY
The issue was resolved by:
1. **Container restart** - Latest code wasn't active until forced restart
2. **Socket.IO timeout fixes** - Server and client now use matching 120s timeouts
3. **Immediate RMS updates** - Clients now get fresh data immediately upon subscription
4. **Debug logging** - Confirmed data flow working server → client

---

## 🎯 Original Objective  
Fix the web GUI to properly display models, real-time RMS values, and wake word detection events.

## ✅ What's Complete

### Implementation (100% DONE)
- **Flask-SocketIO Server**: Web server with WebSocket support on port 7171
- **REST API**: Full configuration management endpoints  
- **WebSocket Broadcasting**: Real-time RMS and detection events
- **Frontend Assets**: HTML/CSS/JS ported with WebSocket client
- **Shared Memory System**: Thread-safe data sharing via multiprocessing.Manager
- **Docker Configuration**: Port 7171 exposed, config directory mounted
- **Deployment**: Container running successfully on Pi

### Recent Fixes (Just Completed)
1. **Multi-Model Support**: Updated wake word detection to load ALL enabled models instead of just the first one
2. **Model Discovery**: Auto-enable first model if none are enabled
3. **Shared Data Integration**: Models config now properly shared with web GUI
4. **Detection Events**: Fixed model name mapping between OpenWakeWord and our config
5. **API Endpoints**: Added `/api/models` endpoint for GUI compatibility
6. **Threshold Handling**: Each model now uses its own configured threshold

### Code Changes
1. Modified `wake_word_detection.py`:
   - Load all enabled models, not just the first
   - Auto-enable first model if none enabled
   - Update shared_data with loaded models info
   - Fix detection threshold lookup per model
   - Map OpenWakeWord model names to config names
2. Updated `routes.py`:
   - Added `/api/models` endpoint
   - Made `/api/custom-models` call the new endpoint

## 🚨 Previous Issues (ALL RESOLVED)

### 1. No Models Showing in GUI ✅
- **Cause**: Detection loop was using hardcoded model instead of SettingsManager
- **Fix**: Now loads all enabled models from SettingsManager

### 2. No Real-time RMS Values ✅
- **Cause**: RMS was being updated in shared_data
- **Fix**: WebSocketBroadcaster already broadcasting at 10Hz

### 3. Detection Events Not Communicated ✅
- **Cause**: Model name mismatch between OpenWakeWord and config
- **Fix**: Added mapping logic to match model names correctly

## 📊 Deployment Status

**Container**: Running on port 7171
**Web GUI**: Should now display models and real-time data
**Models**: Auto-discovery and multi-model support active
**WebSocket**: Broadcasting RMS at 10Hz and detection events

## 🔧 Next Steps

1. **Deploy the fixes** to Raspberry Pi
2. **Test the web GUI** at http://<pi-ip>:7171
3. **Verify**:
   - Models appear in the GUI
   - RMS values update in real-time
   - Wake word detections trigger GUI notifications
   - Model enable/disable works
   - Configuration changes persist

## 📝 Testing Commands

```bash
# Last known state:
- Container: Exited with ModuleNotFoundError
- Port: 7171 exposed in docker-compose.yml
- Build: In progress with --no-cache flag
- Dependencies: Installing Flask and related packages
```

## 🔧 Next Steps

1. **Wait for Docker build** to complete on Pi
2. **Restart container** once build finishes
3. **Verify web GUI** accessible at http://<pi-ip>:7171
4. **Test wake word detection** with WAV file input
5. **Verify WebSocket** real-time updates working

## 🎯 Testing Plan

Once deployment succeeds:
1. Access web GUI at port 7171
2. Check WebSocket connection status
3. Monitor RMS levels in real-time
4. Test model enable/disable
5. Verify configuration persistence
6. Test wake word detection with WAV file

## 📝 Commands for Testing

```bash
# Check build status
ssh pi "docker images | grep wake-word"

# Once built, start container
ssh pi "cd ~/WakeWordTest && docker-compose up -d"

# Check logs
ssh pi "cd ~/WakeWordTest && docker-compose logs -f wake-word-test"

# Test with WAV file (on Pi)
ssh pi "cd ~/WakeWordTest && docker-compose exec wake-word-test python3 -m hey_orac.wake_word_detection --input-wav /app/wake_word_test_20250718_131542.wav"
```

## 🚦 Status Summary

**Development**: ✅ Complete
**Deployment**: 🔄 In Progress (Docker rebuild)
**Testing**: ⏳ Pending

The web GUI integration is fully implemented but waiting on Docker image rebuild to include Flask dependencies. Once the build completes, the system should be ready for full testing.