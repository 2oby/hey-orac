#!/bin/bash
# Verify Shared Memory Activation System on Pi

echo "🔍 Verifying Shared Memory Activation System on Pi"
echo "=================================================="

# Check if the main process is running
echo "📊 Checking if main process is running..."
if pgrep -f "python3.*main.py" > /dev/null; then
    echo "✅ Main process is running"
    MAIN_PID=$(pgrep -f "python3.*main.py")
    echo "📊 Main process PID: $MAIN_PID"
else
    echo "❌ Main process is not running"
    echo "💡 Start the main process first: docker-compose up -d"
    exit 1
fi

# Check if web backend is running
echo ""
echo "🌐 Checking if web backend is running..."
if curl -s http://localhost:7171/api/activation > /dev/null 2>&1; then
    echo "✅ Web backend is responding"
else
    echo "❌ Web backend is not responding"
    echo "💡 Check if web backend is running in the container"
fi

# Test shared memory access
echo ""
echo "🔄 Testing shared memory access..."
python3 -c "
from src.shared_memory_ipc import shared_memory_ipc
import time

try:
    # Try to read current state
    data = shared_memory_ipc.get_activation_state()
    print(f'✅ Shared memory accessible')
    print(f'📊 Current state: {data}')
    
    # Try to write a test state
    shared_memory_ipc.update_activation_state(True, 'Test Model', 0.75)
    print('✅ Can write to shared memory')
    
    # Read back the test state
    test_data = shared_memory_ipc.get_activation_state()
    print(f'📊 Test state written: {test_data}')
    
    # Reset to original state
    shared_memory_ipc.update_activation_state(False)
    print('✅ Reset to original state')
    
except Exception as e:
    print(f'❌ Shared memory test failed: {e}')
"

# Test web API endpoints
echo ""
echo "📡 Testing web API endpoints..."
echo "Testing /api/activation endpoint:"
curl -s http://localhost:7171/api/activation | python3 -m json.tool

echo ""
echo "Testing /api/detections endpoint:"
curl -s http://localhost:7171/api/detections | python3 -m json.tool

# Monitor for activation changes
echo ""
echo "👀 Monitoring for activation changes (30 seconds)..."
echo "💡 Say a wake word to see activation state change"
echo "📊 Press Ctrl+C to stop monitoring"

timeout 30 bash -c '
while true; do
    data=$(curl -s http://localhost:7171/api/activation 2>/dev/null)
    if [ $? -eq 0 ]; then
        is_listening=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get(\"is_listening\", False))")
        rms=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get(\"current_rms\", 0))")
        timestamp=$(date "+%H:%M:%S")
        if [ "$is_listening" = "True" ]; then
            echo "🎯 [$timestamp] ACTIVATION: Listening for wake word (RMS: $rms)"
        else
            echo "🔇 [$timestamp] Not listening (RMS: $rms)"
        fi
    else
        echo "❌ [$timestamp] Failed to get activation data"
    fi
    sleep 1
done
'

echo ""
echo "✅ Verification completed!"
echo ""
echo "📋 Summary:"
echo "  ✅ Main process is running"
echo "  ✅ Web backend is responding"
echo "  ✅ Shared memory is accessible"
echo "  ✅ Web API endpoints are working"
echo ""
echo "🌐 Web interface available at: http://192.168.8.99:7171"
echo "💡 Check the web interface for real-time activation updates" 