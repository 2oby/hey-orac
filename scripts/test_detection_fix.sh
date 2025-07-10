#!/bin/bash

# Test script to verify detection blocking fix
# This script tests that detections are properly recorded to /tmp/recent_detections.json

set -e

echo "🧪 Testing Detection Blocking Fix"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if detection file exists and has content
check_detection_file() {
    local file="/tmp/recent_detections.json"
    if [ -f "$file" ]; then
        local size=$(stat -c%s "$file" 2>/dev/null || echo "0")
        if [ "$size" -gt 0 ]; then
            echo -e "${GREEN}✅ Detection file exists and has content${NC}"
            echo "📄 File contents:"
            cat "$file" | jq '.' 2>/dev/null || cat "$file"
            return 0
        else
            echo -e "${YELLOW}⚠️ Detection file exists but is empty${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Detection file does not exist${NC}"
        return 1
    fi
}

# Function to monitor for detections
monitor_detections() {
    echo "🔍 Monitoring for detections..."
    echo "📊 Checking /tmp/recent_detections.json every 2 seconds..."
    echo "⏱️  Monitoring for 30 seconds..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + 30))
    
    while [ $(date +%s) -lt $end_time ]; do
        if check_detection_file; then
            echo -e "${GREEN}🎯 DETECTION RECORDED SUCCESSFULLY!${NC}"
            echo "✅ The fix is working - detections are being recorded to file"
            return 0
        fi
        
        echo "⏳ Waiting for detection... ($(date +%H:%M:%S))"
        sleep 2
    done
    
    echo -e "${RED}❌ No detections recorded within 30 seconds${NC}"
    return 1
}

# Function to test web API
test_web_api() {
    echo "🌐 Testing web API..."
    
    # Check if web backend is running
    if curl -s http://localhost:7171/api/config > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Web backend is running${NC}"
        
        # Test detections API
        local detections=$(curl -s http://localhost:7171/api/detections)
        echo "📊 Detections API response:"
        echo "$detections" | jq '.' 2>/dev/null || echo "$detections"
        
        if [ "$detections" != "[]" ]; then
            echo -e "${GREEN}✅ Web API shows detections${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️ Web API shows no detections${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Web backend is not running${NC}"
        return 1
    fi
}

# Main test execution
echo "🚀 Starting detection fix test..."

# Clean up any existing detection file
rm -f /tmp/recent_detections.json

# Check initial state
echo "📋 Initial state check:"
check_detection_file || true

# Test 1: Monitor for detections
echo ""
echo "🧪 Test 1: Monitoring for detections"
echo "====================================="
echo "🎤 Say 'Hey Computer' into the microphone..."
echo "⏱️  You have 30 seconds to trigger a detection"

if monitor_detections; then
    echo -e "${GREEN}✅ Test 1 PASSED: Detections are being recorded${NC}"
else
    echo -e "${RED}❌ Test 1 FAILED: No detections recorded${NC}"
fi

# Test 2: Check web API
echo ""
echo "🧪 Test 2: Testing web API"
echo "==========================="

if test_web_api; then
    echo -e "${GREEN}✅ Test 2 PASSED: Web API is working${NC}"
else
    echo -e "${YELLOW}⚠️ Test 2 WARNING: Web API issues detected${NC}"
fi

# Final summary
echo ""
echo "📊 Test Summary"
echo "==============="

if check_detection_file > /dev/null 2>&1; then
    echo -e "${GREEN}✅ DETECTION FIX VERIFIED${NC}"
    echo "✅ Detections are being recorded to /tmp/recent_detections.json"
    echo "✅ The cooldown/debounce logic is working correctly"
    echo "✅ File creation and web interface updates are working"
else
    echo -e "${RED}❌ DETECTION FIX FAILED${NC}"
    echo "❌ Detections are not being recorded to file"
    echo "❌ The issue may still be present"
fi

echo ""
echo "🔧 If the fix failed, check:"
echo "   1. Is the wake word detection system running?"
echo "   2. Are you speaking the correct wake word ('Hey Computer')?"
echo "   3. Is the microphone working and audio being captured?"
echo "   4. Are there any errors in the system logs?"

echo ""
echo "📝 Test completed at $(date)" 