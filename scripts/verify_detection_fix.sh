#!/bin/bash

# Verification script for detection blocking fix
# This script checks if detections are being recorded without interfering with the main loop

set -e

echo "🔍 Verifying Detection Blocking Fix"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check detection file status
check_detection_file() {
    local file="/tmp/recent_detections.json"
    echo -e "${BLUE}📊 Checking detection file: $file${NC}"
    
    if [ -f "$file" ]; then
        local size=$(stat -c%s "$file" 2>/dev/null || echo "0")
        local mod_time=$(stat -c%Y "$file" 2>/dev/null || echo "0")
        local current_time=$(date +%s)
        local age=$((current_time - mod_time))
        
        echo "   📄 File exists: ✅"
        echo "   📏 File size: ${size} bytes"
        echo "   ⏰ Last modified: ${age} seconds ago"
        
        if [ "$size" -gt 0 ]; then
            echo -e "${GREEN}   ✅ File has content${NC}"
            
            # Show file contents (last 3 detections)
            echo "   📋 Recent detections:"
            if command -v jq >/dev/null 2>&1; then
                jq '.[-3:]' "$file" 2>/dev/null || cat "$file"
            else
                tail -3 "$file" 2>/dev/null || cat "$file"
            fi
            return 0
        else
            echo -e "${YELLOW}   ⚠️ File is empty${NC}"
            return 1
        fi
    else
        echo -e "${RED}   ❌ File does not exist${NC}"
        return 1
    fi
}

# Function to test web API
test_web_api() {
    echo -e "${BLUE}🌐 Testing web API...${NC}"
    
    # Check if web backend is running
    if curl -s http://localhost:7171/api/config > /dev/null 2>&1; then
        echo "   ✅ Web backend is running"
        
        # Test detections API
        local detections=$(curl -s http://localhost:7171/api/detections 2>/dev/null || echo "[]")
        echo "   📊 Detections API response:"
        
        if command -v jq >/dev/null 2>&1; then
            echo "$detections" | jq '.' 2>/dev/null || echo "$detections"
        else
            echo "$detections"
        fi
        
        # Check if there are any detections
        local detection_count=$(echo "$detections" | jq 'length' 2>/dev/null || echo "0")
        if [ "$detection_count" -gt 0 ]; then
            echo -e "${GREEN}   ✅ Web API shows $detection_count detection(s)${NC}"
            return 0
        else
            echo -e "${YELLOW}   ⚠️ Web API shows no detections${NC}"
            return 1
        fi
    else
        echo -e "${RED}   ❌ Web backend is not running${NC}"
        return 1
    fi
}

# Function to check system logs for detection activity
check_system_logs() {
    echo -e "${BLUE}📋 Checking system logs for detection activity...${NC}"
    
    # Check for recent detection logs
    local log_files=(
        "/app/logs/custom_detections.log"
        "/app/logs/pipeline_detections.log"
        "/app/logs/hey-orac.log"
    )
    
    for log_file in "${log_files[@]}"; do
        if [ -f "$log_file" ]; then
            echo "   📄 Checking: $log_file"
            local recent_detections=$(grep -i "detected" "$log_file" | tail -3 2>/dev/null || echo "No detections found")
            if [ "$recent_detections" != "No detections found" ]; then
                echo -e "${GREEN}   ✅ Recent detections found in logs${NC}"
                echo "$recent_detections"
            else
                echo -e "${YELLOW}   ⚠️ No recent detections in logs${NC}"
            fi
        else
            echo "   📄 File not found: $log_file"
        fi
    done
}

# Function to provide instructions for testing
show_test_instructions() {
    echo -e "${BLUE}📝 How to test the fix:${NC}"
    echo ""
    echo "1. 🎤 Say 'Hey Computer' into the microphone"
    echo "2. 🔍 Watch for detection logs in the console"
    echo "3. 📊 Check if /tmp/recent_detections.json gets created"
    echo "4. 🌐 Check if the web interface shows the red pulse"
    echo ""
    echo "Expected behavior after the fix:"
    echo "✅ Detection logs appear in console"
    echo "✅ /tmp/recent_detections.json file is created with detection data"
    echo "✅ Web interface shows red pulsing animation"
    echo "✅ /api/detections endpoint returns detection data"
    echo ""
    echo "If you see detection logs but no file creation, the fix failed."
    echo "If you see both logs and file creation, the fix is working!"
}

# Main verification
echo "🚀 Starting verification..."

# Check current state
echo ""
echo "📋 Current Detection Status"
echo "==========================="
check_detection_file

echo ""
echo "🌐 Web API Status"
echo "================"
test_web_api

echo ""
echo "📋 System Logs Status"
echo "===================="
check_system_logs

echo ""
echo "📊 Verification Summary"
echo "======================"

# Determine if the fix is working
if check_detection_file > /dev/null 2>&1; then
    echo -e "${GREEN}✅ DETECTION FIX APPEARS TO BE WORKING${NC}"
    echo "✅ Detection file exists and has content"
    echo "✅ The cooldown/debounce logic is not blocking file creation"
else
    echo -e "${YELLOW}⚠️ DETECTION STATUS UNCLEAR${NC}"
    echo "⚠️ No detection file found - this could mean:"
    echo "   - No detections have occurred yet"
    echo "   - The fix may not be working"
    echo "   - The system needs to be tested with actual speech"
fi

echo ""
show_test_instructions

echo ""
echo "🔧 Troubleshooting:"
echo "   - If you see detection logs but no file: fix failed"
echo "   - If you see both logs and file: fix is working"
echo "   - If you see neither: system may not be detecting wake words"
echo ""
echo "📝 Verification completed at $(date)" 