#!/bin/bash

# Simple detection monitoring script
# Watches for changes to the detection file without interfering with the main loop

set -e

echo "👁️ Detection Monitor"
echo "===================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DETECTION_FILE="/tmp/recent_detections.json"
LAST_SIZE=0

echo "🔍 Monitoring for detection file changes..."
echo "📄 File: $DETECTION_FILE"
echo "⏱️  Press Ctrl+C to stop monitoring"
echo ""

# Function to check file size
get_file_size() {
    if [ -f "$DETECTION_FILE" ]; then
        stat -c%s "$DETECTION_FILE" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Function to show file contents
show_detections() {
    if [ -f "$DETECTION_FILE" ]; then
        echo -e "${GREEN}📋 Detection file contents:${NC}"
        if command -v jq >/dev/null 2>&1; then
            jq '.' "$DETECTION_FILE" 2>/dev/null || cat "$DETECTION_FILE"
        else
            cat "$DETECTION_FILE"
        fi
    else
        echo -e "${YELLOW}⚠️ Detection file does not exist${NC}"
    fi
}

# Main monitoring loop
while true; do
    CURRENT_SIZE=$(get_file_size)
    
    if [ "$CURRENT_SIZE" != "$LAST_SIZE" ]; then
        if [ "$CURRENT_SIZE" -gt "$LAST_SIZE" ]; then
            echo -e "${GREEN}🎯 DETECTION RECORDED!${NC}"
            echo "📊 File size changed: $LAST_SIZE → $CURRENT_SIZE bytes"
            echo "⏰ Time: $(date '+%H:%M:%S')"
            show_detections
            echo ""
        elif [ "$CURRENT_SIZE" -lt "$LAST_SIZE" ]; then
            echo -e "${YELLOW}📄 Detection file was cleared${NC}"
            echo "⏰ Time: $(date '+%H:%M:%S')"
            echo ""
        fi
        LAST_SIZE=$CURRENT_SIZE
    fi
    
    sleep 1
done 