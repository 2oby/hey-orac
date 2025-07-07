#!/usr/bin/env bash
set -euo pipefail

# Test audio device in isolation by stopping the main service
# Usage: ./scripts/test_audio_isolated.sh

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎤 Audio Device Isolation Test${NC}"
echo -e "${BLUE}============================${NC}"

# Check if we're connected to the Pi
echo -e "${YELLOW}👉 Checking connection to pi...${NC}"
if ! ssh -q -o BatchMode=yes -o ConnectTimeout=5 "pi" exit; then
    echo -e "${RED}❌ Cannot connect to pi. Please check your SSH configuration.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Connected to pi${NC}"

# Stop the main service
echo -e "${YELLOW}🛑 Stopping main service to free audio device...${NC}"
ssh pi "cd ~/hey-orac && docker-compose stop hey-orac"

# Wait for device to be released
echo -e "${YELLOW}⏳ Waiting for audio device to be released...${NC}"
sleep 5

# Test audio device access
echo -e "${YELLOW}🧪 Testing audio device access...${NC}"
ssh pi "cd ~/hey-orac && docker-compose run --rm --entrypoint '' hey-orac python src/test_audio_device.py"

# Check test result
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Audio device test passed!${NC}"
else
    echo -e "${RED}❌ Audio device test failed!${NC}"
fi

# Restart the main service
echo -e "${YELLOW}🔄 Restarting main service...${NC}"
ssh pi "cd ~/hey-orac && docker-compose up -d hey-orac"

# Wait for service to start
echo -e "${YELLOW}⏳ Waiting for service to start...${NC}"
sleep 5

# Check service status
echo -e "${YELLOW}📊 Checking service status...${NC}"
ssh pi "cd ~/hey-orac && docker-compose ps"

echo -e "${GREEN}🎉 Audio device isolation test completed!${NC}" 