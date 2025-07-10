#!/bin/bash

# Test USB Speaker on Pi
# This script pushes the speaker test to the Pi and runs it when container is not running

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PI_USER="2oby"
PI_HOST="192.168.8.99"
PI_PATH="/home/2oby/hey-orac"

echo -e "${BLUE}🔊 USB Speaker Test on Pi${NC}"
echo "=================================="

# Check if we can connect to Pi
echo -e "${YELLOW}Checking Pi connection...${NC}"
if ! ssh -o ConnectTimeout=5 ${PI_USER}@${PI_HOST} "echo 'Connection successful'" 2>/dev/null; then
    echo -e "${RED}❌ Cannot connect to Pi at ${PI_USER}@${PI_HOST}${NC}"
    echo "Please check:"
    echo "  - Pi is powered on and connected to network"
    echo "  - SSH is enabled on Pi"
    echo "  - SSH key is set up correctly"
    exit 1
fi

echo -e "${GREEN}✅ Pi connection successful${NC}"

# Stop the container if it's running
echo -e "${YELLOW}Stopping container if running...${NC}"
ssh ${PI_USER}@${PI_HOST} "cd ${PI_PATH} && docker-compose down" 2>/dev/null || true

# Push latest code to Pi
echo -e "${YELLOW}Pushing latest code to Pi...${NC}"
ssh ${PI_USER}@${PI_HOST} "cd ${PI_PATH} && git pull origin master"

# Install numpy if needed (for test tone generation)
echo -e "${YELLOW}Installing numpy for test tone generation...${NC}"
ssh ${PI_USER}@${PI_HOST} "pip3 install numpy" 2>/dev/null || true

# Make the test script executable
echo -e "${YELLOW}Making test script executable...${NC}"
ssh ${PI_USER}@${PI_HOST} "cd ${PI_PATH} && chmod +x src/test_speaker.py"

# Run the speaker test
echo -e "${YELLOW}Running USB speaker test...${NC}"
echo -e "${BLUE}🎵 You should hear test sounds if the speaker is working${NC}"
echo ""

ssh ${PI_USER}@${PI_HOST} "cd ${PI_PATH} && python3 src/test_speaker.py"

echo ""
echo -e "${GREEN}✅ Speaker test completed!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  - If you heard audio: Speaker is working correctly"
echo "  - If no audio: Check USB speaker connection and volume"
echo "  - Run 'docker-compose up' to restart the main application" 