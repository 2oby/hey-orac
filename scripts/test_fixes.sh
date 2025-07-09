#!/bin/bash

# Test script for the three fixes: Custom Models, Audio Feedback, and LED Control
# This script tests all the improvements made to the Hey Orac system

set -e

echo "🧪 Testing Hey Orac Fixes"
echo "========================="
echo "Testing:"
echo "1. Custom Model Loading"
echo "2. Audio Feedback (MP3 playback)"
echo "3. LED Control (USB microphone)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "SUCCESS")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "ERROR")
            echo -e "${RED}❌ $message${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
    esac
}

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_status "ERROR" "Please run this script from the project root directory"
    exit 1
fi

# Check if container is running
if ! docker-compose ps | grep -q "hey-orac.*Up"; then
    print_status "WARNING" "Container is not running. Starting it..."
    docker-compose up -d
    sleep 5
fi

echo ""
print_status "INFO" "Test 1: Custom Model Loading"
echo "=================================="

# Test custom model loading
print_status "INFO" "Testing custom model loading..."
docker-compose exec -T hey-orac python src/test_custom_models.py

echo ""
print_status "INFO" "Test 2: Audio Feedback System"
echo "====================================="

# Test audio feedback
print_status "INFO" "Testing audio feedback system..."
docker-compose exec -T hey-orac python src/audio_feedback.py

# Check if audio file exists
print_status "INFO" "Checking audio assets..."
docker-compose exec -T hey-orac bash -c 'ls -la /app/assets/audio/'

# Test audio playback
print_status "INFO" "Testing audio playback..."
docker-compose exec -T hey-orac bash -c 'which mpg123 || which aplay || which paplay || which ffplay || echo "No audio players found"'

echo ""
print_status "INFO" "Test 3: LED Control System"
echo "================================="

# Test LED controller
print_status "INFO" "Testing LED controller..."
docker-compose exec -T hey-orac python src/led_controller.py

# Check USB devices
print_status "INFO" "Checking USB devices..."
docker-compose exec -T hey-orac bash -c 'lsusb'

# Check if USB HID devices are accessible
print_status "INFO" "Checking USB HID access..."
docker-compose exec -T hey-orac bash -c 'ls -la /dev/hid* 2>/dev/null || echo "No HID devices found"'

echo ""
print_status "INFO" "Test 4: Integration Test"
echo "================================="

# Test the main application with all components
print_status "INFO" "Testing main application integration..."
docker-compose exec -T hey-orac python src/main.py --test-wakeword

echo ""
print_status "INFO" "Test Results Summary"
echo "=========================="

print_status "INFO" "All tests completed. Check the output above for:"
echo "1. ✅ Custom model loading and detection"
echo "2. ✅ Audio feedback system (MP3 playback)"
echo "3. ✅ LED control system (USB device detection)"
echo "4. ✅ Integration with main application"
echo ""

print_status "SUCCESS" "Fix testing completed!" 