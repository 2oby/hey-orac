#!/bin/bash

# Comprehensive Audio Debugging Script for Docker on Raspberry Pi
# This script runs all diagnostic tests to identify PyAudio/PortAudio issues

set -e

echo "🔍 Comprehensive Audio Debugging Script"
echo "======================================"
echo "Target: PyAudio Device Detection in Docker on Raspberry Pi"
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

# Function to run command and capture output
run_command() {
    local cmd="$1"
    local description="$2"
    
    echo ""
    print_status "INFO" "Running: $description"
    echo "Command: $cmd"
    echo "---"
    
    if eval "$cmd"; then
        print_status "SUCCESS" "$description completed"
    else
        print_status "ERROR" "$description failed"
        return 1
    fi
    echo "---"
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
print_status "INFO" "Step 1: Verifying ALSA Functionality in Container"
echo "================================================================"

# Test ALSA cards
run_command "docker-compose exec -T hey-orac bash -c 'cat /proc/asound/cards'" \
    "Checking ALSA cards in container"

# Test arecord
run_command "docker-compose exec -T hey-orac bash -c 'arecord -l'" \
    "Listing ALSA recording devices"

# Test aplay
run_command "docker-compose exec -T hey-orac bash -c 'aplay -l'" \
    "Listing ALSA playback devices"

# Test ALSA version
run_command "docker-compose exec -T hey-orac bash -c 'cat /proc/asound/version'" \
    "Checking ALSA version"

# Test USB devices
run_command "docker-compose exec -T hey-orac bash -c 'lsusb'" \
    "Listing USB devices"

echo ""
print_status "INFO" "Step 2: Testing Audio Device Access"
echo "================================================"

# Check /dev/snd contents
run_command "docker-compose exec -T hey-orac bash -c 'ls -la /dev/snd/'" \
    "Checking audio device files"

# Check user and groups
run_command "docker-compose exec -T hey-orac bash -c 'whoami && groups'" \
    "Checking user and group membership"

# Test ALSA recording
run_command "docker-compose exec -T hey-orac bash -c 'timeout 3 arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 /tmp/test.wav || echo \"Recording test completed\"'" \
    "Testing ALSA recording (3 second timeout)"

echo ""
print_status "INFO" "Step 3: Testing PyAudio with Enhanced Diagnostics"
echo "=============================================================="

# Run enhanced PyAudio test
run_command "docker-compose exec -T hey-orac python src/test_pyaudio_minimal.py" \
    "Running enhanced PyAudio diagnostics"

echo ""
print_status "INFO" "Step 4: Testing PortAudio Directly (C)"
echo "=================================================="

# Install PortAudio development tools
run_command "docker-compose exec -T hey-orac bash -c 'apt-get update && apt-get install -y libportaudio2 portaudio19-dev gcc'" \
    "Installing PortAudio development tools"

# Compile and run C PortAudio test
run_command "docker-compose exec -T hey-orac bash -c 'cd /app/src && gcc -o test_portaudio test_portaudio.c -lportaudio'" \
    "Compiling PortAudio test"

run_command "docker-compose exec -T hey-orac bash -c 'cd /app/src && ./test_portaudio'" \
    "Running PortAudio test"

echo ""
print_status "INFO" "Step 5: Checking PyAudio Library Dependencies"
echo "========================================================="

# Check PyAudio library dependencies
run_command "docker-compose exec -T hey-orac bash -c 'find /app/venv -name \"*pyaudio*\" -type f | head -5'" \
    "Finding PyAudio library files"

run_command "docker-compose exec -T hey-orac bash -c 'ldd /app/venv/lib/python3.12/site-packages/pyaudio/_portaudio.cpython-312-arm-linux-gnueabihf.so 2>/dev/null || echo \"PyAudio library not found or not compiled\""' \
    "Checking PyAudio library dependencies"

echo ""
print_status "INFO" "Step 6: Testing Host Audio System"
echo "=============================================="

# Test host ALSA
run_command "cat /proc/asound/cards" \
    "Checking host ALSA cards"

run_command "arecord -l" \
    "Checking host ALSA recording devices"

run_command "lsusb" \
    "Checking host USB devices"

echo ""
print_status "INFO" "Step 7: Summary and Recommendations"
echo "================================================"

echo ""
print_status "INFO" "Debugging completed. Check the output above for:"
echo "1. ✅ ALSA functionality in container"
echo "2. ✅ Audio device access and permissions"
echo "3. ❌ PyAudio device detection (likely the issue)"
echo "4. ❌ PortAudio device detection (if same as PyAudio)"
echo "5. ✅ Host audio system functionality"
echo ""

print_status "INFO" "Next steps based on results:"
echo "- If ALSA works but PyAudio doesn't: Focus on PyAudio installation"
echo "- If PortAudio works but PyAudio doesn't: Focus on Python bindings"
echo "- If neither works: Focus on PortAudio ALSA backend"
echo "- If host works but container doesn't: Focus on Docker configuration"

echo ""
print_status "SUCCESS" "Debugging script completed!" 