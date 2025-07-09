#!/bin/bash

# Test script for Hey Orac fixes
# Tests custom model loading, audio feedback, and other fixes

set -e

echo "🧪 Testing Hey Orac Fixes"
echo "=========================="

# Test 1: Custom Model Loading
echo ""
echo "🔍 Test 1: Custom Model Loading"
echo "-------------------------------"
docker exec hey-orac python src/test_custom_models.py

# Test 2: Audio Feedback System
echo ""
echo "🔍 Test 2: Audio Feedback System"
echo "--------------------------------"
docker exec hey-orac python src/audio_feedback.py

# Test 3: Integration Test
echo ""
echo "🔍 Test 3: Integration Test"
echo "---------------------------"
docker exec hey-orac python -c "
import sys
import yaml
from wake_word_engines.openwakeword_engine import OpenWakeWordEngine
from audio_feedback import create_audio_feedback

print('✅ All imports successful')
print('✅ System ready for testing')
"

echo ""
echo "✅ All tests completed successfully!" 