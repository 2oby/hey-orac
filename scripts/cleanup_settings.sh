#!/bin/bash

# Cleanup script to remove old flat key format from settings
# This script removes the old flat keys that were causing the RMS filter issue

echo "🧹 Cleaning up settings file..."

# Get the current settings
ssh pi "docker exec hey-orac cat /tmp/settings/config.json" > /tmp/current_settings.json

# Create a cleaned version by removing flat keys
cat /tmp/current_settings.json | jq 'del(.volume_monitoring.rms_filter | select(. != null)) | del(.debounce_ms | select(. != null)) | del(.cooldown_s | select(. != null))' > /tmp/cleaned_settings.json

# Update the settings file on the Pi
ssh pi "docker exec hey-orac bash -c 'cat > /tmp/settings/config.json' < /tmp/cleaned_settings.json"

echo "✅ Settings cleaned up!"
echo "📋 Removed flat keys:"
echo "   - volume_monitoring.rms_filter"
echo "   - debounce_ms" 
echo "   - cooldown_s"
echo ""
echo "✅ Kept nested keys:"
echo "   - volume_monitoring.rms_filter (nested)"
echo "   - wake_word.debounce (nested)"
echo "   - wake_word.cooldown (nested)"

# Clean up local temp files
rm -f /tmp/current_settings.json /tmp/cleaned_settings.json

echo ""
echo "🎯 Now the web GUI changes will properly affect the audio pipeline!" 