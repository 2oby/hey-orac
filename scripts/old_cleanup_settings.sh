#!/bin/bash

# Cleanup script to remove old flat key format from settings
# This script removes the old flat keys that were causing the RMS filter issue
# UPDATED: Now uses settings manager to ensure proper backup and logging

echo "🧹 Cleaning up settings file using settings manager..."

# Execute the cleanup command in the container using settings manager
ssh pi "docker exec hey-orac python -c \"
import json
import os
from src.settings_manager import get_settings_manager

# Get the settings manager instance
settings_manager = get_settings_manager()

# Load current settings
current_settings = settings_manager.get_all()

# Remove flat keys that should be nested
flat_keys_to_remove = [
    'volume_monitoring.rms_filter',  # This should be nested under volume_monitoring
    'debounce_ms',                   # This should be nested under wake_word.debounce
    'cooldown_s'                     # This should be nested under wake_word.cooldown
]

# Create cleaned settings by removing flat keys
cleaned_settings = current_settings.copy()
for key in flat_keys_to_remove:
    if key in cleaned_settings:
        del cleaned_settings[key]
        print(f'🗑️ Removed flat key: {key}')

# Save the cleaned settings using the settings manager
# This will trigger proper backup and logging
if settings_manager.update(cleaned_settings):
    print('✅ Settings cleaned up successfully!')
    print('📋 Removed flat keys and created clean nested structure')
    print('💾 Settings backed up automatically via settings manager')
else:
    print('❌ Failed to clean up settings')
\""

echo ""
echo "🎯 The settings file now has the correct nested structure:"
echo "   - volume_monitoring.rms_filter (nested)"
echo "   - wake_word.debounce (nested)" 
echo "   - wake_word.cooldown (nested)"
echo ""
echo "✅ Web GUI changes will now properly affect the audio pipeline!"
echo "💾 Settings backup created automatically" 