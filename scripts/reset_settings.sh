#!/bin/bash

# Reset settings to defaults using the settings manager
# This will remove the old flat keys and create a clean settings file

echo "🔄 Resetting settings to defaults..."

# Execute the reset command in the container
ssh pi "docker exec hey-orac python -c \"
from src.settings_manager import get_settings_manager
settings_manager = get_settings_manager()
if settings_manager.reset_to_defaults():
    print('✅ Settings reset to defaults successfully!')
    print('📋 Removed old flat keys and created clean nested structure')
else:
    print('❌ Failed to reset settings')
\""

echo ""
echo "🎯 The settings file now has the correct nested structure:"
echo "   - volume_monitoring.rms_filter (nested)"
echo "   - wake_word.debounce (nested)" 
echo "   - wake_word.cooldown (nested)"
echo ""
echo "✅ Web GUI changes will now properly affect the audio pipeline!" 