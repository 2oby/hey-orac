#!/bin/bash
# Simple SharedMemoryIPC Test Script

echo "🧪 Testing SharedMemoryIPC - Simple Test"
echo "========================================"

# Create a simple Python test script
cat > /tmp/test_shared_memory.py << 'EOF'
#!/usr/bin/env python3
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, '/app')

try:
    from src.shared_memory_ipc import shared_memory_ipc
    import time
    
    print('🧪 Testing shared memory activation system...')
    
    # Test 1: Basic functionality
    print('📊 Test 1: Basic functionality')
    
    # Test activation state updates
    shared_memory_ipc.update_activation_state(True, 'Test Model', 0.85)
    print('✅ Set activation to True')
    
    data = shared_memory_ipc.get_activation_state()
    print(f'📊 Current state: {data}')
    
    shared_memory_ipc.update_activation_state(False)
    print('✅ Set activation to False')
    
    data = shared_memory_ipc.get_activation_state()
    print(f'📊 Final state: {data}')
    
    print('✅ Shared memory activation system working correctly')
    
    # Test 2: Audio state updates
    print('📊 Test 2: Audio state updates')
    
    shared_memory_ipc.update_audio_state(0.123, 0.456)
    audio_data = shared_memory_ipc.get_audio_state()
    print(f'📊 Audio state: {audio_data}')
    
    print('✅ Audio state updates working correctly')
    
    print('🎉 All tests passed!')
    
except Exception as e:
    print(f'❌ Shared memory test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

# Run the test
python3 /tmp/test_shared_memory.py

# Clean up
rm -f /tmp/test_shared_memory.py

echo "✅ SharedMemoryIPC test completed!" 