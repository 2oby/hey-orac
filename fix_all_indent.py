#!/usr/bin/env python3

# Read the file
with open('src/hey_orac/wake_word_detection.py', 'r') as f:
    lines = f.readlines()

# Fix all lines inside the try block (1294-1391 in 0-indexed = 1295-1392)
# They should all be indented by 20 spaces minimum (5 levels = try block indent)

for i in range(1294, min(1392, len(lines))):
    line = lines[i]
    stripped = line.lstrip()
    
    if not stripped:  # Empty line
        lines[i] = '\n'
        continue
    
    # Determine proper indentation based on context
    if i == 1294:  # Check STT health comment
        lines[i] = '                    # Check STT health periodically\n'
    elif i == 1295:  # if current_time
        lines[i] = '                    if current_time - last_health_check >= HEALTH_CHECK_INTERVAL:\n'
    elif i >= 1296 and i <= 1303:  # inside the if block
        if stripped.startswith('if stt_client'):
            lines[i] = '                        if stt_client:\n'
        elif stripped.startswith('stt_health_status'):
            lines[i] = '                            stt_health_status = check_all_stt_health(active_model_configs, stt_client)\n'
        elif stripped.startswith('if shared_data'):
            lines[i] = '                            if shared_data.get(\'stt_health\') != stt_health_status:\n'
        elif stripped.startswith('shared_data[\'stt_health\']'):
            lines[i] = '                                shared_data[\'stt_health\'] = stt_health_status\n'
        elif stripped.startswith('shared_data[\'status_changed\']'):
            lines[i] = '                                shared_data[\'status_changed\'] = True\n'
        elif stripped.startswith('logger.info') and 'STT health' in stripped:
            lines[i] = '                                logger.info(f"🏥 STT health status changed to: {stt_health_status}")\n'
        elif stripped.startswith('else:'):
            lines[i] = '                            else:\n'
        elif stripped.startswith('logger.debug') and 'Periodic' in stripped:
            lines[i] = '                                logger.debug(f"🏥 Periodic STT health check: {stt_health_status}")\n'
    elif i == 1304:  # last_health_check
        lines[i] = '                        last_health_check = current_time\n'
    elif i == 1306:  # Read one chunk comment
        lines[i] = '                    # Read one chunk of audio data (1280 samples) with timeout protection\n'
    elif i == 1307:  # inner try
        lines[i] = '                    try:\n'
    elif i >= 1308 and i <= 1312:  # inside inner try
        if stripped.startswith('if hasattr'):
            lines[i] = '                        if hasattr(signal, \'SIGALRM\'):\n'
        elif stripped.startswith('signal.alarm(2)'):
            lines[i] = '                            signal.alarm(2)  # 2 second timeout\n'
        elif stripped.startswith('data = stream'):
            lines[i] = '                        data = stream.read(1280, exception_on_overflow=False)\n'
        elif stripped.startswith('signal.alarm(0)'):
            lines[i] = '                            signal.alarm(0)  # Cancel the alarm\n'
    elif i == 1313:  # except TimeoutError
        lines[i] = '                    except TimeoutError:\n'
    elif i >= 1314 and i <= 1316:  # inside except
        if stripped.startswith('logger.error') and 'Audio stream' in stripped:
            lines[i] = '                        logger.error("Audio stream read timed out - possible frozen audio thread")\n'
        elif stripped.startswith('# Force'):
            lines[i] = '                        # Force exit to trigger container restart\n'
        elif stripped.startswith('sys.exit'):
            lines[i] = '                        sys.exit(1)\n'
    elif i == 1318:  # if data is None
        lines[i] = '                    if data is None or len(data) == 0:\n'
    elif i >= 1319 and i <= 1320:  # inside if
        if stripped.startswith('logger.warning'):
            lines[i] = '                        logger.warning("No audio data read from stream")\n'
        elif stripped.startswith('continue'):
            lines[i] = '                        continue\n'
    # Continue with proper indentation for remaining lines...
    else:
        # Default: add 4 spaces if not enough indentation
        current_indent = len(line) - len(line.lstrip())
        if current_indent < 20:
            lines[i] = '    ' + line

# Fix lines 1322 onwards - these should have 20 spaces minimum
for i in range(1322, min(1392, len(lines))):
    line = lines[i]
    stripped = line.lstrip()
    
    if not stripped:
        continue
    
    # Count current indentation
    current_indent = len(line) - len(line.lstrip())
    
    # These lines should have at least 20 spaces
    if current_indent < 20:
        # Add the missing spaces
        missing = 20 - current_indent
        lines[i] = ' ' * missing + line

# Write the fixed file
with open('src/hey_orac/wake_word_detection.py', 'w') as f:
    f.writelines(lines)

print("Fixed all indentation in try block")
