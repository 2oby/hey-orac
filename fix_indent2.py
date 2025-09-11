#!/usr/bin/env python3

# Read the file
with open('src/hey_orac/wake_word_detection.py', 'r') as f:
    lines = f.readlines()

# Fix specific lines that have indent problems:
# Line 1398 (index 1397) should be indented 24 spaces (inside if block)
# Line 1399 needs 24 spaces too
# Lines 1401+ in the for loop need proper indentation

# First, fix line 1398-1399 (comment and triggered_models)
if len(lines) > 1397:
    lines[1397] = '                        # MULTI-TRIGGER MODE: Check each model independently\n'
if len(lines) > 1398:
    lines[1398] = '                        triggered_models = []\n'

# Lines 1401-1424 are inside a for loop and need proper indentation
for i in range(1401, min(1425, len(lines))):
    line = lines[i]
    stripped = line.lstrip()
    
    if i == 1401:  # for wakeword, score in prediction.items():
        lines[i] = '                        for wakeword, score in prediction.items():\n'
    elif i == 1402:  # comment line
        lines[i] = '                            # Map OpenWakeWord model name to our config name\n'
    elif i == 1403:  # config_name = None
        lines[i] = '                            config_name = None\n'
    elif i == 1404:  # if wakeword in model_name_mapping:
        lines[i] = '                            if wakeword in model_name_mapping:\n'
    elif i == 1405:
        lines[i] = '                                config_name = model_name_mapping[wakeword]\n'
    elif i == 1406:  # elif wakeword in active_model_configs:
        lines[i] = '                            elif wakeword in active_model_configs:\n'
    elif i == 1407:
        lines[i] = '                                config_name = wakeword\n'
    elif i == 1408:  # blank line
        lines[i] = '                            \n'
    elif i == 1409:  # if config_name and config_name in active_model_configs:
        lines[i] = '                            if config_name and config_name in active_model_configs:\n'
    elif i >= 1410 and i <= 1420:  # inside the nested if block
        # These lines need 32 spaces (8 levels of indentation)
        if stripped.startswith('model_config'):
            lines[i] = '                                model_config = active_model_configs[config_name]\n'
        elif stripped.startswith('detection_threshold'):
            lines[i] = '                                detection_threshold = model_config.threshold\n'
        elif stripped == '\n':
            lines[i] = '                                \n'
        elif stripped.startswith('if score >='):
            lines[i] = '                                if score >= detection_threshold:\n'
        elif stripped.startswith('triggered_models.append'):
            lines[i] = '                                    triggered_models.append({\n'
        elif stripped.startswith("'"):
            # Dictionary entries - need 40 spaces
            lines[i] = '                                        ' + stripped
        elif stripped.startswith('}'):
            lines[i] = '                                    })\n'

# Fix lines 1426-1500 (Process each triggered model section)
for i in range(1426, min(1501, len(lines))):
    line = lines[i]
    stripped = line.lstrip()
    
    if i == 1426:  # for trigger_info in triggered_models:
        lines[i] = '                        for trigger_info in triggered_models:\n'
    elif i >= 1427 and i <= 1500:
        # These are inside the for loop, need proper indentation
        if stripped.startswith('logger.info'):
            lines[i] = '                            ' + stripped
        elif stripped.startswith('#'):
            lines[i] = '                            ' + stripped
        elif stripped.startswith('detection_event'):
            lines[i] = '                            ' + stripped
        elif stripped.startswith("'"):
            lines[i] = '                                ' + stripped
        elif stripped.startswith('}'):
            lines[i] = '                            }\n'
        elif stripped.startswith('shared_data'):
            lines[i] = '                            ' + stripped
        elif stripped.startswith('try:'):
            lines[i] = '                            try:\n'
        elif stripped.startswith('event_queue'):
            lines[i] = '                                event_queue.put_nowait(detection_event)\n'
        elif stripped.startswith('except:'):
            lines[i] = '                            except:\n'
        elif stripped.startswith('pass'):
            lines[i] = '                                pass  # Queue full, skip\n'
        elif stripped.startswith('if '):
            lines[i] = '                            ' + stripped
        elif stripped.startswith('heartbeat_sender'):
            lines[i] = '                                ' + stripped
        elif stripped.startswith('webhook_data'):
            lines[i] = '                                ' + stripped
        elif stripped.startswith('response'):
            lines[i] = '                                ' + stripped
        elif stripped.startswith('except '):
            lines[i] = '                            ' + stripped
        elif stripped.startswith('with '):
            lines[i] = '                                ' + stripped
        elif stripped.startswith('stt_language'):
            lines[i] = '                                    ' + stripped
        elif stripped.startswith('speech_recorder'):
            lines[i] = '                                ' + stripped

# Similar fixes for else block (lines 1502+)
for i in range(1502, min(1650, len(lines))):
    line = lines[i]
    stripped = line.lstrip()
    
    if i == 1502:  # else:
        lines[i] = '                    else:\n'
    elif i == 1503:  # comment
        lines[i] = '                        # SINGLE-TRIGGER MODE: Original "winner takes all" behavior\n'
    elif i >= 1504 and i <= 1642:
        # These are inside the else block
        if stripped.startswith('max_confidence'):
            lines[i] = '                        ' + stripped
        elif stripped.startswith('best_model'):
            lines[i] = '                        ' + stripped
        elif stripped.startswith('#'):
            lines[i] = '                        ' + stripped
        elif stripped.startswith('for '):
            lines[i] = '                        ' + stripped
        elif stripped.startswith('if ') and i in [1509, 1518, 1522, 1525, 1530, 1536, 1543]:
            lines[i] = '                            ' + stripped
        elif stripped.startswith('config_name'):
            lines[i] = '                            ' + stripped
        elif stripped.startswith('elif '):
            lines[i] = '                        ' + stripped
        elif stripped.startswith('else:'):
            lines[i] = '                        ' + stripped
        elif stripped.startswith('logger.'):
            # Need to check context - some are deeper indented
            if i >= 1526 and i <= 1528:
                lines[i] = '                                ' + stripped
            elif i == 1537:
                lines[i] = '                                ' + stripped
            else:
                lines[i] = '                            ' + stripped
        elif stripped.startswith('model_config'):
            lines[i] = '                            ' + stripped
        elif stripped.startswith('detection_threshold'):
            lines[i] = '                            ' + stripped
        elif stripped.startswith('detection_event'):
            lines[i] = '                            ' + stripped
        elif stripped.startswith("'"):
            lines[i] = '                                ' + stripped
        elif stripped.startswith('}'):
            lines[i] = '                            }\n'

# Write the fixed file
with open('src/hey_orac/wake_word_detection.py', 'w') as f:
    f.writelines(lines)

print("Fixed indentation for multi-trigger and single-trigger sections")
