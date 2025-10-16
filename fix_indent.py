#!/usr/bin/env python3
import sys

# Read the file
with open('src/hey_orac/wake_word_detection.py', 'r') as f:
    lines = f.readlines()

# Fix indentation for lines 1398-1501 (they need 4 more spaces)
# and lines 1503-end of else block (they need proper indentation)
for i in range(1397, min(1501, len(lines))):  # Lines 1398-1501 in 0-indexed
    if i == 1397:  # Line 1398 (0-indexed 1397)
        # This is the "if multi_trigger_enabled:" line, already fixed
        continue
    elif i == 1501:  # Line 1502 (0-indexed 1501) - the else:
        # This should be aligned with the if
        continue
    else:
        # Add 4 spaces to lines inside the if block
        # Only add if the line starts with exactly 20 spaces
        if lines[i].startswith(' ' * 20) and not lines[i].startswith(' ' * 24):
            lines[i] = '    ' + lines[i]

# Also need to fix lines inside the else block (1503 onwards)
in_else_block = False
for i in range(1501, len(lines)):
    if i == 1501 and lines[i].strip().startswith('else:'):
        in_else_block = True
        continue
    if in_else_block:
        # Check if we've hit another major block that would end the else
        stripped = lines[i].strip()
        if stripped and not lines[i].startswith(' ' * 20):
            # We've left the else block
            break
        # Add 4 spaces to lines that need it
        if lines[i].startswith(' ' * 20) and not lines[i].startswith(' ' * 24):
            lines[i] = '    ' + lines[i]

# Write the fixed file
with open('src/hey_orac/wake_word_detection.py', 'w') as f:
    f.writelines(lines)

print("Fixed indentation")
