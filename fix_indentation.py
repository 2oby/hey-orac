#!/usr/bin/env python3
"""Fix indentation issues in wake_word_detection.py"""

# Read the file
with open('src/hey_orac/wake_word_detection.py', 'r') as f:
    lines = f.readlines()

# Fix lines 1375-1451 (need to be indented by 4 more spaces)
for i in range(1374, min(1451, len(lines))):  # lines 1375-1451 (0-indexed is 1374-1450)
    if lines[i].strip() and not lines[i].startswith('            else:'):
        # Add 4 spaces to non-empty lines that aren't the else statement
        lines[i] = '    ' + lines[i]

# Fix lines 1453-1573 (these should keep current indentation as they're in the else block)
# These are already correctly indented for the else block

# Write the file back
with open('src/hey_orac/wake_word_detection.py', 'w') as f:
    f.writelines(lines)

print("Fixed indentation issues")