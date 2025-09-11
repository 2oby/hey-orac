#!/usr/bin/env python3

# Read the file
with open('src/hey_orac/wake_word_detection.py', 'r') as f:
    lines = f.readlines()

# Fix indentation for lines inside the try block (1283-1391)
# These lines need to be indented 20 spaces (inside the try block)
for i in range(1293, min(1392, len(lines))):  # Lines 1294-1391 (0-indexed)
    line = lines[i]
    stripped = line.lstrip()
    
    # Skip if already properly indented or empty
    if not stripped or line.startswith(' ' * 20):
        continue
    
    # Add 4 spaces to lines that need it
    if line.startswith(' ' * 16):
        lines[i] = '    ' + line

# Also need to fix the except block (line 1392)
if len(lines) > 1391 and lines[1391].strip().startswith('except'):
    lines[1391] = '                except Exception as e:\n'

# Write the fixed file
with open('src/hey_orac/wake_word_detection.py', 'w') as f:
    f.writelines(lines)

print("Fixed indentation in try block")
