#!/usr/bin/env python3
"""
Explicit beep test - makes it very clear when to listen for audio
"""

import subprocess
import time
import sys

print("🔊 EXPLICIT BEEP TEST")
print("=" * 40)
print("This test will make 5 beeps with clear timing.")
print("Listen carefully for each beep sound.")
print("")

# Test 1: echo beep
print("🎵 TEST 1: echo beep")
print("You should hear a beep in 3 seconds...")
time.sleep(3)
print("🔊 BEEPING NOW!")
subprocess.run(['echo', '-e', '\\a'])
time.sleep(2)

# Test 2: speaker-test beep
print("\n🎵 TEST 2: speaker-test beep")
print("You should hear a 1-second tone in 3 seconds...")
time.sleep(3)
print("🔊 BEEPING NOW!")
subprocess.run(['speaker-test', '-t', 'sine', '-f', '1000', '-l', '1'])
time.sleep(2)

# Test 3: Multiple echo beeps
print("\n🎵 TEST 3: Multiple echo beeps")
print("You should hear 3 beeps in 3 seconds...")
time.sleep(3)
print("🔊 BEEPING NOW!")
for i in range(3):
    print(f"  Beep {i+1}/3")
    subprocess.run(['echo', '-e', '\\a'])
    time.sleep(1)

print("\n" + "=" * 40)
print("📊 TEST COMPLETE")
print("=" * 40)
print("Did you hear ANY of the beeps?")
print("- If YES: Pi audio is working")
print("- If NO: Pi audio is not working")
print("")
print("Note: Some Pi models may not have audio output")
print("capability without additional hardware.") 