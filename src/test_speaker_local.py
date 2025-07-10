#!/usr/bin/env python3
"""
Local version of speaker test for development
Run this locally to test the script before pushing to Pi
"""

import os
import sys
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_audio_players():
    """Check which audio players are available locally"""
    players = ['ffplay', 'mpg123', 'aplay', 'paplay', 'speaker-test']
    available = []
    
    logger.info("Checking available audio players...")
    for player in players:
        try:
            result = subprocess.run([player, '--help'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode in [0, 1]:  # Help usually returns 1
                available.append(player)
                logger.info(f"✅ {player} - Available")
            else:
                logger.warning(f"❌ {player} - Not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning(f"❌ {player} - Not found")
    
    return available

def test_mp3_playback():
    """Test MP3 playback if file exists"""
    mp3_file = 'assets/audio/computerbeep.mp3'
    if os.path.exists(mp3_file):
        logger.info(f"🎵 Testing MP3 playback: {mp3_file}")
        
        # Try ffplay first
        try:
            cmd = ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'error', mp3_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("✅ MP3 playback successful with ffplay")
                return True
        except Exception as e:
            logger.warning(f"ffplay failed: {e}")
        
        # Try mpg123
        try:
            cmd = ['mpg123', '-q', mp3_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("✅ MP3 playback successful with mpg123")
                return True
        except Exception as e:
            logger.warning(f"mpg123 failed: {e}")
    
    logger.warning("❌ MP3 playback test failed")
    return False

def main():
    """Main function for local testing"""
    logger.info("🔊 Local Speaker Test")
    logger.info("This is a local test of the speaker test script")
    logger.info("")
    
    # Check available players
    available_players = check_audio_players()
    
    if not available_players:
        logger.error("❌ No audio players available!")
        return False
    
    # Test MP3 playback
    test_mp3_playback()
    
    logger.info("")
    logger.info("✅ Local test completed!")
    logger.info("If this works, the script should work on the Pi")
    
    return True

if __name__ == "__main__":
    main() 