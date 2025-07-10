#!/usr/bin/env python3
"""
Standalone script to test audio output through USB speaker
Run this on the Pi when the container is not running to test speaker functionality
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeakerTester:
    """Test audio output through USB speaker"""
    
    def __init__(self):
        self.audio_players = [
            'ffplay',      # FFmpeg player (usually available)
            'mpg123',      # MP3 player
            'aplay',       # ALSA player
            'paplay',      # PulseAudio player
            'speaker-test' # ALSA speaker test
        ]
        
        self.test_sounds = [
            # MP3 files
            'assets/audio/computerbeep.mp3',
            # WAV files (we'll generate test tones)
            '/tmp/test_tone.wav',
            # System sounds
            '/usr/share/sounds/alsa/Front_Left.wav',
            '/usr/share/sounds/alsa/Front_Right.wav'
        ]
        
    def check_audio_players(self):
        """Check which audio players are available"""
        available_players = []
        logger.info("Checking available audio players...")
        
        for player in self.audio_players:
            try:
                result = subprocess.run([player, '--help'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 or result.returncode == 1:  # Help usually returns 1
                    available_players.append(player)
                    logger.info(f"✅ {player} - Available")
                else:
                    logger.warning(f"❌ {player} - Not available")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning(f"❌ {player} - Not found")
        
        return available_players
    
    def generate_test_tone(self, frequency=440, duration=2, sample_rate=44100):
        """Generate a test tone WAV file"""
        try:
            import numpy as np
            import wave
            
            # Generate sine wave
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * frequency * t)
            
            # Convert to 16-bit PCM
            tone = (tone * 32767).astype(np.int16)
            
            # Save as WAV file
            with wave.open('/tmp/test_tone.wav', 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(tone.tobytes())
            
            logger.info(f"✅ Generated test tone: {frequency}Hz for {duration}s")
            return True
        except ImportError:
            logger.warning("❌ numpy not available, skipping test tone generation")
            return False
        except Exception as e:
            logger.error(f"❌ Error generating test tone: {e}")
            return False
    
    def check_audio_devices(self):
        """Check available audio devices"""
        logger.info("Checking audio devices...")
        
        # Check ALSA devices
        try:
            result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("ALSA Audio Devices:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
        except Exception as e:
            logger.warning(f"Could not check ALSA devices: {e}")
        
        # Check USB devices
        try:
            result = subprocess.run(['lsusb'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("USB Devices:")
                for line in result.stdout.split('\n'):
                    if 'audio' in line.lower() or 'speaker' in line.lower() or 'mic' in line.lower():
                        logger.info(f"  {line}")
        except Exception as e:
            logger.warning(f"Could not check USB devices: {e}")
    
    def play_audio_file(self, file_path, player='ffplay'):
        """Play an audio file using specified player"""
        if not os.path.exists(file_path):
            logger.warning(f"❌ Audio file not found: {file_path}")
            return False
        
        logger.info(f"🎵 Playing {file_path} with {player}")
        
        try:
            if player == 'ffplay':
                cmd = [player, '-nodisp', '-autoexit', '-loglevel', 'error', file_path]
            elif player == 'mpg123':
                cmd = [player, '-q', file_path]
            elif player == 'aplay':
                cmd = [player, file_path]
            elif player == 'paplay':
                cmd = [player, file_path]
            elif player == 'speaker-test':
                cmd = [player, '-t', 'sine', '-f', '440', '-l', '1']
            else:
                logger.error(f"Unknown player: {player}")
                return False
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully played with {player}")
                return True
            else:
                logger.error(f"❌ {player} failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning(f"⏰ {player} timed out (this might be normal)")
            return True  # Timeout might be normal for some players
        except Exception as e:
            logger.error(f"❌ Error playing with {player}: {e}")
            return False
    
    def test_speaker_output(self):
        """Comprehensive speaker test"""
        logger.info("🔊 Starting USB Speaker Test")
        logger.info("=" * 50)
        
        # Check audio devices first
        self.check_audio_devices()
        logger.info("")
        
        # Check available players
        available_players = self.check_audio_players()
        if not available_players:
            logger.error("❌ No audio players available!")
            return False
        
        logger.info("")
        
        # Generate test tone
        tone_generated = self.generate_test_tone()
        logger.info("")
        
        # Test each available player
        for player in available_players:
            logger.info(f"🎵 Testing {player}...")
            
            # Test with MP3 file if available
            mp3_file = 'assets/audio/computerbeep.mp3'
            if os.path.exists(mp3_file):
                logger.info(f"  Testing MP3: {mp3_file}")
                self.play_audio_file(mp3_file, player)
                time.sleep(1)
            
            # Test with generated tone if available
            if tone_generated and os.path.exists('/tmp/test_tone.wav'):
                logger.info("  Testing generated tone")
                self.play_audio_file('/tmp/test_tone.wav', player)
                time.sleep(1)
            
            # Test with system sounds if available
            system_sound = '/usr/share/sounds/alsa/Front_Left.wav'
            if os.path.exists(system_sound):
                logger.info(f"  Testing system sound: {system_sound}")
                self.play_audio_file(system_sound, player)
                time.sleep(1)
            
            logger.info("")
        
        # Special test for speaker-test
        if 'speaker-test' in available_players:
            logger.info("🎵 Testing speaker-test with different frequencies...")
            frequencies = [440, 880, 220]  # A4, A5, A3
            for freq in frequencies:
                logger.info(f"  Testing {freq}Hz tone")
                try:
                    cmd = ['speaker-test', '-t', 'sine', '-f', str(freq), '-l', '1']
                    subprocess.run(cmd, timeout=3)
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"  Error with {freq}Hz: {e}")
        
        logger.info("=" * 50)
        logger.info("🔊 USB Speaker Test Complete!")
        return True
    
    def test_volume_control(self):
        """Test volume control functionality"""
        logger.info("🔊 Testing volume control...")
        
        try:
            # Check current volume
            result = subprocess.run(['amixer', 'get', 'Master'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Current volume levels:")
                for line in result.stdout.split('\n'):
                    if '[' in line and ']' in line:
                        logger.info(f"  {line.strip()}")
            
            # Test setting volume to 50%
            logger.info("Setting volume to 50%...")
            subprocess.run(['amixer', 'set', 'Master', '50%'], 
                         capture_output=True, text=True)
            
            # Play a test sound
            if os.path.exists('assets/audio/computerbeep.mp3'):
                logger.info("Playing test sound at 50% volume...")
                self.play_audio_file('assets/audio/computerbeep.mp3', 'ffplay')
            
            # Restore volume to 100%
            logger.info("Restoring volume to 100%...")
            subprocess.run(['amixer', 'set', 'Master', '100%'], 
                         capture_output=True, text=True)
            
        except Exception as e:
            logger.error(f"Error testing volume control: {e}")

def main():
    """Main function"""
    logger.info("🔊 USB Speaker Test Script")
    logger.info("This script tests audio output through the USB speaker")
    logger.info("Make sure the USB speaker is connected and the container is not running")
    logger.info("")
    
    tester = SpeakerTester()
    
    # Run comprehensive test
    success = tester.test_speaker_output()
    
    if success:
        logger.info("")
        logger.info("🎉 Speaker test completed successfully!")
        logger.info("If you heard audio, the speaker is working correctly.")
        logger.info("If you didn't hear audio, check:")
        logger.info("  - USB speaker connection")
        logger.info("  - Volume settings")
        logger.info("  - Audio device configuration")
        logger.info("  - Container not interfering with audio")
    else:
        logger.error("❌ Speaker test failed!")
        logger.error("Check audio device configuration and connections")

if __name__ == "__main__":
    main() 