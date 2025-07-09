#!/usr/bin/env python3
"""
Audio Feedback Module for Hey Orac
Handles playing audio feedback (beeps, tones) when wake words are detected
"""

import os
import logging
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class AudioFeedback:
    """Handles audio feedback for wake word detection"""
    
    def __init__(self, assets_path: str = None):
        """
        Initialize audio feedback system
        
        Args:
            assets_path: Path to audio assets directory (auto-detected if None)
        """
        if assets_path is None:
            # Auto-detect assets path for both local development and Docker
            possible_paths = [
                Path("assets/audio"),  # Local development
                Path("../assets/audio"),  # From src/ directory
                Path("/app/assets/audio"),  # Docker container
                Path("/app/src/assets/audio"),  # Alternative Docker path
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.assets_path = path
                    logger.info(f"🔍 Found audio assets at: {path}")
                    break
            else:
                # Default to local assets if none found
                self.assets_path = Path("assets/audio")
                logger.warning(f"⚠️ No audio assets directory found, using default: {self.assets_path}")
        else:
            self.assets_path = Path(assets_path)
        
        self.beep_sound = self.assets_path / "computerbeep.mp3"
        
        # Verify audio file exists
        if not self.beep_sound.exists():
            logger.warning(f"❌ Beep sound not found at {self.beep_sound}")
            logger.info(f"🔍 Checking for alternative audio files in {self.assets_path}...")
            
            # Look for any audio files in the directory
            audio_files = list(self.assets_path.glob("*.mp3")) + list(self.assets_path.glob("*.wav"))
            if audio_files:
                self.beep_sound = audio_files[0]
                logger.info(f"✅ Found alternative audio file: {self.beep_sound}")
            else:
                logger.error(f"❌ No audio files found in {self.assets_path}")
                self.beep_sound = None
        else:
            logger.info(f"✅ Audio feedback initialized with beep sound: {self.beep_sound}")
            logger.info(f"   File size: {self.beep_sound.stat().st_size} bytes")
            logger.info(f"   File exists: {self.beep_sound.exists()}")
            logger.info(f"   File readable: {os.access(self.beep_sound, os.R_OK)}")
    
    def play_beep(self, async_play: bool = True) -> bool:
        """
        Play the beep sound when wake word is detected
        
        Args:
            async_play: If True, play sound in background thread
            
        Returns:
            True if beep was played successfully, False otherwise
        """
        if not self.beep_sound or not self.beep_sound.exists():
            logger.warning("No beep sound available")
            return False
        
        try:
            if async_play:
                # Play in background thread to avoid blocking wake word detection
                thread = threading.Thread(target=self._play_audio_file, args=(self.beep_sound,))
                thread.daemon = True
                thread.start()
                logger.info("Beep sound playing in background")
            else:
                # Play synchronously
                self._play_audio_file(self.beep_sound)
                logger.info("Beep sound played")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to play beep sound: {e}")
            return False
    
    def _play_audio_file(self, audio_file: Path) -> None:
        """
        Play an audio file using system audio player
        
        Args:
            audio_file: Path to audio file to play
        """
        try:
            logger.info(f"🔊 Attempting to play audio file: {audio_file}")
            logger.info(f"   File exists: {audio_file.exists()}")
            logger.info(f"   File size: {audio_file.stat().st_size} bytes")
            logger.info(f"   File readable: {os.access(audio_file, os.R_OK)}")
            
            # Try different audio players in order of preference
            players = [
                ["mpg123", "-q", str(audio_file)],  # mpg123 for MP3 files
                ["aplay", str(audio_file)],         # aplay for WAV files
                ["paplay", str(audio_file)],        # paplay (PulseAudio)
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(audio_file)]  # ffplay
            ]
            
            for i, player_cmd in enumerate(players):
                try:
                    logger.info(f"🔊 Trying player {i+1}/{len(players)}: {player_cmd[0]}")
                    result = subprocess.run(
                        player_cmd,
                        capture_output=True,
                        text=True,
                        timeout=5  # 5 second timeout
                    )
                    if result.returncode == 0:
                        logger.info(f"✅ Audio played successfully with {player_cmd[0]}")
                        return
                    else:
                        logger.warning(f"⚠️ Player {player_cmd[0]} failed: {result.stderr}")
                except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
                    logger.warning(f"⚠️ Player {player_cmd[0]} not available: {e}")
                    continue
            
            # Fallback: Try to generate a simple beep tone
            logger.warning("⚠️ All audio players failed, trying fallback beep tone...")
            self._play_fallback_beep()
            
        except Exception as e:
            logger.error(f"❌ Error playing audio file {audio_file}: {e}")
            logger.warning("⚠️ Trying fallback beep tone...")
            self._play_fallback_beep()
    
    def _play_fallback_beep(self) -> None:
        """
        Play a simple beep tone using speaker-test as fallback
        """
        try:
            # Try speaker-test to generate a beep tone
            beep_commands = [
                ["speaker-test", "-t", "sine", "-f", "1000", "-l", "1", "-D", "default"],  # 1kHz tone
                ["speaker-test", "-t", "sine", "-f", "800", "-l", "1"],  # 800Hz tone
                ["speaker-test", "-t", "sine", "-f", "600", "-l", "1"],  # 600Hz tone
            ]
            
            for cmd in beep_commands:
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=3  # 3 second timeout
                    )
                    if result.returncode == 0:
                        logger.info("Fallback beep tone played successfully")
                        return
                except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
                    logger.debug(f"Fallback beep command failed: {e}")
                    continue
            
            logger.warning("No fallback beep method available")
            
        except Exception as e:
            logger.error(f"Error playing fallback beep: {e}")
    
    def play_wake_word_detected(self) -> bool:
        """
        Play feedback when wake word is detected
        
        Returns:
            True if feedback was played successfully
        """
        logger.info("🎵 Playing wake word detected feedback")
        return self.play_beep(async_play=True)
    
    def test_audio_feedback(self) -> bool:
        """
        Test the audio feedback system
        
        Returns:
            True if test was successful
        """
        logger.info("🔊 Testing audio feedback system...")
        
        if not self.beep_sound or not self.beep_sound.exists():
            logger.error("❌ Beep sound file not found")
            return False
        
        logger.info(f"✅ Beep sound found: {self.beep_sound}")
        
        # Test playing the beep
        success = self.play_beep(async_play=False)
        if success:
            logger.info("✅ Audio feedback test successful")
        else:
            logger.error("❌ Audio feedback test failed")
        
        return success

def create_audio_feedback() -> AudioFeedback:
    """
    Factory function to create audio feedback instance
    
    Returns:
        AudioFeedback instance
    """
    return AudioFeedback()

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    feedback = create_audio_feedback()
    feedback.test_audio_feedback() 