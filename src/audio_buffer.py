#!/usr/bin/env python3
"""
Audio buffer for capturing pre-roll and post-roll audio around wake-word detections
"""

import numpy as np
import logging
from collections import deque
from typing import Optional, Tuple
import wave
import io

logger = logging.getLogger(__name__)

class AudioBuffer:
    """Ring buffer for audio data with pre-roll and post-roll capture."""
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1, 
                 preroll_seconds: float = 1.0, postroll_seconds: float = 2.0):
        """
        Initialize audio buffer.
        
        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
            preroll_seconds: Seconds of audio to capture before wake-word
            postroll_seconds: Seconds of audio to capture after wake-word
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.preroll_seconds = preroll_seconds
        self.postroll_seconds = postroll_seconds
        
        # Calculate buffer sizes
        self.preroll_samples = int(sample_rate * preroll_seconds)
        self.postroll_samples = int(sample_rate * postroll_seconds)
        
        # Ring buffer for pre-roll audio (most recent audio)
        self.ring_buffer = deque(maxlen=self.preroll_samples)
        
        # Post-roll buffer
        self.postroll_buffer = []
        self.capturing_postroll = False
        self.postroll_samples_needed = self.postroll_samples
        
        logger.info(f"Audio buffer initialized:")
        logger.info(f"  Sample rate: {sample_rate}")
        logger.info(f"  Pre-roll: {preroll_seconds}s ({self.preroll_samples} samples)")
        logger.info(f"  Post-roll: {postroll_seconds}s ({self.postroll_samples} samples)")
    
    def add_audio(self, audio_chunk: np.ndarray) -> None:
        """
        Add audio chunk to the buffer.
        
        Args:
            audio_chunk: Audio data as numpy array
        """
        # Add to ring buffer (pre-roll)
        self.ring_buffer.extend(audio_chunk)
        
        # Add to post-roll buffer if capturing
        if self.capturing_postroll:
            self.postroll_buffer.extend(audio_chunk)
            self.postroll_samples_needed -= len(audio_chunk)
            
            if self.postroll_samples_needed <= 0:
                self.capturing_postroll = False
                logger.info("✅ Post-roll capture completed")
    
    def start_postroll_capture(self) -> None:
        """Start capturing post-roll audio after wake-word detection."""
        self.postroll_buffer = []
        self.capturing_postroll = True
        self.postroll_samples_needed = self.postroll_samples
        logger.info("🎙️ Started post-roll audio capture")
    
    def is_capturing_postroll(self) -> bool:
        """Check if currently capturing post-roll audio."""
        return self.capturing_postroll
    
    def get_complete_audio_clip(self) -> Optional[np.ndarray]:
        """
        Get complete audio clip with pre-roll and post-roll.
        
        Returns:
            Complete audio clip as numpy array, or None if not ready
        """
        if self.capturing_postroll:
            return None  # Still capturing post-roll
        
        if len(self.postroll_buffer) == 0:
            return None  # No post-roll captured yet
        
        # Combine pre-roll and post-roll
        preroll_audio = np.array(list(self.ring_buffer))
        postroll_audio = np.array(self.postroll_buffer)
        
        # Ensure we have enough pre-roll audio
        if len(preroll_audio) < self.preroll_samples:
            logger.warning(f"Insufficient pre-roll audio: {len(preroll_audio)} < {self.preroll_samples}")
            # Pad with zeros if needed
            padding_needed = self.preroll_samples - len(preroll_audio)
            preroll_audio = np.concatenate([np.zeros(padding_needed, dtype=preroll_audio.dtype), preroll_audio])
        
        # Combine audio
        complete_audio = np.concatenate([preroll_audio, postroll_audio])
        
        logger.info(f"📦 Complete audio clip: {len(complete_audio)} samples ({len(complete_audio)/self.sample_rate:.2f}s)")
        return complete_audio
    
    def save_audio_clip(self, filename: str) -> bool:
        """
        Save complete audio clip to WAV file.
        
        Args:
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        audio_clip = self.get_complete_audio_clip()
        if audio_clip is None:
            logger.error("No complete audio clip available")
            return False
        
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_clip.tobytes())
            
            logger.info(f"💾 Saved audio clip to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving audio clip: {e}")
            return False
    
    def reset(self) -> None:
        """Reset the buffer."""
        self.ring_buffer.clear()
        self.postroll_buffer = []
        self.capturing_postroll = False
        self.postroll_samples_needed = self.postroll_samples
        logger.debug("Audio buffer reset")
    
    def get_buffer_status(self) -> dict:
        """Get current buffer status."""
        return {
            'preroll_samples': len(self.ring_buffer),
            'preroll_seconds': len(self.ring_buffer) / self.sample_rate,
            'postroll_capturing': self.capturing_postroll,
            'postroll_samples': len(self.postroll_buffer),
            'postroll_samples_needed': self.postroll_samples_needed,
            'complete_clip_ready': not self.capturing_postroll and len(self.postroll_buffer) > 0
        } 