"""
Audio endpointing for detecting speech boundaries.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class EndpointConfig:
    """Configuration for speech endpointing.

    Note: These values are set to minimum - whisper.cpp's VAD handles
    the real end-of-speech detection. We just need a trigger to send.
    Whisper VAD settings: --vad-min-silence-duration-ms 100
    """
    silence_threshold: float = 0.01  # RMS threshold for silence
    silence_duration: float = 0.1    # Minimal - just detect any pause (was 0.3)
    grace_period: float = 0.1        # Minimal grace period (was 0.4)
    max_duration: float = 15.0       # Maximum recording duration
    pre_roll: float = 1.0           # Pre-roll duration to include


class SpeechEndpointer:
    """
    Detects speech boundaries for capturing utterances after wake word.
    
    Uses RMS-based silence detection with configurable thresholds.
    """
    
    def __init__(self, config: EndpointConfig, sample_rate: int = 16000):
        """
        Initialize the endpointer.
        
        Args:
            config: Endpointing configuration
            sample_rate: Audio sample rate in Hz
        """
        self.config = config
        self.sample_rate = sample_rate
        
        # Convert durations to sample counts
        self.silence_samples = int(config.silence_duration * sample_rate)
        self.grace_samples = int(config.grace_period * sample_rate)
        self.max_samples = int(config.max_duration * sample_rate)
        
        # State tracking
        self.reset()
    
    def reset(self) -> None:
        """Reset endpointer state."""
        self.is_speech_started = False
        self.silence_counter = 0
        self.total_samples = 0
        self.in_grace_period = False
    
    def process_audio(self, audio_chunk: np.ndarray) -> Tuple[bool, bool]:
        """
        Process an audio chunk and determine speech state.
        
        Args:
            audio_chunk: Audio samples (int16 or float32)
            
        Returns:
            Tuple of (is_speech_active, should_end_capture)
        """
        # Calculate RMS
        if audio_chunk.dtype == np.int16:
            # Normalize int16 to float for RMS calculation
            normalized = audio_chunk.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(normalized ** 2)))
        else:
            # Already float
            rms = float(np.sqrt(np.mean(audio_chunk ** 2)))
        
        self.total_samples += len(audio_chunk)
        
        # Check if we've exceeded maximum duration
        if self.total_samples >= self.max_samples:
            logger.info("Maximum recording duration reached")
            return False, True
        
        # Detect speech/silence
        is_silence = rms < self.config.silence_threshold
        
        if not self.is_speech_started:
            # Waiting for speech to start
            if not is_silence:
                self.is_speech_started = True
                self.silence_counter = 0
                logger.debug(f"Speech started (RMS: {rms:.4f})")
                return True, False
            else:
                return False, False
        
        else:
            # Speech has started, monitoring for end
            if is_silence:
                self.silence_counter += len(audio_chunk)
                
                if self.silence_counter >= self.silence_samples:
                    if not self.in_grace_period:
                        # Enter grace period
                        self.in_grace_period = True
                        logger.debug("Entering grace period")
                    
                    elif self.silence_counter >= (self.silence_samples + self.grace_samples):
                        # End of speech detected
                        logger.info(f"Speech ended after {self.total_samples/self.sample_rate:.2f}s")
                        return False, True
                
                return True, False  # Still in speech (or grace period)
            
            else:
                # Reset silence counter
                if self.silence_counter > 0:
                    logger.debug(f"Speech resumed (RMS: {rms:.4f})")
                self.silence_counter = 0
                self.in_grace_period = False
                return True, False
    
    def get_speech_duration(self) -> float:
        """Get current speech duration in seconds."""
        return self.total_samples / self.sample_rate