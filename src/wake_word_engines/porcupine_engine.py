#!/usr/bin/env python3
"""
Porcupine wake-word detection engine implementation
"""

import pvporcupine
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from wake_word_interface import WakeWordEngine

logger = logging.getLogger(__name__)

class PorcupineEngine(WakeWordEngine):
    """Porcupine wake-word detection engine."""
    
    def __init__(self):
        self.porcupine = None
        self.is_initialized = False
        self.wake_word_name = "Unknown"
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the Porcupine engine.
        
        Args:
            config: Configuration dictionary with Porcupine settings
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            model_path = config.get('model_path', '')
            sensitivity = config.get('sensitivity', 0.6)
            keyword = config.get('keyword', 'ORAC')
            access_key = config.get('access_key', '')
            
            # Check if access key is provided
            if not access_key:
                logger.error("Porcupine access key is required. Get a free key from https://console.picovoice.ai/")
                return False
            
            model_file = Path(model_path)
            
            if not model_file.exists():
                logger.warning(f"Model file not found: {model_path}")
                logger.info("Using built-in 'Picovoice' wake word")
                
                self.porcupine = pvporcupine.create(
                    access_key=access_key,
                    keywords=['picovoice'],
                    sensitivities=[sensitivity]
                )
                self.wake_word_name = "PICOVOICE"
            else:
                logger.info(f"Loading custom wake-word model: {model_path}")
                self.porcupine = pvporcupine.create(
                    access_key=access_key,
                    model_path=model_path,
                    sensitivities=[sensitivity]
                )
                self.wake_word_name = keyword
            
            self.is_initialized = True
            logger.info(f"Porcupine engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine engine: {e}")
            return False
    
    def process_audio(self, audio_chunk: bytes) -> bool:
        """
        Process an audio chunk and detect wake-word.
        
        Args:
            audio_chunk: Raw audio data as bytes
            
        Returns:
            True if wake-word detected, False otherwise
        """
        if not self.is_ready():
            return False
        
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Process with Porcupine
            keyword_index = self.porcupine.process(audio_data)
            
            if keyword_index >= 0:
                logger.info(f"🎯 Wake-word detected: {self.wake_word_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error processing audio with Porcupine: {e}")
            
        return False
    
    def get_sample_rate(self) -> int:
        """Get the required sample rate for Porcupine."""
        if self.porcupine:
            return self.porcupine.sample_rate
        return 16000  # Default
    
    def get_frame_length(self) -> int:
        """Get the required frame length for Porcupine."""
        if self.porcupine:
            return self.porcupine.frame_length
        return 512  # Default
    
    def get_wake_word_name(self) -> str:
        """Get the wake word name."""
        return self.wake_word_name
    
    def is_ready(self) -> bool:
        """Check if the engine is ready."""
        return self.is_initialized and self.porcupine is not None
    
    def cleanup(self) -> None:
        """Clean up Porcupine resources."""
        if self.porcupine:
            try:
                self.porcupine.delete()
            except Exception as e:
                logger.warning(f"Error cleaning up Porcupine: {e}") 