#!/usr/bin/env python3
"""
Wake-word detection using Porcupine
"""

import pvporcupine
import logging
import numpy as np
from typing import Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)

class WakeWordDetector:
    """Porcupine wake-word detector."""
    
    def __init__(self, model_path: str, sensitivity: float = 0.6, keyword: str = "ORAC"):
        """
        Initialize the wake-word detector.
        
        Args:
            model_path: Path to the Porcupine model file (.ppn)
            sensitivity: Detection sensitivity (0.0-1.0)
            keyword: Wake-word keyword for logging
        """
        self.model_path = model_path
        self.sensitivity = sensitivity
        self.keyword = keyword
        self.porcupine = None
        self.is_initialized = False
        
        try:
            self._initialize_porcupine()
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            self.is_initialized = False
    
    def _initialize_porcupine(self):
        """Initialize the Porcupine wake-word detector."""
        model_file = Path(self.model_path)
        
        if not model_file.exists():
            logger.warning(f"Model file not found: {self.model_path}")
            logger.info("Attempting to use built-in 'Picovoice' wake word for testing...")
            # Use built-in wake word for testing
            self.porcupine = pvporcupine.create(
                access_key=None,  # Use built-in model
                keywords=['picovoice'],
                sensitivities=[self.sensitivity]
            )
            self.keyword = "PICOVOICE"  # Update for built-in model
        else:
            logger.info(f"Loading custom wake-word model: {self.model_path}")
            self.porcupine = pvporcupine.create(
                access_key=None,  # Will need access key for custom models
                model_path=self.model_path,
                sensitivities=[self.sensitivity]
            )
        
        self.is_initialized = True
        logger.info(f"Porcupine initialized successfully. Keyword: {self.keyword}")
        logger.info(f"Sample rate: {self.porcupine.sample_rate}")
        logger.info(f"Frame length: {self.porcupine.frame_length}")
    
    def process_audio(self, audio_chunk: np.ndarray) -> bool:
        """
        Process an audio chunk and detect wake-word.
        
        Args:
            audio_chunk: Audio data as numpy array (16-bit PCM)
            
        Returns:
            True if wake-word detected, False otherwise
        """
        if not self.is_initialized or self.porcupine is None:
            logger.warning("Porcupine not initialized, cannot process audio")
            return False
        
        try:
            # Process the audio chunk
            keyword_index = self.porcupine.process(audio_chunk)
            
            if keyword_index >= 0:
                logger.info(f"🎯 Wake-word detected: {self.keyword}")
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
    
    def is_ready(self) -> bool:
        """Check if the detector is ready to process audio."""
        return self.is_initialized and self.porcupine is not None
    
    def __del__(self):
        """Cleanup Porcupine resources."""
        if self.porcupine:
            try:
                self.porcupine.delete()
            except Exception as e:
                logger.warning(f"Error cleaning up Porcupine: {e}") 