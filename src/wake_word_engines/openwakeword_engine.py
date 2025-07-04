#!/usr/bin/env python3
"""
OpenWakeWord wake-word detection engine implementation
Open-source alternative to Porcupine with no licensing restrictions
"""

import openwakeword
import numpy as np
from typing import Dict, Any, Optional
import logging
from wake_word_interface import WakeWordEngine

logger = logging.getLogger(__name__)

class OpenWakeWordEngine(WakeWordEngine):
    """OpenWakeWord wake-word detection engine."""
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.wake_word_name = "Unknown"
        self.threshold = 0.5
        self.sample_rate = 16000
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the OpenWakeWord engine.
        
        Args:
            config: Configuration dictionary with OpenWakeWord settings
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            keyword = config.get('keyword', 'hey_orac')
            self.threshold = config.get('threshold', 0.5)
            
            # Initialize OpenWakeWord model with correct API
            # The API has changed - we need to use the correct initialization
            self.model = openwakeword.Model(
                inference_framework="onnx"
            )
            
            self.wake_word_name = keyword.upper().replace('_', ' ')
            self.is_initialized = True
            
            logger.info(f"OpenWakeWord engine initialized successfully")
            logger.info(f"Wake word: {self.wake_word_name}")
            logger.info(f"Threshold: {self.threshold}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenWakeWord engine: {e}")
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
            # Convert bytes to numpy array (16-bit PCM)
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Predict with OpenWakeWord
            prediction = self.model.predict(audio_float)
            
            # Check if any wake word was detected above threshold
            for wake_word, confidence in prediction.items():
                if confidence > self.threshold:
                    logger.info(f"🎯 OpenWakeWord detected: {wake_word} (confidence: {confidence:.3f})")
                    return True
                
        except Exception as e:
            logger.error(f"Error processing audio with OpenWakeWord: {e}")
            
        return False
    
    def get_sample_rate(self) -> int:
        """Get the required sample rate for OpenWakeWord."""
        return self.sample_rate
    
    def get_frame_length(self) -> int:
        """Get the required frame length for OpenWakeWord."""
        return 1024  # OpenWakeWord typically uses 1024 samples
    
    def get_wake_word_name(self) -> str:
        """Get the wake word name."""
        return self.wake_word_name
    
    def is_ready(self) -> bool:
        """Check if the engine is ready."""
        return self.is_initialized and self.model is not None
    
    def cleanup(self) -> None:
        """Clean up OpenWakeWord resources."""
        if self.model:
            try:
                # OpenWakeWord doesn't have explicit cleanup, but we can clear the reference
                self.model = None
            except Exception as e:
                logger.warning(f"Error cleaning up OpenWakeWord: {e}") 