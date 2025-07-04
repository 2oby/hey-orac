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
            # The Model class takes: wakeword_model_paths, class_mapping_dicts, enable_speex_noise_suppression, vad_threshold
            # If no paths provided, it loads all pre-trained models
            self.model = openwakeword.Model(
                wakeword_model_paths=[],  # Empty list loads all pre-trained models
                class_mapping_dicts=[],   # Empty list uses default mappings
                enable_speex_noise_suppression=False,  # Disable for now
                vad_threshold=0.0  # Disable VAD for now
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
            
            # OpenWakeWord expects 1280 samples of 16khz, 16-bit audio data
            # We need to ensure we have the right amount of data
            if len(audio_data) < 1280:
                # Pad with zeros if we don't have enough data
                audio_data = np.pad(audio_data, (0, 1280 - len(audio_data)), 'constant')
            elif len(audio_data) > 1280:
                # Truncate if we have too much data
                audio_data = audio_data[:1280]
            
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
        return 1280  # OpenWakeWord expects 1280 samples (80ms at 16kHz)
    
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