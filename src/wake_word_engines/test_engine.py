#!/usr/bin/env python3
"""
Test wake-word detection engine for development and testing
Simulates wake-word detection without requiring external dependencies
"""

import time
import random
from typing import Dict, Any
import logging
from wake_word_interface import WakeWordEngine

logger = logging.getLogger(__name__)

class TestEngine(WakeWordEngine):
    """Test wake-word detection engine for development."""
    
    def __init__(self):
        self.is_initialized = False
        self.wake_word_name = "TEST_WORD"
        self.detection_probability = 0.01  # 1% chance per audio chunk
        self.last_detection_time = 0
        self.min_detection_interval = 2.0  # Minimum seconds between detections
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the test engine.
        
        Args:
            config: Configuration dictionary (ignored for test engine)
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Get test-specific settings
            self.wake_word_name = config.get('keyword', 'TEST_WORD')
            self.detection_probability = config.get('detection_probability', 0.01)
            self.min_detection_interval = config.get('min_detection_interval', 2.0)
            
            self.is_initialized = True
            logger.info(f"Test engine initialized successfully")
            logger.info(f"Wake word: {self.wake_word_name}")
            logger.info(f"Detection probability: {self.detection_probability}")
            logger.info(f"Min detection interval: {self.min_detection_interval}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize test engine: {e}")
            return False
    
    def process_audio(self, audio_chunk: bytes) -> bool:
        """
        Process an audio chunk and simulate wake-word detection.
        
        Args:
            audio_chunk: Raw audio data as bytes (ignored for test engine)
            
        Returns:
            True if wake-word detected, False otherwise
        """
        if not self.is_ready():
            return False
        
        try:
            current_time = time.time()
            
            # Check if enough time has passed since last detection
            if current_time - self.last_detection_time < self.min_detection_interval:
                return False
            
            # Simulate detection based on probability
            if random.random() < self.detection_probability:
                self.last_detection_time = current_time
                logger.info(f"🎯 TEST: Wake-word detected: {self.wake_word_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error in test engine: {e}")
            
        return False
    
    def get_sample_rate(self) -> int:
        """Get the required sample rate for test engine."""
        return 16000  # Standard rate
    
    def get_frame_length(self) -> int:
        """Get the required frame length for test engine."""
        return 512  # Standard chunk size
    
    def get_wake_word_name(self) -> str:
        """Get the wake word name."""
        return self.wake_word_name
    
    def is_ready(self) -> bool:
        """Check if the engine is ready."""
        return self.is_initialized
    
    def cleanup(self) -> None:
        """Clean up test engine resources (none needed)."""
        pass 