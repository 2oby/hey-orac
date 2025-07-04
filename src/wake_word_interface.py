#!/usr/bin/env python3
"""
Abstract interface for wake-word detection engines
Allows easy swapping between different wake-word detection libraries
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class WakeWordEngine(ABC):
    """Abstract base class for wake-word detection engines."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the wake-word detection engine.
        
        Args:
            config: Configuration dictionary for the engine
            
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def process_audio(self, audio_chunk: bytes) -> bool:
        """
        Process an audio chunk and detect wake-word.
        
        Args:
            audio_chunk: Raw audio data as bytes
            
        Returns:
            True if wake-word detected, False otherwise
        """
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get the required sample rate for this engine."""
        pass
    
    @abstractmethod
    def get_frame_length(self) -> int:
        """Get the required frame length for this engine."""
        pass
    
    @abstractmethod
    def get_wake_word_name(self) -> str:
        """Get the name of the wake word this engine detects."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the engine is ready to process audio."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources used by the engine."""
        pass

class WakeWordDetector:
    """Main wake-word detector that uses pluggable engines."""
    
    def __init__(self, engine_name: str = "porcupine"):
        """
        Initialize the wake-word detector with specified engine.
        
        Args:
            engine_name: Name of the engine to use ("porcupine", "snowboy", etc.)
        """
        self.engine_name = engine_name
        self.engine: Optional[WakeWordEngine] = None
        self.is_initialized = False
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the wake-word detection engine.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create engine instance based on name
            if self.engine_name == "porcupine":
                from wake_word_engines.porcupine_engine import PorcupineEngine
                self.engine = PorcupineEngine()
            elif self.engine_name == "openwakeword":
                from wake_word_engines.openwakeword_engine import OpenWakeWordEngine
                self.engine = OpenWakeWordEngine()
            elif self.engine_name == "snowboy":
                from wake_word_engines.snowboy_engine import SnowboyEngine
                self.engine = SnowboyEngine()
            elif self.engine_name == "pocketsphinx":
                from wake_word_engines.pocketsphinx_engine import PocketSphinxEngine
                self.engine = PocketSphinxEngine()
            elif self.engine_name == "test":
                from wake_word_engines.test_engine import TestEngine
                self.engine = TestEngine()
            else:
                logger.error(f"Unknown wake-word engine: {self.engine_name}")
                return False
            
            # Initialize the engine
            if self.engine.initialize(config):
                self.is_initialized = True
                logger.info(f"Wake-word engine '{self.engine_name}' initialized successfully")
                logger.info(f"Wake word: {self.engine.get_wake_word_name()}")
                logger.info(f"Sample rate: {self.engine.get_sample_rate()}")
                logger.info(f"Frame length: {self.engine.get_frame_length()}")
                return True
            else:
                logger.error(f"Failed to initialize wake-word engine: {self.engine_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing wake-word engine: {e}")
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
            logger.warning("Wake-word detector not ready")
            return False
        
        try:
            return self.engine.process_audio(audio_chunk)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return False
    
    def get_sample_rate(self) -> int:
        """Get the required sample rate."""
        if self.engine:
            return self.engine.get_sample_rate()
        return 16000  # Default
    
    def get_frame_length(self) -> int:
        """Get the required frame length."""
        if self.engine:
            return self.engine.get_frame_length()
        return 512  # Default
    
    def get_wake_word_name(self) -> str:
        """Get the wake word name."""
        if self.engine:
            return self.engine.get_wake_word_name()
        return "Unknown"
    
    def is_ready(self) -> bool:
        """Check if the detector is ready."""
        return self.is_initialized and self.engine and self.engine.is_ready()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.engine:
            try:
                self.engine.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up engine: {e}")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup() 