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
import os

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
            config: Configuration dictionary containing:
                - keyword: Wake word to detect
                - threshold: Detection threshold (0.0-1.0)
                - custom_model_path: Path to custom model (optional)
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            keyword = config.get('keyword', 'hey_jarvis')
            self.threshold = config.get('threshold', 0.5)
            self.wake_word_name = keyword
            
            # Check if custom model is specified
            custom_model_path = config.get('custom_model_path')
            
            if custom_model_path and os.path.exists(custom_model_path):
                logger.info(f"Loading custom OpenWakeWord model: {custom_model_path}")
                # Load custom model
                self.model = openwakeword.Model(
                    wakeword_model_paths=[custom_model_path],
                    class_mapping_dicts=[{0: keyword}],
                    enable_speex_noise_suppression=False,
                    vad_threshold=0.5
                )
            else:
                # Use pre-trained models
                logger.info(f"Loading pre-trained OpenWakeWord model for: {keyword}")
                
                # Check if keyword is available in pre-trained models
                available_models = openwakeword.models
                if keyword not in available_models:
                    logger.warning(f"Keyword '{keyword}' not found in pre-trained models. Available: {list(available_models.keys())}")
                    logger.info("Falling back to 'hey_jarvis' model")
                    keyword = 'hey_jarvis'
                
                self.wake_word_name = keyword
                model_path = available_models[keyword]['model_path']
                
                self.model = openwakeword.Model(
                    wakeword_model_paths=[model_path],
                    class_mapping_dicts=[{0: keyword}],
                    enable_speex_noise_suppression=False,
                    vad_threshold=0.5
                )
            
            self.is_initialized = True
            logger.info(f"✅ OpenWakeWord engine initialized successfully for '{self.wake_word_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenWakeWord engine: {e}")
            return False
    
    def process_audio(self, audio_chunk: np.ndarray) -> bool:
        """
        Process audio chunk and detect wake word.
        
        Args:
            audio_chunk: Audio data as numpy array (int16)
        
        Returns:
            bool: True if wake word detected, False otherwise
        """
        if not self.is_initialized or self.model is None:
            return False
        
        try:
            # Convert audio to float32 and normalize
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            
            # Get predictions
            predictions = self.model.predict(audio_float)
            
            # Debug: Log prediction type and content
            logger.debug(f"Prediction type: {type(predictions)}")
            logger.debug(f"Prediction content: {predictions}")
            
            # Handle different prediction formats
            if isinstance(predictions, dict):
                # Dictionary format: {'hey_computer': 0.123, ...}
                confidence = predictions.get(self.wake_word_name, 0.0)
            elif isinstance(predictions, (list, tuple)):
                # List/tuple format: [0.123, 0.456, ...]
                confidence = float(predictions[0]) if predictions else 0.0
            elif isinstance(predictions, (int, float)):
                # Direct numeric value
                confidence = float(predictions)
            else:
                # Try to convert to float
                try:
                    confidence = float(predictions)
                except (ValueError, TypeError):
                    logger.error(f"Unexpected prediction format: {type(predictions)} - {predictions}")
                    return False
            
            # Log the confidence score for debugging
            logger.info(f"Wake word confidence: {confidence:.4f} (threshold: {self.threshold})")
            
            # Check if confidence exceeds threshold
            detected = confidence >= self.threshold
            
            if detected:
                logger.info(f"🎯 Wake word detected! Confidence: {confidence:.4f}")
            
            return detected
            
        except Exception as e:
            logger.error(f"Error processing audio with OpenWakeWord: {e}")
            return False
    
    def get_wake_word_name(self) -> str:
        """Get the name of the wake word being detected."""
        return self.wake_word_name
    
    def get_sample_rate(self) -> int:
        """Get the required sample rate for this engine."""
        return self.sample_rate
    
    def get_frame_length(self) -> int:
        """Get the required frame length for this engine."""
        return 1280  # OpenWakeWord requires 1280 samples (80ms at 16kHz)
    
    def is_ready(self) -> bool:
        """Check if the engine is ready to process audio."""
        return self.is_initialized and self.model is not None
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.model = None
        self.is_initialized = False 