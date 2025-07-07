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
            
            # Log available models
            logger.info(f"Available OpenWakeWord models: {list(openwakeword.models.keys())}")
            
            # Check if custom model is specified
            custom_model_path = config.get('custom_model_path')
            
            if custom_model_path and os.path.exists(custom_model_path):
                logger.info(f"Loading custom OpenWakeWord model: {custom_model_path}")
                self.model = openwakeword.Model(
                    wakeword_model_paths=[custom_model_path],
                    class_mapping_dicts=[{0: keyword}],
                    enable_speex_noise_suppression=False,
                    vad_threshold=0.5
                )
            else:
                logger.info(f"Loading pre-trained OpenWakeWord model for: {keyword}")
                available_models = openwakeword.models
                if keyword not in available_models:
                    logger.warning(f"Keyword '{keyword}' not found in pre-trained models. Available: {list(available_models.keys())}")
                    logger.info("Falling back to 'alexa' model")
                    keyword = 'alexa'  # Use a known model
                    self.wake_word_name = keyword
                    
                model_path = available_models[keyword]['model_path']
                logger.info(f"Model path: {model_path}")
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found at: {model_path}")
                    return False
                    
                self.model = openwakeword.Model(
                    wakeword_model_paths=[model_path],
                    class_mapping_dicts=[{0: keyword}],
                    enable_speex_noise_suppression=False,
                    vad_threshold=0.5
                )
                
            # Verify model initialization
            logger.info(f"Model object: {self.model}")
            self.is_initialized = True
            logger.info(f"✅ OpenWakeWord engine initialized successfully for '{self.wake_word_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenWakeWord engine: {e}", exc_info=True)
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
            logger.error("Engine not initialized or model is None")
            return False
        
        try:
            # Log raw audio stats
            logger.debug(f"Raw audio chunk: shape={audio_chunk.shape}, dtype={audio_chunk.dtype}, min={np.min(audio_chunk)}, max={np.max(audio_chunk)}")
            
            # Convert audio to float32 and normalize
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            
            # Log preprocessed audio stats
            logger.debug(f"Preprocessed audio: shape={audio_float.shape}, dtype={audio_float.dtype}, min={np.min(audio_float)}, max={np.max(audio_float)}")
            
            # Verify audio is in expected range [-1, 1]
            if np.any(np.abs(audio_float) > 1.0):
                logger.warning("Preprocessed audio exceeds expected range [-1, 1]")
            
            # Get predictions
            predictions = self.model.predict(audio_float)
            
            logger.debug(f"Prediction type: {type(predictions)}")
            logger.debug(f"Prediction content: {predictions}")
            
            # Handle different prediction formats
            if isinstance(predictions, dict):
                # Try multiple possible keys for the wake word
                possible_keys = [
                    self.wake_word_name,  # e.g., 'alexa'
                    f"{self.wake_word_name}_v0.1",  # e.g., 'alexa_v0.1'
                    f"{self.wake_word_name}_v1.0",  # e.g., 'alexa_v1.0'
                    list(predictions.keys())[0] if predictions else None  # First key if available
                ]
                
                confidence = 0.0
                for key in possible_keys:
                    if key and key in predictions:
                        confidence = predictions[key]
                        logger.debug(f"Found confidence using key '{key}': {confidence}")
                        break
                
                if confidence == 0.0:
                    logger.debug(f"Available prediction keys: {list(predictions.keys())}")
                    logger.debug(f"Looking for wake word: {self.wake_word_name}")
                    
            elif isinstance(predictions, (list, tuple)):
                confidence = float(predictions[0]) if predictions else 0.0
            elif isinstance(predictions, (int, float)):
                confidence = float(predictions)
            else:
                try:
                    confidence = float(predictions)
                except (ValueError, TypeError):
                    logger.error(f"Unexpected prediction format: {type(predictions)} - {predictions}")
                    return False
            
            logger.info(f"Wake word confidence: {confidence:.4f} (threshold: {self.threshold})")
            
            detected = confidence >= self.threshold
            
            if detected:
                logger.info(f"🎯 Wake word detected! Confidence: {confidence:.4f}")
            
            return detected
            
        except Exception as e:
            logger.error(f"Error processing audio with OpenWakeWord: {e}", exc_info=True)
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