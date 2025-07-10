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
        self.debug_counter = 0  # For tracking debug output frequency
        
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
            
            # Check if custom model is specified
            custom_model_path = config.get('custom_model_path')
            
            if custom_model_path and os.path.exists(custom_model_path):
                # For custom models, use the model name from the file path
                model_name = os.path.splitext(os.path.basename(custom_model_path))[0]
                self.wake_word_name = model_name
                logger.info(f"🔍 Using custom model name: {model_name}")
            else:
                # For pre-trained models, use the config keyword
                self.wake_word_name = keyword
            
            # Enhanced debugging: Log OpenWakeWord version and available models
            logger.info("🔍 DEBUGGING: OpenWakeWord initialization")
            logger.info(f"   OpenWakeWord version: {openwakeword.__version__ if hasattr(openwakeword, '__version__') else 'Unknown'}")
            logger.info(f"   Available models: {list(openwakeword.models.keys())}")
            logger.info(f"   Requested keyword: {keyword}")
            logger.info(f"   Threshold: {self.threshold}")
            
            # Check if custom model is specified
            custom_model_path = config.get('custom_model_path')
            
            if custom_model_path and os.path.exists(custom_model_path):
                logger.info(f"🔍 DEBUGGING: Loading custom model from {custom_model_path}")
                
                # Verify the custom model file
                if not custom_model_path.endswith(('.onnx', '.tflite')):
                    logger.error(f"❌ Custom model must be .onnx or .tflite format: {custom_model_path}")
                    return False
                
                # For custom models, use the correct OpenWakeWord API
                try:
                    logger.info(f"🔍 DEBUGGING: Using correct OpenWakeWord API for custom models")
                    logger.info(f"   Custom model path: {custom_model_path}")
                    logger.info(f"   Keyword: {keyword}")
                    
                    # Use the correct API: wakeword_model_paths and class_mapping_dicts
                    self.model = openwakeword.Model(
                        wakeword_model_paths=[custom_model_path],
                        class_mapping_dicts=[{0: keyword}],
                        vad_threshold=0.5,
                        enable_speex_noise_suppression=False
                    )
                    
                    logger.info(f"✅ Custom model loaded successfully: {custom_model_path}")
                    
                    # Test the custom model immediately
                    test_audio = np.zeros(1280, dtype=np.float32)
                    test_predictions = self.model.predict(test_audio)
                    logger.info(f"🔍 Custom model test prediction: {test_predictions}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load custom model: {e}")
                    logger.error(f"❌ Custom model path: {custom_model_path}")
                    logger.error(f"❌ Custom model exists: {os.path.exists(custom_model_path)}")
                    logger.error(f"❌ Custom model size: {os.path.getsize(custom_model_path) if os.path.exists(custom_model_path) else 'N/A'}")
                    logger.info("🔄 Falling back to pre-trained models...")
                    
                    # Fallback to pre-trained models
                    self.model = openwakeword.Model(
                        vad_threshold=0.5,
                        enable_speex_noise_suppression=False
                    )
                    
            else:
                if custom_model_path:
                    logger.warning(f"⚠️ Custom model path specified but file not found: {custom_model_path}")
                    logger.info("🔄 Falling back to pre-trained models...")
                
                logger.info(f"🔍 DEBUGGING: Loading pre-trained models using documented approach")
                
                # Use the documented approach: load ALL available models
                # This matches ARCHITECTURE_UPDATE.md implementation
                logger.info("🔍 DEBUGGING: Creating OpenWakeWord Model with all available models...")
                logger.info(f"   vad_threshold: 0.5")
                logger.info(f"   enable_speex_noise_suppression: False")
                logger.info(f"   wakeword_models: None (load all available)")
                
                # Initialize model with all available models (documented approach)
                # Note: We don't call download_models() as it's not available in this version
                logger.info("🔍 DEBUGGING: Creating OpenWakeWord Model with available models...")
                self.model = openwakeword.Model(
                    vad_threshold=0.5,
                    enable_speex_noise_suppression=False
                )
                
            # Enhanced model verification
            logger.info(f"🔍 DEBUGGING: Model object created: {self.model}")
            logger.info(f"   Model type: {type(self.model)}")
            logger.info(f"   Model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")
            
            # Test model with dummy audio immediately after loading
            logger.info("🔍 DEBUGGING: Testing model with dummy audio after loading...")
            test_audio = np.zeros(1280, dtype=np.float32)
            try:
                test_predictions = self.model.predict(test_audio)
                logger.info(f"   Test prediction type: {type(test_predictions)}")
                logger.info(f"   Test prediction content: {test_predictions}")
                logger.info(f"   Test prediction keys (if dict): {list(test_predictions.keys())}")
                
                # Check prediction_buffer after first prediction
                if hasattr(self.model, 'prediction_buffer'):
                    logger.info(f"   ✅ prediction_buffer available after first prediction")
                    logger.info(f"   prediction_buffer keys: {list(self.model.prediction_buffer.keys())}")
                    for key, scores in self.model.prediction_buffer.items():
                        logger.info(f"     Model '{key}': {len(scores)} scores, latest: {scores[-1] if scores else 'N/A'}")
                else:
                    logger.warning(f"   ⚠️ prediction_buffer still not available after first prediction")
                    
            except Exception as e:
                logger.error(f"❌ Error testing model after loading: {e}")
                return False
                
            self.is_initialized = True
            logger.info(f"✅ OpenWakeWord engine initialized successfully for '{self.wake_word_name}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenWakeWord engine: {e}", exc_info=True)
            return False
    
    def process_audio(self, audio_chunk: np.ndarray) -> bool:
        """
        Process an audio chunk and detect wake words.
        
        Args:
            audio_chunk: Audio data as numpy array (int16)
            
        Returns:
            bool: True if wake word detected, False otherwise
        """
        if not self.is_initialized or self.model is None:
            return False
            
        try:
            # Convert audio to float32 if needed
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Get predictions from the model
            predictions = self.model.predict(audio_chunk)
            
            # Find the highest confidence score
            max_confidence = 0.0
            best_model = None
            
            for model_name, confidence in predictions.items():
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_model = model_name
            
            # Check if any model exceeds the threshold
            if max_confidence >= self.threshold:
                # Only log when there's a detection
                logger.info(f"🎯 WAKE WORD DETECTED! Confidence: {max_confidence:.6f} (threshold: {self.threshold:.6f}) - Source: {best_model}")
                logger.info(f"   All model scores: {[f'{k}: {v:.6f}' for k, v in predictions.items()]}")
                return True
            else:
                # Only log every 100th chunk to reduce verbosity (about every 8 seconds)
                self.debug_counter += 1
                if self.debug_counter % 100 == 0:
                    logger.debug(f"🎯 Wake word confidence: {max_confidence:.6f} (threshold: {self.threshold:.6f}) - Source: best available model '{best_model}'")
                    logger.debug(f"   All model scores: {[f'{k}: {v:.6f}' for k, v in predictions.items()]}")
                
                return False
                
        except Exception as e:
            logger.error(f"❌ Error processing audio: {e}")
            return False
    
    def get_latest_confidence(self) -> float:
        """Get the latest confidence score for the wake word."""
        if not self.is_initialized or self.model is None:
            return 0.0
            
        try:
            # Get the latest confidence from the prediction buffer
            if hasattr(self.model, 'prediction_buffer') and self.model.prediction_buffer:
                # Get the confidence for our specific wake word
                if self.wake_word_name in self.model.prediction_buffer:
                    scores = self.model.prediction_buffer[self.wake_word_name]
                    if scores:
                        return scores[-1]  # Return the latest score
                
                # If our specific wake word isn't found, return the highest confidence
                max_confidence = 0.0
                for model_name, scores in self.model.prediction_buffer.items():
                    if scores and scores[-1] > max_confidence:
                        max_confidence = scores[-1]
                return max_confidence
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Error getting latest confidence: {e}")
            return 0.0
    
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