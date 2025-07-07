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
                self.model = openwakeword.Model(
                    wakeword_model_paths=[custom_model_path],
                    class_mapping_dicts=[{0: keyword}],
                    enable_speex_noise_suppression=False,
                    vad_threshold=0.5
                )
            else:
                logger.info(f"🔍 DEBUGGING: Loading pre-trained model for '{keyword}'")
                available_models = openwakeword.models
                if keyword not in available_models:
                    logger.warning(f"❌ Keyword '{keyword}' not found in pre-trained models!")
                    logger.info(f"   Available models: {list(available_models.keys())}")
                    logger.info("   Falling back to 'alexa' model")
                    keyword = 'alexa'  # Use a known model
                    self.wake_word_name = keyword
                    
                model_path = available_models[keyword]['model_path']
                logger.info(f"🔍 DEBUGGING: Model path: {model_path}")
                logger.info(f"   Model file exists: {os.path.exists(model_path)}")
                logger.info(f"   Model file size: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'} bytes")
                
                if not os.path.exists(model_path):
                    logger.error(f"❌ Model file not found at: {model_path}")
                    return False
                    
                # Enhanced model loading with detailed debugging
                logger.info("🔍 DEBUGGING: Creating OpenWakeWord Model object...")
                logger.info(f"   wakeword_model_paths: [{model_path}]")
                logger.info(f"   class_mapping_dicts: [{{0: '{keyword}'}}]")
                logger.info(f"   enable_speex_noise_suppression: False")
                logger.info(f"   vad_threshold: 0.5")
                
                self.model = openwakeword.Model(
                    wakeword_model_paths=[model_path],
                    class_mapping_dicts=[{0: keyword}],
                    enable_speex_noise_suppression=False,
                    vad_threshold=0.5
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
                logger.info(f"   Test prediction keys (if dict): {list(test_predictions.keys()) if isinstance(test_predictions, dict) else 'N/A'}")
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
            self.debug_counter += 1
            
            # Enhanced audio debugging (only log every 100th chunk to avoid spam)
            if self.debug_counter % 100 == 0:
                logger.info(f"🔍 DEBUGGING: Processing audio chunk #{self.debug_counter}")
                logger.info(f"   Raw audio: shape={audio_chunk.shape}, dtype={audio_chunk.dtype}")
                logger.info(f"   Raw audio: min={np.min(audio_chunk)}, max={np.max(audio_chunk)}, mean={np.mean(audio_chunk):.2f}")
                logger.info(f"   Raw audio: RMS={np.sqrt(np.mean(audio_chunk.astype(np.float32)**2)):.2f}")
            
            # Convert audio to float32 and normalize
            audio_float = audio_chunk.astype(np.float32) / 32768.0
            
            # Enhanced preprocessing debugging
            if self.debug_counter % 100 == 0:
                logger.info(f"🔍 DEBUGGING: Audio preprocessing")
                logger.info(f"   Float audio: shape={audio_float.shape}, dtype={audio_float.dtype}")
                logger.info(f"   Float audio: min={np.min(audio_float):.6f}, max={np.max(audio_float):.6f}, mean={np.mean(audio_float):.6f}")
                logger.info(f"   Float audio: RMS={np.sqrt(np.mean(audio_float**2)):.6f}")
                
                # Check for clipping or unusual values
                if np.any(np.abs(audio_float) > 1.0):
                    logger.warning(f"⚠️ Audio clipping detected: max abs value = {np.max(np.abs(audio_float)):.6f}")
                if np.all(audio_float == 0):
                    logger.warning(f"⚠️ Silent audio detected")
                elif np.sqrt(np.mean(audio_float**2)) < 0.001:
                    logger.warning(f"⚠️ Very quiet audio detected: RMS = {np.sqrt(np.mean(audio_float**2)):.6f}")
            
            # Get predictions with enhanced debugging
            if self.debug_counter % 100 == 0:
                logger.info(f"🔍 DEBUGGING: Calling model.predict()...")
            
            predictions = self.model.predict(audio_float)
            
            # Enhanced prediction debugging
            if self.debug_counter % 100 == 0:
                logger.info(f"🔍 DEBUGGING: Model prediction results")
                logger.info(f"   Prediction type: {type(predictions)}")
                logger.info(f"   Prediction content: {predictions}")
                
                if isinstance(predictions, dict):
                    logger.info(f"   Prediction keys: {list(predictions.keys())}")
                    logger.info(f"   Prediction values: {[f'{k}: {v:.6f}' for k, v in predictions.items()]}")
                elif isinstance(predictions, (list, tuple)):
                    logger.info(f"   Prediction list length: {len(predictions)}")
                    logger.info(f"   Prediction list values: {[f'{v:.6f}' for v in predictions]}")
                else:
                    logger.info(f"   Prediction scalar value: {predictions}")
            
            # Enhanced confidence extraction with detailed debugging
            confidence = 0.0
            confidence_source = "unknown"
            
            if isinstance(predictions, dict):
                # Try multiple possible keys for the wake word
                possible_keys = [
                    self.wake_word_name,  # e.g., 'alexa'
                    f"{self.wake_word_name}_v0.1",  # e.g., 'alexa_v0.1'
                    f"{self.wake_word_name}_v1.0",  # e.g., 'alexa_v1.0'
                    list(predictions.keys())[0] if predictions else None  # First key if available
                ]
                
                if self.debug_counter % 100 == 0:
                    logger.info(f"🔍 DEBUGGING: Extracting confidence from dict")
                    logger.info(f"   Looking for keys: {possible_keys}")
                    logger.info(f"   Available keys: {list(predictions.keys())}")
                
                for key in possible_keys:
                    if key and key in predictions:
                        confidence = predictions[key]
                        confidence_source = f"dict key '{key}'"
                        if self.debug_counter % 100 == 0:
                            logger.info(f"   Found confidence using key '{key}': {confidence:.6f}")
                        break
                
                if confidence == 0.0 and self.debug_counter % 100 == 0:
                    logger.warning(f"⚠️ No matching key found for wake word '{self.wake_word_name}'")
                    logger.info(f"   Available keys: {list(predictions.keys())}")
                    # Try using the first available key as fallback
                    if predictions:
                        first_key = list(predictions.keys())[0]
                        confidence = predictions[first_key]
                        confidence_source = f"fallback key '{first_key}'"
                        logger.info(f"   Using fallback key '{first_key}': {confidence:.6f}")
                    
            elif isinstance(predictions, (list, tuple)):
                confidence = float(predictions[0]) if predictions else 0.0
                confidence_source = "list[0]"
                if self.debug_counter % 100 == 0:
                    logger.info(f"   Extracted confidence from list: {confidence:.6f}")
            elif isinstance(predictions, (int, float)):
                confidence = float(predictions)
                confidence_source = "scalar"
                if self.debug_counter % 100 == 0:
                    logger.info(f"   Extracted confidence from scalar: {confidence:.6f}")
            else:
                try:
                    confidence = float(predictions)
                    confidence_source = "converted scalar"
                    if self.debug_counter % 100 == 0:
                        logger.info(f"   Converted confidence from {type(predictions)}: {confidence:.6f}")
                except (ValueError, TypeError):
                    logger.error(f"❌ Unexpected prediction format: {type(predictions)} - {predictions}")
                    return False
            
            # Enhanced confidence logging
            if self.debug_counter % 100 == 0:
                logger.info(f"🔍 DEBUGGING: Final confidence analysis")
                logger.info(f"   Confidence: {confidence:.6f}")
                logger.info(f"   Confidence source: {confidence_source}")
                logger.info(f"   Threshold: {self.threshold:.6f}")
                logger.info(f"   Detection: {confidence >= self.threshold}")
            
            # Always log confidence for non-zero values or detections
            if confidence > 0.0 or confidence >= self.threshold:
                logger.info(f"🎯 Wake word confidence: {confidence:.6f} (threshold: {self.threshold:.6f}) - Source: {confidence_source}")
            
            detected = confidence >= self.threshold
            
            if detected:
                logger.info(f"🎯 Wake word detected! Confidence: {confidence:.6f} (source: {confidence_source})")
            
            return detected
            
        except Exception as e:
            logger.error(f"❌ Error processing audio with OpenWakeWord: {e}", exc_info=True)
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