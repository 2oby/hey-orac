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
                
                # Verify the custom model file
                if not custom_model_path.endswith(('.onnx', '.tflite')):
                    logger.error(f"❌ Custom model must be .onnx or .tflite format: {custom_model_path}")
                    return False
                
                # For custom models, use the correct OpenWakeWord API
                try:
                    logger.info(f"🔍 DEBUGGING: Using correct OpenWakeWord API for custom models")
                    
                    # Use the correct API: wakeword_model_paths and class_mapping_dicts
                    self.model = openwakeword.Model(
                        wakeword_model_paths=[custom_model_path],
                        class_mapping_dicts=[{0: keyword}],
                        vad_threshold=0.5,
                        enable_speex_noise_suppression=False
                    )
                    
                    logger.info(f"✅ Custom model loaded successfully: {custom_model_path}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load custom model: {e}")
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
                
                # Download models if not present (one-time operation)
                try:
                    openwakeword.utils.download_models()
                    logger.info("✅ Models downloaded/verified")
                except Exception as e:
                    logger.warning(f"⚠️ Could not download models: {e}")
                
                # Initialize model with all available models (documented approach)
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
                logger.info(f"   Test prediction keys (if dict): {list(test_predictions.keys()) if isinstance(test_predictions, dict) else 'N/A'}")
                
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
            
            # Use documented approach: pass int16 audio directly to predict()
            # This matches ARCHITECTURE_UPDATE.md: prediction = self.model.predict(audio_np)
            if self.debug_counter % 100 == 0:
                logger.info(f"🔍 DEBUGGING: Audio processing (documented approach)")
                logger.info(f"   Raw audio: shape={audio_chunk.shape}, dtype={audio_chunk.dtype}")
                logger.info(f"   Raw audio: min={np.min(audio_chunk)}, max={np.max(audio_chunk)}, mean={np.mean(audio_chunk):.2f}")
                logger.info(f"   Raw audio: RMS={np.sqrt(np.mean(audio_chunk.astype(np.float32)**2)):.2f}")
                
                # Check for clipping or unusual values
                if np.any(np.abs(audio_chunk) > 32000):
                    logger.warning(f"⚠️ Audio clipping detected: max abs value = {np.max(np.abs(audio_chunk))}")
                if np.all(audio_chunk == 0):
                    logger.warning(f"⚠️ Silent audio detected")
                elif np.sqrt(np.mean(audio_chunk.astype(np.float32)**2)) < 100:
                    logger.warning(f"⚠️ Very quiet audio detected: RMS = {np.sqrt(np.mean(audio_chunk.astype(np.float32)**2)):.2f}")
            
            # Get predictions using the documented prediction_buffer approach
            if self.debug_counter % 100 == 0:
                logger.info(f"🔍 DEBUGGING: Calling model.predict() with int16 audio (documented approach)...")
            
            # Call predict with int16 audio (documented approach)
            raw_predictions = self.model.predict(audio_chunk)
            
            # Enhanced debugging of raw predictions
            if self.debug_counter % 100 == 0:
                logger.info(f"🔍 DEBUGGING: Raw model.predict() results")
                logger.info(f"   Raw prediction type: {type(raw_predictions)}")
                logger.info(f"   Raw prediction content: {raw_predictions}")
                
                if isinstance(raw_predictions, dict):
                    logger.info(f"   Raw prediction keys: {list(raw_predictions.keys())}")
                    logger.info(f"   Raw prediction values: {[f'{k}: {v:.6f}' for k, v in raw_predictions.items()]}")
            
            # Extract predictions from prediction_buffer (documented approach)
            predictions = {}
            confidence = 0.0
            confidence_source = "unknown"
            
            if hasattr(self.model, 'prediction_buffer'):
                if self.debug_counter % 100 == 0:
                    logger.info(f"🔍 DEBUGGING: prediction_buffer analysis")
                    logger.info(f"   prediction_buffer type: {type(self.model.prediction_buffer)}")
                    logger.info(f"   prediction_buffer keys: {list(self.model.prediction_buffer.keys())}")
                    logger.info(f"   Raw prediction_buffer content: {self.model.prediction_buffer}")
                
                # Extract predictions from each model's buffer
                for model_name, scores in self.model.prediction_buffer.items():
                    if self.debug_counter % 100 == 0:
                        logger.info(f"   Model '{model_name}' buffer: {len(scores)} scores")
                        logger.info(f"     Raw scores array: {scores}")
                        if len(scores) > 0:
                            logger.info(f"     Latest scores (last 5): {scores[-5:] if len(scores) >= 5 else scores}")
                            logger.info(f"     Most recent score: {scores[-1]}")
                    
                    if len(scores) > 0:
                        # Get the most recent score (latest prediction for this audio chunk)
                        # This is the key insight from ARCHITECTURE_UPDATE.md: scores[-1]
                        latest_score = scores[-1]
                        predictions[model_name] = latest_score
                        
                        if self.debug_counter % 100 == 0:
                            logger.info(f"     Using latest score for '{model_name}': {latest_score:.6f}")
                        
                    else:
                        if self.debug_counter % 100 == 0:
                            logger.warning(f"   ⚠️ Model '{model_name}' has no scores in buffer")
                
                if self.debug_counter % 100 == 0:
                    logger.info(f"🔍 DEBUGGING: Extracted predictions from buffer")
                    logger.info(f"   Final extracted predictions: {predictions}")
                    logger.info(f"   Available model names: {list(predictions.keys())}")
                    logger.info(f"   All model scores: {[f'{k}: {v:.6f}' for k, v in predictions.items()]}")
                
                # Use documented approach: check ALL models for detections above threshold
                # This matches ARCHITECTURE_UPDATE.md: detected_models = [name for name, score in predictions.items() if score > self.threshold]
                detected_models = [name for name, score in predictions.items() if score > self.threshold]
                
                if detected_models:
                    # Use the highest confidence score from detected models
                    best_model = max(detected_models, key=lambda k: predictions[k])
                    confidence = predictions[best_model]
                    confidence_source = f"detected model '{best_model}'"
                    if self.debug_counter % 100 == 0:
                        logger.info(f"   ✅ Detected wake words: {detected_models}")
                        logger.info(f"   Best model: '{best_model}' with confidence: {confidence:.6f}")
                else:
                    # No detections above threshold - use highest score for monitoring
                    if predictions:
                        best_model = max(predictions.keys(), key=lambda k: predictions[k])
                        confidence = predictions[best_model]
                        confidence_source = f"best available model '{best_model}'"
                        if self.debug_counter % 100 == 0:
                            logger.info(f"   No detections above threshold. Best score: {confidence:.6f} from '{best_model}'")
                    else:
                        confidence = 0.0
                        confidence_source = "no predictions available"
                        if self.debug_counter % 100 == 0:
                            logger.warning(f"   ⚠️ No predictions available from any model")
            
            else:
                # Fallback to direct prediction if prediction_buffer doesn't exist
                if self.debug_counter % 100 == 0:
                    logger.warning(f"⚠️ prediction_buffer not available, using direct predictions")
                
                if isinstance(raw_predictions, dict):
                    # Try multiple possible keys for the wake word
                    possible_keys = [
                        f"{self.wake_word_name}_v0.1",  # e.g., 'alexa_v0.1' - PRIMARY
                        self.wake_word_name,  # e.g., 'alexa' - FALLBACK
                        f"{self.wake_word_name}_v1.0",  # e.g., 'alexa_v1.0' - FALLBACK
                        list(raw_predictions.keys())[0] if raw_predictions else None  # First key if available
                    ]
                    
                    if self.debug_counter % 100 == 0:
                        logger.info(f"🔍 DEBUGGING: Fallback - extracting confidence from direct predictions")
                        logger.info(f"   Looking for keys: {possible_keys}")
                        logger.info(f"   Available keys: {list(raw_predictions.keys())}")
                    
                    for key in possible_keys:
                        if key and key in raw_predictions:
                            confidence = raw_predictions[key]
                            confidence_source = f"fallback direct key '{key}'"
                            if self.debug_counter % 100 == 0:
                                logger.info(f"   Found confidence using fallback key '{key}': {confidence:.6f}")
                            break
                    
                    if confidence == 0.0 and self.debug_counter % 100 == 0:
                        logger.warning(f"⚠️ No matching key found in fallback mode")
                        if raw_predictions:
                            first_key = list(raw_predictions.keys())[0]
                            confidence = raw_predictions[first_key]
                            confidence_source = f"fallback first key '{first_key}'"
                            logger.info(f"   Using fallback first key '{first_key}': {confidence:.6f}")
                            
                elif isinstance(raw_predictions, (list, tuple)):
                    confidence = float(raw_predictions[0]) if raw_predictions else 0.0
                    confidence_source = "fallback list[0]"
                    if self.debug_counter % 100 == 0:
                        logger.info(f"   Extracted confidence from fallback list: {confidence:.6f}")
                elif isinstance(raw_predictions, (int, float)):
                    confidence = float(raw_predictions)
                    confidence_source = "fallback scalar"
                    if self.debug_counter % 100 == 0:
                        logger.info(f"   Extracted confidence from fallback scalar: {confidence:.6f}")
                else:
                    try:
                        confidence = float(raw_predictions)
                        confidence_source = "fallback converted scalar"
                        if self.debug_counter % 100 == 0:
                            logger.info(f"   Converted confidence from fallback {type(raw_predictions)}: {confidence:.6f}")
                    except (ValueError, TypeError):
                        logger.error(f"❌ Unexpected prediction format: {type(raw_predictions)} - {raw_predictions}")
                        return False
            
            # Enhanced confidence logging with prediction_buffer state
            if self.debug_counter % 100 == 0:
                logger.info(f"🔍 DEBUGGING: Final confidence analysis")
                logger.info(f"   Confidence: {confidence:.6f}")
                logger.info(f"   Confidence source: {confidence_source}")
                logger.info(f"   Threshold: {self.threshold:.6f}")
                logger.info(f"   Detection: {confidence >= self.threshold}")
                logger.info(f"   All extracted predictions: {predictions}")
                
                # Show current prediction_buffer state if available
                if hasattr(self.model, 'prediction_buffer'):
                    logger.info(f"   Current prediction_buffer state:")
                    for key, scores in self.model.prediction_buffer.items():
                        if len(scores) > 0:
                            logger.info(f"     '{key}': {scores[-1]:.6f} (latest of {len(scores)} scores)")
                            if len(scores) > 1:
                                logger.info(f"       Recent history: {scores[-3:]}")
                        else:
                            logger.info(f"     '{key}': no scores yet")
                else:
                    logger.info(f"   No prediction_buffer available")
            
            # Always log confidence for non-zero values or detections
            if confidence > 0.0 or confidence >= self.threshold:
                logger.info(f"🎯 Wake word confidence: {confidence:.6f} (threshold: {self.threshold:.6f}) - Source: {confidence_source}")
                if hasattr(self.model, 'prediction_buffer'):
                    logger.info(f"   All model scores: {[f'{k}: {v:.6f}' for k, v in predictions.items()]}")
            
            detected = confidence >= self.threshold
            
            if detected:
                logger.info(f"🎯 Wake word detected! Confidence: {confidence:.6f} (source: {confidence_source})")
                logger.info(f"   All model scores at detection: {[f'{k}: {v:.6f}' for k, v in predictions.items()]}")
            
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