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
import time

logger = logging.getLogger(__name__)

class OpenWakeWordEngine(WakeWordEngine):
    """OpenWakeWord wake-word detection engine."""
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.wake_word_name = "Unknown"
        self.threshold = 0.5
        self.sample_rate = 16000
        self.debug_mode = True  # Set to False in production
        self.detection_history = []  # Track recent predictions for pattern analysis
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the OpenWakeWord engine.
        
        Args:
            config: Configuration dictionary containing:
                - keyword: Wake word to detect
                - sensitivity: Model sensitivity (0.0-1.0) - internal model parameter
                - threshold: Detection threshold (0.0-1.0) - confidence level to trigger
                - custom_model_path: Path to custom model (optional)
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # OpenWakeWord-specific validation
            # OpenWakeWord requires: 16kHz sample rate, mono channel, 16-bit PCM
            if self.sample_rate != 16000:
                logger.error(f"❌ OpenWakeWord requires 16kHz sample rate, got {self.sample_rate}Hz")
                return False
            
            logger.info(f"✅ OpenWakeWord audio format validation passed:")
            logger.info(f"   Sample rate: {self.sample_rate}Hz ✓")
            logger.info(f"   Frame length: {self.get_frame_length()} samples ✓")
            logger.info(f"   Expected chunk duration: {self.get_frame_length() / self.sample_rate * 1000:.1f}ms ✓")
            
            # Get sensitivity and threshold from config (0.0-1.0)
            self.sensitivity = config.get('sensitivity', 0.5)
            self.threshold = config.get('threshold', 0.13)
            
            logger.info(f"🔧 Model Sensitivity: {self.sensitivity:.3f} (internal model parameter)")
            logger.info(f"🔧 Detection Threshold: {self.threshold:.3f} (confidence level to trigger)")
            logger.info(f"   High sensitivity = more sensitive model processing")
            logger.info(f"   Low threshold = easier detection (triggers at lower confidence)")
            
            # Check if custom model is specified
            custom_model_path = config.get('custom_model_path')
            
            if custom_model_path and os.path.exists(custom_model_path):
                # For custom models, use the model name from the file path
                model_name = os.path.splitext(os.path.basename(custom_model_path))[0]
                self.wake_word_name = model_name
                logger.info(f"🔍 Using custom model name: {model_name}")
            else:
                # For pre-trained models, use a default name
                self.wake_word_name = "unknown"
                logger.info(f"🔍 Using pre-trained model (no custom model specified)")
            
            # Enhanced debugging: Log OpenWakeWord version and available models
            logger.info("🔍 DEBUGGING: OpenWakeWord initialization")
            logger.info(f"   OpenWakeWord version: {openwakeword.__version__ if hasattr(openwakeword, '__version__') else 'Unknown'}")
            logger.info(f"   Available models: {list(openwakeword.models.keys())}")
            logger.info(f"   Requested keyword: {config.get('keyword', 'N/A')}") # Keep this for pre-trained models
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
                    logger.info(f"   Keyword: {config.get('keyword', 'N/A')}") # Keep this for pre-trained models
                    
                    # ISSUE #1: VAD threshold conflict - we were using both OpenWakeWord's VAD and our own RMS filtering
                    # SOLUTION: Disable OpenWakeWord's VAD since we're doing our own audio filtering
                    
                    # CORRECT: Use class mapping for custom models
                    logger.info(f"🔍 CRITICAL: Loading custom model WITH class mapping")
                    self.model = openwakeword.Model(
                        wakeword_model_paths=[custom_model_path],
                        class_mapping_dicts=[{0: self.wake_word_name}],  # Maps output class 0 to our wake word name
                        vad_threshold=0.0,  # Disabled to prevent conflict with our RMS filtering
                        enable_speex_noise_suppression=False
                    )
                    
                    logger.info(f"✅ Custom model loaded successfully: {custom_model_path}")
                    
                    # Validate the custom model setup
                    self._validate_model_setup()
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load custom model: {e}")
                    logger.error(f"❌ Custom model path: {custom_model_path}")
                    logger.error(f"❌ Custom model exists: {os.path.exists(custom_model_path)}")
                    logger.error(f"❌ Custom model size: {os.path.getsize(custom_model_path) if os.path.exists(custom_model_path) else 'N/A'}")
                    logger.info("🔄 Falling back to pre-trained models...")
                    
                    # Fallback to pre-trained models
                    self.model = openwakeword.Model(
                        vad_threshold=0.0,  # CHANGED: Disabled to prevent conflict
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
                logger.info(f"   vad_threshold: 0.0 (DISABLED)")
                logger.info(f"   enable_speex_noise_suppression: False")
                logger.info(f"   wakeword_models: None (load all available)")
                
                # Initialize model with all available models (documented approach)
                # Note: We don't call download_models() as it's not available in this version
                logger.info("🔍 DEBUGGING: Creating OpenWakeWord Model with available models...")
                self.model = openwakeword.Model(
                    vad_threshold=0.0,  # CHANGED: Disabled to prevent conflict
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
            # ISSUE #2: Audio normalization was missing
            # OpenWakeWord expects float32 audio normalized to [-1, 1] range
            # SOLUTION: Properly normalize int16 audio to float32 [-1, 1]
            
            # CRITICAL: Check raw audio levels before normalization
            if audio_chunk.dtype == np.int16:
                raw_max = np.max(np.abs(audio_chunk))
                raw_min = np.min(audio_chunk)
                raw_range = raw_max - raw_min
                logger.info(f"🔍 CRITICAL: Raw int16 audio - max: {raw_max}, min: {raw_min}, range: {raw_range}")
                logger.info(f"🔍 CRITICAL: Raw audio uses {raw_max/32768*100:.2f}% of available dynamic range")
                
                # Check if audio levels are too low
                if raw_max < 100:
                    logger.warning(f"⚠️ CRITICAL: Audio levels extremely low! Max value {raw_max} < 100")
                    logger.warning(f"⚠️ CRITICAL: Check microphone gain - should be much higher")
                elif raw_max < 1000:
                    logger.warning(f"⚠️ CRITICAL: Audio levels very low! Max value {raw_max} < 1000")
                    logger.warning(f"⚠️ CRITICAL: Consider increasing microphone gain")
                
                # Convert int16 [-32768, 32767] to float32 [-1.0, 1.0]
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
                logger.info(f"🔍 CRITICAL: Audio normalized from int16 to float32, shape: {audio_chunk.shape}, range: [{audio_chunk.min():.6f}, {audio_chunk.max():.6f}]")
            elif audio_chunk.dtype != np.float32:
                # If not int16 or float32, convert to float32
                audio_chunk = audio_chunk.astype(np.float32)
                logger.info(f"🔍 CRITICAL: Audio converted to float32, shape: {audio_chunk.shape}, range: [{audio_chunk.min():.6f}, {audio_chunk.max():.6f}]")
            else:
                # Already float32, use as-is
                audio_chunk = audio_chunk
                logger.info(f"🔍 CRITICAL: Audio already float32, shape: {audio_chunk.shape}, range: [{audio_chunk.min():.6f}, {audio_chunk.max():.6f}]")
            
            # Get predictions from OpenWakeWord model
            predictions = self.model.predict(audio_chunk)
            
            # ISSUE #3: We were only checking for our specific model name, which might not match
            # SOLUTION: Log all predictions to understand what the model is actually returning
            if self.debug_mode:
                # Log all model predictions to see what names are being used
                logger.info(f"🔍 CRITICAL: All predictions: {predictions}")
                # Track prediction history for pattern analysis
                self.detection_history.append({
                    'timestamp': time.time(),
                    'predictions': predictions.copy()
                })
                # Keep only last 50 predictions
                if len(self.detection_history) > 50:
                    self.detection_history.pop(0)
                
                # Log every 100th prediction to see if we're getting non-zero values
                if len(self.detection_history) % 100 == 0:
                    logger.info(f"🔍 CRITICAL: Prediction history summary (last 10):")
                    for i, entry in enumerate(self.detection_history[-10:]):
                        logger.info(f"   Entry {i}: {entry['predictions']}")
            
            # ISSUE #4: Model name might not match what we expect
            # SOLUTION: Check all predictions, not just our expected name
            detected = False
            detection_model = None
            detection_confidence = 0.0
            
            # CRITICAL: Log threshold and all predictions for debugging
            logger.info(f"🔍 CRITICAL: Current threshold: {self.threshold:.6f}")
            logger.info(f"🔍 CRITICAL: Checking predictions against threshold:")
            
            for model_name, confidence in predictions.items():
                logger.info(f"   Model '{model_name}': {confidence:.6f} {'✓' if confidence > self.threshold else '✗'}")
                if confidence > self.threshold:
                    detected = True
                    detection_model = model_name
                    detection_confidence = confidence
                    logger.info(f"🔍 CRITICAL: DETECTION TRIGGERED! Model: {detection_model}, Confidence: {detection_confidence:.6f}")
                    break
            
            # ISSUE #5: Single-chunk detection might be too sensitive or not sensitive enough
            # SOLUTION: Implement a simple sliding window check for more robust detection
            if detected:
                # Check if we've had consistent detections in recent chunks
                if self._check_detection_consistency(detection_model):
                    logger.info(f"🎯 WAKE WORD DETECTED by model '{detection_model}'! "
                               f"Confidence: {detection_confidence:.3f}")
                    # Clear history after successful detection to prevent multiple triggers
                    self.detection_history.clear()
                    return True
            
            return False
                
        except Exception as e:
            logger.error(f"❌ Error processing audio: {e}")
            return False
    
    def _check_detection_consistency(self, model_name: str, window_size: int = 3, min_detections: int = 2) -> bool:
        """
        Check if we've had consistent detections in recent chunks to reduce false positives
        
        Args:
            model_name: The model that triggered detection
            window_size: Number of recent chunks to check
            min_detections: Minimum detections needed in the window
        
        Returns:
            True if detection is consistent enough
        """
        if len(self.detection_history) < window_size:
            # Not enough history yet, allow detection
            return True
        
        # Check last N predictions
        recent_detections = 0
        for entry in self.detection_history[-window_size:]:
            if entry['predictions'].get(model_name, 0.0) > self.threshold:
                recent_detections += 1
        
        return recent_detections >= min_detections
    
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
    
    def validate_audio_format(self, sample_rate: int, channels: int, frame_length: int) -> bool:
        """
        Validate that the audio format is compatible with OpenWakeWord.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            frame_length: Number of samples per frame
            
        Returns:
            bool: True if format is compatible, False otherwise
        """
        if sample_rate != 16000:
            logger.error(f"❌ OpenWakeWord requires 16kHz sample rate, got {sample_rate}Hz")
            return False
        
        if channels != 1:
            logger.error(f"❌ OpenWakeWord requires mono audio, got {channels} channels")
            return False
        
        if frame_length != 1280:
            logger.warning(f"⚠️ OpenWakeWord expects 1280 samples per frame, got {frame_length}")
            logger.warning(f"   This may cause detection issues")
        
        logger.info(f"✅ OpenWakeWord audio format validation passed:")
        logger.info(f"   Sample rate: {sample_rate}Hz ✓")
        logger.info(f"   Channels: {channels} ✓")
        logger.info(f"   Frame length: {frame_length} samples ✓")
        
        return True
    
    def is_ready(self) -> bool:
        """Check if the engine is ready to process audio."""
        return self.is_initialized and self.model is not None
    
    def update_sensitivity(self, new_sensitivity: float) -> bool:
        """
        Update the sensitivity dynamically without reinitializing the model.
        
        Args:
            new_sensitivity: New sensitivity value (0.0-1.0)
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            if not self.is_initialized:
                logger.warning("⚠️ Cannot update sensitivity - engine not initialized")
                return False
            
            old_sensitivity = self.sensitivity
            self.sensitivity = new_sensitivity
            
            logger.info(f"🔧 Model Sensitivity updated: {old_sensitivity:.3f} → {self.sensitivity:.3f}")
            logger.info(f"   High sensitivity = more sensitive model processing")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update sensitivity: {e}")
            return False
    
    def update_threshold(self, new_threshold: float) -> bool:
        """
        Update the detection threshold dynamically without reinitializing the model.
        
        Args:
            new_threshold: New threshold value (0.0-1.0)
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            if not self.is_initialized:
                logger.warning("⚠️ Cannot update threshold - engine not initialized")
                return False
            
            old_threshold = self.threshold
            self.threshold = new_threshold
            
            logger.info(f"🔧 Detection Threshold updated: {old_threshold:.3f} → {self.threshold:.3f}")
            logger.info(f"   Low threshold = easier detection (triggers at lower confidence)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update threshold: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.model = None
        self.is_initialized = False
    
    def _validate_model_setup(self):
        """Validate that the model is properly loaded and working."""
        try:
            logger.info(f"🔍 CRITICAL: Validating model setup for '{self.wake_word_name}'")
            
            # Test with silence
            test_audio = np.zeros(1280, dtype=np.float32)
            predictions = self.model.predict(test_audio)
            
            logger.info(f"✅ Model validation passed")
            logger.info(f"   Available prediction keys: {list(predictions.keys())}")
            logger.info(f"   Expected key: '{self.wake_word_name}'")
            logger.info(f"   Has expected key: {self.wake_word_name in predictions}")
            logger.info(f"   Silence test predictions: {predictions}")
            
            # Test with some noise to see if predictions change
            test_audio = np.random.normal(0, 0.1, 1280).astype(np.float32)
            noise_predictions = self.model.predict(test_audio)
            logger.info(f"   Noise test predictions: {noise_predictions}")
            
            # Check if predictions are different (model is working)
            if predictions == noise_predictions:
                logger.warning(f"⚠️ CRITICAL: Model returns same predictions for silence and noise!")
                logger.warning(f"   This suggests the model may not be processing audio correctly")
            else:
                logger.info(f"✅ Model responds differently to different inputs")
            
            # Log prediction types and ranges
            for key, value in predictions.items():
                logger.info(f"   Key '{key}': {value} (type: {type(value)})")
            
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            raise 