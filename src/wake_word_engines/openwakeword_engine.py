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
import hashlib

logger = logging.getLogger(__name__)

class OpenWakeWordEngine(WakeWordEngine):
    """OpenWakeWord wake-word detection engine."""
    
    def __init__(self):
        self.model = None
        self.is_initialized = False
        self.wake_word_name = "Unknown"
        self.threshold = None
        self.sample_rate = 16000  # This is a constant for OpenWakeWord, not configurable
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
            if self.sample_rate != 16000:
                logger.error(f"❌ OpenWakeWord requires 16kHz sample rate, got {self.sample_rate}Hz")
                return False
            
            logger.info(f"✅ OpenWakeWord audio format validation passed:")
            logger.info(f"   Sample rate: {self.sample_rate}Hz ✓")
            logger.info(f"   Frame length: {self.get_frame_length()} samples ✓")
            logger.info(f"   Expected chunk duration: {self.get_frame_length() / self.sample_rate * 1000:.1f}ms ✓")
            
            # HARDCODED VALUES - Settings system is faulty, using interim hardcoded values
            self.sensitivity = 0.5  # Standard sensitivity
            self.threshold = 0.3   # Proper threshold to prevent false positives
            
            # Conservative amplification for low audio levels
            self.low_audio_threshold = 1000  # Only amplify very quiet audio
            self.amplification_factor = 2.0  # Gentle amplification
            
            logger.info("🔧 HARDCODED VALUES (settings system faulty):")
            logger.info(f"   Sensitivity: {self.sensitivity:.3f} (hardcoded)")
            logger.info(f"   Threshold: {self.threshold:.3f} (hardcoded)")
            logger.info(f"   Low audio threshold: {self.low_audio_threshold}")
            logger.info(f"   Amplification factor: {self.amplification_factor}")
            
            # FORCE DEFAULT MODELS FOR TESTING - Temporarily bypass custom models
            logger.info("🔧 FORCING DEFAULT MODELS FOR TESTING")
            logger.info("   Ignoring any custom model configuration")
            logger.info("   This will test with built-in OpenWakeWord models")
            
            # Comment out custom model loading temporarily
            wakeword_models = []  # Force empty to skip custom models
            custom_model_path = None  # Force None to skip custom models
            
            if False:  # Disable custom model loading block
                # Validate all model files before loading
                valid_models = []
                for model_path in wakeword_models:
                    if os.path.exists(model_path):
                        if self._validate_model_file(model_path):
                            valid_models.append(model_path)
                            logger.info(f"✅ Validated model: {model_path}")
                        else:
                            logger.error(f"❌ Model file validation failed: {model_path}")
                    else:
                        logger.warning(f"⚠️ Model file not found: {model_path}")
                
                if not valid_models:
                    logger.error("❌ No valid model files found")
                    return False
                
                # Extract model name from first file path
                model_name = os.path.splitext(os.path.basename(valid_models[0]))[0]
                self.wake_word_name = model_name
                logger.info(f"🔍 Using custom model name: {model_name}")
                
                # Try loading with different configurations
                logger.info(f"🔍 Attempting to load {len(valid_models)} custom models: {valid_models}")
                
                # First attempt: Try with inference_framework specified
                try:
                    logger.info("🔍 Attempt 1: Loading with inference_framework='onnx'")
                    self.model = openwakeword.Model(
                        wakeword_model_paths=valid_models,
                        inference_framework="onnx"  # Explicitly specify ONNX
                    )
                    logger.info("✅ Models loaded successfully with ONNX framework")
                except Exception as e1:
                    logger.warning(f"⚠️ Attempt 1 failed: {e1}")
                    
                    # Second attempt: Try without class mapping
                    try:
                        logger.info("🔍 Attempt 2: Loading without class mapping")
                        self.model = openwakeword.Model(
                            wakeword_model_paths=valid_models
                        )
                        logger.info("✅ Models loaded successfully without class mapping")
                    except Exception as e2:
                        logger.warning(f"⚠️ Attempt 2 failed: {e2}")
                        
                        # Third attempt: Original method with class mapping
                        try:
                            logger.info("🔍 Attempt 3: Loading with class mapping")
                            self.model = openwakeword.Model(
                                wakeword_model_paths=valid_models,
                                class_mapping_dicts=[{0: self.wake_word_name}],
                                vad_threshold=0.0,
                                enable_speex_noise_suppression=False
                            )
                            logger.info("✅ Models loaded successfully with class mapping")
                        except Exception as e3:
                            logger.error(f"❌ All loading attempts failed")
                            logger.error(f"   Error 1: {e1}")
                            logger.error(f"   Error 2: {e2}")
                            logger.error(f"   Error 3: {e3}")
                            
                            # Try to provide more debugging info
                            self._debug_openwakeword_internals()
                            return False
                
                # Validate the model is working
                if not self._validate_model_functionality():
                    logger.error("❌ Model functionality validation failed")
                    return False
                    
            else:
                logger.info("🔍 LOADING DEFAULT PRE-TRAINED MODELS")
                logger.info("   Available wake words: alexa, hey_mycroft, hey_jarvis, timer, weather, etc.")
                self.model = openwakeword.Model(
                    vad_threshold=0.0,
                    enable_speex_noise_suppression=False
                )
                self.wake_word_name = "default_models"
                logger.info("✅ Default pre-trained models loaded successfully")
                logger.info("   Try saying: 'Hey Jarvis', 'Hey Mycroft', or 'Alexa'")
            
            # Final validation and detailed model inspection
            if not self._validate_model_setup():
                logger.error("❌ Model setup validation failed")
                return False
            
            # CRITICAL DEBUG: Inspect what models are actually loaded
            self._inspect_loaded_models()
                
            self.is_initialized = True
            logger.info(f"✅ OpenWakeWord engine initialized successfully for '{self.wake_word_name}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenWakeWord engine: {e}", exc_info=True)
            return False
    
    def _validate_model_file(self, model_path: str) -> bool:
        """Validate the model file integrity and format."""
        try:
            logger.info(f"🔍 Validating model file: {model_path}")
            
            # Check file exists and is readable
            if not os.path.exists(model_path):
                logger.error(f"❌ Model file does not exist: {model_path}")
                return False
            
            # Check file size
            file_size = os.path.getsize(model_path)
            logger.info(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            
            if file_size < 1000:  # Less than 1KB is definitely wrong
                logger.error(f"❌ Model file is too small: {file_size} bytes")
                return False
            
            # Check file extension
            if not model_path.endswith(('.onnx', '.tflite')):
                logger.error(f"❌ Invalid model file extension: must be .onnx or .tflite")
                return False
            
            # Calculate file hash for integrity check
            with open(model_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            logger.info(f"   File MD5 hash: {file_hash}")
            
            # For ONNX files, try to validate structure
            if model_path.endswith('.onnx'):
                try:
                    import onnx
                    logger.info("   Attempting to validate ONNX model structure...")
                    model = onnx.load(model_path)
                    onnx.checker.check_model(model)
                    logger.info("   ✅ ONNX model structure is valid")
                    
                    # Log model info
                    logger.info(f"   Model producer: {model.producer_name} {model.producer_version}")
                    logger.info(f"   Model version: {model.model_version}")
                    logger.info(f"   Graph inputs: {[input.name for input in model.graph.input]}")
                    logger.info(f"   Graph outputs: {[output.name for output in model.graph.output]}")
                    
                except ImportError:
                    logger.warning("   ⚠️ ONNX library not available, skipping structural validation")
                except Exception as e:
                    logger.error(f"   ❌ ONNX validation failed: {e}")
                    return False
            
            logger.info(f"✅ Model file validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model file validation error: {e}")
            return False
    
    def _debug_openwakeword_internals(self):
        """Debug OpenWakeWord internals to understand the issue."""
        try:
            logger.info("🔍 Debugging OpenWakeWord internals...")
            
            # Check OpenWakeWord version and attributes
            logger.info(f"   OpenWakeWord module: {openwakeword}")
            logger.info(f"   Module file: {openwakeword.__file__ if hasattr(openwakeword, '__file__') else 'Unknown'}")
            logger.info(f"   Version: {openwakeword.__version__ if hasattr(openwakeword, '__version__') else 'Unknown'}")
            
            # Check available attributes
            attrs = [attr for attr in dir(openwakeword) if not attr.startswith('_')]
            logger.info(f"   Available attributes: {attrs}")
            
            # Check Model class
            if hasattr(openwakeword, 'Model'):
                model_attrs = [attr for attr in dir(openwakeword.Model) if not attr.startswith('_')]
                logger.info(f"   Model class attributes: {model_attrs}")
            
            # Try to access models
            if hasattr(openwakeword, 'models'):
                logger.info(f"   Available models: {list(openwakeword.models.keys())}")
            
        except Exception as e:
            logger.error(f"❌ Error debugging OpenWakeWord: {e}")
    
    def _validate_model_functionality(self) -> bool:
        """Validate that the model actually processes audio differently for different inputs."""
        try:
            logger.info("🔍 Validating model functionality with comprehensive tests...")
            
            # Test 1: Complete silence
            silence = np.zeros(1280, dtype=np.float32)
            silence_pred = self.model.predict(silence)
            
            # Test 2: Low frequency sine wave (100 Hz - below speech)
            t = np.linspace(0, 1280/16000, 1280, endpoint=False)
            low_sine = np.sin(2 * np.pi * 100 * t).astype(np.float32) * 0.3
            low_sine_pred = self.model.predict(low_sine)
            
            # Test 3: Speech-like frequency (1000 Hz)
            speech_sine = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.5
            speech_sine_pred = self.model.predict(speech_sine)
            
            # Test 4: High frequency (4000 Hz - sibilant range)
            high_sine = np.sin(2 * np.pi * 4000 * t).astype(np.float32) * 0.3
            high_sine_pred = self.model.predict(high_sine)
            
            # Test 5: White noise
            white_noise = np.random.normal(0, 0.1, 1280).astype(np.float32)
            white_noise_pred = self.model.predict(white_noise)
            
            # Test 6: Pink noise (more speech-like)
            pink_noise = self._generate_pink_noise(1280)
            pink_noise_pred = self.model.predict(pink_noise)
            
            # Test 7: Impulse
            impulse = np.zeros(1280, dtype=np.float32)
            impulse[640] = 1.0  # Spike in the middle
            impulse_pred = self.model.predict(impulse)
            
            # Log all predictions
            logger.info("📊 Model predictions for different inputs:")
            logger.info(f"   1. Silence:        {silence_pred}")
            logger.info(f"   2. 100Hz sine:     {low_sine_pred}")
            logger.info(f"   3. 1kHz sine:      {speech_sine_pred}")
            logger.info(f"   4. 4kHz sine:      {high_sine_pred}")
            logger.info(f"   5. White noise:    {white_noise_pred}")
            logger.info(f"   6. Pink noise:     {pink_noise_pred}")
            logger.info(f"   7. Impulse:        {impulse_pred}")
            
            # Check if predictions vary
            predictions = [silence_pred, low_sine_pred, speech_sine_pred, 
                          high_sine_pred, white_noise_pred, pink_noise_pred, impulse_pred]
            
            # Extract confidence values
            confidence_values = []
            for pred in predictions:
                if isinstance(pred, dict):
                    # Get the first value from the dict
                    conf = list(pred.values())[0] if pred else 0.0
                else:
                    conf = float(pred) if pred else 0.0
                confidence_values.append(conf)
            
            # Check variation
            unique_values = len(set(confidence_values))
            value_range = max(confidence_values) - min(confidence_values)
            
            logger.info(f"📊 Analysis:")
            logger.info(f"   Unique values: {unique_values}")
            logger.info(f"   Value range: {value_range:.6f}")
            logger.info(f"   Min confidence: {min(confidence_values):.6f}")
            logger.info(f"   Max confidence: {max(confidence_values):.6f}")
            
            if unique_values < 3 or value_range < 0.0001:
                logger.error("❌ Model shows insufficient variation in predictions!")
                logger.error("   This indicates the model is not processing audio correctly")
                logger.error("   Possible causes:")
                logger.error("   - Corrupted model file")
                logger.error("   - Incompatible OpenWakeWord version")
                logger.error("   - Model expects different preprocessing")
                return False
            
            logger.info("✅ Model shows appropriate variation in predictions")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model functionality validation error: {e}")
            return False
    
    def _generate_pink_noise(self, n_samples: int) -> np.ndarray:
        """Generate pink noise (1/f noise) which is more speech-like than white noise."""
        # Simple pink noise generation using accumulation method
        white = np.random.normal(0, 0.1, n_samples)
        pink = np.zeros_like(white)
        pink[0] = white[0]
        for i in range(1, n_samples):
            pink[i] = pink[i-1] * 0.9 + white[i] * 0.1
        return pink.astype(np.float32)
    
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
            # Convert and normalize audio
            if audio_chunk.dtype == np.int16:
                raw_max = np.max(np.abs(audio_chunk))
                raw_min = np.min(audio_chunk)
                raw_range = raw_max - raw_min
                
                if self.debug_mode and raw_max < 100:
                    logger.warning(f"⚠️ CRITICAL: Audio levels extremely low! Max value {raw_max} < 100")
                
                # Convert int16 to float32
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
                
                # Apply gentle amplification only if audio is very quiet
                if raw_max < self.low_audio_threshold:
                    audio_chunk = audio_chunk * self.amplification_factor
                    if self.debug_mode:
                        logger.debug(f"🔧 Applied {self.amplification_factor}x amplification for low audio (max={raw_max})")
                    
            elif audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Get predictions from model
            predictions = self.model.predict(audio_chunk)
            
            # Log confidence values periodically (every 100 chunks)
            if hasattr(self, '_confidence_log_counter'):
                self._confidence_log_counter += 1
            else:
                self._confidence_log_counter = 0
            
            # Enhanced debugging with proper confidence logging
            if self._confidence_log_counter % 100 == 0:
                if isinstance(predictions, dict):
                    max_confidence = max(predictions.values()) if predictions else 0.0
                    logger.info(f"🔍 Confidence check (chunk {self._confidence_log_counter}):")
                    for model_name, confidence in predictions.items():
                        status = "ABOVE" if confidence > self.threshold else "below"
                        logger.info(f"   {model_name}: {confidence:.6f} ({status} threshold {self.threshold:.3f})")
                    logger.info(f"   Max confidence: {max_confidence:.6f}")
                else:
                    logger.info(f"🔍 Confidence check - Raw predictions: {predictions}")
            
            # Process predictions
            detected = False
            detection_model = None
            detection_confidence = 0.0
            
            if isinstance(predictions, dict):
                for model_name, confidence in predictions.items():
                    if confidence > self.threshold:
                        detected = True
                        detection_model = model_name
                        detection_confidence = confidence
                        logger.info(f"🎯 POTENTIAL DETECTION! Model: {detection_model}, Confidence: {detection_confidence:.6f} (threshold: {self.threshold:.3f})")
                        break
            
            # Track prediction history
            if self.debug_mode:
                self.detection_history.append({
                    'timestamp': time.time(),
                    'predictions': predictions.copy() if isinstance(predictions, dict) else predictions
                })
                if len(self.detection_history) > 50:
                    self.detection_history.pop(0)
            
            # Apply consistency check if detected
            if detected and self._check_detection_consistency(detection_model):
                logger.info(f"🎯 WAKE WORD DETECTED by '{detection_model}'! Confidence: {detection_confidence:.3f}")
                self.detection_history.clear()
                return True
            
            return False
                
        except Exception as e:
            logger.error(f"❌ Error processing audio: {e}")
            return False
    
    def _check_detection_consistency(self, model_name: str, window_size: int = 3, min_detections: int = 2) -> bool:
        """Check if we've had consistent detections in recent chunks to prevent false positives."""
        if len(self.detection_history) < window_size:
            logger.debug(f"🔍 Consistency check: Not enough history ({len(self.detection_history)} < {window_size}), allowing detection")
            return True
        
        recent_detections = 0
        for entry in self.detection_history[-window_size:]:
            if isinstance(entry['predictions'], dict):
                if entry['predictions'].get(model_name, 0.0) > self.threshold:
                    recent_detections += 1
        
        logger.debug(f"🔍 Consistency check for {model_name}: {recent_detections}/{window_size} recent detections (need {min_detections})")
        return recent_detections >= min_detections
    
    def _validate_model_setup(self):
        """Validate that the model is properly loaded and working."""
        try:
            logger.info(f"🔍 Final model setup validation for '{self.wake_word_name}'")
            
            # Quick test with silence
            silence_audio = np.zeros(1280, dtype=np.float32)
            predictions = self.model.predict(silence_audio)
            
            if isinstance(predictions, dict):
                logger.info(f"✅ Model setup validated - prediction keys: {list(predictions.keys())}")
            else:
                logger.warning(f"⚠️ Unexpected prediction format: {type(predictions)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            return False
    
    def _inspect_loaded_models(self):
        """Inspect what models are actually loaded in OpenWakeWord."""
        try:
            logger.info("🔍 DETAILED MODEL INSPECTION:")
            
            if not self.model:
                logger.error("   ❌ No model object found!")
                return
            
            # Check model attributes
            model_attrs = [attr for attr in dir(self.model) if not attr.startswith('_')]
            logger.info(f"   Model attributes: {model_attrs}")
            
            # Check prediction buffer
            if hasattr(self.model, 'prediction_buffer'):
                buffer_keys = list(self.model.prediction_buffer.keys())
                logger.info(f"   Prediction buffer models: {buffer_keys}")
                
                # This tells us if default models are loaded
                default_models = ['alexa', 'hey_mycroft', 'hey_jarvis', 'timer', 'weather']
                loaded_defaults = [model for model in buffer_keys if model in default_models]
                loaded_custom = [model for model in buffer_keys if model not in default_models]
                
                if loaded_defaults:
                    logger.warning(f"   ⚠️ DEFAULT MODELS LOADED: {loaded_defaults}")
                    logger.warning("   This might be the problem - default models instead of custom!")
                
                if loaded_custom:
                    logger.info(f"   ✅ CUSTOM MODELS LOADED: {loaded_custom}")
                else:
                    logger.error("   ❌ NO CUSTOM MODELS FOUND!")
            
            # Check model paths
            if hasattr(self.model, 'model_paths'):
                logger.info(f"   Model paths: {self.model.model_paths}")
            
            # Check loaded model objects
            if hasattr(self.model, 'models'):
                logger.info(f"   Loaded model objects: {list(self.model.models.keys())}")
            
            # Check if wakeword_models was used
            if hasattr(self.model, 'wakeword_models'):
                logger.info(f"   Wakeword models attribute: {self.model.wakeword_models}")
            
        except Exception as e:
            logger.error(f"❌ Error inspecting models: {e}")
    
    def get_latest_confidence(self) -> float:
        """Get the latest confidence score for the wake word."""
        if not self.is_initialized or self.model is None:
            return 0.0
            
        try:
            if hasattr(self.model, 'prediction_buffer') and self.model.prediction_buffer:
                if self.wake_word_name in self.model.prediction_buffer:
                    scores = self.model.prediction_buffer[self.wake_word_name]
                    if scores:
                        return scores[-1]
                
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
        """Validate that the audio format is compatible with OpenWakeWord."""
        if sample_rate != 16000:
            logger.error(f"❌ OpenWakeWord requires 16kHz sample rate, got {sample_rate}Hz")
            return False
        
        if channels != 1:
            logger.error(f"❌ OpenWakeWord requires mono audio, got {channels} channels")
            return False
        
        if frame_length != 1280:
            logger.warning(f"⚠️ OpenWakeWord expects 1280 samples per frame, got {frame_length}")
        
        return True
    
    def is_ready(self) -> bool:
        """Check if the engine is ready to process audio."""
        return self.is_initialized and self.model is not None
    
    def update_sensitivity(self, new_sensitivity: float) -> bool:
        """Update the sensitivity dynamically."""
        try:
            if not self.is_initialized:
                return False
            
            self.sensitivity = new_sensitivity
            logger.info(f"🔧 Sensitivity updated to {self.sensitivity:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update sensitivity: {e}")
            return False
    
    def update_threshold(self, new_threshold: float) -> bool:
        """Update the detection threshold dynamically."""
        try:
            if not self.is_initialized:
                return False
            
            self.threshold = new_threshold
            logger.info(f"🔧 Threshold updated to {self.threshold:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update threshold: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.model = None
        self.is_initialized = False