"""
Wake word detection using OpenWakeWord.
"""

import openwakeword
from openwakeword.model import Model
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


logger = logging.getLogger(__name__)


@dataclass
class WakeWordDetection:
    """Information about a wake word detection event."""
    model_name: str
    confidence: float
    timestamp: float
    audio_offset: int  # Sample offset in audio stream


class WakeDetector:
    """
    Manages wake word detection using OpenWakeWord models.
    
    Features:
    - Multiple model support
    - Configurable detection thresholds
    - Detection event tracking
    - Performance metrics
    """
    
    def __init__(self, models: Optional[List[str]] = None, inference_framework: str = 'tflite', 
                 vad_threshold: float = 0.5, cooldown: float = 2.0):
        """
        Initialize the wake word detector.
        
        Args:
            models: List of model names to load, or None for all pre-trained
            inference_framework: 'tflite' or 'onnx'
            vad_threshold: Voice activity detection threshold (0.0-1.0)
            cooldown: Seconds between detections
        """
        self.inference_framework = inference_framework
        self.models_to_load = models
        self.vad_threshold = vad_threshold
        
        # Model and thresholds
        self.model: Optional[Model] = None
        self.thresholds: Dict[str, float] = {}
        
        # Performance tracking
        self.inference_times: List[float] = []
        self.max_inference_history = 100
        
        # Detection state
        self.last_detections: Dict[str, float] = {}  # model -> last detection time
        self.detection_cooldown = cooldown
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Download and initialize OpenWakeWord models."""
        try:
            # Download pre-trained models if needed
            logger.info("Downloading OpenWakeWord models...")
            openwakeword.utils.download_models()
            
            # Create model instance
            logger.info(f"Initializing models with {self.inference_framework} framework...")
            
            if self.models_to_load:
                # Load specific models
                self.model = Model(
                    wakeword_models=self.models_to_load,
                    inference_framework=self.inference_framework,
                    vad_threshold=self.vad_threshold
                )
            else:
                # Load all pre-trained models
                self.model = Model(
                    inference_framework=self.inference_framework,
                    vad_threshold=self.vad_threshold
                )
            
            # Get loaded models
            if hasattr(self.model, 'models') and self.model.models:
                loaded_models = list(self.model.models.keys())
            else:
                # Get from prediction buffer
                loaded_models = list(self.model.prediction_buffer.keys())
            
            logger.info(f"Loaded models: {loaded_models}")
            
            # Set default thresholds
            for model_name in loaded_models:
                self.thresholds[model_name] = 0.5  # Default threshold
            
            # Test with dummy audio
            self._test_models()
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _test_models(self) -> None:
        """Test models with dummy audio to ensure they're working."""
        try:
            test_audio = np.zeros(1280, dtype=np.float32)
            predictions = self.model.predict(test_audio)
            logger.info(f"Model test successful, predictions: {list(predictions.keys())}")
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            raise
    
    def set_threshold(self, model_name: str, threshold: float) -> None:
        """
        Set detection threshold for a specific model.
        
        Args:
            model_name: Name of the model
            threshold: Detection threshold (0.0 to 1.0)
        """
        if model_name in self.thresholds:
            self.thresholds[model_name] = threshold
            logger.info(f"Set threshold for {model_name} to {threshold}")
        else:
            logger.warning(f"Model {model_name} not found")
    
    def process_audio(self, audio_chunk: np.ndarray) -> Optional[WakeWordDetection]:
        """
        Process audio chunk for wake word detection.
        
        Args:
            audio_chunk: Audio samples as float32 array
            
        Returns:
            WakeWordDetection if wake word detected, None otherwise
        """
        if self.model is None:
            return None
        
        # Measure inference time
        start_time = time.perf_counter()
        
        try:
            # Get predictions
            predictions = self.model.predict(audio_chunk)
            
            # Track inference time
            inference_time = time.perf_counter() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > self.max_inference_history:
                self.inference_times.pop(0)
            
            # Check for detections
            current_time = time.time()
            best_detection = None
            best_confidence = 0.0
            
            for model_name, confidence in predictions.items():
                threshold = self.thresholds.get(model_name, 0.5)
                
                # Check if above threshold
                if confidence >= threshold:
                    # Check cooldown
                    last_detection = self.last_detections.get(model_name, 0)
                    if current_time - last_detection >= self.detection_cooldown:
                        # Valid detection
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_detection = WakeWordDetection(
                                model_name=model_name,
                                confidence=float(confidence),
                                timestamp=current_time,
                                audio_offset=0  # TODO: Calculate actual offset
                            )
            
            if best_detection:
                # Update last detection time
                self.last_detections[best_detection.model_name] = current_time
                logger.info(
                    f"Wake word detected: {best_detection.model_name} "
                    f"(confidence: {best_detection.confidence:.3f})"
                )
                return best_detection
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None
    
    def get_model_names(self) -> List[str]:
        """Get list of loaded model names."""
        return list(self.thresholds.keys())
    
    def get_average_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times) * 1000  # Convert to ms
    
    def reset_model(self, model_name: Optional[str] = None) -> None:
        """
        Reset model state.
        
        Args:
            model_name: Specific model to reset, or None for all
        """
        if model_name:
            self.last_detections.pop(model_name, None)
            if hasattr(self.model, 'reset'):
                # Some versions have a reset method
                try:
                    self.model.reset()
                except:
                    pass
        else:
            self.last_detections.clear()
            if hasattr(self.model, 'reset'):
                try:
                    self.model.reset()
                except:
                    pass