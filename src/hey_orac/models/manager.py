"""
ModelManager for dynamic loading and management of TFLite and ONNX wake word models.
Optimized for Raspberry Pi with TFLite runtime efficiency.
"""

import os
import hashlib
import logging
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
from openwakeword.model import Model

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    path: str
    format: str  # 'tflite' or 'onnx'
    checksum: str
    load_time: float
    last_inference_time: float
    inference_count: int
    name: str


class ModelManager:
    """
    Manages dynamic loading and hot-reloading of TFLite and ONNX wake word models.
    Optimized for Raspberry Pi with TFLite runtime efficiency.
    """
    
    def __init__(self, models_dir: str = "/app/models", default_framework: str = "tflite"):
        """
        Initialize ModelManager.
        
        Args:
            models_dir: Directory containing custom models
            default_framework: Default inference framework ('tflite' or 'onnx')
        """
        self.models_dir = Path(models_dir)
        self.default_framework = default_framework
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe model storage
        self._models: Dict[str, ModelInfo] = {}
        self._active_model: Optional[Model] = None
        self._model_lock = threading.RLock()
        
        # Change detection
        self._file_checksums: Dict[str, str] = {}
        self._change_callbacks: List[Callable] = []
        
        # Performance metrics
        self._metrics = {
            'total_inferences': 0,
            'total_inference_time': 0.0,
            'model_loads': 0,
            'hot_reloads': 0,
            'tflite_inferences': 0,
            'onnx_inferences': 0
        }
        
        logger.info(f"ModelManager initialized with models_dir: {models_dir}")
        logger.info(f"Default framework: {default_framework}")
        logger.info(f"TFLite runtime optimization enabled for Raspberry Pi")
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_model_format(self, file_path: str) -> str:
        """Determine model format from file extension."""
        ext = Path(file_path).suffix.lower()
        if ext == '.tflite':
            return 'tflite'
        elif ext == '.onnx':
            return 'onnx'
        else:
            raise ValueError(f"Unsupported model format: {ext}")
    
    def discover_models(self) -> List[str]:
        """
        Discover all available model files in the models directory.
        
        Returns:
            List of model file paths
        """
        model_files = []
        
        # Search for .tflite and .onnx files
        for pattern in ['*.tflite', '*.onnx']:
            model_files.extend(self.models_dir.glob(pattern))
        
        # Convert to strings and sort (prioritize .tflite for Raspberry Pi)
        model_paths = [str(p) for p in model_files]
        model_paths.sort(key=lambda x: (not x.endswith('.tflite'), x))
        
        logger.info(f"Discovered {len(model_paths)} models: {[Path(p).name for p in model_paths]}")
        return model_paths
    
    def load_model(self, model_paths: Union[str, List[str]], 
                   framework: Optional[str] = None,
                   force_reload: bool = False) -> bool:
        """
        Load a single model or multiple models using OpenWakeWord.
        Optimized for TFLite on Raspberry Pi.
        
        Args:
            model_paths: Path to model file or list of model paths
            framework: Inference framework ('tflite' or 'onnx')
            force_reload: Force reload even if model hasn't changed
            
        Returns:
            True if successful, False otherwise
        """
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        
        framework = framework or self.default_framework
        
        with self._model_lock:
            try:
                start_time = time.time()
                
                # Check if reload is needed
                if not force_reload and self._active_model is not None:
                    current_checksums = {path: self._calculate_file_checksum(path) 
                                       for path in model_paths if os.path.exists(path)}
                    
                    # Compare with stored checksums
                    if all(self._file_checksums.get(path) == current_checksums.get(path) 
                           for path in model_paths):
                        logger.info("Models unchanged, skipping reload")
                        return True
                
                # Extract model names for OpenWakeWord
                model_names = []
                custom_models = []
                
                for path in model_paths:
                    if not os.path.exists(path):
                        logger.warning(f"Model file not found: {path}")
                        continue
                    
                    model_name = Path(path).stem
                    
                    # Check if it's a pre-trained model name
                    if model_name in ['hey_jarvis', 'alexa', 'hey_mycroft', 'hey_computer']:
                        model_names.append(model_name)
                    else:
                        # Custom model path
                        custom_models.append(path)
                
                logger.info(f"Loading models with {framework} framework...")
                logger.info(f"Pre-trained models: {model_names}")
                logger.info(f"Custom models: {[Path(p).name for p in custom_models]}")
                
                # Create OpenWakeWord model with TFLite optimization
                if framework == 'tflite':
                    logger.info("Using TFLite runtime - optimized for Raspberry Pi")
                    self._metrics['tflite_inferences'] += 1
                else:
                    logger.info(f"Using {framework} runtime")
                    self._metrics['onnx_inferences'] += 1
                
                # Load the model
                self._active_model = Model(
                    wakeword_models=model_names if model_names else None,
                    custom_model_paths=custom_models if custom_models else None,
                    inference_framework=framework
                )
                
                load_time = time.time() - start_time
                
                # Update model info
                for path in model_paths:
                    if os.path.exists(path):
                        checksum = self._calculate_file_checksum(path)
                        self._file_checksums[path] = checksum
                        
                        model_info = ModelInfo(
                            path=path,
                            format=self._get_model_format(path),
                            checksum=checksum,
                            load_time=load_time,
                            last_inference_time=0.0,
                            inference_count=0,
                            name=Path(path).stem
                        )
                        self._models[path] = model_info
                
                # Update metrics
                self._metrics['model_loads'] += 1
                if not force_reload:
                    self._metrics['hot_reloads'] += 1
                
                logger.info(f"✅ Models loaded successfully in {load_time:.3f}s")
                
                # Test model with dummy data
                self._test_model_inference()
                
                # Notify change callbacks
                for callback in self._change_callbacks:
                    try:
                        callback(self._active_model)
                    except Exception as e:
                        logger.error(f"Error in change callback: {e}")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to load models: {e}")
                return False
    
    def _test_model_inference(self) -> bool:
        """Test model inference with dummy data."""
        if not self._active_model:
            return False
        
        try:
            # Create dummy audio data (1280 samples for OpenWakeWord)
            dummy_audio = np.zeros(1280, dtype=np.float32)
            
            start_time = time.time()
            prediction = self._active_model.predict(dummy_audio)
            inference_time = time.time() - start_time
            
            logger.info(f"✅ Model inference test successful ({inference_time:.3f}s)")
            logger.info(f"Available models: {list(prediction.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model inference test failed: {e}")
            return False
    
    @contextmanager
    def get_model(self):
        """
        Context manager to safely get the active model for inference.
        Thread-safe access to the model.
        """
        with self._model_lock:
            if self._active_model is None:
                raise RuntimeError("No model loaded")
            yield self._active_model
    
    def predict(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Perform inference on audio data with performance tracking.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary of model predictions
        """
        with self._model_lock:
            if self._active_model is None:
                raise RuntimeError("No model loaded")
            
            start_time = time.time()
            prediction = self._active_model.predict(audio_data)
            inference_time = time.time() - start_time
            
            # Update metrics
            self._metrics['total_inferences'] += 1
            self._metrics['total_inference_time'] += inference_time
            
            # Update model-specific metrics
            for model_info in self._models.values():
                model_info.last_inference_time = inference_time
                model_info.inference_count += 1
            
            return prediction
    
    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        metrics = self._metrics.copy()
        
        # Calculate averages
        if metrics['total_inferences'] > 0:
            metrics['avg_inference_time'] = (
                metrics['total_inference_time'] / metrics['total_inferences']
            )
        else:
            metrics['avg_inference_time'] = 0.0
        
        # Add model-specific metrics
        metrics['loaded_models'] = len(self._models)
        metrics['active_model'] = self._active_model is not None
        
        return metrics
    
    def register_change_callback(self, callback: Callable):
        """Register a callback to be called when models change."""
        self._change_callbacks.append(callback)
    
    def check_for_changes(self) -> bool:
        """
        Check if any model files have changed.
        
        Returns:
            True if changes detected, False otherwise
        """
        for path, stored_checksum in self._file_checksums.items():
            if os.path.exists(path):
                current_checksum = self._calculate_file_checksum(path)
                if current_checksum != stored_checksum:
                    logger.info(f"Model file changed: {Path(path).name}")
                    return True
        return False
    
    def auto_reload_if_changed(self) -> bool:
        """
        Automatically reload models if files have changed.
        
        Returns:
            True if reloaded, False if no changes or reload failed
        """
        if self.check_for_changes():
            logger.info("🔄 Auto-reloading models due to file changes...")
            model_paths = list(self._file_checksums.keys())
            return self.load_model(model_paths, force_reload=True)
        return False
    
    def get_model_info(self) -> Dict[str, ModelInfo]:
        """Get information about all loaded models."""
        with self._model_lock:
            return self._models.copy()
    
    def unload_models(self):
        """Unload all models and free memory."""
        with self._model_lock:
            self._active_model = None
            self._models.clear()
            self._file_checksums.clear()
            logger.info("All models unloaded")