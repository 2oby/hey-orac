#!/usr/bin/env python3
"""
Wake Word Monitor - New Implementation
Part of the ORAC Voice-Control Architecture refactoring
Handles wake word detection with configuration-driven model management
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from settings_manager import get_settings_manager
from wake_word_interface import WakeWordDetector
from shared_memory_ipc import shared_memory_ipc

logger = logging.getLogger(__name__)


class WakeWordMonitor_new:
    """
    New wake word monitor implementation with configuration-driven model management.
    Now includes actual model loading and detection capabilities.
    """
    
    def __init__(self):
        """Initialize the new wake word monitor."""
        self.settings_manager = get_settings_manager()
        self.available_models = []
        self.model_configs = {}
        self.global_settings = {}
        
        # Model loading and detection
        self.active_detectors = {}  # Dict of model_name -> WakeWordDetector
        self.is_detection_enabled = True
        self.last_detection_time = 0
        self.detection_count = 0
        
        # Load configuration
        self._load_configuration()
        self._build_model_configs()
        
        # Load active models
        self._load_active_models()
        
        # Register settings watcher for real-time updates
        self.settings_manager.add_watcher(self._on_settings_changed)
        
        logger.info("✅ WakeWordMonitor_new initialized")
        logger.info(f"📊 Discovered {len(self.available_models)} available models")
        logger.info(f"🎯 Loaded {len(self.active_detectors)} active detectors")
    
    def _load_configuration(self):
        """Load global configuration settings."""
        logger.info("🔧 Loading wake word configuration...")
        
        # Load global settings
        self.global_settings = {
            'cooldown': self.settings_manager.get('wake_word.cooldown', 1.5),
            'engine': self.settings_manager.get('wake_word.engine', 'openwakeword'),
            'model_path': self.settings_manager.get('wake_word.model_path', ''),
            'custom_model_path': self.settings_manager.get('wake_word.custom_model_path', ''),
            'keyword': self.settings_manager.get('wake_word.keyword', 'hey_jarvis')
        }
        
        logger.info(f"🔧 Global Configuration:")
        logger.info(f"   Cooldown: {self.global_settings['cooldown']}s")
        logger.info(f"   Engine: {self.global_settings['engine']}")
        logger.info(f"   Keyword: {self.global_settings['keyword']}")
        logger.info(f"   Model Path: {self.global_settings['model_path']}")
        logger.info(f"   Custom Model Path: {self.global_settings['custom_model_path']}")
    
    def _build_model_configs(self):
        """Build configuration for each discovered model."""
        logger.info("🔧 Building model configurations...")
        
        # Get available models from settings manager
        self.available_models = self.settings_manager.get_available_models()
        
        for model_name in self.available_models:
            # Get complete model config from settings manager
            model_config = self.settings_manager.get_model_config(model_name)
            
            if model_config:
                self.model_configs[model_name] = model_config
                
                logger.info(f"🔧 Model '{model_name}' configuration:")
                logger.info(f"   Sensitivity: {model_config['sensitivity']:.3f}")
                logger.info(f"   Threshold: {model_config['threshold']:.3f}")
                logger.info(f"   API URL: {model_config['api_url']}")
                logger.info(f"   Active: {model_config['active']}")
                logger.info(f"   Files: {list(model_config['file_paths'].keys())}")
    
    def _load_active_models(self):
        """Load all currently active models."""
        logger.info("🔧 Loading active models...")
        
        active_models = self.get_active_models()
        logger.info(f"🎯 Found {len(active_models)} active models: {active_models}")
        
        for model_name in active_models:
            if self._load_single_model(model_name):
                logger.info(f"✅ Successfully loaded model: {model_name}")
            else:
                logger.error(f"❌ Failed to load model: {model_name}")
    
    def _load_single_model(self, model_name: str) -> bool:
        """Load a single model and create its detector."""
        try:
            logger.info(f"🔧 Loading model: {model_name}")
            
            # Get model configuration
            model_config = self.get_model_config(model_name)
            if not model_config:
                logger.error(f"❌ No configuration found for model: {model_name}")
                return False
            
            # Create detector configuration
            detector_config = self._create_detector_config(model_name, model_config)
            
            # Create and initialize detector
            detector = WakeWordDetector()
            if not detector.initialize(detector_config):
                logger.error(f"❌ Failed to initialize detector for model: {model_name}")
                return False
            
            # Store the detector
            self.active_detectors[model_name] = detector
            
            logger.info(f"✅ Model '{model_name}' loaded successfully")
            logger.info(f"   Wake word: {detector.get_wake_word_name()}")
            logger.info(f"   Sample rate: {detector.get_sample_rate()}")
            logger.info(f"   Frame length: {detector.get_frame_length()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model {model_name}: {e}")
            return False
    
    def _create_detector_config(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create detector configuration for a specific model."""
        # Start with global settings
        detector_config = {
            'wake_word': {
                'engine': self.global_settings['engine'],
                'sensitivity': model_config['sensitivity'],
                'threshold': model_config['threshold'],
                'keyword': self.global_settings['keyword'],
                'cooldown': self.global_settings['cooldown']
            }
        }
        
        # Add custom model path if available
        if 'file_paths' in model_config:
            # Check for ONNX file first, then TFLite
            if 'onnx' in model_config['file_paths']:
                detector_config['wake_word']['custom_model_path'] = model_config['file_paths']['onnx']
                logger.info(f"🔧 Using ONNX model for {model_name}: {model_config['file_paths']['onnx']}")
            elif 'tflite' in model_config['file_paths']:
                detector_config['wake_word']['custom_model_path'] = model_config['file_paths']['tflite']
                logger.info(f"🔧 Using TFLite model for {model_name}: {model_config['file_paths']['tflite']}")
        
        logger.info(f"🔧 Detector config for {model_name}:")
        logger.info(f"   Engine: {detector_config['wake_word']['engine']}")
        logger.info(f"   Sensitivity: {detector_config['wake_word']['sensitivity']:.3f}")
        logger.info(f"   Threshold: {detector_config['wake_word']['threshold']:.3f}")
        logger.info(f"   Custom model path: {detector_config['wake_word'].get('custom_model_path', 'None')}")
        
        return detector_config
    
    def _on_settings_changed(self, new_settings: Dict[str, Any]):
        """Handle settings changes from GUI."""
        logger.info("🔧 Settings changed, checking for model updates...")
        
        # Check if active models have changed
        old_active_models = set(self.active_detectors.keys())
        new_active_models = set(self.get_active_models())
        
        if old_active_models != new_active_models:
            logger.info(f"🔄 Active models changed: {old_active_models} → {new_active_models}")
            self._reload_active_models()
        else:
            # Check if model configurations have changed
            for model_name in self.active_detectors:
                if self._has_model_config_changed(model_name):
                    logger.info(f"🔄 Model configuration changed for: {model_name}")
                    self._reload_single_model(model_name)
    
    def _has_model_config_changed(self, model_name: str) -> bool:
        """Check if a model's configuration has changed."""
        old_config = self.model_configs.get(model_name, {})
        new_config = self.settings_manager.get_model_config(model_name)
        
        if not new_config:
            return False
        
        # Compare key settings
        old_sensitivity = old_config.get('sensitivity', 0.0)
        new_sensitivity = new_config.get('sensitivity', 0.0)
        old_threshold = old_config.get('threshold', 0.0)
        new_threshold = new_config.get('threshold', 0.0)
        
        return (abs(old_sensitivity - new_sensitivity) > 0.001 or 
                abs(old_threshold - new_threshold) > 0.001)
    
    def _reload_active_models(self):
        """Reload all active models when settings change."""
        logger.info("🔄 Reloading active models...")
        
        # Clean up old detectors
        for model_name, detector in self.active_detectors.items():
            try:
                detector.cleanup()
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up detector {model_name}: {e}")
        
        self.active_detectors.clear()
        
        # Load new active models
        self._load_active_models()
        
        logger.info(f"✅ Active models reloaded: {list(self.active_detectors.keys())}")
    
    def _reload_single_model(self, model_name: str):
        """Reload a single model when its configuration changes."""
        logger.info(f"🔄 Reloading model: {model_name}")
        
        # Clean up old detector
        if model_name in self.active_detectors:
            try:
                self.active_detectors[model_name].cleanup()
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up detector {model_name}: {e}")
        
        # Reload the model
        if self._load_single_model(model_name):
            logger.info(f"✅ Model {model_name} reloaded successfully")
        else:
            logger.error(f"❌ Failed to reload model {model_name}")
    
    def process_audio(self, audio_data: np.ndarray) -> bool:
        """
        Process audio data through all active models.
        
        Args:
            audio_data: Audio data as numpy array (int16)
            
        Returns:
            True if any model detected a wake word, False otherwise
        """
        if not self.is_detection_enabled or not self.active_detectors:
            return False
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_detection_time < self.global_settings['cooldown']:
            return False
        
        # Process through all active detectors
        for model_name, detector in self.active_detectors.items():
            try:
                if detector.is_ready():
                    # Convert audio data to bytes for the detector
                    audio_bytes = audio_data.tobytes()
                    
                    # Process audio
                    detection_result = detector.process_audio(audio_bytes)
                    
                    if detection_result:
                        logger.info(f"🎯 WAKE WORD DETECTED by model '{model_name}'!")
                        self._handle_detection(model_name, detector, audio_data)
                        return True
                        
            except Exception as e:
                logger.error(f"❌ Error processing audio with model {model_name}: {e}")
        
        return False
    
    def _handle_detection(self, model_name: str, detector: WakeWordDetector, audio_data: np.ndarray):
        """Handle a wake word detection."""
        self.detection_count += 1
        self.last_detection_time = time.time()
        
        # Log detection details
        logger.info(f"🎯 DETECTION #{self.detection_count} - Model: {model_name}")
        logger.info(f"   Wake word: {detector.get_wake_word_name()}")
        logger.info(f"   Audio RMS: {np.sqrt(np.mean(audio_data.astype(np.float32)**2)):.4f}")
        
        # Update shared memory
        try:
            shared_memory_ipc.update_activation_state(True, model_name, 0.0)  # TODO: Get confidence
            logger.info(f"🌐 Updated shared memory with detection from {model_name}")
        except Exception as e:
            logger.warning(f"⚠️ Could not update shared memory: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return self.available_models.copy()
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model."""
        return self.model_configs.get(model_name)
    
    def get_global_settings(self) -> Dict[str, Any]:
        """Get global wake word settings."""
        return self.global_settings.copy()
    
    def get_model_sensitivity(self, model_name: str) -> float:
        """Get sensitivity for a specific model."""
        config = self.get_model_config(model_name)
        return config['sensitivity'] if config else 0.8
    
    def get_model_threshold(self, model_name: str) -> float:
        """Get threshold for a specific model."""
        config = self.get_model_config(model_name)
        return config['threshold'] if config else 0.3
    
    def get_model_api_url(self, model_name: str) -> str:
        """Get API URL for a specific model."""
        config = self.get_model_config(model_name)
        return config['api_url'] if config else "https://api.example.com/webhook"
    
    def get_model_active(self, model_name: str) -> bool:
        """Get active state for a specific model."""
        config = self.get_model_config(model_name)
        return config['active'] if config else False
    
    def get_active_models(self) -> List[str]:
        """Get list of currently active models."""
        return self.settings_manager.get_active_models()
    
    def get_cooldown_seconds(self) -> float:
        """Get global cooldown setting."""
        return self.global_settings['cooldown']
    
    def get_engine_type(self) -> str:
        """Get the wake word engine type."""
        return self.global_settings['engine']
    
    def get_keyword(self) -> str:
        """Get the wake word keyword."""
        return self.global_settings['keyword']
    
    def get_detection_count(self) -> int:
        """Get the total number of detections."""
        return self.detection_count
    
    def get_active_detectors(self) -> Dict[str, WakeWordDetector]:
        """Get the currently active detectors."""
        return self.active_detectors.copy()
    
    def print_configuration_summary(self):
        """Print a summary of the current configuration."""
        logger.info("📊 Wake Word Monitor Configuration Summary:")
        logger.info("=" * 50)
        
        # Global settings
        logger.info("🌐 Global Settings:")
        logger.info(f"   Engine: {self.get_engine_type()}")
        logger.info(f"   Keyword: {self.get_keyword()}")
        logger.info(f"   Cooldown: {self.get_cooldown_seconds()}s")
        
        # Active models
        active_models = self.get_active_models()
        logger.info(f"\n🎯 Active Models ({len(active_models)} models):")
        for model_name in active_models:
            config = self.get_model_config(model_name)
            if config:
                logger.info(f"   ✅ {model_name}:")
                logger.info(f"      Sensitivity: {config['sensitivity']:.3f}")
                logger.info(f"      Threshold: {config['threshold']:.3f}")
                logger.info(f"      API URL: {config['api_url']}")
        
        # Loaded detectors
        logger.info(f"\n🤖 Loaded Detectors ({len(self.active_detectors)} detectors):")
        for model_name, detector in self.active_detectors.items():
            logger.info(f"   ✅ {model_name}:")
            logger.info(f"      Wake word: {detector.get_wake_word_name()}")
            logger.info(f"      Sample rate: {detector.get_sample_rate()}")
            logger.info(f"      Frame length: {detector.get_frame_length()}")
            logger.info(f"      Ready: {detector.is_ready()}")
        
        # All model configurations
        logger.info(f"\n🤖 All Model Configurations ({len(self.available_models)} models):")
        for model_name in self.available_models:
            config = self.get_model_config(model_name)
            status = "✅ ACTIVE" if config['active'] else "❌ INACTIVE"
            logger.info(f"   📁 {model_name} ({status}):")
            logger.info(f"      Sensitivity: {config['sensitivity']:.3f}")
            logger.info(f"      Threshold: {config['threshold']:.3f}")
            logger.info(f"      API URL: {config['api_url']}")
            logger.info(f"      Files: {list(config['file_paths'].keys())}")
        
        logger.info("=" * 50)


def create_wake_word_monitor_new() -> WakeWordMonitor_new:
    """Factory function to create a new wake word monitor instance."""
    return WakeWordMonitor_new()


if __name__ == "__main__":
    # Test the new monitor
    monitor = create_wake_word_monitor_new()
    monitor.print_configuration_summary() 