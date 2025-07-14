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
            'wakeword_models': []  # Will be populated with active model paths
        }
        
        logger.info(f"🔧 Global Configuration:")
        logger.info(f"   Cooldown: {self.global_settings['cooldown']}s")
        logger.info(f"   Engine: {self.global_settings['engine']}")
        logger.info(f"   Model Path: {self.global_settings['model_path']}")
        logger.info(f"   Wakeword Models: {len(self.global_settings['wakeword_models'])} models")
    
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
        """
        Load a single model and create its detector.
        
        TODO: In the future, we want to load multiple models simultaneously 
        for parallel processing instead of loading them one by one.
        This will improve performance and allow for better model comparison.
        """
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
            
            # ENHANCED DEBUGGING: Log the exact config being passed
            logger.info(f"🔧 INITIALIZATION DEBUG for {model_name}:")
            logger.info(f"   Config keys: {list(detector_config.keys())}")
            logger.info(f"   Wake word config: {detector_config.get('wake_word', {})}")
            
            if not detector.initialize(detector_config):
                logger.error(f"❌ Failed to initialize detector for model: {model_name}")
                logger.error(f"   Config that failed: {detector_config}")
                return False
            
            # Store the detector
            self.active_detectors[model_name] = detector
            
            logger.info(f"✅ Model '{model_name}' loaded successfully")
            logger.info(f"   Wake word: {detector.get_wake_word_name()}")
            logger.info(f"   Sample rate: {detector.get_sample_rate()}")
            logger.info(f"   Frame length: {detector.get_frame_length()}")
            logger.info(f"   Is ready: {detector.is_ready()}")
            logger.info(f"   Active detectors count: {len(self.active_detectors)}")
            logger.info(f"   Active detectors keys: {list(self.active_detectors.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model {model_name}: {e}")
            return False
    
    def _create_detector_config(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create detector configuration for a specific model."""
        # Get ALL settings, not just wake_word section
        all_settings = self.settings_manager.get_all()
        
        # Create complete config with all sections
        detector_config = {
            'audio': all_settings.get('audio', {}),
            'detection': all_settings.get('detection', {}),
            'buffer': all_settings.get('buffer', {}),
            'volume_monitoring': all_settings.get('volume_monitoring', {}),
            'wake_word': {
                'engine': self.global_settings['engine'],
                'sensitivity': model_config['sensitivity'],
                'threshold': model_config['threshold'],
                'cooldown': self.global_settings['cooldown']
                # Removed keyword since it's not needed for custom models
            }
        }
        
        # CRITICAL FIX: Use wakeword_models array instead of custom_model_path for better OpenWakeWord compatibility
        if 'file_paths' in model_config:
            model_paths = []
            # Check for ONNX file first, then TFLite (ONNX preferred)
            if 'onnx' in model_config['file_paths']:
                model_paths.append(model_config['file_paths']['onnx'])
                logger.info(f"🔧 Using ONNX model for {model_name}: {model_config['file_paths']['onnx']}")
            elif 'tflite' in model_config['file_paths']:
                model_paths.append(model_config['file_paths']['tflite'])
                logger.info(f"🔧 Using TFLite model for {model_name}: {model_config['file_paths']['tflite']}")
            
            # Use wakeword_models array for OpenWakeWord engine
            if model_paths:
                detector_config['wake_word']['wakeword_models'] = model_paths
        
        logger.info(f"🔧 Detector config for {model_name}:")
        logger.info(f"   Engine: {detector_config['wake_word']['engine']}")
        logger.info(f"   Sensitivity: {detector_config['wake_word']['sensitivity']:.3f}")
        logger.info(f"   Threshold: {detector_config['wake_word']['threshold']:.3f}")
        logger.info(f"   Wakeword models: {detector_config['wake_word'].get('wakeword_models', [])}")
        logger.info(f"   Audio section: {list(detector_config['audio'].keys())}")
        logger.info(f"   Detection section: {list(detector_config['detection'].keys())}")
        logger.info(f"   Buffer section: {list(detector_config['buffer'].keys())}")
        logger.info(f"   Volume monitoring section: {list(detector_config['volume_monitoring'].keys())}")
        
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
        
        has_changed = (abs(old_sensitivity - new_sensitivity) > 0.001 or 
                      abs(old_threshold - new_threshold) > 0.001)
        
        if has_changed:
            # Update the internal cache with the new config
            self.model_configs[model_name] = new_config
            logger.info(f"🔄 Updated model config cache for {model_name}:")
            logger.info(f"   Old threshold: {old_threshold:.3f} → New threshold: {new_threshold:.3f}")
            logger.info(f"   Old sensitivity: {old_sensitivity:.3f} → New sensitivity: {new_sensitivity:.3f}")
        
        return has_changed
    
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
        
        # Rebuild model configs from settings manager
        new_config = self.settings_manager.get_model_config(model_name)
        if new_config:
            self.model_configs[model_name] = new_config
            logger.info(f"🔄 Updated model config for {model_name}:")
            logger.info(f"   Threshold: {new_config['threshold']:.3f}")
            logger.info(f"   Sensitivity: {new_config['sensitivity']:.3f}")
        
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
        
        # ENHANCED DEBUGGING: Track confidence for troubleshooting
        if not hasattr(self, '_confidence_check_counter'):
            self._confidence_check_counter = 0
        self._confidence_check_counter += 1
        
        # Process through all active detectors
        for model_name, detector in self.active_detectors.items():
            try:
                if detector.is_ready():
                    # Process audio directly with numpy array (same as working implementation)
                    detection_result = detector.process_audio(audio_data)
                    
                    # Log confidence every 100 chunks for debugging
                    if self._confidence_check_counter % 100 == 0:
                        confidence = 0.0
                        all_confidences = {}
                        
                        # Try to get confidence from detector engine
                        if hasattr(detector.engine, 'get_latest_confidence'):
                            confidence = detector.engine.get_latest_confidence()
                        
                        # CRITICAL DEBUG: Check if multiple models are loaded in the engine
                        if hasattr(detector.engine, 'model') and detector.engine.model:
                            openwakeword_model = detector.engine.model
                            
                            # Check for prediction_buffer (contains all model predictions)
                            if hasattr(openwakeword_model, 'prediction_buffer'):
                                logger.info(f"🔍 FULL MODEL DEBUG for {model_name}:")
                                logger.info(f"   Prediction buffer keys: {list(openwakeword_model.prediction_buffer.keys())}")
                                
                                for model_key, scores in openwakeword_model.prediction_buffer.items():
                                    if scores and len(scores) > 0:
                                        latest_score = scores[-1]
                                        all_confidences[model_key] = latest_score
                                        logger.info(f"   Model '{model_key}': {latest_score:.6f}")
                                    else:
                                        logger.info(f"   Model '{model_key}': No scores")
                            
                            # Check which models are actually loaded
                            if hasattr(openwakeword_model, 'models'):
                                logger.info(f"   Loaded model objects: {list(openwakeword_model.models.keys())}")
                            
                            # Check model paths
                            if hasattr(openwakeword_model, 'model_paths'):
                                logger.info(f"   Model paths: {openwakeword_model.model_paths}")
                        
                        logger.info(f"🔍 CONFIDENCE SUMMARY - {model_name}: primary={confidence:.6f}, all_models={all_confidences}")
                    
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
        
        # Get actual confidence from detector if available
        confidence = 0.0
        if hasattr(detector.engine, 'get_latest_confidence'):
            confidence = detector.engine.get_latest_confidence()
        
        # Log detection details
        logger.info(f"🎯 DETECTION #{self.detection_count} - Model: {model_name}")
        logger.info(f"   Wake word: {detector.get_wake_word_name()}")
        logger.info(f"   Confidence: {confidence:.6f}")
        logger.info(f"   Audio RMS: {np.sqrt(np.mean(audio_data.astype(np.float32)**2)):.4f}")
        
        # Create detection event for web interface
        try:
            detection_event = {
                'model_name': model_name,
                'wake_word': detector.get_wake_word_name(),
                'confidence': confidence,  # FIXED: Get actual confidence from detector
                'timestamp': int(time.time() * 1000),  # Convert to milliseconds
                'detection_count': self.detection_count,
                'audio_rms': float(np.sqrt(np.mean(audio_data.astype(np.float32)**2)))
            }
            
            # Write detection to file for web API
            import json
            detection_file = "/tmp/recent_detections.json"
            with open(detection_file, 'w') as f:
                json.dump([detection_event], f)  # Store as array for compatibility
            
            logger.info(f"📁 Created detection file: {detection_file}")
            logger.info(f"   Detection event: {detection_event}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create detection file: {e}")
        
        # Update shared memory (but don't set is_listening=True for detections)
        try:
            # Keep the current listening state, just update the detection info
            shared_memory_ipc.update_activation_state(False, model_name, 0.0)  # Don't change listening state
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
        if not config:
            raise ValueError(f"❌ No configuration found for model: {model_name}")
        return config['sensitivity']
    
    def get_model_threshold(self, model_name: str) -> float:
        """Get threshold for a specific model."""
        config = self.get_model_config(model_name)
        if not config:
            raise ValueError(f"❌ No configuration found for model: {model_name}")
        return config['threshold']
    
    def get_model_api_url(self, model_name: str) -> str:
        """Get API URL for a specific model."""
        config = self.get_model_config(model_name)
        if not config:
            raise ValueError(f"❌ No configuration found for model: {model_name}")
        return config['api_url']
    
    def get_model_active(self, model_name: str) -> bool:
        """Get active state for a specific model."""
        config = self.get_model_config(model_name)
        if not config:
            raise ValueError(f"❌ No configuration found for model: {model_name}")
        return config['active']
    
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
        # Return the first active model name as the keyword
        active_models = self.get_active_models()
        if active_models:
            return active_models[0]
        return "Unknown"
    
    def get_detection_count(self) -> int:
        """Get the total number of detections."""
        return self.detection_count
    
    def get_active_detectors(self) -> Dict[str, WakeWordDetector]:
        """Get the currently active detectors."""
        return self.active_detectors.copy()
    
    def validate_audio_format_for_engines(self, sample_rate: int, channels: int, frame_length: int) -> bool:
        """
        Validate audio format compatibility with all active wake word engines.
        This method is called by the audio pipeline to ensure compatibility.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels
            frame_length: Number of samples per frame
            
        Returns:
            bool: True if all active engines are compatible with the audio format
        """
        if not self.active_detectors:
            logger.warning("⚠️ No active detectors to validate audio format against")
            return True
        
        logger.info(f"🔧 Validating audio format with {len(self.active_detectors)} active engines:")
        logger.info(f"   Sample rate: {sample_rate}Hz")
        logger.info(f"   Channels: {channels}")
        logger.info(f"   Frame length: {frame_length} samples")
        
        for model_name, detector in self.active_detectors.items():
            if hasattr(detector.engine, 'validate_audio_format'):
                if not detector.engine.validate_audio_format(sample_rate, channels, frame_length):
                    logger.error(f"❌ Audio format validation failed for engine: {model_name}")
                    return False
                logger.info(f"   ✅ {model_name}: Audio format compatible")
            else:
                logger.warning(f"   ⚠️ {model_name}: No audio format validation available")
        
        logger.info("✅ All active engines are compatible with the audio format")
        return True
    
    def print_configuration_summary(self):
        """Print a summary of the current configuration."""
        logger.info("📊 Wake Word Monitor Configuration Summary:")
        logger.info("=" * 50)
        
        # Global settings
        logger.info("🌐 Global Settings:")
        logger.info(f"   Engine: {self.get_engine_type()}")
        logger.info(f"   Active Models: {', '.join(self.get_active_models())}")
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
            if config:
                status = "✅ ACTIVE" if config['active'] else "❌ INACTIVE"
                logger.info(f"   📁 {model_name} ({status}):")
                logger.info(f"      Sensitivity: {config['sensitivity']:.3f}")
                logger.info(f"      Threshold: {config['threshold']:.3f}")
                logger.info(f"      API URL: {config['api_url']}")
                logger.info(f"      Files: {list(config['file_paths'].keys())}")
            else:
                logger.warning(f"   ⚠️ {model_name}: No configuration available")
        
        logger.info("=" * 50)
    
    def start_monitoring(self, stream_callback: Callable[[np.ndarray], bool] = None) -> bool:
        """
        Start the main audio monitoring loop.
        
        Args:
            stream_callback: Optional callback for audio stream processing
            
        Returns:
            bool: True if monitoring started successfully
        """
        try:
            import pyaudio
            
            if not self.active_detectors:
                logger.error("❌ No active detectors loaded, cannot start monitoring")
                return False
            
            # Get audio settings
            audio_settings = self.settings_manager.get('audio', {})
            sample_rate = audio_settings.get('sample_rate', 16000)
            channels = audio_settings.get('channels', 1)
            chunk_size = audio_settings.get('chunk_size', 1280)
            device_index = audio_settings.get('device_index', None)
            
            logger.info(f"🎤 Starting audio monitoring:")
            logger.info(f"   Sample rate: {sample_rate}Hz")
            logger.info(f"   Channels: {channels}")
            logger.info(f"   Chunk size: {chunk_size} samples")
            logger.info(f"   Device index: {device_index}")
            
            # Validate audio format with all active engines
            if not self.validate_audio_format_for_engines(sample_rate, channels, chunk_size):
                logger.error("❌ Audio format validation failed")
                return False
            
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            
            # Open audio stream
            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk_size,
                stream_callback=None  # Use blocking read for better control
            )
            
            logger.info("✅ Audio stream opened successfully")
            logger.info("🎯 Starting wake word detection loop...")
            logger.info("Press Ctrl+C to stop")
            
            chunk_count = 0
            try:
                while True:
                    # Read audio chunk
                    audio_data = stream.read(chunk_size, exception_on_overflow=False)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    
                    chunk_count += 1
                    
                    # Process audio through wake word detection
                    detection_result = self.process_audio(audio_np)
                    
                    # Call stream callback if provided
                    if stream_callback:
                        try:
                            stream_callback(audio_np)
                        except Exception as e:
                            logger.warning(f"⚠️ Stream callback error: {e}")
                    
                    # Log progress every 1000 chunks
                    if chunk_count % 1000 == 0:
                        logger.info(f"📊 Processed {chunk_count} chunks ({chunk_count * chunk_size / sample_rate:.1f}s)")
                        
            except KeyboardInterrupt:
                logger.info("🛑 Monitoring stopped by user")
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                return False
            finally:
                # Clean up audio resources
                try:
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    logger.info("🧹 Audio resources cleaned up")
                except Exception as e:
                    logger.warning(f"⚠️ Error cleaning up audio: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring: {e}")
            return False


def create_wake_word_monitor_new() -> WakeWordMonitor_new:
    """Factory function to create a new wake word monitor instance."""
    return WakeWordMonitor_new()


def run_wake_word_monitor_new() -> int:
    """
    Run the new wake word monitor as a standalone application.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        logger.info("🚀 Starting Wake Word Monitor New...")
        
        # Create and initialize monitor
        monitor = create_wake_word_monitor_new()
        
        # Print configuration summary
        monitor.print_configuration_summary()
        
        # Start monitoring
        if monitor.start_monitoring():
            logger.info("✅ Wake word monitoring completed successfully")
            return 0
        else:
            logger.error("❌ Wake word monitoring failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Fatal error in wake word monitor: {e}")
        return 1


if __name__ == "__main__":
    # Run the new monitor as a standalone application
    exit_code = run_wake_word_monitor_new()
    exit(exit_code) 