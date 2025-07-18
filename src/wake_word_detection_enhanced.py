#!/usr/bin/env python3
"""
Enhanced OpenWakeWord script with TFLite optimization for Raspberry Pi.
Uses ModelManager for dynamic model loading and MetricsCollector for performance monitoring.
"""

import sys
import os
import argparse
import time
import wave
import json
import threading
from datetime import datetime
from pathlib import Path

# Use environment variable for unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

import logging
import numpy as np
from audio_utils import AudioManager

# Import new TFLite-optimized components
from hey_orac.models.manager import ModelManager
from hey_orac.config.manager import SettingsManager
from hey_orac.metrics.collector import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)


class EnhancedWakeWordDetector:
    """Enhanced wake word detector with TFLite optimization and hot-reload support."""
    
    def __init__(self, config_path: str = "/app/config/settings.json"):
        """Initialize the enhanced wake word detector."""
        self.config_path = config_path
        
        # Initialize components
        self.settings_manager = SettingsManager(config_path)
        self.model_manager = None
        self.metrics_collector = None
        self.audio_manager = None
        
        # Runtime state
        self.running = False
        self.hot_reload_thread = None
        self.stop_event = threading.Event()
        
        logger.info("🚀 Enhanced WakeWordDetector initialized")
    
    def initialize_components(self):
        """Initialize all components based on configuration."""
        logger.info("🔧 Initializing components...")
        
        # Get configuration
        with self.settings_manager.get_config() as config:
            models_dir = config.system.models_dir
            metrics_enabled = config.system.metrics_enabled
            hot_reload_enabled = config.system.hot_reload_enabled
        
        # Initialize ModelManager with TFLite optimization
        self.model_manager = ModelManager(
            models_dir=models_dir,
            default_framework="tflite"
        )
        
        # Initialize MetricsCollector
        if metrics_enabled:
            self.metrics_collector = MetricsCollector(collection_interval=1.0)
            logger.info("✅ MetricsCollector initialized")
        
        # Initialize AudioManager
        self.audio_manager = AudioManager()
        logger.info("✅ AudioManager initialized")
        
        # Register change callbacks
        self.settings_manager.register_change_callback(self._on_config_change)
        self.model_manager.register_change_callback(self._on_model_change)
        
        # Load initial models
        self._load_models_from_config()
        
        # Start hot-reload thread if enabled
        if hot_reload_enabled:
            self._start_hot_reload_thread()
        
        logger.info("✅ All components initialized successfully")
    
    def _load_models_from_config(self):
        """Load models based on current configuration."""
        logger.info("📂 Loading models from configuration...")
        
        with self.settings_manager.get_config() as config:
            enabled_models = [model for model in config.models if model.enabled]
            
            if not enabled_models:
                logger.warning("No enabled models found in configuration")
                return False
            
            # Prepare model paths
            model_paths = []
            for model in enabled_models:
                if os.path.exists(model.path):
                    model_paths.append(model.path)
                else:
                    # Assume it's a pre-trained model name
                    model_paths.append(model.name)
            
            # Load models with TFLite framework
            success = self.model_manager.load_model(
                model_paths=model_paths,
                framework="tflite"
            )
            
            if success:
                logger.info(f"✅ Loaded {len(model_paths)} models successfully")
                
                # Record model loading metrics
                if self.metrics_collector:
                    for model in enabled_models:
                        model_size = 0
                        if os.path.exists(model.path):
                            model_size = os.path.getsize(model.path)
                        
                        self.metrics_collector.record_model_load(
                            load_time=0.1,  # Placeholder - actual time tracked in ModelManager
                            model_path=model.path,
                            model_size=model_size,
                            model_format=model.framework
                        )
            else:
                logger.error("❌ Failed to load models")
                return False
        
        return True
    
    def _on_config_change(self, new_config):
        """Handle configuration changes."""
        logger.info("🔄 Configuration changed, reloading models...")
        self._load_models_from_config()
    
    def _on_model_change(self, new_model):
        """Handle model changes."""
        logger.info("🔄 Model changed, updating detection parameters...")
        # Could update detection thresholds or other parameters here
    
    def _start_hot_reload_thread(self):
        """Start background thread for hot-reload monitoring."""
        self.hot_reload_thread = threading.Thread(
            target=self._hot_reload_loop,
            daemon=True
        )
        self.hot_reload_thread.start()
        logger.info("🔄 Hot-reload monitoring started")
    
    def _hot_reload_loop(self):
        """Background loop for hot-reload monitoring."""
        with self.settings_manager.get_config() as config:
            reload_interval = config.system.hot_reload_interval
        
        while not self.stop_event.wait(reload_interval):
            try:
                # Check for configuration changes
                if self.settings_manager.reload_if_changed():
                    logger.info("🔄 Configuration reloaded due to file changes")
                
                # Check for model file changes
                if self.model_manager.auto_reload_if_changed():
                    logger.info("🔄 Models reloaded due to file changes")
                
            except Exception as e:
                logger.error(f"Error in hot-reload loop: {e}")
    
    def find_and_setup_audio(self):
        """Find USB microphone and set up audio stream."""
        logger.info("🎤 Setting up audio capture...")
        
        # Find USB microphone
        usb_mic = self.audio_manager.find_usb_microphone()
        if not usb_mic:
            logger.error("❌ No USB microphone found")
            raise RuntimeError("No USB microphone detected")
        
        logger.info(f"✅ Using USB microphone: {usb_mic.name} (index {usb_mic.index})")
        
        # Get audio configuration
        with self.settings_manager.get_config() as config:
            audio_config = config.audio
        
        # Start audio stream
        stream = self.audio_manager.start_stream(
            device_index=usb_mic.index,
            sample_rate=audio_config.sample_rate,
            channels=audio_config.channels,
            chunk_size=audio_config.chunk_size
        )
        
        if not stream:
            logger.error("❌ Failed to start audio stream")
            raise RuntimeError("Failed to start audio stream")
        
        logger.info("✅ Audio stream started successfully")
        return stream, audio_config
    
    def run_detection_loop(self, stream, audio_config):
        """Run the main wake word detection loop."""
        logger.info("🎯 Starting wake word detection loop...")
        
        chunk_count = 0
        detection_count = 0
        
        # Get detection thresholds from config
        with self.settings_manager.get_config() as config:
            model_thresholds = {model.name: model.threshold for model in config.models}
        
        while self.running:
            try:
                # Read audio chunk
                data = stream.read(audio_config.chunk_size, exception_on_overflow=False)
                if not data:
                    continue
                
                # Convert to numpy array and handle stereo/mono
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                if len(audio_array) > audio_config.chunk_size:
                    # Stereo to mono conversion
                    stereo_data = audio_array.reshape(-1, 2)
                    audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
                else:
                    audio_data = audio_array.astype(np.float32)
                
                # Calculate audio level for metrics
                audio_level = np.sqrt(np.mean(audio_data**2))
                
                # Perform inference with metrics tracking
                start_time = time.time()
                
                try:
                    with self.model_manager.get_model() as model:
                        prediction = model.predict(audio_data)
                    
                    inference_time = time.time() - start_time
                    
                    # Record metrics
                    if self.metrics_collector:
                        self.metrics_collector.record_inference_time(inference_time, "tflite")
                        self.metrics_collector.record_audio_metrics(
                            chunk_size=audio_config.chunk_size,
                            audio_level=audio_level
                        )
                    
                except Exception as e:
                    logger.error(f"Error during inference: {e}")
                    continue
                
                # Process predictions
                chunk_count += 1
                max_confidence = 0.0
                best_model = None
                
                for wake_word, score in prediction.items():
                    if score > max_confidence:
                        max_confidence = score
                        best_model = wake_word
                
                # Check for wake word detection
                threshold = model_thresholds.get(best_model, 0.3)
                
                if max_confidence >= threshold:
                    detection_count += 1
                    logger.info(f"🎯 WAKE WORD DETECTED! '{best_model}' with confidence {max_confidence:.6f}")
                    logger.info(f"   Threshold: {threshold:.3f}, Inference time: {inference_time:.4f}s")
                    
                    # Record detection metrics
                    if self.metrics_collector:
                        self.metrics_collector.record_wake_word_detection(best_model, max_confidence)
                
                # Periodic status logging
                if chunk_count % 100 == 0:
                    logger.info(f"📊 Processed {chunk_count} chunks, {detection_count} detections")
                    logger.info(f"   Audio level: {audio_level:.4f}, Inference time: {inference_time:.4f}s")
                    logger.info(f"   Best confidence: {best_model} = {max_confidence:.6f}")
                    
                    # Log metrics summary
                    if self.metrics_collector:
                        metrics = self.metrics_collector.get_metrics_summary()
                        logger.info(f"   Metrics: {metrics['tflite']['avg_inference_time']:.4f}s avg, "
                                  f"{metrics['system']['cpu_usage_percent']:.1f}% CPU")
                
                # Check for moderate confidence (debugging)
                elif max_confidence > 0.1:
                    logger.debug(f"🔍 Moderate confidence: {best_model} = {max_confidence:.6f}")
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                continue
    
    def run(self):
        """Run the enhanced wake word detector."""
        logger.info("🚀 Starting Enhanced Wake Word Detector with TFLite optimization")
        
        try:
            # Initialize all components
            self.initialize_components()
            
            # Set up audio
            stream, audio_config = self.find_and_setup_audio()
            
            # Start detection
            self.running = True
            self.run_detection_loop(stream, audio_config)
            
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        except Exception as e:
            logger.error(f"❌ Error during execution: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the detector and clean up."""
        logger.info("🛑 Stopping Enhanced Wake Word Detector...")
        
        self.running = False
        self.stop_event.set()
        
        # Wait for hot-reload thread to finish
        if self.hot_reload_thread:
            self.hot_reload_thread.join(timeout=1.0)
        
        # Clean up components
        if self.metrics_collector:
            self.metrics_collector.stop()
        
        if self.audio_manager:
            self.audio_manager.__del__()
        
        logger.info("✅ Enhanced Wake Word Detector stopped")
    
    def get_metrics(self):
        """Get current metrics."""
        if self.metrics_collector:
            return self.metrics_collector.get_metrics_summary()
        return {}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced OpenWakeWord with TFLite optimization')
    parser.add_argument('--config', default='/app/config/settings.json',
                       help='Path to configuration file')
    parser.add_argument('--models-dir', default='/app/models',
                       help='Directory containing custom models')
    parser.add_argument('--metrics-port', type=int, default=8000,
                       help='Port for metrics endpoint')
    parser.add_argument('--test-integration', action='store_true',
                       help='Run integration tests')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    if args.test_integration:
        # Run integration tests
        logger.info("🧪 Running TFLite integration tests...")
        from test_tflite_integration import TFLiteIntegrationTest
        
        test_runner = TFLiteIntegrationTest()
        success = test_runner.run_all_tests()
        
        return 0 if success else 1
    
    # Run the enhanced detector
    detector = EnhancedWakeWordDetector(config_path=args.config)
    
    try:
        detector.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        detector.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())