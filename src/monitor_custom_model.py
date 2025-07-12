#!/usr/bin/env python3
"""
Monitor Custom Wake Word Models
Handles monitoring of custom wake word models (Hey Computer, etc.)
"""

import logging
import time
import numpy as np
import json
import os
from base_monitor import BaseWakeWordMonitor
from shared_memory_ipc import shared_memory_ipc
from settings_manager import get_settings_manager

logger = logging.getLogger(__name__)


class CustomModelMonitor(BaseWakeWordMonitor):
    """
    Monitor for custom wake word models with advanced features.
    """
    
    def __init__(self, config: dict, usb_device, audio_manager, custom_model_path: str = None):
        """
        Initialize the custom model monitor.
        
        Args:
            config: Configuration dictionary
            usb_device: USB audio device to use
            audio_manager: Initialized audio manager
            custom_model_path: Path to custom model file (optional)
        """
        super().__init__(config, usb_device, audio_manager)
        self.custom_model_path = custom_model_path
        self.settings_manager = get_settings_manager()
        
        # Detection timing controls
        self.last_detection_time = 0
        self.last_detection_chunk = 0
        self.detection_cooldown_seconds = 1.5
        self.debounce_seconds = 0.2
        self.detection_debounce_chunks = 0
        
        # Volume filtering
        self.rms_filter_value = 50
        self.silence_threshold = 0.5
        self.volume_window_size = 10
        self.volume_history = []
        
        # Confidence logging
        self.confidence_log_count = 0
        
        # Register settings watcher
        self.settings_manager.add_watcher(self._on_settings_changed)
    
    def _initialize_detector(self) -> bool:
        """Initialize the wake word detector for custom models."""
        # Get selected model and its per-model settings
        model_name = self.settings_manager.get("wake_word.model", "Hay--compUta_v_lrg")
        sensitivity = self.settings_manager.get_model_sensitivity(model_name, 0.8)
        threshold = self.settings_manager.get_model_threshold(model_name, 0.3)
        
        logger.info(f"🔧 DEBUG: Model: {model_name}, Sensitivity: {sensitivity:.6f}, Threshold: {threshold:.6f}")

        # Create custom config for custom models
        custom_config = self.config.copy()
        if self.custom_model_path:
            custom_config['wake_word']['custom_model_path'] = self.custom_model_path
            logger.info(f"📁 Using custom model: {self.custom_model_path}")
        
        # Inject per-model sensitivity and threshold
        if 'wake_word' not in custom_config:
            custom_config['wake_word'] = {}
        custom_config['wake_word']['sensitivity'] = sensitivity
        custom_config['wake_word']['threshold'] = threshold
        
        logger.info(f"🔧 DEBUG: Config sensitivity: {custom_config['wake_word'].get('sensitivity', 'NOT SET')}")
        logger.info(f"🔧 DEBUG: Config threshold: {custom_config['wake_word'].get('threshold', 'NOT SET')}")
        
        # Initialize timing controls
        self._update_timing_controls()
        
        return self.wake_detector.initialize(custom_config)
    
    def _update_timing_controls(self):
        """Update timing controls from settings."""
        self.detection_cooldown_seconds = self.settings_manager.get("wake_word.cooldown", 1.5)
        self.debounce_seconds = self.settings_manager.get("wake_word.debounce", 0.2)
        self.detection_debounce_chunks = int(self.debounce_seconds * self.wake_detector.get_sample_rate() / self.wake_detector.get_frame_length())
        self.rms_filter_value = self.settings_manager.get("volume_monitoring.rms_filter", 50)
        
        logger.info(f"🔧 Timing controls updated: Cooldown={self.detection_cooldown_seconds}s, Debounce={self.debounce_seconds}s, RMS Filter={self.rms_filter_value}")
    
    def _on_settings_changed(self, new_settings):
        """Handle settings changes."""
        logger.info("🔧 Settings changed, updating timing controls...")
        self._update_timing_controls()
    
    def _should_allow_detection(self) -> bool:
        """Check if detection should be allowed based on timing controls."""
        current_time = time.time()
        current_chunk = self.chunk_count
        
        # Check cooldown period
        time_since_last = current_time - self.last_detection_time
        if time_since_last < self.detection_cooldown_seconds:
            logger.debug(f"🛡️ Cooldown active: {self.detection_cooldown_seconds - time_since_last:.2f}s remaining")
            return False
        
        # Check debounce period
        chunks_since_last = current_chunk - self.last_detection_chunk
        if chunks_since_last < self.detection_debounce_chunks:
            logger.debug(f"🛡️ Debounce active: {self.detection_debounce_chunks - chunks_since_last} chunks remaining")
            return False
        
        # Check RMS filter
        if hasattr(self, 'volume_history') and len(self.volume_history) > 0:
            avg_volume = np.mean(self.volume_history)
            if avg_volume < self.rms_filter_value:
                logger.debug(f"🛡️ RMS filter active: avg_volume={avg_volume:.2f} < {self.rms_filter_value}")
                return False
        
        return True
    
    def _process_audio_chunk(self, audio_data: np.ndarray) -> bool:
        """Process a single audio chunk for wake word detection with custom model features."""
        # DEBUG: Log every 50 chunks to track progress
        if self.chunk_count % 50 == 0:
            logger.info(f"🔍 DEBUG: Processing chunk {self.chunk_count} - Audio data length: {len(audio_data)}")
        
        # Update volume history for RMS filtering
        rms_level = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
        self.volume_history.append(rms_level)
        if len(self.volume_history) > self.volume_window_size:
            self.volume_history.pop(0)
        
        # Update shared memory with RMS data for web interface
        try:
            shared_memory_ipc.update_audio_state(rms_level)
            # DEBUG: Log shared memory updates every 100 chunks
            if self.chunk_count % 100 == 0:
                logger.info(f"🔗 DEBUG: Updated shared memory - RMS: {rms_level:.4f}, Chunk: {self.chunk_count}")
        except Exception as e:
            logger.warning(f"⚠️ Could not update audio state: {e}")
        
        # Log confidence scores periodically
        self.confidence_log_count += 1
        if self.confidence_log_count % 50 == 0:
            self._log_confidence_scores(audio_data)
        
        # Process audio for wake-word detection
        detection_result = self.wake_detector.process_audio(audio_data)
        
        if detection_result:
            return self._handle_detection(audio_data)
        
        return False
    
    def _log_confidence_scores(self, audio_data: np.ndarray):
        """Log confidence scores for debugging."""
        if hasattr(self.wake_detector, 'engine') and hasattr(self.wake_detector.engine, 'get_latest_confidence'):
            confidence = self.wake_detector.engine.get_latest_confidence()
            rms_level = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
            logger.debug(f"🔍 Confidence: {confidence:.6f}, RMS: {rms_level:.4f}, Chunk: {self.chunk_count}")
    
    def _handle_detection(self, audio_data: np.ndarray) -> bool:
        """Handle a wake word detection with custom model features."""
        # Check if detection should be allowed (timing controls)
        if not self._should_allow_detection():
            return False
        
        # Call parent detection handling
        result = super()._handle_detection(audio_data)
        
        if result:
            # Update timing controls AFTER successful detection
            current_time = time.time()
            self.last_detection_time = current_time
            self.last_detection_chunk = self.chunk_count
            
            # Log custom model specific details
            logger.info(f"⏱️ Time since last detection: {current_time - self.last_detection_time:.2f}s")
            logger.info(f"📦 Chunks since last detection: {self.chunk_count - self.last_detection_chunk}")
            
            # Log detection details with enhanced parameters
            logger.info(f"🎯 Detection details:")
            logger.info(f"   Model: {self.wake_detector.get_wake_word_name()}")
            logger.info(f"   Sensitivity: {self.settings_manager.get_model_sensitivity(self.settings_manager.get('wake_word.model', 'Hay--compUta_v_lrg'), 0.4):.6f}")
            logger.info(f"   Confidence: {self.wake_detector.engine.get_latest_confidence():.6f}" if hasattr(self.wake_detector, 'engine') else "   Confidence: N/A")
            logger.info(f"   Chunk number: {self.chunk_count}")
            logger.info(f"   Time since last detection: {current_time - self.last_detection_time:.2f}s")
            logger.info(f"   Chunks since last detection: {self.chunk_count - self.last_detection_chunk}")
            logger.info(f"   Cooldown setting: {self.detection_cooldown_seconds}s")
            logger.info(f"   Debounce setting: {self.debounce_seconds}s ({self.detection_debounce_chunks} chunks)")
            logger.info(f"   RMS filter setting: {self.rms_filter_value}")
            logger.info(f"   Audio RMS level: {np.sqrt(np.mean(audio_data.astype(np.float32)**2)):.4f}")
            logger.info(f"   Audio max level: {np.max(np.abs(audio_data))}")
            logger.info(f"   Buffer status: {self.audio_buffer.get_buffer_status()}")
            
            # Don't update activation state here - let the system stay in normal monitoring mode
            # The activation state should only be set when actually listening for commands
            
            logger.info(f"🛡️ Cooldown active for {self.detection_cooldown_seconds}s...")
        
        return result
    
    def _update_activation_state(self, is_listening: bool):
        """Update activation state in shared memory."""
        try:
            model_name = self.wake_detector.get_wake_word_name()
            confidence = self.wake_detector.engine.get_latest_confidence() if hasattr(self.wake_detector, 'engine') else 0.0
            
            # Update shared memory with activation state
            shared_memory_ipc.update_activation_state(is_listening, model_name, confidence)
            
            logger.info(f"🌐 ACTIVATION: Updated shared memory - Listening: {is_listening}, Model: {model_name}, Confidence: {confidence:.3f}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not update activation state: {e}")
    
    def _get_detection_log_file(self) -> str:
        """Get the path to the custom detection log file."""
        return "/tmp/custom_detections.log"
    
    def _get_audio_clip_filename(self) -> str:
        """Get the filename for custom audio clips."""
        return f"/tmp/custom_wake_word_detection_{self.detection_count}.wav"
    
    def _log_progress(self):
        """Log progress with custom model specific information."""
        if self.chunk_count % 1000 == 0:
            buffer_status = self.audio_buffer.get_buffer_status()
            time_since_last = time.time() - self.last_detection_time
            logger.info(f"📊 Progress Report:")
            logger.info(f"   Processed chunks: {self.chunk_count}")
            logger.info(f"   Detections: {self.detection_count}")
            logger.info(f"   Time since last detection: {time_since_last:.1f}s")
            logger.info(f"   Cooldown remaining: {max(0, self.detection_cooldown_seconds - time_since_last):.1f}s")
            logger.info(f"   Buffer pre-roll samples: {buffer_status['preroll_samples']}")
            logger.info(f"   Buffer post-roll samples: {buffer_status['postroll_samples']}")
            logger.info(f"   Runtime: {self.chunk_count * self.wake_detector.get_frame_length() / self.wake_detector.get_sample_rate():.1f}s")
    
    def run(self) -> int:
        """Run the monitoring loop with custom model specific setup."""
        logger.info("🎯 Starting continuous custom model monitoring...")
        logger.info("📊 Debug info will be logged every 100 chunks")
        logger.info("🔍 Custom model confidence will be logged every 50 chunks")
        logger.info(f"🛡️ Detection cooldown: {self.detection_cooldown_seconds}s, Debounce: {self.detection_debounce_chunks} chunks")
        logger.info(f"🔧 Using settings values - Cooldown: {self.detection_cooldown_seconds}s, Debounce: {self.debounce_seconds}s, RMS Filter: {self.rms_filter_value}")
        
        # Initialize activation state to not listening
        self._update_activation_state(False)
        
        # DEBUG: Log that we're about to start the main loop
        logger.info("🔍 DEBUG: About to start main monitoring loop...")
        
        return super().run()


def monitor_custom_models(config: dict, usb_device, audio_manager, custom_model_path: str = None) -> int:
    """
    Monitor custom wake word models.
    
    Args:
        config: Configuration dictionary
        usb_device: USB audio device to use
        audio_manager: Initialized audio manager
        custom_model_path: Path to custom model file (optional)
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("🎯 Starting custom model monitoring...")
    
    # DEBUG: Log initialization steps
    logger.info("🔍 DEBUG: Creating CustomModelMonitor instance...")
    
    # Create and initialize the monitor
    monitor = CustomModelMonitor(config, usb_device, audio_manager, custom_model_path)
    
    logger.info("🔍 DEBUG: About to initialize monitor...")
    if not monitor.initialize():
        logger.error("❌ Failed to initialize custom model monitor")
        return 1
    
    logger.info("🔍 DEBUG: Monitor initialized successfully, about to start run()...")
    
    # Run the monitoring loop
    return monitor.run()


def test_custom_model_with_speech(config: dict, usb_device, audio_manager, 
                                custom_model_path: str, duration: int = 30) -> int:
    """
    Test a custom model with speech input.
    
    Args:
        config: Configuration dictionary
        usb_device: USB audio device to use
        audio_manager: Initialized audio manager
        custom_model_path: Path to custom model file
        duration: Test duration in seconds
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info(f"🧪 Testing custom model with speech: {custom_model_path}")
    logger.info(f"⏱️ Test duration: {duration} seconds")
    
    # Create and initialize the monitor
    monitor = CustomModelMonitor(config, usb_device, audio_manager, custom_model_path)
    if not monitor.initialize():
        logger.error("❌ Failed to initialize custom model monitor for testing")
        return 1
    
    # Run test for specified duration
    start_time = time.time()
    detection_count = 0
    
    try:
        while time.time() - start_time < duration:
            try:
                # Read audio chunk
                audio_chunk = monitor.stream.read(monitor.wake_detector.get_frame_length(), exception_on_overflow=False)
                monitor.chunk_count += 1
                
                # Convert to numpy array
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # Process audio for detection
                if monitor._process_audio_chunk(audio_data):
                    detection_count += 1
                    logger.info(f"🎯 Detection #{detection_count} during test!")
                
            except Exception as e:
                logger.error(f"❌ Error during test: {e}")
                continue
        
        logger.info(f"✅ Test completed: {detection_count} detections in {duration} seconds")
        return 0 if detection_count > 0 else 1
        
    except KeyboardInterrupt:
        logger.info("🛑 Test stopped by user")
        return 0
    finally:
        monitor._cleanup()


if __name__ == "__main__":
    # This file is designed to be imported and used by main.py
    # It can also be run independently for testing
    logger.info("🎯 Custom model monitor module loaded") 