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
from rms_monitor import rms_monitor
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
        # Get selected model and its sensitivity
        model_name = self.settings_manager.get("wake_word.model", "Hay--compUta_v_lrg")
        sensitivity = self.settings_manager.get_model_sensitivity(model_name, 0.4)
        
        logger.info(f"🔧 DEBUG: Model: {model_name}, Sensitivity: {sensitivity:.6f}")

        # Create custom config for custom models
        custom_config = self.config.copy()
        if self.custom_model_path:
            custom_config['wake_word']['custom_model_path'] = self.custom_model_path
            logger.info(f"📁 Using custom model: {self.custom_model_path}")
        
        # Inject per-model sensitivity
        if 'wake_word' not in custom_config:
            custom_config['wake_word'] = {}
        custom_config['wake_word']['sensitivity'] = sensitivity
        
        logger.info(f"🔧 DEBUG: Config sensitivity: {custom_config['wake_word'].get('sensitivity', 'NOT SET')}")
        
        # Initialize timing controls
        self._update_timing_controls()
        
        return self.wake_detector.initialize(custom_config)
    
    def _update_timing_controls(self):
        """Update timing controls from settings."""
        self.detection_cooldown_seconds = self.settings_manager.get('wake_word.cooldown', 1.5)
        self.debounce_seconds = self.settings_manager.get('wake_word.debounce', 0.2)
        self.detection_debounce_chunks = int((self.wake_detector.get_sample_rate() * self.debounce_seconds) / self.wake_detector.get_frame_length())
        
        # Update RMS filter
        self.rms_filter_value = self.settings_manager.get('detection.rms_filter', 50)
        self.silence_threshold = (self.rms_filter_value / 100.0) * 0.99 + 0.01
    
    def _on_settings_changed(self, new_settings):
        """Handle settings changes."""
        # Update detection parameters from new settings
        self.detection_cooldown_seconds = new_settings.get('wake_word', {}).get('cooldown', 1.5)
        self.debounce_seconds = new_settings.get('wake_word', {}).get('debounce', 0.2)
        self.detection_debounce_chunks = int((self.wake_detector.get_sample_rate() * self.debounce_seconds) / self.wake_detector.get_frame_length())
        
        self.rms_filter_value = new_settings.get('detection', {}).get('rms_filter', 50)
        self.silence_threshold = (self.rms_filter_value / 100.0) * 0.99 + 0.01
        
        # Check if sensitivity changed for the current model
        current_model = new_settings.get('wake_word', {}).get('model', 'Hay--compUta_v_lrg')
        sensitivities = new_settings.get('wake_word', {}).get('sensitivities', {})
        if current_model in sensitivities:
            new_sensitivity = sensitivities[current_model]
            logger.info(f"🔄 Sensitivity changed for active model {current_model}: {new_sensitivity:.6f}")
            if self.wake_detector.update_sensitivity(new_sensitivity):
                logger.info(f"✅ Sensitivity updated to {new_sensitivity:.6f}")
            else:
                logger.warning(f"⚠️ Failed to update sensitivity to {new_sensitivity:.6f}")
        
        logger.info(f"🔄 Settings updated - Cooldown: {self.detection_cooldown_seconds}s, Debounce: {self.debounce_seconds}s, RMS Filter: {self.rms_filter_value}")
    
    def _should_allow_detection(self) -> bool:
        """Check if detection should be allowed based on timing controls."""
        current_time = time.time()
        
        # Check cooldown
        if self.last_detection_time == 0:
            time_since_last_detection = -1
        else:
            time_since_last_detection = current_time - self.last_detection_time
            if time_since_last_detection >= self.detection_cooldown_seconds:
                time_since_last_detection = -1
        
        # Check debounce
        chunks_since_last_detection = self.chunk_count - self.last_detection_chunk
        
        # Check if we're in post-roll capture
        not_postrolling = not self.audio_buffer.is_capturing_postroll()
        
        # Determine if detection should be allowed
        cooldown_ok = (time_since_last_detection == -1) or (time_since_last_detection >= self.detection_cooldown_seconds)
        debounce_ok = chunks_since_last_detection >= self.detection_debounce_chunks
        should_allow = cooldown_ok and debounce_ok and not_postrolling
        
        # Debug logging
        if not should_allow:
            logger.info(f"🛡️ Detection blocked by timing controls:")
            logger.info(f"   Model: {self.wake_detector.get_wake_word_name()}")
            logger.info(f"   Time since last detection: {time_since_last_detection:.2f}s")
            logger.info(f"   Chunks since last detection: {chunks_since_last_detection}")
            logger.info(f"   Cooldown setting: {self.detection_cooldown_seconds}s")
            logger.info(f"   Debounce setting: {self.debounce_seconds}s ({self.detection_debounce_chunks} chunks)")
            logger.info(f"   Cooldown OK: {cooldown_ok}")
            logger.info(f"   Debounce OK: {debounce_ok}")
            logger.info(f"   Not postrolling: {not_postrolling}")
            if not cooldown_ok:
                logger.info(f"   Reason: Cooldown period active (need {self.detection_cooldown_seconds - time_since_last_detection:.1f}s more)")
            elif not debounce_ok:
                logger.info(f"   Reason: Debounce period active (need {self.detection_debounce_chunks - chunks_since_last_detection} more chunks)")
            elif not not_postrolling:
                logger.info(f"   Reason: Post-roll capture in progress")
        
        return should_allow
    
    def _process_audio_chunk(self, audio_data: np.ndarray) -> bool:
        """Process audio chunk with custom model specific logic."""
        # Update RMS monitor for web interface
        rms_level = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
        rms_monitor.update_rms(rms_level)
        
        # Volume filtering - skip silent audio
        self.volume_history.append(rms_level)
        if len(self.volume_history) > self.volume_window_size:
            self.volume_history.pop(0)
        
        avg_volume = np.mean(self.volume_history) if self.volume_history else 0
        
        # Skip processing if audio is too quiet
        if avg_volume < self.silence_threshold:
            if self.chunk_count % 100 == 0:
                logger.debug(f"🔇 Skipping silent audio: avg_volume={avg_volume:.3f} < threshold={self.silence_threshold}")
            return False
        
        # Enhanced custom model debugging - log confidence scores periodically
        if self.chunk_count % 50 == 0:
            self._log_confidence_scores(audio_data)
        
        # Call parent method for detection processing
        return super()._process_audio_chunk(audio_data)
    
    def _log_confidence_scores(self, audio_data: np.ndarray):
        """Log confidence scores for debugging."""
        self.confidence_log_count += 1
        try:
            # Try to get detailed confidence scores for debugging
            if hasattr(self.wake_detector, 'engine') and self.wake_detector.engine:
                # Convert audio to float for model prediction
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                # Get raw predictions from the model
                predictions = self.wake_detector.engine.model.predict(audio_float)
                
                if isinstance(predictions, dict):
                    # Log all prediction values
                    for model_name, confidence in predictions.items():
                        logger.info(f"🔍 Custom Model Confidence #{self.confidence_log_count}: {model_name} = {confidence:.6f}")
                else:
                    logger.info(f"🔍 Custom Model Confidence #{self.confidence_log_count}: {predictions}")
        except Exception as e:
            logger.debug(f"⚠️ Could not get detailed confidence scores: {e}")
        
        # Log basic confidence scores
        confidence = self.wake_detector.engine.get_latest_confidence()
        audio_rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
        audio_max = np.max(np.abs(audio_data))
        logger.info(f"🔍 Custom Model Confidence #{self.chunk_count//50}: {self.wake_detector.get_wake_word_name()} = {confidence:.6f}")
        logger.info(f"📊 Audio levels - RMS: {audio_rms:.4f}, Max: {audio_max}")
    
    def _handle_detection(self, audio_data: np.ndarray) -> bool:
        """Handle detection with custom model specific logic."""
        # Update timing controls
        current_time = time.time()
        self.last_detection_time = current_time
        self.last_detection_chunk = self.chunk_count
        
        # Call parent method for standard detection handling
        result = super()._handle_detection(audio_data)
        
        if result:
            # Log custom model specific details
            logger.info(f"⏱️ Time since last detection: {current_time - self.last_detection_time:.2f}s")
            logger.info(f"📦 Chunks since last detection: {self.chunk_count - self.last_detection_chunk}")
            
            # Log detection details with enhanced parameters
            logger.info(f"�� Detection details:")
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
            
            # Notify web interface of detection
            self._notify_web_interface()
            
            logger.info(f"🛡️ Cooldown active for {self.detection_cooldown_seconds}s...")
        
        return result
    
    def _notify_web_interface(self):
        """Notify web interface of detection."""
        logger.info("🌐 DEBUG: About to write detection to file...")
        try:
            # Record detection to a file for web interface to read
            detection_data = {
                'model_name': str(self.wake_detector.get_wake_word_name()),
                'confidence': float(self.wake_detector.engine.get_latest_confidence()) if hasattr(self.wake_detector, 'engine') else 0.0,
                'timestamp': int(time.time() * 1000)  # milliseconds since epoch
            }
            logger.info(f"🌐 DEBUG: Detection data prepared: {detection_data}")
            
            # Write to detection log file
            detection_file = '/tmp/recent_detections.json'
            logger.info(f"🌐 DEBUG: Writing to file: {detection_file}")
            detections = []
            
            # Read existing detections if file exists
            if os.path.exists(detection_file):
                logger.info(f"🌐 DEBUG: Existing file found, reading...")
                try:
                    with open(detection_file, 'r') as f:
                        detections = json.load(f)
                    logger.info(f"🌐 DEBUG: Read {len(detections)} existing detections")
                except Exception as e:
                    logger.warning(f"🌐 DEBUG: Failed to read existing file: {e}")
                    detections = []
            else:
                logger.info(f"🌐 DEBUG: No existing file, starting fresh")
            
            # Add new detection
            detections.append(detection_data)
            logger.info(f"🌐 DEBUG: Added detection, total now: {len(detections)}")
            
            # Keep only last 50 detections
            if len(detections) > 50:
                detections = detections[-50:]
                logger.info(f"🌐 DEBUG: Trimmed to last 50 detections")
            
            # Write back to file
            logger.info(f"🌐 DEBUG: Writing {len(detections)} detections to file...")
            with open(detection_file, 'w') as f:
                json.dump(detections, f)
            logger.info(f"🌐 Detection recorded to file: {self.wake_detector.get_wake_word_name()}")
        except Exception as e:
            logger.warning(f"⚠️ Could not record detection to file: {e}")
    
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
    
    # Create and initialize the monitor
    monitor = CustomModelMonitor(config, usb_device, audio_manager, custom_model_path)
    if not monitor.initialize():
        logger.error("❌ Failed to initialize custom model monitor")
        return 1
    
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