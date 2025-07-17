"""
Main application for Hey ORAC wake word detection.
"""

import logging
import time
import threading
from typing import Optional, Dict, Any
from pathlib import Path

from .audio.microphone import AudioCapture
from .audio.endpointing import SpeechEndpointer, EndpointConfig
from .models.wake_detector import WakeDetector


logger = logging.getLogger(__name__)


class HeyOracApplication:
    """
    Main application class for wake word detection.
    
    Coordinates audio capture, wake word detection, and speech capture.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the application.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration (for now, use defaults)
        self.config = self._load_config(config_path)
        
        # Components
        self.audio_capture = AudioCapture(
            sample_rate=16000,
            chunk_size=1280,
            ring_buffer_seconds=10.0
        )
        
        self.wake_detector = WakeDetector(
            models=['hey_jarvis'],  # Start with just hey_jarvis
            inference_framework='tflite'
        )
        
        self.endpointer = SpeechEndpointer(
            EndpointConfig(
                silence_threshold=0.01,
                silence_duration=0.3,
                max_duration=15.0,
                pre_roll=1.0
            )
        )
        
        # State
        self.is_running = False
        self.detection_thread: Optional[threading.Thread] = None
        
        # Metrics
        self.total_detections = 0
        self.total_audio_processed = 0
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        # TODO: Implement config loading from JSON
        return {
            'models': {
                'hey_jarvis': {
                    'enabled': True,
                    'threshold': 0.5
                }
            },
            'audio': {
                'sample_rate': 16000,
                'chunk_size': 1280
            }
        }
    
    def start(self) -> bool:
        """
        Start the application.
        
        Returns:
            True if started successfully
        """
        logger.info("Starting Hey ORAC application...")
        
        # Start audio capture
        if not self.audio_capture.start():
            logger.error("Failed to start audio capture")
            return False
        
        # Set model thresholds from config
        for model_name, model_config in self.config['models'].items():
            if model_config.get('enabled', False):
                threshold = model_config.get('threshold', 0.5)
                self.wake_detector.set_threshold(model_name, threshold)
        
        # Start detection thread
        self.is_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.start()
        
        logger.info("Application started successfully")
        return True
    
    def stop(self) -> None:
        """Stop the application."""
        logger.info("Stopping Hey ORAC application...")
        
        self.is_running = False
        
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
            self.detection_thread = None
        
        self.audio_capture.stop()
        
        logger.info("Application stopped")
    
    def _detection_loop(self) -> None:
        """Main detection loop."""
        logger.info("Detection loop started")
        
        chunk_count = 0
        last_log_time = time.time()
        
        while self.is_running:
            try:
                # Get audio chunk
                audio_chunk = self.audio_capture.get_audio_chunk()
                if audio_chunk is None:
                    time.sleep(0.01)
                    continue
                
                chunk_count += 1
                self.total_audio_processed += len(audio_chunk)
                
                # Process for wake word
                detection = self.wake_detector.process_audio(audio_chunk)
                
                if detection:
                    self.total_detections += 1
                    logger.info(
                        f"🎯 WAKE WORD DETECTED! Model: {detection.model_name}, "
                        f"Confidence: {detection.confidence:.3f}"
                    )
                    
                    # TODO: Start speech capture
                    # For now, just log the detection
                    self._handle_wake_detection(detection)
                
                # Periodic status logging
                current_time = time.time()
                if current_time - last_log_time >= 10.0:
                    rms = self.audio_capture.get_rms()
                    avg_inference = self.wake_detector.get_average_inference_time()
                    logger.info(
                        f"Status - Chunks: {chunk_count}, RMS: {rms:.4f}, "
                        f"Inference: {avg_inference:.1f}ms, Detections: {self.total_detections}"
                    )
                    last_log_time = current_time
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(0.1)
        
        logger.info("Detection loop stopped")
    
    def _handle_wake_detection(self, detection) -> None:
        """
        Handle wake word detection event.
        
        Args:
            detection: WakeWordDetection instance
        """
        # Get pre-roll audio
        pre_roll = self.audio_capture.get_pre_roll(1.0)
        logger.info(f"Captured {len(pre_roll)} samples of pre-roll audio")
        
        # TODO: Implement speech capture and streaming
        # For now, just log
        logger.info("Ready for speech input...")
    
    def get_status(self) -> Dict[str, Any]:
        """Get application status."""
        return {
            'is_running': self.is_running,
            'total_detections': self.total_detections,
            'total_audio_processed': self.total_audio_processed,
            'audio_rms': self.audio_capture.get_rms() if self.audio_capture else 0.0,
            'models': self.wake_detector.get_model_names() if self.wake_detector else [],
            'average_inference_ms': self.wake_detector.get_average_inference_time() if self.wake_detector else 0.0
        }