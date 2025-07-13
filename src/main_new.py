#!/usr/bin/env python3
"""
Hey Orac - Wake-word Detection Service (Minimal Version)
Phase 1a of the ORAC Voice-Control Architecture - Minimal Implementation
"""

import logging
import sys
import yaml
from typing import Dict, Any

# Core imports
from audio_utils import AudioManager
from wake_word_interface import WakeWordDetector
from audio_pipeline_new import create_audio_pipeline
from wake_word_monitor_new import create_wake_word_monitor_new

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see debug messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add debug log to verify logging is working
logger.debug("🔧 Main new application initialized with DEBUG logging enabled")


class HeyOracApp:
    """
    Minimal Hey Orac application with just the essential monitoring loop.
    Now properly integrates wake_word_monitor_new and audio_pipeline_new.
    """
    
    def __init__(self, config_path: str = "/app/config.yaml"):
        """Initialize the Hey Orac application."""
        self.config_path = config_path
        self.config = self._load_config()
        self.audio_manager = None
        self.wake_detector = None
        self.usb_device = None
        self.wake_word_monitor = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def initialize(self) -> bool:
        """Initialize the application components."""
        try:
            # Initialize audio manager
            self.audio_manager = AudioManager()
            
            # Find USB microphone
            devices = self.audio_manager.list_input_devices()
            if not devices:
                logger.error("❌ No audio devices found!")
                return False
            
            self.usb_device = None
            for device in devices:
                if device.is_usb:
                    self.usb_device = device
                    break
            
            if not self.usb_device:
                logger.error("❌ No USB microphone found!")
                return False
            
            logger.info(f"🎤 Using USB microphone: {self.usb_device.name}")
            
            # Initialize wake word monitor (new implementation) - this handles all wake word detection
            self.wake_word_monitor = create_wake_word_monitor_new()
            logger.info("✅ Wake word monitor (new) initialized successfully")
            
            # Print configuration summary
            self.wake_word_monitor.print_configuration_summary()
            
            # Get sample rate and frame length from the monitor's active detectors
            sample_rate = 16000  # Default for OpenWakeWord
            frame_length = 1280   # Default for OpenWakeWord
            
            # Try to get from active detectors if available
            active_detectors = self.wake_word_monitor.get_active_detectors()
            if active_detectors:
                # Get the first active detector for audio format info
                first_detector = list(active_detectors.values())[0]
                sample_rate = first_detector.get_sample_rate()
                frame_length = first_detector.get_frame_length()
                logger.info(f"🔧 Using audio format from active detector: {sample_rate}Hz, {frame_length} samples")
            else:
                logger.warning("⚠️ No active detectors found, using default audio format")
            
            # Store audio format for pipeline creation
            self.sample_rate = sample_rate
            self.frame_length = frame_length
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize application: {e}")
            return False
    
    def run_monitoring(self) -> int:
        """Run the basic monitoring loop."""
        logger.info("🎯 Starting new audio pipeline with RMS monitoring and wake word monitor...")
        
        # Create audio pipeline with wake word monitor for validation
        audio_pipeline = create_audio_pipeline(
            audio_manager=self.audio_manager,
            usb_device=self.usb_device,
            sample_rate=self.sample_rate,
            frame_length=self.frame_length,
            channels=1,  # Mono audio
            wake_word_monitor=self.wake_word_monitor
        )
        
        # Set up wake word callback using the new monitor
        def wake_word_callback(audio_data, chunk_count, rms_level, avg_volume):
            """Wake word detection callback using the new monitor."""
            try:
                # DEBUG: Add logging to see if callback is being called
                logger.debug(f"🎯 DEBUG: Audio callback called - Chunk: {chunk_count}, RMS: {rms_level:.4f}")
                logger.info(f"🎯 INFO: Audio callback called - Chunk: {chunk_count}, RMS: {rms_level:.4f}")
                
                # Process audio through the new monitor's detection system
                detection_result = self.wake_word_monitor.process_audio(audio_data)
                
                if detection_result:
                    logger.info(f"🎯 Wake word detected! Chunk: {chunk_count}, RMS: {rms_level:.4f}")
                    logger.info(f"   Total detections: {self.wake_word_monitor.get_detection_count()}")
                else:
                    # Log progress periodically (every 1000 chunks to reduce spam)
                    if chunk_count % 1000 == 0:
                        logger.debug(f"🔍 Processing audio - Chunk: {chunk_count}, RMS: {rms_level:.4f}")
                        logger.debug(f"   Active detectors: {list(self.wake_word_monitor.get_active_detectors().keys())}")
                
            except Exception as e:
                logger.error(f"❌ Error in wake word callback: {e}")
        
        # Set the callback
        audio_pipeline.set_wake_word_callback(wake_word_callback)
        logger.info("✅ Wake word callback set with new monitor integration")
        
        return audio_pipeline.run()


def main():
    """Main application entry point."""
    # Create application instance
    app = HeyOracApp()
    
    # Initialize application
    if not app.initialize():
        return 1
    
    # Run the monitoring loop
    return app.run_monitoring()


if __name__ == "__main__":
    main() 