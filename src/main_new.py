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
from monitor_custom_model import CustomModelMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HeyOracApp:
    """
    Minimal Hey Orac application with just the essential monitoring loop.
    """
    
    def __init__(self, config_path: str = "/app/config.yaml"):
        """Initialize the Hey Orac application."""
        self.config_path = config_path
        self.config = self._load_config()
        self.audio_manager = None
        self.wake_detector = None
        self.usb_device = None
        self.custom_monitor = None
        
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
            
            # Initialize wake word detector
            self.wake_detector = WakeWordDetector()
            if not self.wake_detector.initialize(self.config):
                logger.error("❌ Failed to initialize wake word detector")
                return False
            
            logger.info("✅ Wake word detector initialized successfully")
            
            # Initialize custom model monitor for neural engine processing
            self.custom_monitor = CustomModelMonitor(
                config=self.config,
                usb_device=self.usb_device,
                audio_manager=self.audio_manager
            )
            
            if not self.custom_monitor.initialize():
                logger.error("❌ Failed to initialize custom model monitor")
                return False
            
            logger.info("✅ Custom model monitor initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize application: {e}")
            return False
    
    def _wake_word_callback(self, audio_data, chunk_count, rms_level, avg_volume):
        """
        Callback function that receives audio from the pipeline when RMS is above threshold.
        This is where the neural engine processes the audio for wake word detection.
        """
        try:
            # Process audio through the custom model monitor (neural engine)
            # This will only be called when audio is above the RMS filter threshold
            detection_result = self.custom_monitor._process_audio_chunk(audio_data)
            
            if detection_result:
                logger.info(f"🎯 Wake word detected in callback! Chunk: {chunk_count}, RMS: {rms_level:.4f}")
            else:
                # Only log occasionally to avoid spam
                if chunk_count % 100 == 0:
                    logger.debug(f"🔍 Processing audio chunk {chunk_count} - RMS: {rms_level:.4f}, Avg: {avg_volume:.4f}")
                    
        except Exception as e:
            logger.error(f"❌ Error in wake word callback: {e}")
    
    def run_monitoring(self) -> int:
        """Run the basic monitoring loop."""
        logger.info("🎯 Starting new audio pipeline with RMS monitoring and neural engine...")
        
        # Create audio pipeline
        audio_pipeline = create_audio_pipeline(
            audio_manager=self.audio_manager,
            usb_device=self.usb_device,
            sample_rate=self.wake_detector.get_sample_rate(),
            frame_length=self.wake_detector.get_frame_length(),
            channels=self.wake_detector.get_channels()
        )
        
        # Set up wake word callback to connect audio pipeline to neural engine
        audio_pipeline.set_wake_word_callback(self._wake_word_callback)
        
        logger.info("✅ Wake word callback connected to audio pipeline")
        logger.info("🎯 Neural engine will only receive audio when RMS is above filter threshold")
        
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