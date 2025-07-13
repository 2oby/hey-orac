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
    """
    
    def __init__(self, config_path: str = "/app/config.yaml"):
        """Initialize the Hey Orac application."""
        self.config_path = config_path
        self.config = self._load_config()
        self.audio_manager = None
        self.wake_detector = None
        self.usb_device = None
        
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
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize application: {e}")
            return False
    
    def run_monitoring(self) -> int:
        """Run the basic monitoring loop."""
        logger.info("🎯 Starting new audio pipeline with RMS monitoring...")
        
        # Create audio pipeline
        audio_pipeline = create_audio_pipeline(
            audio_manager=self.audio_manager,
            usb_device=self.usb_device,
            sample_rate=self.wake_detector.get_sample_rate(),
            frame_length=self.wake_detector.get_frame_length(),
            channels=self.wake_detector.get_channels()
        )
        
        # TODO: Set up wake word callback
        # For now, just run the audio pipeline without wake word detection
        # def wake_word_callback(audio_data, chunk_count, rms_level, avg_volume):
        #     # Wake word detection logic will go here
        #     pass
        # audio_pipeline.set_wake_word_callback(wake_word_callback)
        
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