#!/usr/bin/env python3
"""
Monitor Default Wake Word Models
Handles monitoring of pre-trained wake word models (Hey Jarvis, etc.)
"""

import logging
from base_monitor import BaseWakeWordMonitor

logger = logging.getLogger(__name__)


class DefaultModelMonitor(BaseWakeWordMonitor):
    """
    Monitor for default/pre-trained wake word models.
    """
    
    def _initialize_detector(self) -> bool:
        """Initialize the wake word detector for default models."""
        return self.wake_detector.initialize(self.config)
    
    def _should_allow_detection(self) -> bool:
        """Default models have no timing controls - always allow detection."""
        return True
    
    def _get_detection_log_file(self) -> str:
        """Get the path to the default detection log file."""
        return "/app/logs/default_detections.log"
    
    def _get_audio_clip_filename(self) -> str:
        """Get the filename for default audio clips."""
        return f"/tmp/default_wake_word_detection_{self.detection_count}.wav"


def monitor_default_models(config: dict, usb_device, audio_manager) -> int:
    """
    Monitor default/pre-trained wake word models.
    
    Args:
        config: Configuration dictionary
        usb_device: USB audio device to use
        audio_manager: Initialized audio manager
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("🎯 Starting default model monitoring...")
    
    # Create and initialize the monitor
    monitor = DefaultModelMonitor(config, usb_device, audio_manager)
    if not monitor.initialize():
        logger.error("❌ Failed to initialize default model monitor")
        return 1
    
    # Run the monitoring loop
    return monitor.run()


if __name__ == "__main__":
    # This file is designed to be imported and used by main.py
    # It can also be run independently for testing
    logger.info("🎯 Default model monitor module loaded") 