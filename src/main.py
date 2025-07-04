#!/usr/bin/env python3
"""
Hey Orac - Wake-word Detection Service
Phase 1a of the ORAC Voice-Control Architecture
"""

import argparse
import logging
import sys
import yaml
import time
from pathlib import Path
from audio_utils import AudioManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def test_audio_file(audio_path: str, config: dict) -> None:
    """Test wake-word detection with audio file."""
    logger.info(f"Testing wake-word detection with {audio_path}")
    # TODO: Implement audio file testing
    logger.info("Audio file testing not yet implemented")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Hey Orac Wake-word Detection Service")
    parser.add_argument(
        "--config", 
        default="/app/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-audio",
        help="Test wake-word detection with audio file"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.list_devices:
        logger.info("Listing audio devices...")
        audio_manager = AudioManager()
        devices = audio_manager.list_input_devices()
        
        print("\n=== Available Audio Input Devices ===")
        for device in devices:
            print(f"Device {device.index}: {device.name}")
            print(f"  - Max Input Channels: {device.max_input_channels}")
            print(f"  - Default Sample Rate: {device.default_sample_rate}")
            print(f"  - Host API: {device.host_api}")
            print(f"  - USB Device: {'Yes' if device.is_usb else 'No'}")
            print()
        
        # Find USB microphone
        usb_device = audio_manager.find_usb_microphone()
        if usb_device:
            print(f"✅ Found USB microphone: {usb_device.name} (index {usb_device.index})")
        else:
            print("⚠️  No USB microphone found")
        
        return
    
    if args.test_audio:
        test_audio_file(args.test_audio, config)
        return
    
    # Main service loop
    logger.info("Starting Hey Orac wake-word detection service...")
    logger.info(f"Configuration: {config}")
    
    # TODO: Implement main service loop
    logger.info("Main service loop not yet implemented")
    
    try:
        # Placeholder for main loop - with sleep to prevent CPU runaway
        logger.info("Main service loop not yet implemented - sleeping to prevent CPU runaway")
        while True:
            time.sleep(1)  # Sleep 1 second to prevent 100% CPU usage
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 