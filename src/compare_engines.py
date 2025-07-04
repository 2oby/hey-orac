#!/usr/bin/env python3
"""
Compare different wake-word detection engines side by side
"""

import yaml
import time
import logging
from pathlib import Path
from audio_utils import AudioManager
from wake_word_interface import WakeWordDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "/app/config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}

def test_engine(engine_name: str, config: dict, duration: int = 30) -> None:
    """
    Test a specific wake-word detection engine.
    
    Args:
        engine_name: Name of the engine to test
        config: Configuration dictionary
        duration: Test duration in seconds
    """
    logger.info(f"🧪 Testing {engine_name.upper()} engine for {duration} seconds...")
    
    # Create detector with specified engine
    detector = WakeWordDetector(engine_name=engine_name)
    
    if not detector.initialize(config['wake_word']):
        logger.error(f"❌ Failed to initialize {engine_name} engine")
        return
    
    if not detector.is_ready():
        logger.error(f"❌ {engine_name} engine not ready")
        return
    
    # Initialize audio manager
    audio_manager = AudioManager()
    usb_device = audio_manager.find_usb_microphone()
    if not usb_device:
        logger.error("❌ No USB microphone found")
        return
    
    logger.info(f"🎤 Listening for '{detector.get_wake_word_name()}' on {usb_device.name}")
    logger.info(f"📊 Sample rate: {detector.get_sample_rate()}, Frame length: {detector.get_frame_length()}")
    
    detections = 0
    start_time = time.time()
    
    try:
        # Start live audio stream
        stream = audio_manager.start_stream(
            device_index=usb_device.index,
            sample_rate=detector.get_sample_rate(),
            channels=1,
            chunk_size=detector.get_frame_length()
        )
        
        if not stream:
            logger.error("❌ Failed to start audio stream")
            return
        
        # Process audio in real-time
        while time.time() - start_time < duration:
            try:
                # Read audio chunk
                audio_data = stream.read(detector.get_frame_length(), exception_on_overflow=False)
                
                # Process with wake-word detector
                if detector.process_audio(audio_data):
                    detections += 1
                    
            except KeyboardInterrupt:
                logger.info("Stopping test...")
                break
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                break
        
        # Cleanup
        stream.stop_stream()
        stream.close()
        
    except KeyboardInterrupt:
        logger.info(f"{engine_name.upper()} test stopped by user")
    except Exception as e:
        logger.error(f"Error in {engine_name} test: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"✅ {engine_name.upper()} test completed:")
    logger.info(f"   - Duration: {elapsed_time:.1f}s")
    logger.info(f"   - Detections: {detections}")
    logger.info(f"   - Rate: {detections/elapsed_time:.2f} detections/second")
    
    detector.cleanup()

def main():
    """Main function to compare engines."""
    config = load_config()
    if not config:
        return
    
    logger.info("🔬 Wake-word Engine Comparison Test")
    logger.info("=" * 50)
    
    # Test different engines
    engines_to_test = ["test", "openwakeword"]
    
    for engine in engines_to_test:
        logger.info(f"\n{'='*20} {engine.upper()} {'='*20}")
        test_engine(engine, config, duration=15)  # 15 seconds per engine
        time.sleep(2)  # Brief pause between tests
    
    logger.info("\n🎉 Engine comparison completed!")

if __name__ == "__main__":
    main() 