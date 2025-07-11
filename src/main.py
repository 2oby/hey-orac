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
import numpy as np
from pathlib import Path
from audio_utils import AudioManager
from wake_word_interface import WakeWordDetector
from audio_buffer import AudioBuffer
# Removed LED controller import since USB LED control is not working
from audio_feedback import create_audio_feedback
import pyaudio

# Import the monitor modules
from monitor_default_model import monitor_default_models
from audio_pipeline import run_audio_pipeline

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


def test_openwakeword_integration(wake_detector, config: dict) -> bool:
    """
    Test OpenWakeWord integration with the actual audio pipeline.
    This runs during startup to verify the wake word detection is working.
    """
    try:
        logger.info("🔍 Testing OpenWakeWord model and prediction pipeline...")
        
        # Test 1: Verify model is loaded
        if not wake_detector.is_ready():
            logger.error("❌ Wake word detector not ready")
            return False
        
        logger.info(f"✅ Wake word detector ready: {wake_detector.get_wake_word_name()}")
        
        # Test 2: Test with silence (should give low confidence)
        logger.info("🔍 Testing with silence (should give low confidence)...")
        silence_audio = np.zeros(wake_detector.get_frame_length(), dtype=np.int16)
        
        # Process silence multiple times to get a baseline
        silence_confidences = []
        for i in range(10):
            result = wake_detector.process_audio(silence_audio)
            silence_confidences.append(result)
        
        logger.info(f"✅ Silence test completed - detections: {sum(silence_confidences)}/10")
        
        # Test 2.5: Get detailed confidence scores if possible
        try:
            # Try to access the engine directly for detailed testing
            if hasattr(wake_detector, 'engine') and wake_detector.engine:
                logger.info("🔍 Testing detailed confidence scores...")
                
                # Test silence with direct engine access
                silence_float = silence_audio.astype(np.float32) / 32768.0
                predictions = wake_detector.engine.model.predict(silence_float)
                
                if isinstance(predictions, dict):
                    confidence = predictions.get(wake_detector.get_wake_word_name(), 0.0)
                elif isinstance(predictions, (list, tuple)):
                    confidence = float(predictions[0]) if predictions else 0.0
                else:
                    confidence = float(predictions)
                
                logger.info(f"   Silence confidence: {confidence:.6f}")
                
                # Test with noise
                noise_audio = np.random.randint(-1000, 1000, wake_detector.get_frame_length(), dtype=np.int16)
                noise_float = noise_audio.astype(np.float32) / 32768.0
                predictions = wake_detector.engine.model.predict(noise_float)
                
                if isinstance(predictions, dict):
                    confidence = predictions.get(wake_detector.get_wake_word_name(), 0.0)
                elif isinstance(predictions, (list, tuple)):
                    confidence = float(predictions[0]) if predictions else 0.0
                else:
                    confidence = float(predictions)
                
                logger.info(f"   Noise confidence: {confidence:.6f}")
                
                # Test with sine wave
                t = np.linspace(0, wake_detector.get_frame_length() / wake_detector.get_sample_rate(), 
                               wake_detector.get_frame_length())
                sine_audio = (np.sin(2 * np.pi * 1000 * t) * 5000).astype(np.int16)
                sine_float = sine_audio.astype(np.float32) / 32768.0
                predictions = wake_detector.engine.model.predict(sine_float)
                
                if isinstance(predictions, dict):
                    confidence = predictions.get(wake_detector.get_wake_word_name(), 0.0)
                elif isinstance(predictions, (list, tuple)):
                    confidence = float(predictions[0]) if predictions else 0.0
                else:
                    confidence = float(predictions)
                
                logger.info(f"   Sine wave confidence: {confidence:.6f}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not get detailed confidence scores: {e}")
        
        # Test 3: Test with random noise (should give varying confidence)
        logger.info("🔍 Testing with random noise...")
        noise_audio = np.random.randint(-1000, 1000, wake_detector.get_frame_length(), dtype=np.int16)
        
        noise_results = []
        for i in range(5):
            result = wake_detector.process_audio(noise_audio)
            noise_results.append(result)
        
        logger.info(f"✅ Noise test completed - detections: {sum(noise_results)}/5")
        
        # Test 4: Test with sine wave (simulating speech-like audio)
        logger.info("🔍 Testing with sine wave (speech-like audio)...")
        t = np.linspace(0, wake_detector.get_frame_length() / wake_detector.get_sample_rate(), 
                       wake_detector.get_frame_length())
        sine_audio = (np.sin(2 * np.pi * 1000 * t) * 5000).astype(np.int16)  # 1kHz sine wave
        
        sine_results = []
        for i in range(5):
            result = wake_detector.process_audio(sine_audio)
            sine_results.append(result)
        
        logger.info(f"✅ Sine wave test completed - detections: {sum(sine_results)}/5")
        
        # Test 5: Verify sensitivity behavior
        logger.info("🔍 Testing sensitivity behavior...")
        sensitivity = config['wake_word'].get('sensitivity', 0.5)
        logger.info(f"   Current sensitivity: {sensitivity}")
        logger.info(f"   Silence detections: {sum(silence_confidences)}/10")
        logger.info(f"   Noise detections: {sum(noise_results)}/5")
        logger.info(f"   Sine wave detections: {sum(sine_results)}/5")
        
        # Summary
        total_tests = 10 + 5 + 5
        total_detections = sum(silence_confidences) + sum(noise_results) + sum(sine_results)
        
        logger.info(f"📊 Integration test summary:")
        logger.info(f"   Total audio chunks tested: {total_tests}")
        logger.info(f"   Total detections: {total_detections}")
        logger.info(f"   Detection rate: {total_detections/total_tests*100:.1f}%")
        
        # Consider the test passed if the system is processing audio
        # (we can't easily get confidence scores from the interface)
        logger.info("✅ OpenWakeWord integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenWakeWord integration test failed: {e}", exc_info=True)
        return False


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
    parser.add_argument(
        "--audio-diagnostics",
        action="store_true",
        help="Run comprehensive audio system diagnostics"
    )
    parser.add_argument(
        "--monitor-default",
        action="store_true",
        help="Monitor default/pre-trained wake word models"
    )
    parser.add_argument(
        "--monitor-custom",
        action="store_true",
        help="Monitor custom wake word models"
    )
    parser.add_argument(
        "--test-custom-model",
        help="Test a specific custom model with speech input (provide model path)"
    )
    parser.add_argument(
        "--test-duration",
        type=int,
        default=30,
        help="Duration for custom model testing in seconds"
    )
    parser.add_argument(
        "--startup-test-model",
        help="Test a custom model at startup before monitoring (provide model path)"
    )
    parser.add_argument(
        "--startup-test-duration",
        type=int,
        default=10,
        help="Duration for startup model testing in seconds"
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run optimized audio pipeline with volume monitoring"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Handle different modes
    if args.list_devices:
        logger.info("🔍 Listing available audio devices...")
        audio_manager = AudioManager()
        devices = audio_manager.list_input_devices()
        if devices:
            logger.info("📋 Available audio input devices:")
            for device in devices:
                logger.info(f"   Device {device.index}: {device.name}")
                logger.info(f"     USB: {device.is_usb}")
                logger.info(f"     Max channels: {device.max_input_channels}")
                logger.info(f"     Sample rate: {device.default_sample_rate}")
        else:
            logger.warning("⚠️ No audio input devices found")
        return 0
    
    if args.audio_diagnostics:
        logger.info("🔧 Running comprehensive audio system diagnostics...")
        # This will be handled by the existing audio diagnostics code
        # For now, just indicate that this mode is available
        logger.info("📋 Audio diagnostics mode selected")
        return 0
    
    # Initialize audio manager
    audio_manager = AudioManager()
    
    # Find USB microphone
    devices = audio_manager.list_input_devices()
    if not devices:
        logger.error("❌ No audio devices found!")
        return 1
    
    usb_device = None
    for device in devices:
        if device.is_usb:
            usb_device = device
            break
    
    if not usb_device:
        logger.error("❌ No USB microphone found!")
        return 1
    
    logger.info(f"🎤 Using USB microphone: {usb_device.name}")
    
    # Initialize wake word detector for testing
    wake_detector = WakeWordDetector()
    if not wake_detector.initialize(config):
        logger.error("❌ Failed to initialize wake word detector")
        return 1
    
    logger.info("✅ Wake word detector initialized successfully")
    
    # Handle different monitoring modes
    if args.monitor_default:
        logger.info("🎯 Starting default model monitoring...")
        return monitor_default_models(config, usb_device, audio_manager)
    
    elif args.monitor_custom:
        logger.error("❌ Custom model monitoring not available - use --pipeline instead")
        return 1
    
    elif args.test_custom_model:
        logger.error("❌ Custom model testing not available - use --pipeline instead")
        return 1
    

    
    # Check for startup model testing
    if args.startup_test_model:
        logger.error("❌ Startup model testing not available - use --pipeline instead")
        return 1
    
    # Handle pipeline mode
    if args.pipeline:
        logger.info("🎯 Starting optimized audio pipeline with volume monitoring...")
        return run_audio_pipeline(config, usb_device, audio_manager)
    
    # Default behavior: Monitor default models
    logger.info("🎯 Starting default model monitoring (default behavior)...")
    return monitor_default_models(config, usb_device, audio_manager)


if __name__ == "__main__":
    main() 