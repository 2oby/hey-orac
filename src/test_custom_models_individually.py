#!/usr/bin/env python3
"""
Test each custom model individually with actual speech input.
"""

import sys
import os
import time
import logging
import numpy as np
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wake_word_engines.openwakeword_engine import OpenWakeWordEngine
from audio_utils import AudioManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_custom_model(model_path: str, model_name: str, duration: int = 10):
    """
    Test a single custom model with actual speech input.
    
    Args:
        model_path: Path to the custom model file
        model_name: Name of the model for logging
        duration: Test duration in seconds
    """
    logger.info(f"🎯 Testing custom model: {model_name}")
    logger.info(f"📁 Model path: {model_path}")
    logger.info(f"⏱️  Test duration: {duration} seconds")
    
    # Initialize audio manager
    audio_manager = AudioManager()
    audio_manager.initialize()
    
    # Find USB microphone
    devices = audio_manager.get_input_devices()
    if not devices:
        logger.error("❌ No audio devices found!")
        return False
    
    usb_device = None
    for device in devices:
        if device.get('is_usb', False):
            usb_device = device
            break
    
    if not usb_device:
        logger.error("❌ No USB microphone found!")
        return False
    
    logger.info(f"🎤 Using USB microphone: {usb_device['name']}")
    
    # Initialize wake word engine with custom model
    engine = OpenWakeWordEngine()
    
    config = {
        'keyword': model_name.lower().replace('_', '').replace('-', ''),
        'threshold': 0.1,  # Lower threshold for testing
        'custom_model_path': model_path
    }
    
    if not engine.initialize(config):
        logger.error(f"❌ Failed to initialize engine for {model_name}")
        return False
    
    logger.info(f"✅ Engine initialized for {model_name}")
    
    # Start audio stream
    device_index = usb_device['index']
    if not audio_manager.start_stream(device_index):
        logger.error("❌ Failed to start audio stream")
        return False
    
    logger.info(f"🎤 Audio stream started on device {device_index}")
    
    # Test with actual speech
    logger.info(f"🎤 Say 'Hey Computer' into the microphone for {duration} seconds...")
    logger.info(f"📊 Monitoring for detections...")
    
    start_time = time.time()
    detections = 0
    chunks_processed = 0
    
    try:
        while time.time() - start_time < duration:
            # Read audio chunk
            audio_chunk = audio_manager.read_chunk()
            if audio_chunk is None:
                continue
            
            chunks_processed += 1
            
            # Process with wake word engine
            if engine.process_audio(audio_chunk):
                detections += 1
                logger.info(f"🎯 DETECTION #{detections} - {model_name} detected!")
            
            # Log progress every 2 seconds
            elapsed = time.time() - start_time
            if chunks_processed % 25 == 0:  # ~2 seconds at 80ms chunks
                logger.info(f"⏱️  {elapsed:.1f}s elapsed - {chunks_processed} chunks processed - {detections} detections")
    
    except KeyboardInterrupt:
        logger.info("⏹️  Test interrupted by user")
    
    finally:
        audio_manager.stop_stream()
        audio_manager.cleanup()
    
    # Results
    elapsed = time.time() - start_time
    logger.info(f"📊 Test Results for {model_name}:")
    logger.info(f"   Duration: {elapsed:.1f} seconds")
    logger.info(f"   Chunks processed: {chunks_processed}")
    logger.info(f"   Detections: {detections}")
    logger.info(f"   Detection rate: {detections/elapsed:.2f} detections/second")
    
    return detections > 0

def main():
    """Test all custom models individually."""
    logger.info("🎯 Custom Model Individual Testing")
    logger.info("=" * 50)
    
    # Custom models to test
    models = [
        {
            'path': 'third_party/openwakeword/custom_models/Hay--compUta_v_lrg.onnx',
            'name': 'Hay_compUta_v_lrg'
        },
        {
            'path': 'third_party/openwakeword/custom_models/Hey_computer.onnx',
            'name': 'Hey_computer'
        },
        {
            'path': 'third_party/openwakeword/custom_models/hey-CompUter_lrg.onnx',
            'name': 'hey_CompUter_lrg'
        }
    ]
    
    results = {}
    
    for i, model in enumerate(models, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"🧪 TEST {i}/{len(models)}: {model['name']}")
        logger.info(f"{'='*50}")
        
        # Check if model file exists
        if not os.path.exists(model['path']):
            logger.error(f"❌ Model file not found: {model['path']}")
            results[model['name']] = False
            continue
        
        # Test the model
        success = test_custom_model(model['path'], model['name'], duration=10)
        results[model['name']] = success
        
        # Wait between tests
        if i < len(models):
            logger.info("⏳ Waiting 3 seconds before next test...")
            time.sleep(3)
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 FINAL RESULTS")
    logger.info(f"{'='*50}")
    
    for model_name, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        logger.info(f"   {model_name}: {status}")
    
    working_count = sum(1 for success in results.values() if success)
    logger.info(f"\n🎯 Summary: {working_count}/{len(results)} models working")
    
    if working_count > 0:
        logger.info("✅ At least one custom model is working!")
    else:
        logger.info("❌ No custom models detected speech")

if __name__ == "__main__":
    main() 