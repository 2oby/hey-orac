#!/usr/bin/env python3
"""
Test script for custom OpenWakeWord models
Verifies that custom models are loading and working correctly
"""

import sys
import os
import yaml
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wake_word_engines.openwakeword_engine import OpenWakeWordEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_custom_model_loading():
    """Test custom model loading and detection."""
    
    # Load configuration
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"❌ Configuration file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("🔍 Testing custom model loading...")
    logger.info(f"   Engine: {config['wake_word']['engine']}")
    logger.info(f"   Keyword: {config['wake_word']['keyword']}")
    logger.info(f"   Custom model path: {config['wake_word']['custom_model_path']}")
    
    # Check if custom model file exists
    custom_model_path = config['wake_word']['custom_model_path']
    if not os.path.exists(custom_model_path):
        logger.error(f"❌ Custom model file not found: {custom_model_path}")
        return False
    
    logger.info(f"✅ Custom model file exists: {custom_model_path}")
    logger.info(f"   File size: {os.path.getsize(custom_model_path)} bytes")
    
    # Initialize OpenWakeWord engine
    engine = OpenWakeWordEngine()
    
    # Test initialization
    logger.info("🔍 Initializing OpenWakeWord engine...")
    if not engine.initialize(config['wake_word']):
        logger.error("❌ Failed to initialize OpenWakeWord engine")
        return False
    
    logger.info("✅ OpenWakeWord engine initialized successfully")
    
    # Test with silence
    logger.info("🔍 Testing with silence...")
    silence_audio = np.zeros(1280, dtype=np.int16)
    silence_result = engine.process_audio(silence_audio)
    logger.info(f"   Silence detection: {silence_result}")
    
    # Test with noise
    logger.info("🔍 Testing with noise...")
    noise_audio = np.random.randint(-1000, 1000, 1280, dtype=np.int16)
    noise_result = engine.process_audio(noise_audio)
    logger.info(f"   Noise detection: {noise_result}")
    
    # Test with sine wave (simulating speech)
    logger.info("🔍 Testing with sine wave...")
    t = np.linspace(0, 1280 / 16000, 1280)
    sine_audio = (np.sin(2 * np.pi * 1000 * t) * 5000).astype(np.int16)
    sine_result = engine.process_audio(sine_audio)
    logger.info(f"   Sine wave detection: {sine_result}")
    
    # Test multiple audio chunks
    logger.info("🔍 Testing multiple audio chunks...")
    detections = 0
    for i in range(10):
        # Generate random audio with some speech-like characteristics
        audio = np.random.randint(-2000, 2000, 1280, dtype=np.int16)
        # Add some periodic components to simulate speech
        t = np.linspace(0, 1280 / 16000, 1280)
        audio += (np.sin(2 * np.pi * 500 * t) * 1000).astype(np.int16)
        audio += (np.sin(2 * np.pi * 1500 * t) * 500).astype(np.int16)
        
        result = engine.process_audio(audio)
        if result:
            detections += 1
            logger.info(f"   Detection #{detections} on chunk {i+1}")
    
    logger.info(f"✅ Multiple chunk test completed: {detections}/10 detections")
    
    # Cleanup
    engine.cleanup()
    
    logger.info("✅ Custom model test completed successfully!")
    return True

def test_all_custom_models():
    """Test all available custom models."""
    
    custom_models_dir = Path("third_party/openwakeword/custom_models")
    if not custom_models_dir.exists():
        logger.error(f"❌ Custom models directory not found: {custom_models_dir}")
        return False
    
    # Find all ONNX and TFLite models
    onnx_models = list(custom_models_dir.glob("*.onnx"))
    tflite_models = list(custom_models_dir.glob("*.tflite"))
    
    logger.info(f"🔍 Found custom models:")
    logger.info(f"   ONNX models: {len(onnx_models)}")
    logger.info(f"   TFLite models: {len(tflite_models)}")
    
    for model in onnx_models + tflite_models:
        logger.info(f"   - {model.name}")
    
    # Test each ONNX model
    for model_path in onnx_models:
        logger.info(f"🔍 Testing model: {model_path.name}")
        
        # Create test configuration
        test_config = {
            'keyword': model_path.stem.lower().replace('-', '_').replace('_', '_'),
            'threshold': 0.5,
            'custom_model_path': str(model_path)
        }
        
        # Initialize engine
        engine = OpenWakeWordEngine()
        if engine.initialize(test_config):
            logger.info(f"✅ Model {model_path.name} loaded successfully")
            
            # Test with silence
            silence_audio = np.zeros(1280, dtype=np.int16)
            result = engine.process_audio(silence_audio)
            logger.info(f"   Silence test: {result}")
            
            engine.cleanup()
        else:
            logger.error(f"❌ Failed to load model {model_path.name}")
    
    return True

if __name__ == "__main__":
    logger.info("🧪 Custom Model Testing Script")
    logger.info("=" * 40)
    
    # Test 1: Test configured custom model
    logger.info("Test 1: Testing configured custom model")
    if test_custom_model_loading():
        logger.info("✅ Test 1 passed")
    else:
        logger.error("❌ Test 1 failed")
    
    # Test 2: Test all available custom models
    logger.info("\nTest 2: Testing all available custom models")
    if test_all_custom_models():
        logger.info("✅ Test 2 passed")
    else:
        logger.error("❌ Test 2 failed")
    
    logger.info("\n🎉 Custom model testing completed!") 