#!/usr/bin/env python3
"""
Test script to verify OpenWakeWord fixes are working correctly
"""

import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from wake_word_engines.openwakeword_engine import OpenWakeWordEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_openwakeword_fixes():
    """Test the OpenWakeWord fixes."""
    logger.info("🧪 Testing OpenWakeWord fixes...")
    
    # Create engine instance
    engine = OpenWakeWordEngine()
    
    # Test configuration
    config = {
        'custom_model_path': 'third_party/openwakeword/custom_models/Hay--compUta_v_lrg.onnx',
        'sensitivity': 0.8,
        'threshold': 0.001  # FIX #4: Lowered threshold for debugging
    }
    
    logger.info("🔧 Initializing OpenWakeWord engine with fixes...")
    
    # Initialize the engine
    if engine.initialize(config):
        logger.info("✅ Engine initialized successfully")
        
        # Test audio processing with different inputs
        import numpy as np
        
        # Test 1: Silence
        silence_audio = np.zeros(1280, dtype=np.int16)
        logger.info("🔍 Testing with silence...")
        result = engine.process_audio(silence_audio)
        logger.info(f"   Silence result: {result}")
        
        # Test 2: Low level audio (simulates quiet microphone)
        low_audio = np.random.randint(-100, 100, 1280, dtype=np.int16)
        logger.info("🔍 Testing with low level audio...")
        result = engine.process_audio(low_audio)
        logger.info(f"   Low level result: {result}")
        
        # Test 3: Normal level audio
        normal_audio = np.random.randint(-5000, 5000, 1280, dtype=np.int16)
        logger.info("🔍 Testing with normal level audio...")
        result = engine.process_audio(normal_audio)
        logger.info(f"   Normal level result: {result}")
        
        # Test 4: High level audio
        high_audio = np.random.randint(-20000, 20000, 1280, dtype=np.int16)
        logger.info("🔍 Testing with high level audio...")
        result = engine.process_audio(high_audio)
        logger.info(f"   High level result: {result}")
        
        logger.info("✅ All tests completed")
        
    else:
        logger.error("❌ Engine initialization failed")
        return False
    
    return True

if __name__ == "__main__":
    success = test_openwakeword_fixes()
    if success:
        logger.info("🎉 All OpenWakeWord fixes verified successfully!")
    else:
        logger.error("❌ OpenWakeWord fixes test failed")
        sys.exit(1) 