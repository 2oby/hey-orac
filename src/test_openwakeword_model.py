#!/usr/bin/env python3
"""
Test script to verify OpenWakeWord model loading and prediction functionality
"""

import openwakeword
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if the hey_jarvis model can be loaded successfully."""
    try:
        logger.info("Testing OpenWakeWord model loading...")
        
        # Check available models
        available_models = openwakeword.models
        logger.info(f"Available models: {list(available_models.keys())}")
        
        # Check if hey_jarvis exists
        if 'hey_jarvis' not in available_models:
            logger.error("❌ hey_jarvis model not found in available models!")
            return False
        
        logger.info("✅ hey_jarvis model found in available models")
        
        # Get model path
        model_info = available_models['hey_jarvis']
        model_path = model_info['model_path']
        logger.info(f"Model path: {model_path}")
        
        # Try to load the model
        logger.info("Loading OpenWakeWord model...")
        model = openwakeword.Model(
            wakeword_model_paths=[model_path],
            class_mapping_dicts=[{0: 'hey_jarvis'}],
            enable_speex_noise_suppression=False,
            vad_threshold=0.5
        )
        
        logger.info("✅ Model loaded successfully!")
        
        # Test with dummy audio (silence)
        logger.info("Testing with dummy audio (silence)...")
        dummy_audio = np.zeros(1280, dtype=np.float32)  # 1280 samples = 80ms at 16kHz
        
        predictions = model.predict(dummy_audio)
        logger.info(f"Prediction type: {type(predictions)}")
        logger.info(f"Prediction content: {predictions}")
        
        # Test with some random audio
        logger.info("Testing with random audio...")
        random_audio = np.random.randn(1280).astype(np.float32) * 0.1
        
        predictions = model.predict(random_audio)
        logger.info(f"Random audio prediction: {predictions}")
        
        # Test with a sine wave (simulating speech-like audio)
        logger.info("Testing with sine wave...")
        t = np.linspace(0, 0.08, 1280)  # 80ms
        sine_audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32) * 0.1
        
        predictions = model.predict(sine_audio)
        logger.info(f"Sine wave prediction: {predictions}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        logger.info("🎉 All tests passed!")
    else:
        logger.error("💥 Tests failed!") 