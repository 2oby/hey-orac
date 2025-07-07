#!/usr/bin/env python3
"""
Basic OpenWakeWord test script
Tests installation and basic model loading
"""

import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_openwakeword_installation():
    """Test if OpenWakeWord is properly installed."""
    try:
        import openwakeword
        logger.info("✅ OpenWakeWord imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import OpenWakeWord: {e}")
        return False

def test_available_models():
    """Test what models are available."""
    try:
        import openwakeword
        models = openwakeword.models
        logger.info(f"✅ Available OpenWakeWord models: {list(models.keys())}")
        
        # Check if 'alexa' model is available
        if 'alexa' in models:
            logger.info("✅ 'alexa' model is available")
            model_info = models['alexa']
            logger.info(f"   Model path: {model_info.get('model_path', 'N/A')}")
            return True
        else:
            logger.warning("⚠️ 'alexa' model not found in available models")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking models: {e}")
        return False

def test_model_loading():
    """Test loading the alexa model."""
    try:
        import openwakeword
        
        # Check if alexa model exists
        if 'alexa' not in openwakeword.models:
            logger.error("❌ 'alexa' model not available")
            return False
            
        model_path = openwakeword.models['alexa']['model_path']
        logger.info(f"Loading model from: {model_path}")
        
        # Try to load the model
        model = openwakeword.Model(
            wakeword_model_paths=[model_path],
            class_mapping_dicts=[{0: 'alexa'}],
            enable_speex_noise_suppression=False,
            vad_threshold=0.5
        )
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"   Model object: {model}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}", exc_info=True)
        return False

def test_basic_prediction():
    """Test basic prediction with dummy audio."""
    try:
        import openwakeword
        
        # Load model
        model_path = openwakeword.models['alexa']['model_path']
        model = openwakeword.Model(
            wakeword_model_paths=[model_path],
            class_mapping_dicts=[{0: 'alexa'}],
            enable_speex_noise_suppression=False,
            vad_threshold=0.5
        )
        
        # Create dummy audio (silence)
        dummy_audio = np.zeros(1280, dtype=np.float32)  # 80ms at 16kHz
        
        logger.info(f"Testing prediction with dummy audio: shape={dummy_audio.shape}")
        
        # Get prediction
        prediction = model.predict(dummy_audio)
        
        logger.info(f"✅ Prediction successful")
        logger.info(f"   Prediction type: {type(prediction)}")
        logger.info(f"   Prediction value: {prediction}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing prediction: {e}", exc_info=True)
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Starting OpenWakeWord basic tests...")
    
    tests = [
        ("Installation", test_openwakeword_installation),
        ("Available Models", test_available_models),
        ("Model Loading", test_model_loading),
        ("Basic Prediction", test_basic_prediction),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running test: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n📊 Test Results Summary:")
    logger.info("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n   Total: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! OpenWakeWord is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 