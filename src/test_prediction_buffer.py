#!/usr/bin/env python3
"""
Test script to verify OpenWakeWord prediction_buffer functionality
This will help us understand how the prediction_buffer works and what data it contains.
"""

import sys
import logging
import numpy as np
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_prediction_buffer():
    """Test OpenWakeWord prediction_buffer functionality"""
    try:
        logger.info("🔍 Testing OpenWakeWord prediction_buffer functionality...")
        
        # Import OpenWakeWord
        import openwakeword
        
        logger.info(f"✅ OpenWakeWord imported successfully")
        logger.info(f"   Version: {openwakeword.__version__ if hasattr(openwakeword, '__version__') else 'Unknown'}")
        logger.info(f"   Available models: {list(openwakeword.models.keys())}")
        
        # Load a test model
        test_model_name = 'alexa'  # Use a known model
        if test_model_name not in openwakeword.models:
            logger.error(f"❌ Test model '{test_model_name}' not available")
            return False
            
        model_path = openwakeword.models[test_model_name]['model_path']
        logger.info(f"🔧 Loading test model: {test_model_name}")
        logger.info(f"   Model path: {model_path}")
        
        # Create model
        model = openwakeword.Model(
            wakeword_model_paths=[model_path],
            class_mapping_dicts=[{0: test_model_name}],
            enable_speex_noise_suppression=False,
            vad_threshold=0.5
        )
        
        logger.info(f"✅ Model created successfully")
        logger.info(f"   Model type: {type(model)}")
        logger.info(f"   Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        
        # Check if prediction_buffer exists
        if hasattr(model, 'prediction_buffer'):
            logger.info(f"✅ prediction_buffer available")
            logger.info(f"   prediction_buffer type: {type(model.prediction_buffer)}")
            logger.info(f"   prediction_buffer keys: {list(model.prediction_buffer.keys())}")
            logger.info(f"   prediction_buffer content: {model.prediction_buffer}")
        else:
            logger.error(f"❌ prediction_buffer not available")
            return False
        
        # Test with different audio types
        test_audio_types = [
            ("silence", np.zeros(1280, dtype=np.float32)),
            ("noise", np.random.randn(1280).astype(np.float32) * 0.1),
            ("sine_wave", (np.sin(2 * np.pi * 1000 * np.linspace(0, 1280/16000, 1280)) * 0.5).astype(np.float32))
        ]
        
        for audio_name, audio_data in test_audio_types:
            logger.info(f"\n🔍 Testing with {audio_name}...")
            logger.info(f"   Audio shape: {audio_data.shape}")
            logger.info(f"   Audio RMS: {np.sqrt(np.mean(audio_data**2)):.6f}")
            
            # Call predict
            raw_predictions = model.predict(audio_data)
            logger.info(f"   Raw predictions type: {type(raw_predictions)}")
            logger.info(f"   Raw predictions: {raw_predictions}")
            
            # Check prediction_buffer after prediction
            if hasattr(model, 'prediction_buffer'):
                logger.info(f"   prediction_buffer after prediction:")
                for key, scores in model.prediction_buffer.items():
                    logger.info(f"     '{key}': {len(scores)} scores")
                    if len(scores) > 0:
                        logger.info(f"       Latest score: {scores[-1]:.6f}")
                        logger.info(f"       Last 5 scores: {[f'{s:.6f}' for s in scores[-5:]]}")
                        logger.info(f"       All scores: {[f'{s:.6f}' for s in scores]}")
                    else:
                        logger.info(f"       No scores yet")
            else:
                logger.error(f"   ❌ prediction_buffer not available after prediction")
        
        logger.info(f"\n✅ prediction_buffer test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ prediction_buffer test failed: {e}", exc_info=True)
        return False

def main():
    """Main test function"""
    logger.info("🧪 Starting OpenWakeWord prediction_buffer test...")
    
    success = test_prediction_buffer()
    
    if success:
        logger.info("✅ All tests passed!")
        return True
    else:
        logger.error("❌ Tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 