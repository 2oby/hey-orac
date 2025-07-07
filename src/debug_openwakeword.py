#!/usr/bin/env python3
"""
Comprehensive OpenWakeWord debugging script
Tests all aspects of OpenWakeWord to identify the root cause of 0.0000 confidence issue
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

def test_openwakeword_installation():
    """Test 1: Basic OpenWakeWord installation and import."""
    logger.info("="*60)
    logger.info("TEST 1: OpenWakeWord Installation")
    logger.info("="*60)
    
    try:
        import openwakeword
        logger.info("✅ OpenWakeWord imported successfully")
        
        # Check version
        version = getattr(openwakeword, '__version__', 'Unknown')
        logger.info(f"   Version: {version}")
        
        # Check available models
        models = openwakeword.models
        logger.info(f"   Available models: {list(models.keys())}")
        
        return True, openwakeword
        
    except ImportError as e:
        logger.error(f"❌ Failed to import OpenWakeWord: {e}")
        return False, None

def test_model_availability(openwakeword):
    """Test 2: Check if requested models are available."""
    logger.info("="*60)
    logger.info("TEST 2: Model Availability")
    logger.info("="*60)
    
    models = openwakeword.models
    test_keywords = ['hey_jarvis', 'alexa', 'hey_mycroft', 'timer', 'weather']
    
    for keyword in test_keywords:
        if keyword in models:
            model_info = models[keyword]
            model_path = model_info['model_path']
            exists = Path(model_path).exists()
            size = Path(model_path).stat().st_size if exists else 0
            
            logger.info(f"✅ {keyword}: Available")
            logger.info(f"   Path: {model_path}")
            logger.info(f"   Exists: {exists}")
            logger.info(f"   Size: {size:,} bytes")
        else:
            logger.warning(f"⚠️ {keyword}: Not available")
    
    return True

def test_model_loading(openwakeword):
    """Test 3: Test loading different models."""
    logger.info("="*60)
    logger.info("TEST 3: Model Loading")
    logger.info("="*60)
    
    models = openwakeword.models
    test_keywords = ['hey_jarvis', 'alexa', 'hey_mycroft']
    
    loaded_models = {}
    
    for keyword in test_keywords:
        if keyword not in models:
            logger.warning(f"⚠️ Skipping {keyword} - not available")
            continue
            
        try:
            logger.info(f"🔍 Loading {keyword} model...")
            
            model_path = models[keyword]['model_path']
            model = openwakeword.Model(
                wakeword_model_paths=[model_path],
                class_mapping_dicts=[{0: keyword}],
                enable_speex_noise_suppression=False,
                vad_threshold=0.5
            )
            
            logger.info(f"✅ {keyword} model loaded successfully")
            logger.info(f"   Model type: {type(model)}")
            logger.info(f"   Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
            
            loaded_models[keyword] = model
            
        except Exception as e:
            logger.error(f"❌ Failed to load {keyword} model: {e}")
    
    return loaded_models

def test_audio_formats(openwakeword, loaded_models):
    """Test 4: Test different audio formats and preprocessing."""
    logger.info("="*60)
    logger.info("TEST 4: Audio Format Testing")
    logger.info("="*60)
    
    if not loaded_models:
        logger.error("❌ No models loaded for testing")
        return False
    
    # Test different audio types
    test_audios = {
        'silence': np.zeros(1280, dtype=np.float32),
        'noise': np.random.randn(1280).astype(np.float32) * 0.1,
        'sine_1khz': (np.sin(2 * np.pi * 1000 * np.linspace(0, 0.08, 1280)) * 0.1).astype(np.float32),
        'sine_500hz': (np.sin(2 * np.pi * 500 * np.linspace(0, 0.08, 1280)) * 0.1).astype(np.float32),
        'loud_sine': (np.sin(2 * np.pi * 1000 * np.linspace(0, 0.08, 1280)) * 0.5).astype(np.float32),
    }
    
    # Test int16 conversion and normalization
    int16_audios = {
        'int16_silence': np.zeros(1280, dtype=np.int16),
        'int16_noise': (np.random.randn(1280) * 1000).astype(np.int16),
        'int16_sine': (np.sin(2 * np.pi * 1000 * np.linspace(0, 0.08, 1280)) * 5000).astype(np.int16),
    }
    
    # Convert int16 to float32 (like the main code does)
    for name, audio in int16_audios.items():
        float_audio = audio.astype(np.float32) / 32768.0
        test_audios[f"converted_{name}"] = float_audio
    
    # Test each model with each audio type
    for model_name, model in loaded_models.items():
        logger.info(f"🔍 Testing {model_name} model...")
        
        for audio_name, audio in test_audios.items():
            try:
                logger.info(f"   Testing {audio_name}...")
                logger.info(f"     Audio: shape={audio.shape}, dtype={audio.dtype}")
                logger.info(f"     Audio: min={np.min(audio):.6f}, max={np.max(audio):.6f}, RMS={np.sqrt(np.mean(audio**2)):.6f}")
                
                predictions = model.predict(audio)
                
                logger.info(f"     Prediction type: {type(predictions)}")
                logger.info(f"     Prediction content: {predictions}")
                
                if isinstance(predictions, dict):
                    logger.info(f"     Prediction keys: {list(predictions.keys())}")
                    logger.info(f"     Prediction values: {[f'{k}: {v:.6f}' for k, v in predictions.items()]}")
                    
                    # Try to extract confidence
                    confidence = predictions.get(model_name, 0.0)
                    if confidence == 0.0 and predictions:
                        first_key = list(predictions.keys())[0]
                        confidence = predictions[first_key]
                        logger.info(f"     Using fallback key '{first_key}': {confidence:.6f}")
                    else:
                        logger.info(f"     Confidence for '{model_name}': {confidence:.6f}")
                        
                elif isinstance(predictions, (list, tuple)):
                    confidence = float(predictions[0]) if predictions else 0.0
                    logger.info(f"     Confidence from list: {confidence:.6f}")
                else:
                    confidence = float(predictions)
                    logger.info(f"     Confidence from scalar: {confidence:.6f}")
                
                # Check if confidence is always 0.0000
                if confidence == 0.0:
                    logger.warning(f"     ⚠️ ZERO CONFIDENCE DETECTED!")
                elif confidence > 0.0:
                    logger.info(f"     ✅ Non-zero confidence: {confidence:.6f}")
                
            except Exception as e:
                logger.error(f"     ❌ Error testing {audio_name}: {e}")
    
    return True

def test_class_mapping(openwakeword, loaded_models):
    """Test 5: Test different class mapping configurations."""
    logger.info("="*60)
    logger.info("TEST 5: Class Mapping Testing")
    logger.info("="*60)
    
    if not loaded_models:
        logger.error("❌ No models loaded for testing")
        return False
    
    # Test different class mapping configurations
    test_mappings = [
        {0: 'hey_jarvis'},
        {0: 'alexa'},
        {0: 'test_wakeword'},
        {1: 'hey_jarvis'},  # Try different class index
        {0: 'hey_jarvis', 1: 'alexa'},  # Multiple mappings
    ]
    
    # Use alexa model for testing
    if 'alexa' not in loaded_models:
        logger.error("❌ Alexa model not available for class mapping test")
        return False
    
    model_path = openwakeword.models['alexa']['model_path']
    test_audio = np.random.randn(1280).astype(np.float32) * 0.1
    
    for i, mapping in enumerate(test_mappings):
        try:
            logger.info(f"🔍 Testing class mapping {i+1}: {mapping}")
            
            model = openwakeword.Model(
                wakeword_model_paths=[model_path],
                class_mapping_dicts=[mapping],
                enable_speex_noise_suppression=False,
                vad_threshold=0.5
            )
            
            predictions = model.predict(test_audio)
            logger.info(f"   Prediction: {predictions}")
            
            if isinstance(predictions, dict):
                logger.info(f"   Available keys: {list(predictions.keys())}")
                for key, value in predictions.items():
                    logger.info(f"   {key}: {value:.6f}")
            
        except Exception as e:
            logger.error(f"   ❌ Error with mapping {mapping}: {e}")
    
    return True

def test_frame_lengths(openwakeword, loaded_models):
    """Test 6: Test different frame lengths."""
    logger.info("="*60)
    logger.info("TEST 6: Frame Length Testing")
    logger.info("="*60)
    
    if not loaded_models:
        logger.error("❌ No models loaded for testing")
        return False
    
    # Test different frame lengths
    frame_lengths = [512, 1024, 1280, 1600, 2048]
    
    # Use alexa model for testing
    if 'alexa' not in loaded_models:
        logger.error("❌ Alexa model not available for frame length test")
        return False
    
    model_path = openwakeword.models['alexa']['model_path']
    
    for frame_length in frame_lengths:
        try:
            logger.info(f"🔍 Testing frame length: {frame_length}")
            
            model = openwakeword.Model(
                wakeword_model_paths=[model_path],
                class_mapping_dicts=[{0: 'alexa'}],
                enable_speex_noise_suppression=False,
                vad_threshold=0.5
            )
            
            # Create test audio with this frame length
            test_audio = np.random.randn(frame_length).astype(np.float32) * 0.1
            
            predictions = model.predict(test_audio)
            logger.info(f"   Prediction: {predictions}")
            
            if isinstance(predictions, dict):
                confidence = predictions.get('alexa', 0.0)
                logger.info(f"   Confidence: {confidence:.6f}")
            
        except Exception as e:
            logger.error(f"   ❌ Error with frame length {frame_length}: {e}")
    
    return True

def test_config_loading():
    """Test 7: Test configuration loading and validation."""
    logger.info("="*60)
    logger.info("TEST 7: Configuration Testing")
    logger.info("="*60)
    
    try:
        config_path = Path(__file__).parent / "config.yaml"
        if not config_path.exists():
            logger.error(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"   Wake word engine: {config['wake_word']['engine']}")
        logger.info(f"   Keyword: {config['wake_word']['keyword']}")
        logger.info(f"   Threshold: {config['wake_word']['threshold']}")
        logger.info(f"   Custom model path: {config['wake_word'].get('custom_model_path', 'None')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading configuration: {e}")
        return False

def main():
    """Run all debugging tests."""
    logger.info("🔍 COMPREHENSIVE OPENWAKEWORD DEBUGGING")
    logger.info("="*60)
    
    # Test 1: Installation
    success, openwakeword = test_openwakeword_installation()
    if not success:
        logger.error("❌ OpenWakeWord installation test failed - stopping")
        return False
    
    # Test 2: Model availability
    test_model_availability(openwakeword)
    
    # Test 3: Model loading
    loaded_models = test_model_loading(openwakeword)
    
    # Test 4: Audio formats
    test_audio_formats(openwakeword, loaded_models)
    
    # Test 5: Class mapping
    test_class_mapping(openwakeword, loaded_models)
    
    # Test 6: Frame lengths
    test_frame_lengths(openwakeword, loaded_models)
    
    # Test 7: Configuration
    test_config_loading()
    
    logger.info("="*60)
    logger.info("🎉 DEBUGGING COMPLETED")
    logger.info("="*60)
    logger.info("Check the output above to identify the root cause of the 0.0000 confidence issue.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 