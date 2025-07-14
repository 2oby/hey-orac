#!/usr/bin/env python3
"""
Test script to verify OpenWakeWord with built-in models
This helps establish baseline behavior before testing custom models
"""

import logging
import sys
import time
import numpy as np
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_builtin_models():
    """Test OpenWakeWord with built-in models to establish baseline."""
    try:
        logger.info("🚀 Testing OpenWakeWord with built-in models...")
        
        # Import the engine
        from wake_word_engines.openwakeword_engine import OpenWakeWordEngine
        
        # Create engine instance
        engine = OpenWakeWordEngine()
        
        # Test configuration with proper thresholds
        test_config = {
            'sensitivity': 0.5,  # Standard sensitivity
            'threshold': 0.5,    # Standard threshold for built-in models
            'keyword': 'alexa'   # Test with a built-in wake word
        }
        
        logger.info("🔧 Testing with configuration:")
        logger.info(f"   Sensitivity: {test_config['sensitivity']}")
        logger.info(f"   Threshold: {test_config['threshold']}")
        logger.info(f"   Keyword: {test_config['keyword']}")
        
        # Initialize engine
        if not engine.initialize(test_config):
            logger.error("❌ Failed to initialize engine with built-in models")
            return False
        
        logger.info("✅ Engine initialized successfully")
        logger.info(f"   Wake word: {engine.get_wake_word_name()}")
        logger.info(f"   Sample rate: {engine.get_sample_rate()}")
        logger.info(f"   Frame length: {engine.get_frame_length()}")
        logger.info(f"   Ready: {engine.is_ready()}")
        
        # Test with different audio patterns
        logger.info("\n📊 Testing with various audio patterns...")
        
        test_patterns = [
            ("silence", np.zeros(1280, dtype=np.int16)),
            ("low_noise", np.random.randint(-100, 100, 1280, dtype=np.int16)),
            ("medium_noise", np.random.randint(-1000, 1000, 1280, dtype=np.int16)),
            ("speech_like", generate_speech_like_audio()),
            ("loud_noise", np.random.randint(-10000, 10000, 1280, dtype=np.int16))
        ]
        
        results = {}
        for pattern_name, audio_data in test_patterns:
            logger.info(f"\n🔍 Testing with {pattern_name}...")
            
            # Test multiple chunks to see confidence patterns
            confidences = []
            detections = 0
            
            for i in range(10):  # Test 10 chunks
                detection = engine.process_audio(audio_data)
                confidence = engine.get_latest_confidence()
                confidences.append(confidence)
                if detection:
                    detections += 1
                    logger.info(f"   🎯 Detection #{detections} with confidence {confidence:.6f}")
            
            avg_confidence = np.mean(confidences)
            max_confidence = np.max(confidences)
            
            logger.info(f"   Results for {pattern_name}:")
            logger.info(f"     Detections: {detections}/10")
            logger.info(f"     Avg confidence: {avg_confidence:.6f}")
            logger.info(f"     Max confidence: {max_confidence:.6f}")
            logger.info(f"     Confidence range: {np.min(confidences):.6f} - {np.max(confidences):.6f}")
            
            results[pattern_name] = {
                'detections': detections,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'confidences': confidences
            }
        
        # Analyze results
        logger.info("\n📈 ANALYSIS:")
        
        # Check if silence gives low confidence (should be close to 0)
        silence_confidence = results['silence']['max_confidence']
        if silence_confidence > 0.1:
            logger.warning(f"⚠️ Silence gives high confidence ({silence_confidence:.6f}) - potential issue!")
        else:
            logger.info(f"✅ Silence gives appropriate low confidence ({silence_confidence:.6f})")
        
        # Check if noise levels correlate with confidence
        noise_levels = ['low_noise', 'medium_noise', 'loud_noise']
        for noise in noise_levels:
            conf = results[noise]['max_confidence']
            detections = results[noise]['detections']
            logger.info(f"   {noise}: {detections} detections, max confidence {conf:.6f}")
        
        # Check for inappropriate detections
        total_detections = sum(r['detections'] for r in results.values())
        if total_detections > 5:  # Arbitrary threshold
            logger.warning(f"⚠️ Too many detections ({total_detections}) for test patterns!")
            logger.warning("   This suggests the threshold might be too low")
        else:
            logger.info(f"✅ Reasonable number of detections ({total_detections}) for test patterns")
        
        # Test threshold behavior
        logger.info(f"\n🔧 Testing threshold behavior...")
        original_threshold = engine.threshold
        
        # Test with higher threshold
        engine.update_threshold(0.8)
        high_thresh_detections = 0
        for i in range(5):
            if engine.process_audio(test_patterns[3][1]):  # speech_like
                high_thresh_detections += 1
        
        # Test with lower threshold  
        engine.update_threshold(0.1)
        low_thresh_detections = 0
        for i in range(5):
            if engine.process_audio(test_patterns[3][1]):  # speech_like
                low_thresh_detections += 1
        
        # Restore original threshold
        engine.update_threshold(original_threshold)
        
        logger.info(f"   High threshold (0.8): {high_thresh_detections}/5 detections")
        logger.info(f"   Low threshold (0.1): {low_thresh_detections}/5 detections")
        logger.info(f"   Original threshold ({original_threshold}): restored")
        
        if low_thresh_detections <= high_thresh_detections:
            logger.warning("⚠️ Threshold behavior seems inverted!")
        else:
            logger.info("✅ Threshold behavior is correct (lower threshold = more detections)")
        
        logger.info("\n🎯 RECOMMENDATIONS:")
        
        # Suggest appropriate thresholds
        silence_max = results['silence']['max_confidence']
        noise_max = max(results[noise]['max_confidence'] for noise in noise_levels)
        
        recommended_threshold = max(noise_max * 2, 0.3)  # At least 2x noise level, minimum 0.3
        logger.info(f"   Recommended threshold: {recommended_threshold:.3f}")
        logger.info(f"   Current threshold: {engine.threshold:.3f}")
        
        if engine.threshold < recommended_threshold:
            logger.warning(f"   ⚠️ Current threshold might be too low!")
            logger.warning(f"   Consider increasing to {recommended_threshold:.3f} to reduce false positives")
        
        logger.info("\n✅ Built-in model testing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing built-in models: {e}", exc_info=True)
        return False
    finally:
        if 'engine' in locals():
            engine.cleanup()

def generate_speech_like_audio() -> np.ndarray:
    """Generate audio that resembles speech patterns."""
    # Create a mix of frequencies typical in speech
    duration = 1280 / 16000  # 80ms
    t = np.linspace(0, duration, 1280, endpoint=False)
    
    # Mix of fundamental and harmonics typical in speech
    audio = (
        0.3 * np.sin(2 * np.pi * 150 * t) +      # Fundamental frequency
        0.2 * np.sin(2 * np.pi * 300 * t) +      # First harmonic
        0.1 * np.sin(2 * np.pi * 450 * t) +      # Second harmonic
        0.05 * np.sin(2 * np.pi * 1200 * t) +    # Higher frequency content
        0.05 * np.random.normal(0, 0.1, 1280)    # Add some noise
    )
    
    # Apply envelope to make it more speech-like
    envelope = np.exp(-t * 5)  # Decay envelope
    audio = audio * envelope
    
    # Convert to int16
    audio = (audio * 5000).astype(np.int16)
    return audio

if __name__ == "__main__":
    success = test_builtin_models()
    sys.exit(0 if success else 1)