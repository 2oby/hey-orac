#!/usr/bin/env python3
"""
Test script to validate the threshold fix for false positives
"""

import logging
import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_threshold_fix():
    """Test that the threshold fix prevents false positives."""
    try:
        logger.info("🚀 Testing threshold fix...")
        
        # Test with settings manager
        from settings_manager import get_settings_manager
        settings_manager = get_settings_manager()
        
        # Check default thresholds
        global_threshold = settings_manager.get('wake_word.threshold', 0.0)
        logger.info(f"📊 Global threshold: {global_threshold}")
        
        # Check model-specific thresholds
        models = settings_manager.get('wake_word.models', {})
        for model_name, config in models.items():
            threshold = config.get('threshold', 0.0)
            sensitivity = config.get('sensitivity', 0.0)
            active = config.get('active', False)
            logger.info(f"   {model_name}: threshold={threshold:.3f}, sensitivity={sensitivity:.3f}, active={active}")
        
        # Test wake word monitor
        logger.info("\n🔧 Testing wake word monitor with new settings...")
        from wake_word_monitor_new import WakeWordMonitor_new
        
        monitor = WakeWordMonitor_new()
        
        # Print configuration summary
        monitor.print_configuration_summary()
        
        # Test with quiet audio (should not trigger with proper thresholds)
        logger.info("\n🔍 Testing with quiet audio patterns...")
        
        # Test patterns that should NOT trigger detection
        test_patterns = [
            ("silence", np.zeros(1280, dtype=np.int16)),
            ("very_quiet_noise", np.random.randint(-10, 10, 1280, dtype=np.int16)),
            ("quiet_noise", np.random.randint(-100, 100, 1280, dtype=np.int16)),
            ("room_noise", np.random.randint(-500, 500, 1280, dtype=np.int16))
        ]
        
        total_false_positives = 0
        
        for pattern_name, audio_data in test_patterns:
            logger.info(f"\n   Testing {pattern_name}...")
            detections = 0
            
            # Test 20 chunks of each pattern
            for i in range(20):
                if monitor.process_audio(audio_data):
                    detections += 1
                    logger.warning(f"     ⚠️ FALSE POSITIVE detected in chunk {i+1}")
            
            logger.info(f"     Result: {detections}/20 detections")
            total_false_positives += detections
        
        # Evaluate results
        logger.info(f"\n📈 EVALUATION RESULTS:")
        logger.info(f"   Total false positives: {total_false_positives}")
        
        if total_false_positives == 0:
            logger.info("✅ EXCELLENT: No false positives detected!")
            logger.info("   The threshold fix appears to be working correctly")
        elif total_false_positives < 5:
            logger.info("✅ GOOD: Very few false positives detected")
            logger.info("   The threshold fix is working well")
        elif total_false_positives < 20:
            logger.warning("⚠️ MODERATE: Some false positives detected")
            logger.warning("   Consider increasing thresholds further")
        else:
            logger.error("❌ POOR: Many false positives detected")
            logger.error("   The threshold fix may need more adjustment")
        
        # Test with more realistic speech-like audio
        logger.info(f"\n🎤 Testing with speech-like audio (should have higher confidence)...")
        
        # Generate speech-like audio
        duration = 1280 / 16000  # 80ms
        t = np.linspace(0, duration, 1280, endpoint=False)
        
        speech_audio = (
            0.3 * np.sin(2 * np.pi * 200 * t) +      # Fundamental frequency
            0.2 * np.sin(2 * np.pi * 400 * t) +      # First harmonic
            0.1 * np.sin(2 * np.pi * 600 * t) +      # Second harmonic
            0.05 * np.random.normal(0, 0.1, 1280)    # Add some noise
        )
        
        # Convert to int16 with reasonable amplitude
        speech_audio = (speech_audio * 8000).astype(np.int16)
        
        speech_detections = 0
        for i in range(10):
            if monitor.process_audio(speech_audio):
                speech_detections += 1
        
        logger.info(f"   Speech-like audio detections: {speech_detections}/10")
        
        # Test threshold adjustment
        logger.info(f"\n🔧 Testing threshold adjustment...")
        
        active_models = monitor.get_active_models()
        if active_models:
            test_model = active_models[0]
            original_threshold = monitor.get_model_threshold(test_model)
            
            # Test with higher threshold
            logger.info(f"   Original threshold for {test_model}: {original_threshold:.3f}")
            
            # Note: We can't directly update thresholds without settings manager integration
            # But we can test if the system is responsive to configuration changes
            
        logger.info("\n✅ Threshold fix testing completed")
        return total_false_positives < 10  # Success if fewer than 10 false positives
        
    except Exception as e:
        logger.error(f"❌ Error testing threshold fix: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_threshold_fix()
    sys.exit(0 if success else 1)