#!/usr/bin/env python3
"""
Test script to verify custom TFLite model loading (Hay--compUta_v_lrg.tflite).
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hey_orac.models.manager import ModelManager
from hey_orac.config.manager import SettingsManager
from hey_orac.metrics.collector import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_custom_model_loading():
    """Test loading the custom TFLite model."""
    logger.info("🧪 Testing custom TFLite model loading...")
    
    # Define model path
    model_path = "/Users/2oby/pCloud Box/Projects/WakeWordTest/models/Hay--compUta_v_lrg.tflite"
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        return False
    
    logger.info(f"✅ Found model file: {Path(model_path).name}")
    logger.info(f"📊 Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    try:
        # Initialize ModelManager
        models_dir = "/Users/2oby/pCloud Box/Projects/WakeWordTest/models"
        model_manager = ModelManager(models_dir=models_dir, default_framework="tflite")
        
        # Load the custom model
        logger.info("🔄 Loading custom model...")
        success = model_manager.load_model(
            model_paths=[model_path],
            framework="tflite"
        )
        
        if not success:
            logger.error("❌ Failed to load custom model")
            return False
        
        logger.info("✅ Custom model loaded successfully")
        
        # Test inference
        logger.info("🧪 Testing model inference...")
        dummy_audio = np.zeros(1280, dtype=np.float32)
        
        with model_manager.get_model() as model:
            prediction = model.predict(dummy_audio)
            
            logger.info(f"✅ Inference successful!")
            logger.info(f"📊 Available models: {list(prediction.keys())}")
            
            # Log all predictions
            for model_name, score in prediction.items():
                logger.info(f"   {model_name}: {score:.6f}")
        
        # Get metrics
        metrics = model_manager.get_metrics()
        logger.info(f"📈 Metrics: {metrics['model_loads']} loads, {metrics['total_inferences']} inferences")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during model testing: {e}")
        return False


def test_settings_manager():
    """Test SettingsManager with custom model configuration."""
    logger.info("🧪 Testing SettingsManager with custom model...")
    
    try:
        # Initialize SettingsManager
        config_path = "/Users/2oby/pCloud Box/Projects/WakeWordTest/config/settings.json"
        settings_manager = SettingsManager(config_path)
        
        # Check configuration
        with settings_manager.get_config() as config:
            logger.info(f"✅ Configuration loaded successfully")
            logger.info(f"📊 Version: {config.version}")
            logger.info(f"🎯 Models:")
            
            for model in config.models:
                status = "✅ ENABLED" if model.enabled else "❌ DISABLED"
                logger.info(f"   {model.name}: {model.path} ({model.framework}) {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during settings testing: {e}")
        return False


def test_metrics_collector():
    """Test MetricsCollector functionality."""
    logger.info("🧪 Testing MetricsCollector...")
    
    try:
        # Initialize MetricsCollector
        metrics_collector = MetricsCollector(collection_interval=0.5)
        
        # Record some test metrics
        metrics_collector.record_inference_time(0.05, "hay_computa")
        metrics_collector.record_wake_word_detection("hay_computa", 0.8)
        metrics_collector.record_model_load(0.5, "/app/models/Hay--compUta_v_lrg.tflite", 5242880, "tflite")
        
        # Get metrics summary
        metrics = metrics_collector.get_metrics_summary()
        
        logger.info(f"✅ Metrics collected successfully")
        logger.info(f"📊 TFLite metrics:")
        logger.info(f"   Inference count: {metrics['tflite']['inference_count']}")
        logger.info(f"   Avg inference time: {metrics['tflite']['avg_inference_time']:.4f}s")
        logger.info(f"   Wake word detections: {metrics['wake_words']['total_detections']}")
        
        # Test Prometheus format
        prometheus_metrics = metrics_collector.get_prometheus_metrics()
        logger.info(f"📈 Prometheus metrics available: {len(prometheus_metrics.split(chr(10)))} lines")
        
        # Stop metrics collector
        metrics_collector.stop()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during metrics testing: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting Custom TFLite Model Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Custom Model Loading", test_custom_model_loading),
        ("Settings Manager", test_settings_manager),
        ("Metrics Collector", test_metrics_collector)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"📋 Running test: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
        
        logger.info("-" * 60)
    
    logger.info("📊 Test Results:")
    logger.info(f"   Passed: {passed}")
    logger.info(f"   Failed: {failed}")
    logger.info(f"   Total:  {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"❌ {failed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())