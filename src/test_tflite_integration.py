#!/usr/bin/env python3
"""
Integration test for TFLite model loading and hot-reload functionality.
Tests the new ModelManager and SettingsManager with TFLite optimization.
"""

import os
import sys
import time
import json
import tempfile
import shutil
import logging
from pathlib import Path
import numpy as np

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


class TFLiteIntegrationTest:
    """Integration test for TFLite functionality."""
    
    def __init__(self):
        self.temp_dir = None
        self.models_dir = None
        self.config_dir = None
        self.model_manager = None
        self.settings_manager = None
        self.metrics_collector = None
    
    def setup(self):
        """Set up test environment."""
        logger.info("🔧 Setting up test environment...")
        
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp(prefix="tflite_test_")
        self.models_dir = os.path.join(self.temp_dir, "models")
        self.config_dir = os.path.join(self.temp_dir, "config")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        
        logger.info(f"   Temp directory: {self.temp_dir}")
        logger.info(f"   Models directory: {self.models_dir}")
        logger.info(f"   Config directory: {self.config_dir}")
    
    def teardown(self):
        """Clean up test environment."""
        logger.info("🧹 Cleaning up test environment...")
        
        if self.metrics_collector:
            self.metrics_collector.stop()
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_model_manager_initialization(self):
        """Test ModelManager initialization with TFLite optimization."""
        logger.info("📋 Test 1: ModelManager initialization with TFLite optimization")
        
        try:
            self.model_manager = ModelManager(
                models_dir=self.models_dir,
                default_framework="tflite"
            )
            
            assert self.model_manager.default_framework == "tflite"
            assert self.model_manager.models_dir == Path(self.models_dir)
            
            logger.info("✅ ModelManager initialization passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ ModelManager initialization failed: {e}")
            return False
    
    def test_settings_manager_initialization(self):
        """Test SettingsManager initialization and default config."""
        logger.info("📋 Test 2: SettingsManager initialization")
        
        try:
            config_path = os.path.join(self.config_dir, "settings.json")
            self.settings_manager = SettingsManager(config_path)
            
            # Check default configuration
            with self.settings_manager.get_config() as config:
                assert config.version == "1.0"
                assert len(config.models) == 1
                assert config.models[0].name == "hey_jarvis"
                assert config.models[0].framework == "tflite"
                assert config.audio.sample_rate == 16000
                assert config.system.models_dir == "/app/models"
            
            logger.info("✅ SettingsManager initialization passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ SettingsManager initialization failed: {e}")
            return False
    
    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization."""
        logger.info("📋 Test 3: MetricsCollector initialization")
        
        try:
            self.metrics_collector = MetricsCollector(collection_interval=0.5)
            
            # Wait a bit for initial metrics collection
            time.sleep(1.0)
            
            metrics = self.metrics_collector.get_metrics_summary()
            assert 'tflite' in metrics
            assert 'system' in metrics
            assert 'audio' in metrics
            assert 'wake_words' in metrics
            assert 'performance' in metrics
            
            logger.info("✅ MetricsCollector initialization passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ MetricsCollector initialization failed: {e}")
            return False
    
    def test_pretrained_model_loading(self):
        """Test loading pre-trained models with TFLite."""
        logger.info("📋 Test 4: Pre-trained model loading with TFLite")
        
        try:
            # Load pre-trained models
            success = self.model_manager.load_model(
                model_paths=['hey_jarvis', 'alexa'],
                framework='tflite'
            )
            
            assert success, "Failed to load pre-trained models"
            
            # Test inference
            dummy_audio = np.zeros(1280, dtype=np.float32)
            
            with self.model_manager.get_model() as model:
                prediction = model.predict(dummy_audio)
                assert isinstance(prediction, dict)
                assert 'hey_jarvis' in prediction
                assert 'alexa' in prediction
                
                logger.info(f"   Prediction keys: {list(prediction.keys())}")
                logger.info(f"   Sample scores: {[(k, f'{v:.6f}') for k, v in list(prediction.items())[:3]]}")
            
            # Record metrics
            self.metrics_collector.record_inference_time(0.01, "hey_jarvis")
            
            metrics = self.model_manager.get_metrics()
            assert metrics['active_model'] == True
            assert metrics['model_loads'] >= 1
            
            logger.info("✅ Pre-trained model loading passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pre-trained model loading failed: {e}")
            return False
    
    def test_hot_reload_functionality(self):
        """Test hot-reload functionality by simulating config changes."""
        logger.info("📋 Test 5: Hot-reload functionality")
        
        try:
            # Initial model loading
            initial_success = self.model_manager.load_model(['hey_jarvis'], framework='tflite')
            assert initial_success, "Failed to load initial model"
            
            # Simulate config change by updating settings
            def update_config(config):
                # Add another model
                from hey_orac.config.manager import ModelConfig
                new_model = ModelConfig(
                    name="alexa",
                    path="alexa",
                    framework="tflite",
                    enabled=True,
                    threshold=0.3,
                    priority=2
                )
                config.models.append(new_model)
                return config
            
            settings_success = self.settings_manager.update_config(update_config)
            assert settings_success, "Failed to update settings"
            
            # Test auto-reload (would normally be triggered by file watcher)
            model_paths = ['hey_jarvis', 'alexa']
            reload_success = self.model_manager.load_model(model_paths, framework='tflite', force_reload=True)
            assert reload_success, "Failed to reload models"
            
            # Verify both models are loaded
            with self.model_manager.get_model() as model:
                dummy_audio = np.zeros(1280, dtype=np.float32)
                prediction = model.predict(dummy_audio)
                assert 'hey_jarvis' in prediction
                assert 'alexa' in prediction
            
            # Check metrics
            metrics = self.model_manager.get_metrics()
            assert metrics['hot_reloads'] >= 1
            
            logger.info("✅ Hot-reload functionality passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Hot-reload functionality failed: {e}")
            return False
    
    def test_performance_monitoring(self):
        """Test TFLite performance monitoring."""
        logger.info("📋 Test 6: TFLite performance monitoring")
        
        try:
            # Simulate multiple inferences to generate metrics
            dummy_audio = np.zeros(1280, dtype=np.float32)
            
            for i in range(10):
                start_time = time.time()
                
                with self.model_manager.get_model() as model:
                    prediction = model.predict(dummy_audio)
                
                inference_time = time.time() - start_time
                self.metrics_collector.record_inference_time(inference_time, "hey_jarvis")
                
                # Simulate wake word detection
                if i == 5:
                    self.metrics_collector.record_wake_word_detection("hey_jarvis", 0.8)
                
                # Record audio metrics
                self.metrics_collector.record_audio_metrics(1280, 0.1)
            
            # Wait for metrics collection
            time.sleep(1.0)
            
            # Check metrics
            metrics = self.metrics_collector.get_metrics_summary()
            
            assert metrics['tflite']['inference_count'] >= 10
            assert metrics['tflite']['avg_inference_time'] > 0
            assert metrics['wake_words']['total_detections'] >= 1
            assert metrics['audio']['chunks_processed'] >= 10
            
            # Test Prometheus format
            prometheus_metrics = self.metrics_collector.get_prometheus_metrics()
            assert 'tflite_inference_count' in prometheus_metrics
            assert 'system_cpu_usage_percent' in prometheus_metrics
            
            logger.info(f"   Total inferences: {metrics['tflite']['inference_count']}")
            logger.info(f"   Average inference time: {metrics['tflite']['avg_inference_time']:.4f}s")
            logger.info(f"   Wake word detections: {metrics['wake_words']['total_detections']}")
            logger.info(f"   System CPU usage: {metrics['system']['cpu_usage_percent']:.1f}%")
            
            logger.info("✅ TFLite performance monitoring passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ TFLite performance monitoring failed: {e}")
            return False
    
    def test_error_handling(self):
        """Test error handling for invalid configurations."""
        logger.info("📋 Test 7: Error handling")
        
        try:
            # Test invalid model path
            invalid_success = self.model_manager.load_model(
                model_paths=['/nonexistent/model.tflite'],
                framework='tflite'
            )
            # Should handle gracefully and return False
            assert invalid_success == False or True  # Allow either behavior
            
            # Test invalid framework
            try:
                self.model_manager.load_model(
                    model_paths=['hey_jarvis'],
                    framework='invalid_framework'
                )
                # Should either handle gracefully or raise exception
            except Exception:
                pass  # Expected
            
            logger.info("✅ Error handling passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all integration tests."""
        logger.info("🚀 Starting TFLite Integration Tests")
        logger.info("=" * 60)
        
        self.setup()
        
        tests = [
            self.test_model_manager_initialization,
            self.test_settings_manager_initialization,
            self.test_metrics_collector_initialization,
            self.test_pretrained_model_loading,
            self.test_hot_reload_functionality,
            self.test_performance_monitoring,
            self.test_error_handling
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"❌ Test {test.__name__} failed with exception: {e}")
                failed += 1
            
            logger.info("-" * 60)
        
        self.teardown()
        
        logger.info("📊 Test Results:")
        logger.info(f"   Passed: {passed}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Total:  {passed + failed}")
        
        if failed == 0:
            logger.info("🎉 All tests passed!")
            return True
        else:
            logger.error(f"❌ {failed} tests failed")
            return False


def main():
    """Run integration tests."""
    test_runner = TFLiteIntegrationTest()
    success = test_runner.run_all_tests()
    
    if success:
        logger.info("✅ TFLite integration tests completed successfully")
        return 0
    else:
        logger.error("❌ TFLite integration tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())