#!/usr/bin/env python3
"""
Wake Word Monitor - New Implementation
Part of the ORAC Voice-Control Architecture refactoring
Handles wake word detection with configuration-driven model management
"""

import logging
import os
import glob
from typing import Dict, Any, List, Optional
from pathlib import Path
from settings_manager import get_settings_manager

logger = logging.getLogger(__name__)


class WakeWordMonitor_new:
    """
    New wake word monitor implementation with configuration-driven model management.
    Focuses on settings management and model discovery without loading models yet.
    """
    
    def __init__(self):
        """Initialize the new wake word monitor."""
        self.settings_manager = get_settings_manager()
        self.available_models = []
        self.model_configs = {}
        self.global_settings = {}
        
        # Load configuration
        self._load_configuration()
        self._discover_available_models()
        self._build_model_configs()
        
        logger.info("✅ WakeWordMonitor_new initialized")
        logger.info(f"📊 Discovered {len(self.available_models)} available models")
    
    def _load_configuration(self):
        """Load global configuration settings."""
        logger.info("🔧 Loading wake word configuration...")
        
        # Load global settings
        self.global_settings = {
            'cooldown': self.settings_manager.get('wake_word.cooldown', 1.5),
            'engine': self.settings_manager.get('wake_word.engine', 'openwakeword'),
            'model_path': self.settings_manager.get('wake_word.model_path', ''),
            'custom_model_path': self.settings_manager.get('wake_word.custom_model_path', ''),
            'keyword': self.settings_manager.get('wake_word.keyword', 'hey_jarvis')
        }
        
        logger.info(f"🔧 Global Configuration:")
        logger.info(f"   Cooldown: {self.global_settings['cooldown']}s")
        logger.info(f"   Engine: {self.global_settings['engine']}")
        logger.info(f"   Keyword: {self.global_settings['keyword']}")
        logger.info(f"   Model Path: {self.global_settings['model_path']}")
        logger.info(f"   Custom Model Path: {self.global_settings['custom_model_path']}")
    
    def _discover_available_models(self):
        """Discover available wake word models in the custom models directory."""
        logger.info("🔍 Discovering available wake word models...")
        
        # Define the custom models directory
        custom_models_dir = Path("third_party/openwakeword/custom_models")
        
        if not custom_models_dir.exists():
            logger.warning(f"⚠️ Custom models directory not found: {custom_models_dir}")
            return
        
        # Look for .onnx and .tflite files
        model_files = []
        model_files.extend(glob.glob(str(custom_models_dir / "*.onnx")))
        model_files.extend(glob.glob(str(custom_models_dir / "*.tflite")))
        
        # Extract model names (remove extension and path)
        discovered_models = set()
        for model_file in model_files:
            model_name = Path(model_file).stem  # Remove extension
            discovered_models.add(model_name)
        
        self.available_models = sorted(list(discovered_models))
        
        logger.info(f"🔍 Discovered models: {self.available_models}")
        
        # Log model file details
        for model_name in self.available_models:
            onnx_file = custom_models_dir / f"{model_name}.onnx"
            tflite_file = custom_models_dir / f"{model_name}.tflite"
            
            logger.info(f"📁 Model '{model_name}':")
            logger.info(f"   ONNX: {'✅' if onnx_file.exists() else '❌'} {onnx_file}")
            logger.info(f"   TFLite: {'✅' if tflite_file.exists() else '❌'} {tflite_file}")
    
    def _build_model_configs(self):
        """Build configuration for each discovered model."""
        logger.info("🔧 Building model configurations...")
        
        for model_name in self.available_models:
            # Get model-specific settings from settings manager
            sensitivity = self.settings_manager.get_model_sensitivity(model_name, 0.8)
            threshold = self.settings_manager.get_model_threshold(model_name, 0.3)
            api_url = self.settings_manager.get_model_api_url(model_name, "https://api.example.com/webhook")
            
            self.model_configs[model_name] = {
                'name': model_name,
                'sensitivity': sensitivity,
                'threshold': threshold,
                'api_url': api_url,
                'file_paths': self._get_model_file_paths(model_name)
            }
            
            logger.info(f"🔧 Model '{model_name}' configuration:")
            logger.info(f"   Sensitivity: {sensitivity:.3f}")
            logger.info(f"   Threshold: {threshold:.3f}")
            logger.info(f"   API URL: {api_url}")
            logger.info(f"   Files: {self.model_configs[model_name]['file_paths']}")
    
    def _get_model_file_paths(self, model_name: str) -> Dict[str, str]:
        """Get file paths for a specific model."""
        custom_models_dir = Path("third_party/openwakeword/custom_models")
        
        return {
            'onnx': str(custom_models_dir / f"{model_name}.onnx"),
            'tflite': str(custom_models_dir / f"{model_name}.tflite")
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return self.available_models.copy()
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model."""
        return self.model_configs.get(model_name)
    
    def get_global_settings(self) -> Dict[str, Any]:
        """Get global wake word settings."""
        return self.global_settings.copy()
    
    def get_model_sensitivity(self, model_name: str) -> float:
        """Get sensitivity for a specific model."""
        config = self.get_model_config(model_name)
        return config['sensitivity'] if config else 0.8
    
    def get_model_threshold(self, model_name: str) -> float:
        """Get threshold for a specific model."""
        config = self.get_model_config(model_name)
        return config['threshold'] if config else 0.3
    
    def get_model_api_url(self, model_name: str) -> str:
        """Get API URL for a specific model."""
        config = self.get_model_config(model_name)
        return config['api_url'] if config else "https://api.example.com/webhook"
    
    def get_cooldown_seconds(self) -> float:
        """Get global cooldown setting."""
        return self.global_settings['cooldown']
    

    
    def get_engine_type(self) -> str:
        """Get the wake word engine type."""
        return self.global_settings['engine']
    
    def get_keyword(self) -> str:
        """Get the wake word keyword."""
        return self.global_settings['keyword']
    
    def print_configuration_summary(self):
        """Print a summary of the current configuration."""
        logger.info("📊 Wake Word Monitor Configuration Summary:")
        logger.info("=" * 50)
        
        # Global settings
        logger.info("🌐 Global Settings:")
        logger.info(f"   Engine: {self.get_engine_type()}")
        logger.info(f"   Keyword: {self.get_keyword()}")
        logger.info(f"   Cooldown: {self.get_cooldown_seconds()}s")
        
        # Model configurations
        logger.info(f"\n🤖 Model Configurations ({len(self.available_models)} models):")
        for model_name in self.available_models:
            config = self.get_model_config(model_name)
            logger.info(f"   📁 {model_name}:")
            logger.info(f"      Sensitivity: {config['sensitivity']:.3f}")
            logger.info(f"      Threshold: {config['threshold']:.3f}")
            logger.info(f"      API URL: {config['api_url']}")
            logger.info(f"      Files: {list(config['file_paths'].keys())}")
        
        logger.info("=" * 50)


def create_wake_word_monitor_new() -> WakeWordMonitor_new:
    """Factory function to create a new wake word monitor instance."""
    return WakeWordMonitor_new()


if __name__ == "__main__":
    # Test the new monitor
    monitor = create_wake_word_monitor_new()
    monitor.print_configuration_summary() 