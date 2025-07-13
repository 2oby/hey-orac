#!/usr/bin/env python3
"""
Wake Word Monitor - New Implementation
Part of the ORAC Voice-Control Architecture refactoring
Handles wake word detection with configuration-driven model management
"""

import logging
from typing import Dict, Any, List, Optional
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
    
    def _build_model_configs(self):
        """Build configuration for each discovered model."""
        logger.info("🔧 Building model configurations...")
        
        # Get available models from settings manager
        self.available_models = self.settings_manager.get_available_models()
        
        for model_name in self.available_models:
            # Get complete model config from settings manager
            model_config = self.settings_manager.get_model_config(model_name)
            
            if model_config:
                self.model_configs[model_name] = model_config
                
                logger.info(f"🔧 Model '{model_name}' configuration:")
                logger.info(f"   Sensitivity: {model_config['sensitivity']:.3f}")
                logger.info(f"   Threshold: {model_config['threshold']:.3f}")
                logger.info(f"   API URL: {model_config['api_url']}")
                logger.info(f"   Active: {model_config['active']}")
                logger.info(f"   Files: {list(model_config['file_paths'].keys())}")
    
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
    
    def get_model_active(self, model_name: str) -> bool:
        """Get active state for a specific model."""
        config = self.get_model_config(model_name)
        return config['active'] if config else False
    
    def get_active_models(self) -> List[str]:
        """Get list of currently active models."""
        return self.settings_manager.get_active_models()
    
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
        
        # Active models
        active_models = self.get_active_models()
        logger.info(f"\n🎯 Active Models ({len(active_models)} models):")
        for model_name in active_models:
            config = self.get_model_config(model_name)
            if config:
                logger.info(f"   ✅ {model_name}:")
                logger.info(f"      Sensitivity: {config['sensitivity']:.3f}")
                logger.info(f"      Threshold: {config['threshold']:.3f}")
                logger.info(f"      API URL: {config['api_url']}")
        
        # All model configurations
        logger.info(f"\n🤖 All Model Configurations ({len(self.available_models)} models):")
        for model_name in self.available_models:
            config = self.get_model_config(model_name)
            status = "✅ ACTIVE" if config['active'] else "❌ INACTIVE"
            logger.info(f"   📁 {model_name} ({status}):")
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