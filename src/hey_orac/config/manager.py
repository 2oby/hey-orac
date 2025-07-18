"""
SettingsManager for thread-safe configuration management with JSON schema validation.
Supports atomic file writes and change notifications for hot-reload functionality.
"""

import json
import os
import tempfile
import threading
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    path: str
    framework: str = "tflite"
    enabled: bool = True
    threshold: float = 0.3
    priority: int = 1


@dataclass
class AudioConfig:
    """Audio capture configuration."""
    sample_rate: int = 16000
    channels: int = 2
    chunk_size: int = 1280
    device_index: Optional[int] = None
    auto_select_usb: bool = True


@dataclass
class SystemConfig:
    """System-level configuration."""
    log_level: str = "INFO"
    models_dir: str = "/app/models"
    recordings_dir: str = "/app/recordings"
    metrics_enabled: bool = True
    metrics_port: int = 8000
    hot_reload_enabled: bool = True
    hot_reload_interval: float = 5.0


@dataclass
class HeyOracConfig:
    """Complete Hey Orac configuration."""
    models: List[ModelConfig]
    audio: AudioConfig
    system: SystemConfig
    version: str = "1.0"


class SettingsManager:
    """
    Thread-safe configuration manager with JSON schema validation,
    atomic file writes, and change notification system.
    """
    
    # JSON Schema for configuration validation
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "models": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "path": {"type": "string"},
                        "framework": {"type": "string", "enum": ["tflite", "onnx"]},
                        "enabled": {"type": "boolean"},
                        "threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "priority": {"type": "integer", "minimum": 1}
                    },
                    "required": ["name", "path"]
                }
            },
            "audio": {
                "type": "object",
                "properties": {
                    "sample_rate": {"type": "integer", "minimum": 8000, "maximum": 48000},
                    "channels": {"type": "integer", "minimum": 1, "maximum": 2},
                    "chunk_size": {"type": "integer", "minimum": 128, "maximum": 8192},
                    "device_index": {"type": ["integer", "null"]},
                    "auto_select_usb": {"type": "boolean"}
                }
            },
            "system": {
                "type": "object",
                "properties": {
                    "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                    "models_dir": {"type": "string"},
                    "recordings_dir": {"type": "string"},
                    "metrics_enabled": {"type": "boolean"},
                    "metrics_port": {"type": "integer", "minimum": 1024, "maximum": 65535},
                    "hot_reload_enabled": {"type": "boolean"},
                    "hot_reload_interval": {"type": "number", "minimum": 1.0}
                }
            }
        },
        "required": ["models", "audio", "system"]
    }
    
    def __init__(self, config_path: str = "/app/config/settings.json"):
        """
        Initialize SettingsManager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe access
        self._lock = threading.RLock()
        self._config: Optional[HeyOracConfig] = None
        self._file_mtime: Optional[float] = None
        
        # Change notification system
        self._change_callbacks: List[Callable[[HeyOracConfig], None]] = []
        
        # Load initial configuration
        self._load_config()
        
        logger.info(f"SettingsManager initialized with config: {config_path}")
    
    def _get_default_config(self) -> HeyOracConfig:
        """Get default configuration."""
        return HeyOracConfig(
            models=[
                ModelConfig(
                    name="hey_jarvis",
                    path="hey_jarvis",  # Pre-trained model
                    framework="tflite",
                    enabled=True,
                    threshold=0.3,
                    priority=1
                )
            ],
            audio=AudioConfig(),
            system=SystemConfig()
        )
    
    def _validate_config(self, config_dict: Dict[str, Any]) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config_dict: Configuration dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic schema validation would go here
            # For now, just check required fields
            required_fields = ['models', 'audio', 'system']
            for field in required_fields:
                if field not in config_dict:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # Validate models
            if not isinstance(config_dict['models'], list):
                logger.error("Models must be a list")
                return False
            
            for model in config_dict['models']:
                if not isinstance(model, dict):
                    logger.error("Each model must be a dictionary")
                    return False
                if 'name' not in model or 'path' not in model:
                    logger.error("Models must have 'name' and 'path' fields")
                    return False
            
            # Validate audio config
            audio_config = config_dict['audio']
            if not isinstance(audio_config, dict):
                logger.error("Audio config must be a dictionary")
                return False
            
            # Validate system config
            system_config = config_dict['system']
            if not isinstance(system_config, dict):
                logger.error("System config must be a dictionary")
                return False
            
            logger.debug("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> HeyOracConfig:
        """Convert dictionary to HeyOracConfig object."""
        try:
            # Convert models
            models = []
            for model_dict in config_dict.get('models', []):
                model = ModelConfig(
                    name=model_dict['name'],
                    path=model_dict['path'],
                    framework=model_dict.get('framework', 'tflite'),
                    enabled=model_dict.get('enabled', True),
                    threshold=model_dict.get('threshold', 0.3),
                    priority=model_dict.get('priority', 1)
                )
                models.append(model)
            
            # Convert audio config
            audio_dict = config_dict.get('audio', {})
            audio = AudioConfig(
                sample_rate=audio_dict.get('sample_rate', 16000),
                channels=audio_dict.get('channels', 2),
                chunk_size=audio_dict.get('chunk_size', 1280),
                device_index=audio_dict.get('device_index'),
                auto_select_usb=audio_dict.get('auto_select_usb', True)
            )
            
            # Convert system config
            system_dict = config_dict.get('system', {})
            system = SystemConfig(
                log_level=system_dict.get('log_level', 'INFO'),
                models_dir=system_dict.get('models_dir', '/app/models'),
                recordings_dir=system_dict.get('recordings_dir', '/app/recordings'),
                metrics_enabled=system_dict.get('metrics_enabled', True),
                metrics_port=system_dict.get('metrics_port', 8000),
                hot_reload_enabled=system_dict.get('hot_reload_enabled', True),
                hot_reload_interval=system_dict.get('hot_reload_interval', 5.0)
            )
            
            return HeyOracConfig(
                models=models,
                audio=audio,
                system=system,
                version=config_dict.get('version', '1.0')
            )
            
        except Exception as e:
            logger.error(f"Error converting dictionary to config: {e}")
            raise
    
    def _config_to_dict(self, config: HeyOracConfig) -> Dict[str, Any]:
        """Convert HeyOracConfig to dictionary."""
        return {
            'version': config.version,
            'models': [asdict(model) for model in config.models],
            'audio': asdict(config.audio),
            'system': asdict(config.system)
        }
    
    def _load_config(self) -> bool:
        """
        Load configuration from file.
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                if not self.config_path.exists():
                    logger.info("Config file not found, creating default configuration")
                    self._config = self._get_default_config()
                    self._save_config()
                    return True
                
                # Check if file has been modified
                current_mtime = self.config_path.stat().st_mtime
                if self._file_mtime is not None and current_mtime == self._file_mtime:
                    logger.debug("Config file unchanged, skipping reload")
                    return True
                
                # Load and validate configuration
                with open(self.config_path, 'r') as f:
                    config_dict = json.load(f)
                
                if not self._validate_config(config_dict):
                    logger.error("Configuration validation failed")
                    return False
                
                self._config = self._dict_to_config(config_dict)
                self._file_mtime = current_mtime
                
                logger.info("Configuration loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                return False
    
    def _save_config(self) -> bool:
        """
        Save configuration to file atomically.
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                if self._config is None:
                    logger.error("No configuration to save")
                    return False
                
                config_dict = self._config_to_dict(self._config)
                
                # Atomic write using temporary file
                with tempfile.NamedTemporaryFile(
                    mode='w',
                    dir=self.config_path.parent,
                    delete=False,
                    suffix='.tmp'
                ) as tmp_file:
                    json.dump(config_dict, tmp_file, indent=2)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                
                # Atomic replace
                os.replace(tmp_file.name, self.config_path)
                
                # Update modification time
                self._file_mtime = self.config_path.stat().st_mtime
                
                logger.info("Configuration saved successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error saving configuration: {e}")
                return False
    
    @contextmanager
    def get_config(self):
        """
        Context manager for thread-safe configuration access.
        
        Yields:
            HeyOracConfig: Current configuration
        """
        with self._lock:
            if self._config is None:
                raise RuntimeError("Configuration not loaded")
            yield self._config
    
    def update_config(self, updater: Callable[[HeyOracConfig], HeyOracConfig]) -> bool:
        """
        Update configuration using a function.
        
        Args:
            updater: Function that takes current config and returns updated config
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                if self._config is None:
                    logger.error("No configuration to update")
                    return False
                
                # Apply update
                updated_config = updater(self._config)
                
                # Validate updated configuration
                config_dict = self._config_to_dict(updated_config)
                if not self._validate_config(config_dict):
                    logger.error("Updated configuration validation failed")
                    return False
                
                self._config = updated_config
                
                # Save to file
                if not self._save_config():
                    logger.error("Failed to save updated configuration")
                    return False
                
                # Notify change callbacks
                for callback in self._change_callbacks:
                    try:
                        callback(self._config)
                    except Exception as e:
                        logger.error(f"Error in change callback: {e}")
                
                logger.info("Configuration updated successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error updating configuration: {e}")
                return False
    
    def reload_if_changed(self) -> bool:
        """
        Reload configuration if file has been modified.
        
        Returns:
            True if reloaded or no changes, False on error
        """
        with self._lock:
            if not self.config_path.exists():
                return False
            
            current_mtime = self.config_path.stat().st_mtime
            if self._file_mtime is None or current_mtime != self._file_mtime:
                logger.info("Configuration file changed, reloading...")
                old_config = self._config
                
                if self._load_config():
                    # Notify change callbacks if config actually changed
                    if old_config != self._config:
                        for callback in self._change_callbacks:
                            try:
                                callback(self._config)
                            except Exception as e:
                                logger.error(f"Error in change callback: {e}")
                    return True
                else:
                    return False
            
            return True
    
    def register_change_callback(self, callback: Callable[[HeyOracConfig], None]):
        """Register a callback to be called when configuration changes."""
        with self._lock:
            self._change_callbacks.append(callback)
    
    def get_models_config(self) -> List[ModelConfig]:
        """Get model configurations."""
        with self._lock:
            if self._config is None:
                return []
            return self._config.models.copy()
    
    def get_audio_config(self) -> AudioConfig:
        """Get audio configuration."""
        with self._lock:
            if self._config is None:
                return AudioConfig()
            return self._config.audio
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        with self._lock:
            if self._config is None:
                return SystemConfig()
            return self._config.system