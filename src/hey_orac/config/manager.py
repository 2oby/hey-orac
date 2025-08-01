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
    webhook_url: str = ""
    priority: int = 1
    stt_enabled: bool = True


@dataclass
class AudioPreprocessingConfig:
    """Audio preprocessing configuration."""
    enable_agc: bool = True
    agc_target_level: float = 0.3
    agc_max_gain: float = 10.0
    agc_attack_time: float = 0.01
    agc_release_time: float = 0.1
    enable_compression: bool = True
    compression_threshold: float = 0.5
    compression_ratio: float = 4.0
    enable_limiter: bool = True
    limiter_threshold: float = 0.95
    enable_noise_gate: bool = False
    noise_gate_threshold: float = 0.01


@dataclass
class AudioConfig:
    """Audio capture configuration."""
    sample_rate: int = 16000
    channels: int = 2
    chunk_size: int = 1280
    device_index: Optional[int] = None
    auto_select_usb: bool = True
    preprocessing: Optional[AudioPreprocessingConfig] = None
    
    def __post_init__(self):
        if self.preprocessing is None:
            self.preprocessing = AudioPreprocessingConfig()


@dataclass
class STTConfig:
    """Speech-to-Text configuration."""
    enabled: bool = True
    base_url: str = "http://localhost:8000"
    timeout: int = 30
    language: Optional[str] = None
    pre_roll_duration: float = 1.0
    silence_threshold: float = 0.01
    silence_duration: float = 0.3
    grace_period: float = 0.4
    max_recording_duration: float = 15.0
    enable_per_model: bool = True


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
    rms_filter: float = 50.0
    cooldown: float = 2.0
    vad_threshold: float = 0.5
    # MULTI_TRIGGER: Enable multiple wake word models to trigger simultaneously
    # WARNING: This is for testing purposes only. In normal operation, this should be FALSE
    # to prevent multiple notifications when multiple models trigger at similar confidence levels
    multi_trigger: bool = False
    # Enable audio preprocessing (AGC, compression, limiting)
    enable_audio_preprocessing: bool = False


@dataclass
class HeyOracConfig:
    """Complete Hey Orac configuration."""
    models: List[ModelConfig]
    audio: AudioConfig
    system: SystemConfig
    stt: STTConfig
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
                        "webhook_url": {"type": "string"},
                        "priority": {"type": "integer", "minimum": 1},
                        "stt_enabled": {"type": "boolean"}
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
                    "auto_select_usb": {"type": "boolean"},
                    "preprocessing": {
                        "type": "object",
                        "properties": {
                            "enable_agc": {"type": "boolean"},
                            "agc_target_level": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "agc_max_gain": {"type": "number", "minimum": 0.0, "maximum": 40.0},
                            "agc_attack_time": {"type": "number", "minimum": 0.001, "maximum": 1.0},
                            "agc_release_time": {"type": "number", "minimum": 0.01, "maximum": 5.0},
                            "enable_compression": {"type": "boolean"},
                            "compression_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "compression_ratio": {"type": "number", "minimum": 1.0, "maximum": 20.0},
                            "enable_limiter": {"type": "boolean"},
                            "limiter_threshold": {"type": "number", "minimum": 0.5, "maximum": 1.0},
                            "enable_noise_gate": {"type": "boolean"},
                            "noise_gate_threshold": {"type": "number", "minimum": 0.0, "maximum": 0.5}
                        }
                    }
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
                    "hot_reload_interval": {"type": "number", "minimum": 1.0},
                    "rms_filter": {"type": "number", "minimum": 0.0, "maximum": 5000.0},
                    "cooldown": {"type": "number", "minimum": 0.0, "maximum": 60.0},
                    "vad_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "multi_trigger": {"type": "boolean"}
                }
            },
            "stt": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "base_url": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
                    "language": {"type": ["string", "null"]},
                    "pre_roll_duration": {"type": "number", "minimum": 0.0, "maximum": 5.0},
                    "silence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "silence_duration": {"type": "number", "minimum": 0.1, "maximum": 5.0},
                    "grace_period": {"type": "number", "minimum": 0.0, "maximum": 5.0},
                    "max_recording_duration": {"type": "number", "minimum": 1.0, "maximum": 60.0},
                    "enable_per_model": {"type": "boolean"}
                }
            }
        },
        "required": ["models", "audio", "system", "stt"]
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
        
        # Auto-discover and add new models
        self._auto_discover_models()
        
        logger.info(f"SettingsManager initialized with config: {config_path}")
    
    def _get_default_config(self) -> HeyOracConfig:
        """Get default configuration."""
        return HeyOracConfig(
            models=[],  # Start with empty models - auto-discovery will populate
            audio=AudioConfig(),
            system=SystemConfig(),
            stt=STTConfig()
        )
    
    def _create_from_template(self) -> bool:
        """
        Create settings.json from template file.
        
        Returns:
            True if successful, False if template not found or error
        """
        template_path = self.config_path.parent / "settings.json.template"
        
        try:
            if not template_path.exists():
                logger.debug(f"Template file not found at {template_path}")
                return False
            
            logger.info(f"Creating configuration from template: {template_path}")
            
            # Load and validate template
            with open(template_path, 'r') as f:
                config_dict = json.load(f)
            
            if not self._validate_config(config_dict):
                logger.error("Template validation failed, falling back to defaults")
                return False
            
            # Convert to config object and save
            self._config = self._dict_to_config(config_dict)
            self._save_config()
            
            logger.info("Configuration created from template successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating config from template: {e}")
            return False
    
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
                    webhook_url=model_dict.get('webhook_url', ''),
                    priority=model_dict.get('priority', 1),
                    stt_enabled=model_dict.get('stt_enabled', True)
                )
                models.append(model)
            
            # Convert audio config
            audio_dict = config_dict.get('audio', {})
            
            # Handle preprocessing config
            preprocessing_dict = audio_dict.get('preprocessing', {})
            if preprocessing_dict:
                preprocessing = AudioPreprocessingConfig(
                    enable_agc=preprocessing_dict.get('enable_agc', True),
                    agc_target_level=preprocessing_dict.get('agc_target_level', 0.3),
                    agc_max_gain=preprocessing_dict.get('agc_max_gain', 10.0),
                    agc_attack_time=preprocessing_dict.get('agc_attack_time', 0.01),
                    agc_release_time=preprocessing_dict.get('agc_release_time', 0.1),
                    enable_compression=preprocessing_dict.get('enable_compression', True),
                    compression_threshold=preprocessing_dict.get('compression_threshold', 0.5),
                    compression_ratio=preprocessing_dict.get('compression_ratio', 4.0),
                    enable_limiter=preprocessing_dict.get('enable_limiter', True),
                    limiter_threshold=preprocessing_dict.get('limiter_threshold', 0.95),
                    enable_noise_gate=preprocessing_dict.get('enable_noise_gate', False),
                    noise_gate_threshold=preprocessing_dict.get('noise_gate_threshold', 0.01)
                )
            else:
                preprocessing = None
            
            audio = AudioConfig(
                sample_rate=audio_dict.get('sample_rate', 16000),
                channels=audio_dict.get('channels', 2),
                chunk_size=audio_dict.get('chunk_size', 1280),
                device_index=audio_dict.get('device_index'),
                auto_select_usb=audio_dict.get('auto_select_usb', True),
                preprocessing=preprocessing
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
                hot_reload_interval=system_dict.get('hot_reload_interval', 5.0),
                rms_filter=system_dict.get('rms_filter', 50.0),
                cooldown=system_dict.get('cooldown', 2.0),
                vad_threshold=system_dict.get('vad_threshold', 0.5),
                multi_trigger=system_dict.get('multi_trigger', False),
                enable_audio_preprocessing=system_dict.get('enable_audio_preprocessing', False)
            )
            
            # Convert STT config
            stt_dict = config_dict.get('stt', {})
            stt = STTConfig(
                enabled=stt_dict.get('enabled', True),
                base_url=stt_dict.get('base_url', 'http://localhost:8000'),
                timeout=stt_dict.get('timeout', 30),
                language=stt_dict.get('language'),
                pre_roll_duration=stt_dict.get('pre_roll_duration', 1.0),
                silence_threshold=stt_dict.get('silence_threshold', 0.01),
                silence_duration=stt_dict.get('silence_duration', 0.3),
                grace_period=stt_dict.get('grace_period', 0.4),
                max_recording_duration=stt_dict.get('max_recording_duration', 15.0),
                enable_per_model=stt_dict.get('enable_per_model', True)
            )
            
            return HeyOracConfig(
                models=models,
                audio=audio,
                system=system,
                stt=stt,
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
            'system': asdict(config.system),
            'stt': asdict(config.stt)
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
                    logger.info("Config file not found, attempting to create from template")
                    if self._create_from_template():
                        return True
                    logger.info("Template not found, creating default configuration")
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
    
    def get_stt_config(self) -> STTConfig:
        """Get STT configuration."""
        with self._lock:
            if self._config is None:
                return STTConfig()
            return self._config.stt
    
    def _discover_model_files(self) -> List[str]:
        """
        Discover all model files in the models directory.
        
        Returns:
            List of model file paths
        """
        if self._config is None:
            return []
        
        models_dir = Path(self._config.system.models_dir)
        if not models_dir.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return []
        
        model_files = []
        
        # Search for .tflite and .onnx files in subdirectories
        for pattern in ['**/*.tflite', '**/*.onnx']:
            model_files.extend(models_dir.glob(pattern))
        
        # Convert to strings and sort (prioritize .tflite)
        model_paths = [str(p) for p in model_files]
        model_paths.sort(key=lambda x: (not x.endswith('.tflite'), x))
        
        logger.info(f"Discovered {len(model_paths)} model files: {[Path(p).name for p in model_paths]}")
        return model_paths
    
    def _create_model_config_from_file(self, file_path: str, priority: int) -> ModelConfig:
        """
        Create a ModelConfig from a model file path with default values.
        
        Args:
            file_path: Path to the model file
            priority: Priority value for the model
            
        Returns:
            ModelConfig with default values
        """
        file_path_obj = Path(file_path)
        name = file_path_obj.stem
        
        # Determine framework from extension
        framework = 'tflite' if file_path_obj.suffix.lower() == '.tflite' else 'onnx'
        
        return ModelConfig(
            name=name,
            path=file_path,
            framework=framework,
            enabled=False,  # Default to disabled for new models
            threshold=0.3,
            webhook_url="",
            priority=priority
        )
    
    def _auto_discover_models(self) -> bool:
        """
        Automatically discover new models and add them to configuration.
        Only adds models that are not already in the configuration.
        
        Returns:
            True if new models were added, False otherwise
        """
        with self._lock:
            if self._config is None:
                logger.warning("No configuration loaded, skipping auto-discovery")
                return False
            
            # Get current model paths in config
            existing_paths = {model.path for model in self._config.models}
            
            # Discover all model files
            discovered_paths = self._discover_model_files()
            
            # Find new models (not in existing config)
            new_model_paths = [path for path in discovered_paths if path not in existing_paths]
            
            if not new_model_paths:
                logger.info("No new models discovered")
                return False
            
            logger.info(f"Found {len(new_model_paths)} new models to add to configuration")
            
            # Create new model configs with increasing priority
            max_priority = max((model.priority for model in self._config.models), default=0)
            new_models = []
            
            for i, model_path in enumerate(new_model_paths):
                try:
                    new_model = self._create_model_config_from_file(
                        model_path, 
                        max_priority + i + 1
                    )
                    new_models.append(new_model)
                    logger.info(f"Added new model config: {new_model.name} (path: {model_path})")
                except Exception as e:
                    logger.error(f"Error creating config for model {model_path}: {e}")
            
            if new_models:
                # Check if we have any enabled models already
                has_enabled_models = any(model.enabled for model in self._config.models)
                
                # If no models are currently enabled, enable the first new model found
                if not has_enabled_models and new_models:
                    new_models[0].enabled = True
                    logger.info(f"🎯 Enabled first discovered model: {new_models[0].name} (no other models were enabled)")
                
                # Add new models to configuration
                self._config.models.extend(new_models)
                
                # Save updated configuration
                if self._save_config():
                    logger.info(f"✅ Added {len(new_models)} new models to configuration")
                    return True
                else:
                    logger.error("❌ Failed to save configuration with new models")
                    return False
            
            return False
    
    def refresh_model_discovery(self) -> bool:
        """
        Manually trigger model discovery and configuration update.
        
        Returns:
            True if new models were discovered and added, False otherwise
        """
        logger.info("🔍 Manually refreshing model discovery...")
        return self._auto_discover_models()
    
    def update_model_config(self, model_name: str, **updates) -> bool:
        """
        Update configuration for a specific model.
        
        Args:
            model_name: Name of the model to update
            **updates: Configuration fields to update
            
        Returns:
            True if successful, False otherwise
        """
        def updater(config: HeyOracConfig) -> HeyOracConfig:
            # Find the model to update
            for model in config.models:
                if model.name == model_name:
                    # Update the model with provided values
                    if 'threshold' in updates:
                        model.threshold = float(updates['threshold'])
                    if 'webhook_url' in updates:
                        model.webhook_url = str(updates['webhook_url'])
                    if 'enabled' in updates:
                        model.enabled = bool(updates['enabled'])
                    if 'framework' in updates:
                        model.framework = str(updates['framework'])
                    if 'priority' in updates:
                        model.priority = int(updates['priority'])
                    logger.info(f"Updated model '{model_name}' config: {updates}")
                    break
            else:
                logger.warning(f"Model '{model_name}' not found for update")
            
            return config
        
        return self.update_config(updater)
    
    def update_system_config(self, **updates) -> bool:
        """
        Update system configuration.
        
        Args:
            **updates: System configuration fields to update
            
        Returns:
            True if successful, False otherwise
        """
        def updater(config: HeyOracConfig) -> HeyOracConfig:
            # Update system config with provided values
            if 'log_level' in updates:
                config.system.log_level = str(updates['log_level'])
            if 'models_dir' in updates:
                config.system.models_dir = str(updates['models_dir'])
            if 'recordings_dir' in updates:
                config.system.recordings_dir = str(updates['recordings_dir'])
            if 'metrics_enabled' in updates:
                config.system.metrics_enabled = bool(updates['metrics_enabled'])
            if 'metrics_port' in updates:
                config.system.metrics_port = int(updates['metrics_port'])
            if 'hot_reload_enabled' in updates:
                config.system.hot_reload_enabled = bool(updates['hot_reload_enabled'])
            if 'hot_reload_interval' in updates:
                config.system.hot_reload_interval = float(updates['hot_reload_interval'])
            if 'rms_filter' in updates:
                config.system.rms_filter = float(updates['rms_filter'])
            if 'cooldown' in updates:
                config.system.cooldown = float(updates['cooldown'])
            if 'vad_threshold' in updates:
                config.system.vad_threshold = float(updates['vad_threshold'])
            if 'multi_trigger' in updates:
                config.system.multi_trigger = bool(updates['multi_trigger'])
            
            logger.info(f"Updated system config: {updates}")
            return config
        
        return self.update_config(updater)
    
    def save(self) -> bool:
        """
        Convenience method to save current configuration to file.
        
        Returns:
            True if successful, False otherwise
        """
        return self._save_config()
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelConfig if found, None otherwise
        """
        with self._lock:
            if self._config is None:
                return None
            
            for model in self._config.models:
                if model.name == model_name:
                    return model
            
            return None