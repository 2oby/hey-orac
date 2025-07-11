import json
import os
import time
import threading
import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import fcntl

logger = logging.getLogger(__name__)

class SettingsManager:
    """
    Manages application settings with real-time file-based IPC using tmpfs.
    
    Features:
    - Settings stored in RAM (tmpfs) for fast access
    - File watching for real-time updates
    - Thread-safe operations
    - Automatic backup/restore from permanent storage
    """
    
    def __init__(self, 
                 settings_file: str = "/tmp/settings/config.json",
                 backup_file: str = "/app/settings_backup.json",
                 default_settings: Optional[Dict[str, Any]] = None):
        """
        Initialize settings manager.
        
        Args:
            settings_file: Path to tmpfs settings file
            backup_file: Path to permanent backup file
            default_settings: Default settings if no file exists
        """
        self.settings_file = Path(settings_file)
        self.backup_file = Path(backup_file)
        self.default_settings = default_settings or self._get_default_settings()
        
        # Ensure tmpfs directory exists
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        self._settings = {}
        self._watchers = []
        self._file_watcher_thread = None
        self._stop_watching = False
        
        # Load initial settings
        self._load_settings()
        
        # Start file watcher
        self._start_file_watcher()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default application settings."""
        return {
            "audio": {
                "sample_rate": 16000,
                "chunk_size": 1280,
                "channels": 1,
                "device_index": 0
            },
            "wake_word": {
                "threshold": 0.4,
                "model": "Hay--compUta_v_lrg",
                "cooldown": 1.5,
                "debounce": 0.2
            },
            "detection": {
                "rms_filter": 50,
                "min_audio_level": 100,
                "max_audio_level": 32767
            },
            "web": {
                "port": 7171,
                "host": "0.0.0.0",
                "debug": False
            },
            "system": {
                "log_level": "INFO",
                "max_log_size": "10MB",
                "backup_interval": 300  # 5 minutes
            }
        }
    
    def _load_settings(self) -> None:
        """Load settings from file or create with defaults."""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    with self._lock:
                        self._settings = json.load(f)
                logger.info(f"✅ Settings loaded from {self.settings_file}")
            else:
                # Create with defaults
                with self._lock:
                    self._settings = self.default_settings.copy()
                self._save_settings()
                logger.info(f"✅ Created new settings file with defaults: {self.settings_file}")
        except Exception as e:
            logger.error(f"❌ Failed to load settings: {e}")
            with self._lock:
                self._settings = self.default_settings.copy()
    
    def _save_settings(self) -> bool:
        """Save settings to tmpfs file."""
        try:
            with open(self.settings_file, 'w') as f:
                # Use file locking to prevent corruption
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(self._settings, f, indent=2)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save settings: {e}")
            return False
    
    def _backup_settings(self) -> bool:
        """Backup settings to permanent storage."""
        try:
            self.backup_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.backup_file, 'w') as f:
                json.dump(self._settings, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to backup settings: {e}")
            return False
    
    def _restore_from_backup(self) -> bool:
        """Restore settings from backup file."""
        try:
            if self.backup_file.exists():
                with open(self.backup_file, 'r') as f:
                    with self._lock:
                        self._settings = json.load(f)
                self._save_settings()
                logger.info("✅ Settings restored from backup")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to restore settings: {e}")
        return False
    
    def _start_file_watcher(self) -> None:
        """Start file watcher thread."""
        if self._file_watcher_thread is None:
            self._stop_watching = False
            self._file_watcher_thread = threading.Thread(
                target=self._file_watcher_loop,
                daemon=True,
                name="SettingsFileWatcher"
            )
            self._file_watcher_thread.start()
            logger.info("✅ Settings file watcher started")
    
    def _file_watcher_loop(self) -> None:
        """File watcher loop that monitors for changes."""
        last_modified = 0
        
        while not self._stop_watching:
            try:
                if self.settings_file.exists():
                    current_modified = self.settings_file.stat().st_mtime
                    
                    if current_modified > last_modified:
                        # File was modified, reload settings
                        self._load_settings()
                        last_modified = current_modified
                        
                        # Notify watchers
                        self._notify_watchers()
                        
                        logger.debug(f"🔄 Settings reloaded from file: {self.settings_file}")
                
                time.sleep(0.1)  # Check every 100ms
                
            except Exception as e:
                logger.error(f"❌ File watcher error: {e}")
                time.sleep(1)  # Wait longer on error
    
    def _notify_watchers(self) -> None:
        """Notify all registered watchers of settings changes."""
        for callback in self._watchers:
            try:
                callback(self._settings)
            except Exception as e:
                logger.error(f"❌ Watcher callback error: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value using dot notation (e.g., 'audio.sample_rate')."""
        with self._lock:
            keys = key.split('.')
            value = self._settings
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
    
    def set(self, key: str, value: Any) -> bool:
        """Set a setting value using dot notation."""
        try:
            with self._lock:
                keys = key.split('.')
                current = self._settings
                
                # Navigate to the parent of the target key
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                # Set the value
                current[keys[-1]] = value
                
                # Save to file
                success = self._save_settings()
                
                if success:
                    logger.info(f"✅ Setting updated: {key} = {value}")
                
                return success
                
        except Exception as e:
            logger.error(f"❌ Failed to set setting {key}: {e}")
            return False
    
    def get_all(self) -> Dict[str, Any]:
        """Get all settings."""
        with self._lock:
            return self._settings.copy()
    
    def update(self, settings: Dict[str, Any]) -> bool:
        """Update multiple settings at once."""
        try:
            with self._lock:
                self._settings.update(settings)
                success = self._save_settings()
                
                if success:
                    logger.info(f"✅ Updated {len(settings)} settings")
                
                return success
                
        except Exception as e:
            logger.error(f"❌ Failed to update settings: {e}")
            return False
    
    def add_watcher(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add a callback function to be called when settings change."""
        self._watchers.append(callback)
        logger.info(f"✅ Added settings watcher: {callback.__name__}")
    
    def remove_watcher(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove a settings watcher callback."""
        if callback in self._watchers:
            self._watchers.remove(callback)
            logger.info(f"✅ Removed settings watcher: {callback.__name__}")
    
    def backup(self) -> bool:
        """Manually trigger a backup."""
        return self._backup_settings()
    
    def restore(self) -> bool:
        """Manually trigger a restore from backup."""
        return self._restore_from_backup()
    
    def reset_to_defaults(self) -> bool:
        """Reset settings to defaults."""
        try:
            with self._lock:
                self._settings = self.default_settings.copy()
                success = self._save_settings()
                
                if success:
                    logger.info("✅ Settings reset to defaults")
                
                return success
                
        except Exception as e:
            logger.error(f"❌ Failed to reset settings: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the settings manager."""
        self._stop_watching = True
        if self._file_watcher_thread:
            self._file_watcher_thread.join(timeout=1)
        logger.info("✅ Settings manager stopped")

# Global settings manager instance
_settings_manager = None

def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager

def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting value."""
    return get_settings_manager().get(key, default)

def set_setting(key: str, value: Any) -> bool:
    """Set a setting value."""
    return get_settings_manager().set(key, value) 