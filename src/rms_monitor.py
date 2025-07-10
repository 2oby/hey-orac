import threading
import time
import numpy as np
import json
import os
from typing import Optional, Dict, Any

class RMSMonitor:
    """Monitor and share RMS audio levels for web interface"""
    
    def __init__(self):
        self._rms_data = {
            'current_rms': 0.0,
            'avg_rms': 0.0,
            'max_rms': 0.0,
            'volume_history': [],
            'last_update': time.time(),
            'is_active': False
        }
        self._lock = threading.Lock()
        self._volume_window_size = 50  # Keep last 50 RMS values for averaging
        self._data_file = '/tmp/rms_monitor_data.json'
        
    def update_rms(self, rms_level: float):
        """Update RMS level from audio pipeline"""
        with self._lock:
            self._rms_data['current_rms'] = rms_level
            self._rms_data['last_update'] = time.time()
            self._rms_data['is_active'] = True
            
            # Update volume history
            self._rms_data['volume_history'].append(rms_level)
            if len(self._rms_data['volume_history']) > self._volume_window_size:
                self._rms_data['volume_history'].pop(0)
            
            # Calculate average and max
            if self._rms_data['volume_history']:
                self._rms_data['avg_rms'] = np.mean(self._rms_data['volume_history'])
                self._rms_data['max_rms'] = max(self._rms_data['volume_history'])
            
            # Save to file for cross-process sharing
            try:
                # Convert numpy types to native Python types for JSON serialization
                data_to_save = {
                    'current_rms': float(self._rms_data['current_rms']),
                    'avg_rms': float(self._rms_data['avg_rms']),
                    'max_rms': float(self._rms_data['max_rms']),
                    'volume_history': [float(x) for x in self._rms_data['volume_history']],
                    'last_update': self._rms_data['last_update'],
                    'is_active': self._rms_data['is_active']
                }
                with open(self._data_file, 'w') as f:
                    json.dump(data_to_save, f)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"⚠️ Failed to save RMS data to file: {e}")
            

    
    def get_rms_data(self) -> Dict[str, Any]:
        """Get current RMS data for web interface"""
        with self._lock:
            # Try to load from file first (for cross-process sharing)
            try:
                if os.path.exists(self._data_file):
                    with open(self._data_file, 'r') as f:
                        file_data = json.load(f)
                        # Update local data from file
                        self._rms_data.update(file_data)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"⚠️ Failed to load RMS data from file: {e}")
            
            current_time = time.time()
            time_diff = current_time - self._rms_data['last_update']
            
            # Check if data is stale (older than 30 seconds) - increased from 5s
            if time_diff > 30.0:
                self._rms_data['is_active'] = False
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"⚠️ RMS data is stale ({time_diff:.2f}s), setting is_active=False")
            
            return self._rms_data.copy()
    
    def reset(self):
        """Reset RMS monitor"""
        with self._lock:
            self._rms_data = {
                'current_rms': 0.0,
                'avg_rms': 0.0,
                'max_rms': 0.0,
                'volume_history': [],
                'last_update': time.time(),
                'is_active': False
            }
            # Also clear the file
            try:
                if os.path.exists(self._data_file):
                    os.remove(self._data_file)
            except Exception:
                pass

# Global RMS monitor instance
rms_monitor = RMSMonitor() 