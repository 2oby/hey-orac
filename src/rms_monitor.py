import threading
import time
import numpy as np
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
    
    def get_rms_data(self) -> Dict[str, Any]:
        """Get current RMS data for web interface"""
        with self._lock:
            # Check if data is stale (older than 5 seconds)
            if time.time() - self._rms_data['last_update'] > 5.0:
                self._rms_data['is_active'] = False
            
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

# Global RMS monitor instance
rms_monitor = RMSMonitor() 