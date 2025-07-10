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
            
            # Debug logging (only log every 100 updates to avoid spam)
            if len(self._rms_data['volume_history']) % 100 == 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"🔊 RMS Monitor Updated: current={rms_level:.4f}, avg={self._rms_data['avg_rms']:.4f}, max={self._rms_data['max_rms']:.4f}")
            
            # Additional debug logging for first few updates
            if len(self._rms_data['volume_history']) <= 5:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"🔊 RMS Monitor Update #{len(self._rms_data['volume_history'])}: rms={rms_level:.4f}, is_active=True")
    
    def get_rms_data(self) -> Dict[str, Any]:
        """Get current RMS data for web interface"""
        with self._lock:
            current_time = time.time()
            time_diff = current_time - self._rms_data['last_update']
            
            # Debug logging for timestamp issues
            if time_diff > 5.0:  # Log if data is getting stale
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"⚠️ RMS data getting stale: {time_diff:.2f}s since last update")
            
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

# Global RMS monitor instance
rms_monitor = RMSMonitor() 