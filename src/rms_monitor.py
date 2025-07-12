import threading
import time
import numpy as np
import json
import os
import struct
from typing import Optional, Dict, Any
from multiprocessing import shared_memory

class RMSMonitor:
    """Monitor and share RMS audio levels for web interface using shared memory"""
    
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
        
        # Shared memory setup
        self._shm_name = 'rms_monitor_data'
        self._shm_size = 1024  # Enough for RMS data
        self._shm = None
        self._setup_shared_memory()
        
    def _setup_shared_memory(self):
        """Setup shared memory for cross-process communication"""
        try:
            # Try to connect to existing shared memory
            self._shm = shared_memory.SharedMemory(name=self._shm_name, create=False)
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"📊 Connected to existing shared memory: {self._shm_name}")
        except FileNotFoundError:
            # Create new shared memory
            self._shm = shared_memory.SharedMemory(name=self._shm_name, create=True, size=self._shm_size)
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"📊 Created new shared memory: {self._shm_name}")
        
    def _pack_rms_data(self, rms_level: float, is_active: bool) -> bytes:
        """Pack RMS data into bytes for shared memory"""
        # Pack: rms_level (8 bytes) + is_active (1 byte) + timestamp (8 bytes)
        packed = struct.pack('d?d', rms_level, is_active, time.time())
        return packed
    
    def _unpack_rms_data(self, data: bytes) -> tuple:
        """Unpack RMS data from bytes"""
        # Unpack: rms_level (8 bytes) + is_active (1 byte) + timestamp (8 bytes)
        rms_level, is_active, timestamp = struct.unpack('d?d', data[:17])
        return rms_level, is_active, timestamp
        
    def update_rms(self, rms_level: float):
        """Update RMS level from audio pipeline using shared memory"""
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
            
            # Write to shared memory
            try:
                packed_data = self._pack_rms_data(rms_level, True)
                self._shm.buf[:len(packed_data)] = packed_data
                
                # Debug logging for testing
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"📊 RMS Monitor Updated: {rms_level:.2f}, Active: True (Shared Memory)")
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"❌ Failed to write to shared memory: {e}")
    
    def get_rms_data(self) -> Dict[str, Any]:
        """Get current RMS data for web interface from shared memory"""
        with self._lock:
            try:
                # Read from shared memory
                data = bytes(self._shm.buf[:17])  # Read packed data
                rms_level, is_active, timestamp = self._unpack_rms_data(data)
                
                current_time = time.time()
                time_diff = current_time - timestamp
                
                # Check if data is stale (older than 5 seconds)
                if time_diff > 5.0:
                    is_active = False
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"⚠️ RMS data is stale ({time_diff:.2f}s), setting is_active=False")
                
                return {
                    'current_rms': float(rms_level),
                    'is_active': bool(is_active),
                    'last_update': timestamp
                }
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"❌ Failed to read from shared memory: {e}")
                # Fallback to local data
                return {
                    'current_rms': float(self._rms_data['current_rms']),
                    'is_active': False,
                    'last_update': time.time()
                }
    
    def reset(self):
        """Reset RMS monitor and cleanup shared memory"""
        with self._lock:
            self._rms_data = {
                'current_rms': 0.0,
                'avg_rms': 0.0,
                'max_rms': 0.0,
                'volume_history': [],
                'last_update': time.time(),
                'is_active': False
            }
            
            # Cleanup shared memory
            try:
                if self._shm:
                    self._shm.close()
                    self._shm.unlink()
            except Exception:
                pass

# Global RMS monitor instance
rms_monitor = RMSMonitor() 