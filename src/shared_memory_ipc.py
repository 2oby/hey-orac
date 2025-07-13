import threading
import time
import numpy as np
import json
import os
import struct
from typing import Optional, Dict, Any
from multiprocessing import shared_memory

class SharedMemoryIPC:
    """Interprocess communication system using shared memory for audio state and activation data"""
    
    def __init__(self):
        self._audio_data = {
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
        self._shm_name = 'hey_orac_ipc_data'
        self._shm_size = 1024  # Enough for audio and activation data
        self._shm = None
        self._setup_shared_memory()
        
    def _setup_shared_memory(self):
        """Setup shared memory for cross-process communication"""
        try:
            # Try to connect to existing shared memory
            self._shm = shared_memory.SharedMemory(name=self._shm_name, create=False)
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"🔗 Connected to existing shared memory: {self._shm_name}")
        except FileNotFoundError:
            # Create new shared memory
            self._shm = shared_memory.SharedMemory(name=self._shm_name, create=True, size=self._shm_size)
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"🔗 Created new shared memory: {self._shm_name}")
        
    def _pack_state_data(self, rms_level: float, is_active: bool, is_listening: bool = False) -> bytes:
        """Pack audio and activation data into bytes for shared memory"""
        # Pack: rms_level (8 bytes) + is_active (1 byte) + is_listening (8 bytes) + timestamp (8 bytes)
        # Use 'd' for double (8 bytes), 'B' for unsigned char (1 byte), 'd' for double (8 bytes), 'd' for double (8 bytes)
        packed = struct.pack('dBdd', rms_level, 1 if is_active else 0, 1.0 if is_listening else 0.0, time.time())
        return packed
    
    def _unpack_state_data(self, data: bytes) -> tuple:
        """Unpack audio and activation data from bytes"""
        # Calculate exact size needed for the struct
        struct_size = struct.calcsize('dBdd')
        if len(data) < struct_size:
            raise ValueError(f"Expected at least {struct_size} bytes, got {len(data)}")
        
        rms_level, is_active_int, is_listening_float, timestamp = struct.unpack('dBdd', data[:struct_size])
        is_active = bool(is_active_int)
        is_listening = bool(is_listening_float)
        return rms_level, is_active, is_listening, timestamp
        
    def update_audio_state(self, rms_level: float):
        """Update audio state from audio pipeline using shared memory"""
        with self._lock:
            self._audio_data['current_rms'] = rms_level
            self._audio_data['last_update'] = time.time()
            self._audio_data['is_active'] = True
            
            # Update volume history
            self._audio_data['volume_history'].append(rms_level)
            if len(self._audio_data['volume_history']) > self._volume_window_size:
                self._audio_data['volume_history'].pop(0)
            
            # Calculate average and max
            if self._audio_data['volume_history']:
                self._audio_data['avg_rms'] = np.mean(self._audio_data['volume_history'])
                self._audio_data['max_rms'] = max(self._audio_data['volume_history'])
            
            # Write to shared memory (preserve existing activation state)
            try:
                # Read current activation state directly from shared memory without recursive lock
                struct_size = struct.calcsize('dBdd')
                data = bytes(self._shm.buf[:struct_size])
                rms_level_old, is_active_old, is_listening_old, timestamp_old = self._unpack_state_data(data)
                
                # Keep the existing is_listening state
                is_listening = bool(is_listening_old)
                
                packed_data = self._pack_state_data(rms_level, True, is_listening)
                self._shm.buf[:len(packed_data)] = packed_data
                
                # Removed excessive debug logging
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"❌ Failed to write to shared memory: {e}")
    
    def update_activation_state(self, is_listening: bool, model_name: str = None, confidence: float = 0.0):
        """Update activation state in shared memory with enhanced debugging"""
        with self._lock:
            try:
                # Read current audio data directly from shared memory without recursive lock
                struct_size = struct.calcsize('dBdd')
                data = bytes(self._shm.buf[:struct_size])
                rms_level, is_active_old, is_listening_old, timestamp_old = self._unpack_state_data(data)
                
                # ENHANCED DEBUGGING: Log state transition
                import logging
                logger = logging.getLogger(__name__)
                
                if is_listening != is_listening_old:
                    logger.info(f"🔄 ACTIVATION STATE CHANGE: {is_listening_old} → {is_listening}")
                    logger.info(f"   Model: {model_name}, Confidence: {confidence:.3f}")
                    logger.info(f"   RMS Level: {rms_level:.4f}")
                    logger.info(f"   Timestamp: {time.time()}")
                else:
                    # Removed excessive debug logging
                    pass
                
                # Write updated data with new activation state
                packed_data = self._pack_state_data(rms_level, is_active_old, is_listening)
                self._shm.buf[:len(packed_data)] = packed_data
                
                # Verify the write was successful
                verification_data = bytes(self._shm.buf[:struct_size])
                verification_rms, verification_active, verification_listening, verification_timestamp = self._unpack_state_data(verification_data)
                
                if verification_listening != is_listening:
                    logger.error(f"❌ ACTIVATION: Shared memory write verification failed!")
                    logger.error(f"   Expected: {is_listening}, Got: {verification_listening}")
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"❌ Failed to update activation state: {e}")
                logger.error(f"   Attempted to set is_listening: {is_listening}")
                logger.error(f"   Model: {model_name}, Confidence: {confidence}")
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get current system state (audio + activation) from shared memory with enhanced debugging"""
        with self._lock:
            try:
                # Read from shared memory
                struct_size = struct.calcsize('dBdd')
                data = bytes(self._shm.buf[:struct_size])  # Read packed data
                rms_level, is_active, is_listening, timestamp = self._unpack_state_data(data)
                
                current_time = time.time()
                time_diff = current_time - timestamp
                
                # ENHANCED DEBUGGING: Log state reads
                import logging
                logger = logging.getLogger(__name__)
                
                # Removed excessive read logging
                
                # Check if data is stale (older than 5 seconds)
                if time_diff > 5.0:
                    rms_level = 0.0  # Reset RMS to 0 when data is stale
                    is_active = False
                    is_listening = False
                    logger.warning(f"⚠️ Shared memory data is stale ({time_diff:.2f}s), resetting RMS=0.0, is_active=False, is_listening=False")
                
                return {
                    'current_rms': float(rms_level),
                    'is_active': bool(is_active),
                    'is_listening': bool(is_listening),
                    'last_update': timestamp
                }
                
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"❌ Failed to read from shared memory: {e}")
                # Fallback to safe defaults
                return {
                    'current_rms': 0.0,  # Return 0.0 instead of stale data
                    'is_active': False,
                    'is_listening': False,
                    'last_update': time.time()
                }
    
    def get_audio_state(self) -> Dict[str, Any]:
        """Get audio state from shared memory (alias for get_system_state for clarity)"""
        return self.get_system_state()
    
    def get_activation_state(self) -> Dict[str, Any]:
        """Get activation state from shared memory (alias for get_system_state for clarity)"""
        return self.get_system_state()
    
    def reset(self):
        """Reset IPC system and cleanup shared memory"""
        with self._lock:
            self._audio_data = {
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

# Global IPC instance
shared_memory_ipc = SharedMemoryIPC() 