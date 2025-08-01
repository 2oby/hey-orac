"""
Ring buffer implementation for audio capture with pre-roll capability.
"""

import array
import threading
from typing import Optional, Tuple, Union
import numpy as np


class RingBuffer:
    """
    Thread-safe ring buffer for audio data storage.
    
    Stores audio samples in a circular buffer, allowing for efficient
    continuous recording with the ability to retrieve past audio (pre-roll).
    
    Supports both int16 and float32 audio formats.
    """
    
    def __init__(self, capacity_seconds: float = 10.0, sample_rate: int = 16000, dtype: type = np.float32):
        """
        Initialize the ring buffer.
        
        Args:
            capacity_seconds: Buffer capacity in seconds
            sample_rate: Audio sample rate in Hz
            dtype: Data type for audio samples (np.int16 or np.float32)
        """
        self.sample_rate = sample_rate
        self.capacity_samples = int(capacity_seconds * sample_rate)
        self.dtype = dtype
        
        # Use numpy array for storage (supports both int16 and float32)
        self.buffer = np.zeros(self.capacity_samples, dtype=dtype)
        
        # Write position (where next sample will be written)
        self.write_pos = 0
        
        # Total samples written (to track if buffer has wrapped)
        self.total_written = 0
        
        # Thread safety
        self.lock = threading.Lock()
    
    def write(self, samples: np.ndarray) -> None:
        """
        Write audio samples to the buffer.
        
        Args:
            samples: Audio samples as numpy array (int16 or float32)
        """
        with self.lock:
            # Convert to buffer dtype if needed
            if samples.dtype != self.dtype:
                if self.dtype == np.float32 and samples.dtype == np.int16:
                    # Convert int16 to float32
                    samples = samples.astype(np.float32) / 32768.0
                elif self.dtype == np.int16 and samples.dtype == np.float32:
                    # Convert float32 to int16
                    samples = (samples * 32768.0).astype(np.int16)
                else:
                    samples = samples.astype(self.dtype)
            
            num_samples = len(samples)
            
            # Handle case where samples wrap around buffer end
            if self.write_pos + num_samples <= self.capacity_samples:
                # Simple case: all samples fit before wrap
                self.buffer[self.write_pos:self.write_pos + num_samples] = samples
            else:
                # Wrap case: split samples
                first_part = self.capacity_samples - self.write_pos
                self.buffer[self.write_pos:] = samples[:first_part]
                self.buffer[:num_samples - first_part] = samples[first_part:]
            
            # Update write position and total written
            self.write_pos = (self.write_pos + num_samples) % self.capacity_samples
            self.total_written += num_samples
    
    def read_last(self, duration_seconds: float) -> np.ndarray:
        """
        Read the last N seconds of audio from the buffer.
        
        Args:
            duration_seconds: Duration of audio to read
            
        Returns:
            Audio samples as numpy array (same dtype as buffer)
        """
        with self.lock:
            num_samples = int(duration_seconds * self.sample_rate)
            
            # Limit to what we actually have
            if self.total_written < self.capacity_samples:
                # Buffer hasn't wrapped yet
                available = self.total_written
            else:
                # Buffer has wrapped
                available = self.capacity_samples
            
            num_samples = min(num_samples, available)
            
            if num_samples == 0:
                return np.array([], dtype=self.dtype)
            
            # Calculate read start position
            if self.total_written < self.capacity_samples:
                # Buffer hasn't wrapped
                read_start = max(0, self.write_pos - num_samples)
                if read_start + num_samples <= self.write_pos:
                    # Simple case
                    return self.buffer[read_start:self.write_pos].copy()
                else:
                    # This shouldn't happen if buffer hasn't wrapped
                    return self.buffer[read_start:self.write_pos].copy()
            else:
                # Buffer has wrapped
                read_start = (self.write_pos - num_samples) % self.capacity_samples
                
                if read_start < self.write_pos:
                    # Simple case: continuous read
                    return self.buffer[read_start:self.write_pos].copy()
                else:
                    # Wrap case: need to concatenate
                    part1 = self.buffer[read_start:].copy()
                    part2 = self.buffer[:self.write_pos].copy()
                    return np.concatenate([part1, part2])
    
    def read_last_as_int16(self, duration_seconds: float) -> np.ndarray:
        """
        Read the last N seconds of audio from the buffer as int16.
        
        This is for backward compatibility with components expecting int16.
        
        Args:
            duration_seconds: Duration of audio to read
            
        Returns:
            Audio samples as int16 numpy array
        """
        audio = self.read_last(duration_seconds)
        if self.dtype == np.float32 and len(audio) > 0:
            # Convert float32 to int16
            return (audio * 32768.0).astype(np.int16)
        return audio.astype(np.int16)
    
    def clear(self) -> None:
        """Clear the buffer and reset positions."""
        with self.lock:
            self.buffer = np.zeros(self.capacity_samples, dtype=self.dtype)
            self.write_pos = 0
            self.total_written = 0
    
    def get_fill_level(self) -> float:
        """
        Get the current fill level of the buffer.
        
        Returns:
            Fill level as a fraction (0.0 to 1.0)
        """
        with self.lock:
            if self.total_written < self.capacity_samples:
                return self.total_written / self.capacity_samples
            else:
                return 1.0