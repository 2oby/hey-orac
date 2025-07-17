"""
Ring buffer implementation for audio capture with pre-roll capability.
"""

import array
import threading
from typing import Optional, Tuple
import numpy as np


class RingBuffer:
    """
    Thread-safe ring buffer for audio data storage.
    
    Stores audio samples in a circular buffer, allowing for efficient
    continuous recording with the ability to retrieve past audio (pre-roll).
    """
    
    def __init__(self, capacity_seconds: float = 10.0, sample_rate: int = 16000):
        """
        Initialize the ring buffer.
        
        Args:
            capacity_seconds: Buffer capacity in seconds
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.capacity_samples = int(capacity_seconds * sample_rate)
        
        # Use array.array for efficient storage of 16-bit audio
        self.buffer = array.array('h', [0] * self.capacity_samples)
        
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
            samples: Audio samples as numpy array (int16)
        """
        with self.lock:
            num_samples = len(samples)
            
            # Handle case where samples wrap around buffer end
            if self.write_pos + num_samples <= self.capacity_samples:
                # Simple case: all samples fit before wrap
                self.buffer[self.write_pos:self.write_pos + num_samples] = array.array('h', samples)
            else:
                # Wrap case: split samples
                first_part = self.capacity_samples - self.write_pos
                self.buffer[self.write_pos:] = array.array('h', samples[:first_part])
                self.buffer[:num_samples - first_part] = array.array('h', samples[first_part:])
            
            # Update write position and total written
            self.write_pos = (self.write_pos + num_samples) % self.capacity_samples
            self.total_written += num_samples
    
    def read_last(self, duration_seconds: float) -> np.ndarray:
        """
        Read the last N seconds of audio from the buffer.
        
        Args:
            duration_seconds: Duration of audio to read
            
        Returns:
            Audio samples as numpy array
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
                return np.array([], dtype=np.int16)
            
            # Calculate read start position
            if self.total_written < self.capacity_samples:
                # Buffer hasn't wrapped
                read_start = max(0, self.write_pos - num_samples)
                if read_start + num_samples <= self.write_pos:
                    # Simple case
                    return np.array(self.buffer[read_start:self.write_pos], dtype=np.int16)
                else:
                    # This shouldn't happen if buffer hasn't wrapped
                    return np.array(self.buffer[read_start:self.write_pos], dtype=np.int16)
            else:
                # Buffer has wrapped
                read_start = (self.write_pos - num_samples) % self.capacity_samples
                
                if read_start < self.write_pos:
                    # Simple case: continuous read
                    return np.array(self.buffer[read_start:self.write_pos], dtype=np.int16)
                else:
                    # Wrap case: need to concatenate
                    part1 = np.array(self.buffer[read_start:], dtype=np.int16)
                    part2 = np.array(self.buffer[:self.write_pos], dtype=np.int16)
                    return np.concatenate([part1, part2])
    
    def clear(self) -> None:
        """Clear the buffer and reset positions."""
        with self.lock:
            self.buffer = array.array('h', [0] * self.capacity_samples)
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