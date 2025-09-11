#!/usr/bin/env python3
"""
Audio reader thread for non-blocking audio capture.
Isolates the blocking stream.read() call in a separate thread with queue communication.
"""

import threading
import queue
import logging
import time
import sys
from typing import Optional, Any

logger = logging.getLogger(__name__)


class AudioReaderThread:
    """
    Dedicated thread for reading audio data from stream.
    Communicates with main thread via Queue.
    """
    
    def __init__(self, stream: Any, chunk_size: int = 1280, queue_maxsize: int = 10):
        """
        Initialize the audio reader thread.
        
        Args:
            stream: PyAudio stream object
            chunk_size: Number of samples to read per chunk
            queue_maxsize: Maximum number of chunks to buffer in queue
        """
        self.stream = stream
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue(maxsize=queue_maxsize)
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.last_read_time = time.time()
        self.read_count = 0
        self.error_count = 0
        self.restart_count = 0
        
    def start(self) -> bool:
        """Start the audio reader thread."""
        if self.running:
            logger.warning("Audio reader thread already running")
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        logger.info(f"🎤 Audio reader thread started (restart #{self.restart_count})")
        return True
        
    def stop(self) -> None:
        """Stop the audio reader thread gracefully."""
        if not self.running:
            return
            
        logger.info("📴 Stopping audio reader thread...")
        self.running = False
        
        # Clear the queue to unblock any pending puts
        try:
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass
            
        # Wait for thread to finish with timeout
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("⚠️ Audio reader thread did not stop gracefully")
                
    def restart(self) -> bool:
        """Restart the audio reader thread."""
        logger.info("🔄 Restarting audio reader thread...")
        self.stop()
        time.sleep(0.1)  # Brief pause before restart
        self.restart_count += 1
        return self.start()
        
    def _reader_loop(self) -> None:
        """Main loop that runs in the audio thread."""
        logger.debug("Audio reader loop started")
        
        while self.running:
            try:
                # This is the blocking call that can hang
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                if data is None or len(data) == 0:
                    logger.warning("No audio data read from stream")
                    time.sleep(0.01)  # Brief pause before retry
                    continue
                    
                # Update metrics
                self.last_read_time = time.time()
                self.read_count += 1
                
                # Try to put data in queue with small timeout
                try:
                    self.audio_queue.put(data, timeout=0.1)
                except queue.Full:
                    # Queue is full, drop oldest item and try again
                    logger.debug("Audio queue full, dropping oldest chunk")
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put(data, timeout=0.1)
                    except (queue.Empty, queue.Full):
                        logger.warning("Failed to queue audio data")
                        
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error in audio reader thread: {e}")
                if self.error_count > 10:
                    logger.error("Too many errors in audio reader, stopping thread")
                    self.running = False
                    break
                time.sleep(0.01)  # Brief pause before retry
                
        logger.debug("Audio reader loop ended")
        
    def get_audio(self, timeout: float = 2.0) -> Optional[bytes]:
        """
        Get audio data from the queue with timeout.
        
        Args:
            timeout: Maximum time to wait for data (seconds)
            
        Returns:
            Audio data bytes or None if timeout/error
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def is_healthy(self, max_age: float = 3.0) -> bool:
        """
        Check if the audio thread is healthy.
        
        Args:
            max_age: Maximum age of last read in seconds
            
        Returns:
            True if thread is healthy, False otherwise
        """
        if not self.running or not self.thread or not self.thread.is_alive():
            return False
            
        # Check if we've read data recently
        age = time.time() - self.last_read_time
        if age > max_age:
            logger.warning(f"⚠️ Audio thread unhealthy: last read {age:.1f}s ago")
            return False
            
        return True
        
    def get_stats(self) -> dict:
        """Get statistics about the audio reader thread."""
        return {
            'running': self.running,
            'thread_alive': self.thread.is_alive() if self.thread else False,
            'last_read_age': time.time() - self.last_read_time,
            'read_count': self.read_count,
            'error_count': self.error_count,
            'restart_count': self.restart_count,
            'queue_size': self.audio_queue.qsize()
        }