"""
Microphone audio capture with PyAudio.
"""

import pyaudio
import numpy as np
import logging
import threading
import queue
from typing import Optional, Callable, Tuple
from dataclasses import dataclass

from .ring_buffer import RingBuffer
from .preprocessor import AudioPreprocessor, AudioPreprocessorConfig


logger = logging.getLogger(__name__)


@dataclass
class AudioDevice:
    """Information about an audio device."""
    index: int
    name: str
    channels: int
    sample_rate: int
    is_usb: bool = False


class AudioCapture:
    """
    Manages audio capture from microphone with ring buffer storage.
    
    Features:
    - Automatic USB microphone detection
    - Continuous audio capture to ring buffer
    - RMS level monitoring
    - Thread-safe operation
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1280,
        ring_buffer_seconds: float = 10.0,
        device_index: Optional[int] = None,
        preprocessor_config: Optional[AudioPreprocessorConfig] = None
    ):
        """
        Initialize audio capture.
        
        Args:
            sample_rate: Sample rate in Hz
            chunk_size: Number of samples per chunk
            ring_buffer_seconds: Ring buffer capacity in seconds
            device_index: Specific device index, or None for auto-detect
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.device_index = device_index
        
        # Initialize PyAudio
        self.pyaudio = pyaudio.PyAudio()
        
        # Ring buffer for audio storage (using float32)
        self.ring_buffer = RingBuffer(ring_buffer_seconds, sample_rate, dtype=np.float32)
        
        # Audio preprocessor
        self.preprocessor_config = preprocessor_config or AudioPreprocessorConfig(sample_rate=sample_rate)
        self.preprocessor = AudioPreprocessor(self.preprocessor_config)
        
        # Audio stream
        self.stream: Optional[pyaudio.Stream] = None
        
        # RMS monitoring
        self.current_rms = 0.0
        self.rms_lock = threading.Lock()
        
        # Audio callback queue
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Control flags
        self.is_running = False
        self.capture_thread: Optional[threading.Thread] = None
        
    def find_usb_microphone(self) -> Optional[AudioDevice]:
        """Find and return the first USB microphone device."""
        for i in range(self.pyaudio.get_device_count()):
            info = self.pyaudio.get_device_info_by_index(i)
            
            # Check if it's an input device
            if info['maxInputChannels'] > 0:
                name = info['name'].lower()
                # Common USB microphone identifiers
                if any(usb_id in name for usb_id in ['usb', 'sh-04', 'blue', 'yeti', 'samson']):
                    logger.info(f"Found USB microphone: {info['name']} (index {i})")
                    return AudioDevice(
                        index=i,
                        name=info['name'],
                        channels=min(info['maxInputChannels'], 2),  # Use stereo if available
                        sample_rate=int(info['defaultSampleRate']),
                        is_usb=True
                    )
        return None
    
    def start(self) -> bool:
        """
        Start audio capture.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Audio capture already running")
            return True
        
        # Find device if not specified
        if self.device_index is None:
            device = self.find_usb_microphone()
            if not device:
                logger.error("No USB microphone found")
                return False
            self.device_index = device.index
            device_channels = device.channels
        else:
            # Get device info
            info = self.pyaudio.get_device_info_by_index(self.device_index)
            device_channels = min(info['maxInputChannels'], 2)
        
        try:
            # Open audio stream
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=device_channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.start()
            
            logger.info(f"Audio capture started on device {self.device_index}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            return False
    
    def stop(self) -> None:
        """Stop audio capture."""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        
        logger.info("Audio capture stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for audio data."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Queue audio data for processing
        try:
            self.audio_queue.put_nowait(in_data)
        except queue.Full:
            logger.warning("Audio queue full, dropping frame")
        
        return (None, pyaudio.paContinue)
    
    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        logger.info("Audio capture loop started")
        
        while self.is_running:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Convert to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # Convert stereo to mono if needed
                if len(audio_array) > self.chunk_size:
                    # Stereo data - convert to mono
                    stereo_data = audio_array.reshape(-1, 2)
                    audio_array = np.mean(stereo_data, axis=1).astype(np.int16)
                
                # Convert to float32 early for preprocessing
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Apply preprocessing (AGC, compression, limiting)
                processed_audio = self.preprocessor.process(audio_float)
                
                # Write to ring buffer (already float32)
                self.ring_buffer.write(processed_audio)
                
                # Update RMS from processed audio
                with self.rms_lock:
                    self.current_rms = float(np.sqrt(np.mean(processed_audio ** 2)))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
        
        logger.info("Audio capture loop stopped")
    
    def get_rms(self) -> float:
        """Get current RMS level."""
        with self.rms_lock:
            return self.current_rms
    
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """
        Get a chunk of audio data for processing.
        
        Returns:
            Audio chunk as float32 array, or None if not available
        """
        try:
            # Get the last chunk_size samples (already float32)
            audio_data = self.ring_buffer.read_last(self.chunk_size / self.sample_rate)
            if len(audio_data) == self.chunk_size:
                # Already float32 from preprocessing
                return audio_data
            return None
        except Exception as e:
            logger.error(f"Error getting audio chunk: {e}")
            return None
    
    def get_pre_roll(self, seconds: float) -> np.ndarray:
        """
        Get pre-roll audio from ring buffer.
        
        Args:
            seconds: Duration of pre-roll to retrieve
            
        Returns:
            Audio data as float32 array
        """
        return self.ring_buffer.read_last(seconds)
    
    def get_pre_roll_int16(self, seconds: float) -> np.ndarray:
        """
        Get pre-roll audio from ring buffer as int16.
        
        Args:
            seconds: Duration of pre-roll to retrieve
            
        Returns:
            Audio data as int16 array
        """
        return self.ring_buffer.read_last_as_int16(seconds)
    
    def get_audio_metrics(self) -> dict:
        """
        Get audio quality metrics from preprocessor.
        
        Returns:
            Dictionary of audio metrics
        """
        metrics = self.preprocessor.get_metrics()
        metrics['current_rms'] = self.get_rms()
        metrics['buffer_fill_level'] = self.ring_buffer.get_fill_level()
        return metrics
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
        if hasattr(self, 'pyaudio'):
            self.pyaudio.terminate()