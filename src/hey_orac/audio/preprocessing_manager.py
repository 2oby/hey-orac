"""
Audio Preprocessing Manager

Manages audio preprocessing with graceful fallback support.
Provides a unified interface for getting audio whether preprocessing
is available or not.
"""

import logging
from typing import Optional, Union
import numpy as np

from hey_orac.audio.microphone import AudioCapture
from hey_orac.audio.ring_buffer import RingBuffer
from hey_orac.config.manager import SettingsManager


class PreprocessingManager:
    """Manages audio preprocessing with graceful fallback"""
    
    def __init__(self, settings_manager: SettingsManager, logger: Optional[logging.Logger] = None):
        """
        Initialize the preprocessing manager.
        
        Args:
            settings_manager: Settings manager instance
            logger: Optional logger instance
        """
        self.settings_manager = settings_manager
        self.logger = logger or logging.getLogger(__name__)
        
        self.audio_capture: Optional[AudioCapture] = None
        self.fallback_mode = False
        self.stream = None
        self.ring_buffer: Optional[RingBuffer] = None
        self._initialized = False
        
    def initialize(self, usb_mic: Optional[dict], stream, audio_config: dict) -> bool:
        """
        Initialize preprocessing if available and enabled.
        
        Args:
            usb_mic: USB microphone info if detected
            stream: PyAudio stream for fallback
            audio_config: Audio configuration dict
            
        Returns:
            bool: True if preprocessing is active, False if using fallback
        """
        self.stream = stream
        
        # Check if preprocessing is enabled
        with self.settings_manager.get_config() as config:
            preprocessing_enabled = getattr(config.system, 'enable_audio_preprocessing', False)
            self.logger.debug(f"Config system attributes: {dir(config.system)}")
            self.logger.debug(f"Preprocessing enabled flag: {preprocessing_enabled}")
        
        if not preprocessing_enabled:
            self.logger.info("Audio preprocessing disabled by configuration")
            self.fallback_mode = True
            self._initialized = True
            return False
            
        # Try to initialize AudioCapture with preprocessing
        try:
            if usb_mic is None:
                self.logger.warning("No USB microphone detected, falling back to raw audio")
                self.fallback_mode = True
                self._initialized = True
                return False
                
            self.logger.info("Initializing audio preprocessing...")
            
            # Create AudioCapture instance
            from hey_orac.audio.microphone import AudioCapture
            from hey_orac.audio.preprocessor import AudioPreprocessorConfig
            
            # Get preprocessing config
            preprocessing_config = audio_config.get('preprocessing', {})
            preprocessor_config = AudioPreprocessorConfig(
                enable_agc=preprocessing_config.get('enable_agc', True),
                agc_target_level=preprocessing_config.get('agc_target_level', 0.3),
                agc_max_gain=preprocessing_config.get('agc_max_gain', 10.0),
                agc_attack_time=preprocessing_config.get('agc_attack_time', 0.01),
                agc_release_time=preprocessing_config.get('agc_release_time', 0.1),
                enable_compression=preprocessing_config.get('enable_compression', True),
                compression_threshold=preprocessing_config.get('compression_threshold', 0.5),
                compression_ratio=preprocessing_config.get('compression_ratio', 4.0),
                enable_limiter=preprocessing_config.get('enable_limiter', True),
                limiter_threshold=preprocessing_config.get('limiter_threshold', 0.95),
                enable_noise_gate=preprocessing_config.get('enable_noise_gate', False),
                noise_gate_threshold=preprocessing_config.get('noise_gate_threshold', 0.01)
            )
            
            self.audio_capture = AudioCapture(
                sample_rate=audio_config.get('sample_rate', 16000),
                chunk_size=audio_config.get('chunk_size', 1280),
                ring_buffer_seconds=10.0,
                device_index=usb_mic.index,
                preprocessor_config=preprocessor_config
            )
            
            # Start the audio capture
            if self.audio_capture.start():
                self.ring_buffer = self.audio_capture.ring_buffer
                self.fallback_mode = False
                self._initialized = True
                self.logger.info("✅ Audio preprocessing enabled successfully")
                return True
            else:
                self.logger.error("Failed to start AudioCapture")
                self._cleanup_audio_capture()
                self.fallback_mode = True
                self._initialized = True
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize audio preprocessing: {e}")
            self._cleanup_audio_capture()
            self.fallback_mode = True
            self._initialized = True
            return False
    
    def get_audio_chunk(self, chunk_size: int = 1280) -> Optional[np.ndarray]:
        """
        Get audio chunk with or without preprocessing.
        
        Args:
            chunk_size: Number of samples to read
            
        Returns:
            Audio data as numpy array or None if error
        """
        if not self._initialized:
            self.logger.error("PreprocessingManager not initialized")
            return None
            
        try:
            if not self.fallback_mode and self.audio_capture and self.audio_capture.is_active():
                # Get preprocessed audio from AudioCapture
                chunk = self.audio_capture.get_audio_chunk()
                if chunk is not None and len(chunk) == chunk_size:
                    return chunk
                else:
                    # AudioCapture failed, switch to fallback
                    self.logger.warning("AudioCapture read failed, switching to fallback mode")
                    self._cleanup_audio_capture()
                    self.fallback_mode = True
                    
            # Fallback: read from raw stream
            if self.stream:
                try:
                    raw_data = self.stream.read(chunk_size, exception_on_overflow=False)
                    # Convert to numpy array (assuming int16)
                    audio_data = np.frombuffer(raw_data, dtype=np.int16)
                    # Normalize to float32 [-1, 1] to match preprocessed format
                    return audio_data.astype(np.float32) / 32768.0
                except Exception as e:
                    self.logger.error(f"Error reading from stream: {e}")
                    return None
            else:
                self.logger.error("No audio source available")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting audio chunk: {e}")
            return None
    
    def get_ring_buffer(self) -> Optional[RingBuffer]:
        """
        Get the ring buffer instance.
        
        Returns:
            RingBuffer instance or None if not available
        """
        if not self.fallback_mode and self.audio_capture:
            return self.audio_capture.ring_buffer
        return self.ring_buffer
    
    def is_preprocessing_active(self) -> bool:
        """
        Check if preprocessing is currently active.
        
        Returns:
            bool: True if preprocessing is active
        """
        return (not self.fallback_mode and 
                self.audio_capture is not None and 
                self.audio_capture.is_active())
    
    def get_status(self) -> dict:
        """
        Get current preprocessing status.
        
        Returns:
            dict: Status information
        """
        return {
            'initialized': self._initialized,
            'preprocessing_active': self.is_preprocessing_active(),
            'fallback_mode': self.fallback_mode,
            'audio_capture_active': self.audio_capture.is_active() if self.audio_capture else False
        }
    
    def cleanup(self):
        """Clean up resources"""
        self._cleanup_audio_capture()
        self.stream = None
        self.ring_buffer = None
        self._initialized = False
        
    def _cleanup_audio_capture(self):
        """Clean up AudioCapture resources"""
        if self.audio_capture:
            try:
                self.audio_capture.stop()
            except Exception as e:
                self.logger.error(f"Error stopping AudioCapture: {e}")
            self.audio_capture = None