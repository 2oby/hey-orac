"""
Audio preprocessing module for Hey Orac wake-word detection.

Provides AGC, compression, limiting, and noise reduction to improve
audio quality before STT processing.
"""

import numpy as np
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
import collections

logger = logging.getLogger(__name__)


@dataclass
class AudioPreprocessorConfig:
    """Configuration for audio preprocessing."""
    enable_agc: bool = True
    agc_target_level: float = 0.3  # Target RMS level (0.0-1.0)
    agc_max_gain: float = 10.0  # Maximum gain in dB
    agc_attack_time: float = 0.01  # AGC attack time in seconds
    agc_release_time: float = 0.1  # AGC release time in seconds
    
    enable_compression: bool = True
    compression_threshold: float = 0.5  # Threshold for compression (0.0-1.0)
    compression_ratio: float = 4.0  # Compression ratio (e.g., 4:1)
    
    enable_limiter: bool = True
    limiter_threshold: float = 0.95  # Hard limit threshold (0.0-1.0)
    limiter_lookahead: float = 0.005  # Lookahead time in seconds
    
    enable_noise_gate: bool = False
    noise_gate_threshold: float = 0.01  # Noise gate threshold (0.0-1.0)
    
    sample_rate: int = 16000


class AudioPreprocessor:
    """
    Audio preprocessor with AGC, compression, limiting, and noise reduction.
    
    Processes audio in float32 format (-1.0 to 1.0 range).
    """
    
    def __init__(self, config: Optional[AudioPreprocessorConfig] = None):
        """
        Initialize audio preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or AudioPreprocessorConfig()
        
        # AGC state
        self.agc_gain = 1.0
        self.agc_target_rms = self.config.agc_target_level
        self.agc_attack_coeff = 1.0 - np.exp(-1.0 / (self.config.agc_attack_time * self.config.sample_rate))
        self.agc_release_coeff = 1.0 - np.exp(-1.0 / (self.config.agc_release_time * self.config.sample_rate))
        
        # Compression state
        self.comp_threshold = self.config.compression_threshold
        self.comp_ratio = self.config.compression_ratio
        self.comp_knee_width = 0.1  # Soft knee width
        
        # Limiter state
        self.limiter_threshold = self.config.limiter_threshold
        self.limiter_buffer_size = int(self.config.limiter_lookahead * self.config.sample_rate)
        self.limiter_buffer = collections.deque(maxlen=self.limiter_buffer_size)
        
        # RMS calculation buffer
        self.rms_window_size = int(0.02 * self.config.sample_rate)  # 20ms window
        self.rms_buffer = collections.deque(maxlen=self.rms_window_size)
        
        # Metrics
        self.clipping_count = 0
        self.peak_level = 0.0
        
        logger.info(f"Audio preprocessor initialized with config: {self.config}")
    
    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Process audio chunk through the preprocessing pipeline.
        
        Args:
            audio_chunk: Audio data as float32 numpy array (-1.0 to 1.0)
            
        Returns:
            Processed audio chunk
        """
        if audio_chunk.dtype != np.float32:
            raise ValueError(f"Expected float32 audio, got {audio_chunk.dtype}")
        
        # Make a copy to avoid modifying the original
        audio = audio_chunk.copy()
        
        # Apply noise gate first (if enabled)
        if self.config.enable_noise_gate:
            audio = self._apply_noise_gate(audio)
        
        # Apply AGC (if enabled)
        if self.config.enable_agc:
            audio = self._apply_agc(audio)
        
        # Apply compression (if enabled)
        if self.config.enable_compression:
            audio = self._apply_compression(audio)
        
        # Apply limiter (if enabled)
        if self.config.enable_limiter:
            audio = self._apply_limiter(audio)
        
        # Update metrics
        self._update_metrics(audio)
        
        return audio
    
    def _apply_agc(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Automatic Gain Control to normalize audio levels.
        
        Args:
            audio: Input audio
            
        Returns:
            AGC-processed audio
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Avoid division by zero
        if rms < 1e-10:
            return audio
        
        # Calculate desired gain
        desired_gain = self.agc_target_rms / rms
        
        # Limit maximum gain
        max_gain_linear = 10 ** (self.config.agc_max_gain / 20.0)
        desired_gain = min(desired_gain, max_gain_linear)
        
        # Apply attack/release envelope
        if desired_gain > self.agc_gain:
            # Attack (gain increasing)
            self.agc_gain += self.agc_attack_coeff * (desired_gain - self.agc_gain)
        else:
            # Release (gain decreasing)
            self.agc_gain += self.agc_release_coeff * (desired_gain - self.agc_gain)
        
        # Apply gain
        return audio * self.agc_gain
    
    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply dynamic range compression.
        
        Args:
            audio: Input audio
            
        Returns:
            Compressed audio
        """
        # Get absolute values for compression calculation
        abs_audio = np.abs(audio)
        
        # Calculate compression gain for each sample
        gain = np.ones_like(abs_audio)
        
        # Apply soft knee compression
        for i in range(len(abs_audio)):
            level = abs_audio[i]
            
            if level > self.comp_threshold:
                # Above threshold - apply compression
                excess = level - self.comp_threshold
                compressed_excess = excess / self.comp_ratio
                gain[i] = (self.comp_threshold + compressed_excess) / level
            
            # Soft knee smoothing
            knee_start = self.comp_threshold - self.comp_knee_width / 2
            knee_end = self.comp_threshold + self.comp_knee_width / 2
            
            if knee_start < level < knee_end:
                # In knee region - smooth transition
                knee_pos = (level - knee_start) / self.comp_knee_width
                gain[i] = 1.0 - (1.0 - gain[i]) * (knee_pos ** 2)
        
        # Apply gain
        return audio * gain
    
    def _apply_limiter(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply peak limiting to prevent clipping.
        
        Args:
            audio: Input audio
            
        Returns:
            Limited audio
        """
        # Simple brick-wall limiter with lookahead
        limited = audio.copy()
        
        # Add current chunk to lookahead buffer
        self.limiter_buffer.extend(audio)
        
        # Apply limiting
        for i in range(len(limited)):
            # Check for peaks in lookahead window
            lookahead_start = max(0, i - self.limiter_buffer_size // 2)
            lookahead_end = min(len(limited), i + self.limiter_buffer_size // 2)
            
            if lookahead_end > lookahead_start:
                window = limited[lookahead_start:lookahead_end]
                peak = np.max(np.abs(window))
                
                if peak > self.limiter_threshold:
                    # Calculate limiting gain
                    limiting_gain = self.limiter_threshold / peak
                    
                    # Apply smooth limiting
                    limited[i] *= limiting_gain
        
        # Hard clip as final safety
        limited = np.clip(limited, -self.limiter_threshold, self.limiter_threshold)
        
        return limited
    
    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise gate to reduce low-level noise.
        
        Args:
            audio: Input audio
            
        Returns:
            Gated audio
        """
        # Calculate envelope
        envelope = np.abs(audio)
        
        # Apply gate
        gate_mask = envelope > self.config.noise_gate_threshold
        
        # Smooth gate transitions to avoid clicks
        from scipy.signal import savgol_filter
        try:
            gate_smooth = savgol_filter(gate_mask.astype(float), 
                                       window_length=min(51, len(audio) if len(audio) % 2 == 1 else len(audio) - 1),
                                       polyorder=3)
            return audio * gate_smooth
        except:
            # Fallback to simple gating if smoothing fails
            return audio * gate_mask
    
    def _update_metrics(self, audio: np.ndarray) -> None:
        """
        Update audio quality metrics.
        
        Args:
            audio: Processed audio
        """
        # Update peak level
        current_peak = np.max(np.abs(audio))
        self.peak_level = max(self.peak_level, current_peak)
        
        # Count clipping
        clipped_samples = np.sum(np.abs(audio) >= 0.99)
        if clipped_samples > 0:
            self.clipping_count += clipped_samples
            logger.warning(f"Clipping detected: {clipped_samples} samples")
    
    def get_metrics(self) -> dict:
        """
        Get current audio quality metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "agc_gain": self.agc_gain,
            "peak_level": self.peak_level,
            "clipping_count": self.clipping_count,
            "agc_gain_db": 20 * np.log10(self.agc_gain) if self.agc_gain > 0 else -np.inf
        }
    
    def reset_metrics(self) -> None:
        """Reset audio quality metrics."""
        self.clipping_count = 0
        self.peak_level = 0.0