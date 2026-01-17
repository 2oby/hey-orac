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

# Import scipy.signal at module load time to avoid delays during audio processing
try:
    from scipy.signal import butter, lfilter, savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    butter = None
    lfilter = None
    savgol_filter = None

logger = logging.getLogger(__name__)


@dataclass
class AudioPreprocessorConfig:
    """Configuration for audio preprocessing."""
    # High-pass filter to remove low frequency rumble/hum
    enable_highpass: bool = True
    highpass_cutoff: float = 80.0  # Cutoff frequency in Hz

    # Presence boost for speech clarity (2-4kHz range)
    enable_presence_boost: bool = True
    presence_gain_db: float = 3.0  # Boost in dB (subtle enhancement)

    # AGC settings
    enable_agc: bool = True
    agc_target_level: float = 0.3  # Target RMS level (0.0-1.0)
    agc_max_gain: float = 20.0  # Maximum gain in dB (increased from 10)
    agc_attack_time: float = 0.01  # AGC attack time in seconds
    agc_release_time: float = 0.1  # AGC release time in seconds

    # Compression settings
    enable_compression: bool = True
    compression_threshold: float = 0.5  # Threshold for compression (0.0-1.0)
    compression_ratio: float = 4.0  # Compression ratio (e.g., 4:1)

    # Limiter settings
    enable_limiter: bool = True
    limiter_threshold: float = 0.95  # Hard limit threshold (0.0-1.0)
    limiter_lookahead: float = 0.005  # Lookahead time in seconds

    # Noise gate to cut background noise during silence
    # CAUTION: Enabling noise gate can cause RMS=0 issues if threshold is too high
    enable_noise_gate: bool = False  # Disabled by default - can cause issues
    noise_gate_threshold: float = 0.005  # Very low threshold to avoid gating speech

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

        # High-pass filter coefficients (Butterworth 2nd order)
        if self.config.enable_highpass:
            self._init_highpass_filter()

        # Presence boost filter coefficients (peaking EQ at 3kHz)
        if self.config.enable_presence_boost:
            self._init_presence_filter()

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

    def _init_highpass_filter(self):
        """Initialize high-pass filter coefficients (Butterworth 2nd order)."""
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, disabling high-pass filter")
            self.config.enable_highpass = False
            return
        nyquist = self.config.sample_rate / 2.0
        normalized_cutoff = self.config.highpass_cutoff / nyquist
        # Clamp to valid range
        normalized_cutoff = min(max(normalized_cutoff, 0.001), 0.99)
        self.hp_b, self.hp_a = butter(2, normalized_cutoff, btype='high')
        # Filter state for continuous processing
        self.hp_zi = np.zeros(max(len(self.hp_a), len(self.hp_b)) - 1)

    def _init_presence_filter(self):
        """Initialize presence boost filter (peaking EQ at 3kHz)."""
        # Peaking EQ filter design using Q factor instead of bandwidth
        center_freq = 3000.0  # 3kHz - speech presence range
        Q = 1.0  # Quality factor (lower = wider bandwidth)
        gain_db = self.config.presence_gain_db

        # Calculate filter coefficients (peaking EQ)
        A = 10 ** (gain_db / 40.0)
        omega = 2 * np.pi * center_freq / self.config.sample_rate
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        alpha = sin_omega / (2 * Q)

        b0 = 1 + alpha * A
        b1 = -2 * cos_omega
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_omega
        a2 = 1 - alpha / A

        # Normalize coefficients
        self.pres_b = np.array([b0/a0, b1/a0, b2/a0])
        self.pres_a = np.array([1.0, a1/a0, a2/a0])
        # Filter state for continuous processing
        self.pres_zi = np.zeros(2)
    
    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Process audio chunk through the preprocessing pipeline.

        Processing order:
        1. High-pass filter (remove rumble/hum)
        2. Noise gate (cut background noise)
        3. Presence boost (add clarity)
        4. AGC (normalize levels)
        5. Compression (even out dynamics)
        6. Limiter (prevent clipping)

        Args:
            audio_chunk: Audio data as float32 numpy array (-1.0 to 1.0)

        Returns:
            Processed audio chunk
        """
        if audio_chunk.dtype != np.float32:
            raise ValueError(f"Expected float32 audio, got {audio_chunk.dtype}")

        # Make a copy to avoid modifying the original
        audio = audio_chunk.copy()

        # Apply high-pass filter first (remove low frequency rumble)
        if self.config.enable_highpass:
            audio = self._apply_highpass(audio)

        # Apply noise gate (cut background noise during silence)
        if self.config.enable_noise_gate:
            audio = self._apply_noise_gate(audio)

        # Apply presence boost (add clarity to speech frequencies)
        if self.config.enable_presence_boost:
            audio = self._apply_presence_boost(audio)

        # Apply AGC (normalize levels - boost quiet, tame loud)
        if self.config.enable_agc:
            audio = self._apply_agc(audio)

        # Apply compression (even out dynamics)
        if self.config.enable_compression:
            audio = self._apply_compression(audio)

        # Apply limiter (prevent clipping)
        if self.config.enable_limiter:
            audio = self._apply_limiter(audio)

        # Update metrics
        self._update_metrics(audio)

        return audio

    def _apply_highpass(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter to remove low frequency rumble/hum.

        Args:
            audio: Input audio

        Returns:
            High-pass filtered audio
        """
        if not SCIPY_AVAILABLE or not hasattr(self, 'hp_b'):
            return audio
        try:
            filtered, self.hp_zi = lfilter(self.hp_b, self.hp_a, audio, zi=self.hp_zi)
            return filtered.astype(np.float32)
        except Exception as e:
            logger.warning(f"High-pass filter failed: {e}")
            return audio

    def _apply_presence_boost(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply presence boost filter for speech clarity.

        Args:
            audio: Input audio

        Returns:
            Presence-boosted audio
        """
        if not SCIPY_AVAILABLE or not hasattr(self, 'pres_b'):
            return audio
        try:
            filtered, self.pres_zi = lfilter(self.pres_b, self.pres_a, audio, zi=self.pres_zi)
            return filtered.astype(np.float32)
        except Exception as e:
            logger.warning(f"Presence boost filter failed: {e}")
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

        # Check if any samples pass the gate - if not, return original audio
        # to avoid zeroing everything out
        if not np.any(gate_mask):
            return audio

        # Smooth gate transitions to avoid clicks (only if scipy available)
        if SCIPY_AVAILABLE and savgol_filter is not None and len(audio) >= 7:
            try:
                # Ensure window_length is odd and at least 5, and > polyorder
                window_length = min(51, len(audio))
                if window_length % 2 == 0:
                    window_length -= 1
                window_length = max(window_length, 5)  # Must be > polyorder (3)

                gate_smooth = savgol_filter(gate_mask.astype(float),
                                           window_length=window_length,
                                           polyorder=3)
                # Clip to 0-1 range
                gate_smooth = np.clip(gate_smooth, 0.0, 1.0)
                return (audio * gate_smooth).astype(np.float32)
            except Exception as e:
                logger.debug(f"Noise gate smoothing failed, using simple gate: {e}")

        # Fallback to simple gating
        return (audio * gate_mask).astype(np.float32)
    
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
            Dictionary of metrics (all values JSON-serializable)
        """
        agc_gain_db = float(20 * np.log10(self.agc_gain)) if self.agc_gain > 0 else float('-inf')
        return {
            "agc_gain": float(self.agc_gain),
            "peak_level": float(self.peak_level),
            "clipping_count": int(self.clipping_count),
            "agc_gain_db": agc_gain_db
        }
    
    def reset_metrics(self) -> None:
        """Reset audio quality metrics."""
        self.clipping_count = 0
        self.peak_level = 0.0