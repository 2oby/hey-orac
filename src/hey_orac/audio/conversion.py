"""
Audio format conversion utilities for Hey ORAC.

This module provides audio conversion functions for different components:
- OpenWakeWord: Un-normalized float32 (raw int16 values as float32)
- STT: Normalized float32 (-1.0 to 1.0 range)

CRITICAL: OpenWakeWord requires un-normalized audio! The model was trained on
raw int16 values converted to float32 WITHOUT normalization. Normalizing the
audio (dividing by 32768.0) will break wake word detection.

Historical context: This normalization bug was discovered during development
and caused wake word detection failures. See devlog.md for details.
"""

import numpy as np
from hey_orac import constants


def convert_to_openwakeword_format(
    audio_data: bytes,
    channels: int = None
) -> np.ndarray:
    """
    Convert audio bytes to OpenWakeWord-compatible float32 format.

    CRITICAL: This function does NOT normalize the audio. OpenWakeWord expects
    raw int16 values converted to float32 without division by 32768.0.

    Args:
        audio_data: Raw audio bytes from microphone or WAV file
        channels: Number of audio channels (1=mono, 2=stereo).
                 If None, auto-detect based on array length.

    Returns:
        Mono float32 numpy array suitable for OpenWakeWord

    Examples:
        # Auto-detect stereo/mono from chunk size
        >>> audio = convert_to_openwakeword_format(data)

        # Explicit channel count (useful for WAV files)
        >>> audio = convert_to_openwakeword_format(frames, channels=2)
    """
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Determine if stereo based on channels parameter or array length
    if channels is not None:
        is_stereo = (channels == constants.CHANNELS_STEREO)
    else:
        # Auto-detect: stereo data has 2x samples (left + right)
        is_stereo = len(audio_array) > constants.CHUNK_SIZE

    if is_stereo:
        # Convert stereo to mono by averaging left and right channels
        stereo_data = audio_array.reshape(-1, 2)
        audio_mono = np.mean(stereo_data, axis=1).astype(np.float32)
    else:
        # Already mono, just convert type
        audio_mono = audio_array.astype(np.float32)

    # CRITICAL FIX: NO normalization! OpenWakeWord needs raw int16→float32
    # DO NOT divide by 32768.0 here!
    return audio_mono


def convert_to_normalized_format(
    audio_data: bytes,
    channels: int = None
) -> np.ndarray:
    """
    Convert audio bytes to normalized float32 format for STT.

    This function normalizes the audio to the -1.0 to 1.0 range expected
    by speech-to-text services.

    Args:
        audio_data: Raw audio bytes from microphone or WAV file
        channels: Number of audio channels (1=mono, 2=stereo).
                 If None, auto-detect based on array length.

    Returns:
        Mono float32 numpy array normalized to -1.0 to 1.0 range
    """
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # Determine if stereo
    if channels is not None:
        is_stereo = (channels == constants.CHANNELS_STEREO)
    else:
        is_stereo = len(audio_array) > constants.CHUNK_SIZE

    if is_stereo:
        # Convert stereo to mono
        stereo_data = audio_array.reshape(-1, 2)
        audio_mono = np.mean(stereo_data, axis=1).astype(np.float32)
    else:
        audio_mono = audio_array.astype(np.float32)

    # Normalize to -1.0 to 1.0 range
    return audio_mono / 32768.0
