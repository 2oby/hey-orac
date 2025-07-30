#!/usr/bin/env python3
"""
Enhanced wake word detection with audio preprocessing support.
This module provides helper functions for integrating audio preprocessing
into the main wake_word_detection.py
"""

import logging
import numpy as np
from typing import Optional, Tuple
from hey_orac.audio.microphone import AudioCapture
from hey_orac.audio.preprocessor import AudioPreprocessorConfig

logger = logging.getLogger(__name__)


def initialize_audio_capture_with_preprocessing(
    audio_config,
    device_index: int
) -> Tuple[Optional[AudioCapture], Optional[object]]:
    """
    Initialize AudioCapture with preprocessing based on configuration.
    
    Args:
        audio_config: Audio configuration from settings
        device_index: Audio device index
        
    Returns:
        Tuple of (audio_capture, ring_buffer) or (None, None) on failure
    """
    try:
        # Create preprocessor config from audio settings
        preprocessor_config = None
        if hasattr(audio_config, 'preprocessing') and audio_config.preprocessing:
            preprocessor_config = AudioPreprocessorConfig(
                enable_agc=audio_config.preprocessing.enable_agc,
                agc_target_level=audio_config.preprocessing.agc_target_level,
                agc_max_gain=audio_config.preprocessing.agc_max_gain,
                agc_attack_time=audio_config.preprocessing.agc_attack_time,
                agc_release_time=audio_config.preprocessing.agc_release_time,
                enable_compression=audio_config.preprocessing.enable_compression,
                compression_threshold=audio_config.preprocessing.compression_threshold,
                compression_ratio=audio_config.preprocessing.compression_ratio,
                enable_limiter=audio_config.preprocessing.enable_limiter,
                limiter_threshold=audio_config.preprocessing.limiter_threshold,
                enable_noise_gate=audio_config.preprocessing.enable_noise_gate,
                noise_gate_threshold=audio_config.preprocessing.noise_gate_threshold,
                sample_rate=audio_config.sample_rate
            )
            logger.info("✅ Audio preprocessing enabled:")
            logger.info(f"   AGC: {preprocessor_config.enable_agc} (target: {preprocessor_config.agc_target_level})")
            logger.info(f"   Compression: {preprocessor_config.enable_compression} (ratio: {preprocessor_config.compression_ratio}:1)")
            logger.info(f"   Limiter: {preprocessor_config.enable_limiter} (threshold: {preprocessor_config.limiter_threshold})")
        else:
            logger.info("ℹ️ Audio preprocessing disabled in configuration")
        
        # Initialize AudioCapture with preprocessing
        audio_capture = AudioCapture(
            sample_rate=audio_config.sample_rate,
            chunk_size=audio_config.chunk_size,
            ring_buffer_seconds=10.0,
            device_index=device_index,
            preprocessor_config=preprocessor_config
        )
        
        # Start audio capture
        if not audio_capture.start():
            logger.error("Failed to start audio capture")
            return None, None
        
        logger.info("✅ Audio capture started successfully")
        
        # Get ring buffer reference
        ring_buffer = audio_capture.ring_buffer
        logger.debug(f"Ring buffer initialized with capacity=10.0s, sample_rate={audio_config.sample_rate}Hz")
        
        return audio_capture, ring_buffer
        
    except Exception as e:
        logger.error(f"Error initializing audio capture: {e}")
        return None, None


def get_preprocessed_audio_chunk(
    audio_capture: AudioCapture,
    args,
    stream,
    chunk_size: int = 1280
) -> Optional[np.ndarray]:
    """
    Get audio chunk with appropriate preprocessing based on input source.
    
    Args:
        audio_capture: AudioCapture instance (for microphone input)
        args: Command line arguments
        stream: Audio stream (for WAV file input)
        chunk_size: Number of samples per chunk
        
    Returns:
        Audio data as float32 array in raw int16 range, or None if no data
    """
    if args.input_wav:
        # Read from WAV file stream
        data = stream.read(chunk_size, exception_on_overflow=False)
        if data is None or len(data) == 0:
            return None
        
        # Convert bytes to numpy array
        audio_array = np.frombuffer(data, dtype=np.int16)
        
        # Handle channel conversion for WAV files
        if hasattr(stream, 'channels'):
            if stream.channels == 2 and len(audio_array) > chunk_size:
                # Stereo WAV file - convert to mono
                stereo_data = audio_array.reshape(-1, 2)
                audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
            else:
                # Mono WAV file
                audio_data = audio_array.astype(np.float32)
        else:
            audio_data = audio_array.astype(np.float32)
    else:
        # Get audio chunk from AudioCapture (already preprocessed and float32)
        audio_data = audio_capture.get_audio_chunk()
        if audio_data is None:
            return None
        
        # CRITICAL: OpenWakeWord expects raw int16 values as float32, NOT normalized!
        # Convert back from normalized float32 to raw int16 range
        audio_data = audio_data * 32768.0
    
    return audio_data


def update_audio_metrics(shared_data: dict, audio_capture: Optional[AudioCapture], 
                        audio_data: np.ndarray, args) -> float:
    """
    Update audio metrics in shared data.
    
    Args:
        shared_data: Shared data dictionary
        audio_capture: AudioCapture instance (None for WAV input)
        audio_data: Current audio chunk
        args: Command line arguments
        
    Returns:
        RMS value
    """
    if audio_capture and not args.input_wav:
        # Get RMS from AudioCapture for better accuracy
        rms = audio_capture.get_rms()
        shared_data['rms'] = float(rms)
        
        # Update audio metrics
        metrics = audio_capture.get_audio_metrics()
        shared_data['audio_metrics'] = metrics
        
        # Log preprocessing metrics periodically
        if 'log_counter' not in shared_data:
            shared_data['log_counter'] = 0
        shared_data['log_counter'] += 1
        
        if shared_data['log_counter'] % 500 == 0:  # Every ~10 seconds at 16kHz
            logger.info(f"📊 Audio preprocessing metrics:")
            logger.info(f"   AGC gain: {metrics.get('agc_gain_db', 0):.1f} dB")
            logger.info(f"   Peak level: {metrics.get('peak_level', 0):.3f}")
            logger.info(f"   Clipping count: {metrics.get('clipping_count', 0)}")
    else:
        # Calculate RMS for WAV file or when AudioCapture not available
        rms = np.sqrt(np.mean(audio_data**2))
        shared_data['rms'] = float(rms)
    
    return rms