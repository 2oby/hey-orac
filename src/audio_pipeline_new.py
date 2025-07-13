#!/usr/bin/env python3
"""
Audio Pipeline Module - Pure Audio Processing
Handles continuous audio monitoring, RMS calculation, and threshold filtering.
Delegates wake word detection to external callback.
"""

import logging
import time
import numpy as np
from typing import Callable, Optional, Dict, Any
from shared_memory_ipc import shared_memory_ipc
from settings_manager import get_settings_manager

logger = logging.getLogger(__name__)


class AudioPipeline:
    """
    Pure audio processing pipeline that monitors audio levels and filters based on threshold.
    Delegates wake word detection to external callback function.
    """
    
    def __init__(self, audio_manager, usb_device, sample_rate: int, frame_length: int, channels: int = 1):
        """Initialize the audio pipeline."""
        self.audio_manager = audio_manager
        self.usb_device = usb_device
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.channels = channels
        
        # Get settings manager for configuration
        self.settings_manager = get_settings_manager()
        
        # Audio processing parameters
        self.silence_threshold = self.settings_manager.get("detection.rms_filter", 50)  # Use GUI setting
        self.volume_window_size = self.settings_manager.get("volume_monitoring.window_size", 10)
        self.silence_duration_threshold = self.settings_manager.get("volume_monitoring.silence_duration_threshold", 2.0)  # seconds
        
        # Audio state tracking
        self.volume_history = []
        self.is_audio_active = False
        self.silence_start_time = None
        self.chunk_count = 0
        
        # Audio buffer for future API packaging (commented for now)
        # self.audio_buffer = AudioBuffer(
        #     sample_rate=sample_rate,
        #     preroll_seconds=1.0,
        #     postroll_seconds=2.0
        # )
        
        # Callback for wake word detection
        self.wake_word_callback: Optional[Callable] = None
        
        # Stream reference
        self.stream = None
        
        # Add settings watcher to update threshold dynamically
        self.settings_manager.add_watcher(self._on_settings_changed)
        logger.info(f"✅ Settings watcher registered for dynamic threshold updates")
        
        logger.info(f"🔧 Audio Pipeline Configuration:")
        logger.info(f"   Silence threshold: {self.silence_threshold}")
        logger.info(f"   Volume window size: {self.volume_window_size}")
        logger.info(f"   Silence duration threshold: {self.silence_duration_threshold}s")
        logger.info(f"   Sample rate: {sample_rate}")
        logger.info(f"   Frame length: {frame_length}")
        logger.info(f"   Channels: {channels}")
    
    def set_wake_word_callback(self, callback: Callable):
        """Set the callback function for wake word detection."""
        self.wake_word_callback = callback
        logger.info("✅ Wake word callback set")
    
    def start_stream(self) -> bool:
        """Start the audio stream."""
        try:
            self.stream = self.audio_manager.start_stream(
                device_index=self.usb_device.index,
                sample_rate=self.sample_rate,
                channels=self.channels,
                chunk_size=self.frame_length
            )
            
            if not self.stream:
                logger.error("❌ Failed to start audio stream")
                return False
            
            logger.info("✅ Audio stream started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting audio stream: {e}")
            return False
    
    def _calculate_rms(self, audio_data: np.ndarray) -> float:
        """Calculate RMS level from audio data."""
        return np.sqrt(np.mean(audio_data.astype(np.float32)**2))
    
    def _update_volume_history(self, rms_level: float):
        """Update rolling volume history."""
        self.volume_history.append(rms_level)
        if len(self.volume_history) > self.volume_window_size:
            self.volume_history.pop(0)
    
    def _get_average_volume(self) -> float:
        """Get average volume over the window."""
        return np.mean(self.volume_history) if self.volume_history else 0.0
    
    def _should_pass_audio(self, avg_volume: float, current_time: float) -> bool:
        """
        Determine if audio should be passed to wake word pipeline.
        Uses hysteresis: once above threshold, keep passing until prolonged silence.
        """
        # If audio is above threshold, always pass
        if avg_volume >= self.silence_threshold:
            self.is_audio_active = True
            self.silence_start_time = None
            return True
        
        # If audio is below threshold
        if avg_volume < self.silence_threshold:
            # If we were previously active, start silence timer
            if self.is_audio_active and self.silence_start_time is None:
                self.silence_start_time = current_time
                logger.debug(f"🔇 Silence started at {current_time:.2f}s")
            
            # If silence has been going on for too long, stop passing audio
            if (self.silence_start_time is not None and 
                current_time - self.silence_start_time >= self.silence_duration_threshold):
                self.is_audio_active = False
                self.silence_start_time = None
                logger.debug(f"🔇 Audio pipeline deactivated after {self.silence_duration_threshold}s silence")
                return False
            
            # If we're still in the grace period, keep passing audio
            if self.is_audio_active:
                return True
        
        return False
    
    def _update_shared_memory(self, rms_level: float, avg_volume: float, is_passing_audio: bool):
        """Update shared memory with audio pipeline state."""
        try:
            # Update RMS value and listening state
            shared_memory_ipc.update_audio_state(rms_level)
            
            # Update the listening state based on whether we're passing audio to wake word pipeline
            # This tells the web GUI whether the system is "listening" for wake words
            shared_memory_ipc.update_activation_state(is_passing_audio)
            
            # Log state changes every 500 chunks (reduced frequency)
            if self.chunk_count % 500 == 0:
                logger.info(f"🔍 Audio Pipeline State:")
                logger.info(f"   RMS: {rms_level:.4f}")
                logger.info(f"   Avg Volume: {avg_volume:.4f}")
                logger.info(f"   Threshold: {self.silence_threshold}")
                logger.info(f"   Passing Audio: {is_passing_audio}")
                logger.info(f"   Audio Active: {self.is_audio_active}")
                logger.info(f"   Listening State: {is_passing_audio} (Shared Memory)")
                if self.silence_start_time:
                    silence_duration = time.time() - self.silence_start_time
                    logger.info(f"   Silence Duration: {silence_duration:.2f}s")
        except Exception as e:
            logger.warning(f"⚠️ Could not update shared memory: {e}")
    
    def run(self) -> int:
        """Run the audio processing pipeline."""
        logger.info("🎯 Starting audio pipeline monitoring...")
        
        if not self.start_stream():
            return 1
        
        try:
            while True:
                try:
                    # Read audio chunk
                    audio_chunk = self.stream.read(self.frame_length, exception_on_overflow=False)
                    self.chunk_count += 1
                    
                    # Convert to numpy array
                    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                    
                    # Calculate RMS level
                    rms_level = self._calculate_rms(audio_data)
                    self._update_volume_history(rms_level)
                    avg_volume = self._get_average_volume()
                    
                    # Determine if audio should be passed to wake word pipeline
                    current_time = time.time()
                    should_pass_audio = self._should_pass_audio(avg_volume, current_time)
                    
                    # Update shared memory with audio state
                    self._update_shared_memory(rms_level, avg_volume, should_pass_audio)
                    
                    # TODO: Add audio to buffer for future API packaging
                    # self.audio_buffer.add_audio(audio_data)
                    
                    # Pass audio to wake word pipeline if above threshold
                    if should_pass_audio and self.wake_word_callback:
                        try:
                            self.wake_word_callback(audio_data, self.chunk_count, rms_level, avg_volume)
                        except Exception as e:
                            logger.error(f"❌ Error in wake word callback: {e}")
                    
                    # Progress logging
                    if self.chunk_count % 1000 == 0:
                        logger.info(f"📊 Audio Pipeline Progress:")
                        logger.info(f"   Processed chunks: {self.chunk_count}")
                        logger.info(f"   Current RMS: {rms_level:.4f}")
                        logger.info(f"   Avg Volume: {avg_volume:.4f}")
                        logger.info(f"   Passing Audio: {should_pass_audio}")
                        logger.info(f"   Runtime: {self.chunk_count * self.frame_length / self.sample_rate:.1f}s")
                
                except Exception as e:
                    logger.error(f"❌ Error processing audio chunk {self.chunk_count}: {e}")
                    logger.error(f"   Audio data length: {len(audio_data) if 'audio_data' in locals() else 'N/A'}")
                    logger.error(f"   Stream active: {self.stream.is_active() if self.stream else False}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("🛑 Audio pipeline stopped by user")
        except Exception as e:
            logger.error(f"❌ Audio pipeline error: {e}")
            logger.error(f"   Last chunk processed: {self.chunk_count}")
            return 1
        finally:
            self._cleanup()
        
        return 0
    
    def _on_settings_changed(self, new_settings: Dict[str, Any]):
        """Handle settings changes from GUI."""
        try:
            # Get the new threshold from the settings
            new_threshold = new_settings.get("detection", {}).get("rms_filter", 50)
            if new_threshold != self.silence_threshold:
                old_threshold = self.silence_threshold
                self.silence_threshold = new_threshold
                logger.info(f"🔄 Audio threshold updated: {old_threshold} → {new_threshold}")
                logger.info(f"   New threshold will be used for next audio processing")
        except Exception as e:
            logger.error(f"❌ Error updating audio threshold: {e}")
    
    def _cleanup(self):
        """Clean up audio pipeline resources."""
        logger.info("🧹 Starting audio pipeline cleanup...")
        try:
            # Remove settings watcher
            self.settings_manager.remove_watcher(self._on_settings_changed)
            
            if self.stream:
                logger.info("🛑 Stopping audio stream...")
                self.stream.stop_stream()
                self.stream.close()
                logger.info("✅ Audio stream closed")
            
            logger.info("🛑 Stopping audio manager...")
            self.audio_manager.stop_recording()
            logger.info("✅ Audio manager stopped")
            
            logger.info("✅ Audio pipeline cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


def create_audio_pipeline(audio_manager, usb_device, sample_rate: int, frame_length: int, channels: int = 1) -> AudioPipeline:
    """Factory function to create an audio pipeline instance."""
    return AudioPipeline(audio_manager, usb_device, sample_rate, frame_length, channels) 