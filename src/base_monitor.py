#!/usr/bin/env python3
"""
Base Monitor Class
Contains common functionality for wake word monitoring
"""

import logging
import time
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from audio_utils import AudioManager
from wake_word_interface import WakeWordDetector
from audio_buffer import AudioBuffer
from audio_feedback import create_audio_feedback

logger = logging.getLogger(__name__)


class BaseWakeWordMonitor(ABC):
    """
    Base class for wake word monitoring with common functionality.
    """
    
    def __init__(self, config: dict, usb_device, audio_manager: AudioManager):
        """
        Initialize the base monitor.
        
        Args:
            config: Configuration dictionary
            usb_device: USB audio device to use
            audio_manager: Initialized audio manager
        """
        self.config = config
        self.usb_device = usb_device
        self.audio_manager = audio_manager
        self.wake_detector = None
        self.audio_feedback = None
        self.audio_buffer = None
        self.stream = None
        self.detection_count = 0
        self.chunk_count = 0
        
    def initialize(self) -> bool:
        """
        Initialize the monitor components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        logger.info("🎯 Initializing base monitor...")
        
        # Initialize wake word detector
        logger.info("🔍 DEBUG: Initializing wake word detector...")
        self.wake_detector = WakeWordDetector()
        if not self._initialize_detector():
            logger.error("❌ Failed to initialize wake word detector")
            return False
        
        logger.info(f"✅ Wake word detector initialized: {self.wake_detector.get_wake_word_name()}")
        
        # Initialize audio feedback
        logger.info("🔍 DEBUG: Initializing audio feedback...")
        self.audio_feedback = create_audio_feedback()
        if self.audio_feedback:
            logger.info("✅ Audio feedback system initialized")
        else:
            logger.warning("⚠️ Audio feedback system not available")
        
        # Initialize audio buffer
        logger.info("🔍 DEBUG: Initializing audio buffer...")
        self.audio_buffer = AudioBuffer(
            sample_rate=self.wake_detector.get_sample_rate(),
            preroll_seconds=self.config['buffer'].get('preroll_seconds', 1.0),
            postroll_seconds=self.config['buffer'].get('postroll_seconds', 2.0)
        )
        
        # Start audio stream
        logger.info("🔍 DEBUG: About to start audio stream...")
        if not self._start_audio_stream():
            return False
        
        logger.info("✅ Base monitor initialization completed")
        return True
    
    @abstractmethod
    def _initialize_detector(self) -> bool:
        """
        Initialize the wake word detector with specific configuration.
        Must be implemented by subclasses.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    def _start_audio_stream(self) -> bool:
        """
        Start the audio stream.
        
        Returns:
            True if stream started successfully, False otherwise
        """
        logger.info(f"🎤 Starting audio stream on device {self.usb_device.index} ({self.usb_device.name})")
        logger.info(f"⚙️ Stream parameters: {self.wake_detector.get_sample_rate()}Hz, 1 channel, {self.wake_detector.get_frame_length()} samples/chunk")
        
        logger.info("🔍 DEBUG: About to call audio_manager.start_stream()...")
        self.stream = self.audio_manager.start_stream(
            device_index=self.usb_device.index,
            sample_rate=self.wake_detector.get_sample_rate(),
            channels=1,
            chunk_size=self.wake_detector.get_frame_length()
        )
        
        if not self.stream:
            logger.error("❌ Failed to start audio stream")
            return False
        
        logger.info("✅ Audio stream started successfully")
        logger.info("🔍 DEBUG: Audio stream object created successfully")
        return True
    
    def _process_audio_chunk(self, audio_data: np.ndarray) -> bool:
        """
        Process a single audio chunk for wake word detection.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if wake word detected, False otherwise
        """
        # Add to audio buffer
        self.audio_buffer.add_audio(audio_data)
        
        # Process audio for wake-word detection
        detection_result = self.wake_detector.process_audio(audio_data)
        
        if detection_result:
            return self._handle_detection(audio_data)
        
        return False
    
    def _handle_detection(self, audio_data: np.ndarray) -> bool:
        """
        Handle a wake word detection.
        
        Args:
            audio_data: Audio data that triggered the detection
            
        Returns:
            True if detection was processed, False if blocked by controls
        """
        # Check if detection should be allowed (implemented by subclasses)
        if not self._should_allow_detection():
            return False
        
        self.detection_count += 1
        
        # Log detection
        self._log_detection(audio_data)
        
        # Provide audio feedback
        self._provide_audio_feedback()
        
        # Capture post-roll audio
        self._capture_postroll_audio()
        
        # Save audio clip
        self._save_audio_clip()
        
        return True
    
    @abstractmethod
    def _should_allow_detection(self) -> bool:
        """
        Check if detection should be allowed based on timing controls.
        Must be implemented by subclasses.
        
        Returns:
            True if detection should be allowed, False otherwise
        """
        pass
    
    def _log_detection(self, audio_data: np.ndarray):
        """Log detection details."""
        logger.info("🎯🎯🎯 WAKE WORD DETECTED! 🎯🎯🎯")
        logger.info(f"🎯 DETECTION #{self.detection_count} - {self.wake_detector.get_wake_word_name()} detected!")
        
        # Detection log with timestamp
        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detection_log_line = f"[{detection_time}] WAKE WORD DETECTED: {self.wake_detector.get_wake_word_name()} (Detection #{self.detection_count})"
        logger.info(detection_log_line)
        
        # Write to dedicated detection log file
        self._write_detection_log(detection_log_line)
        
        # Log detection details
        self._log_detection_details(audio_data)
    
    def _write_detection_log(self, detection_log_line: str):
        """Write detection to log file."""
        try:
            log_file = self._get_detection_log_file()
            with open(log_file, "a") as f:
                f.write(f"{detection_log_line}\n")
        except Exception as e:
            logger.warning(f"⚠️ Could not write to detection log: {e}")
    
    @abstractmethod
    def _get_detection_log_file(self) -> str:
        """
        Get the path to the detection log file.
        Must be implemented by subclasses.
        
        Returns:
            Path to the detection log file
        """
        pass
    
    def _log_detection_details(self, audio_data: np.ndarray):
        """Log detailed detection information."""
        logger.info(f"📊 Detection details:")
        logger.info(f"   Chunk number: {self.chunk_count}")
        logger.info(f"   Audio RMS level: {np.sqrt(np.mean(audio_data.astype(np.float32)**2)):.4f}")
        logger.info(f"   Audio max level: {np.max(np.abs(audio_data))}")
        logger.info(f"   Buffer status: {self.audio_buffer.get_buffer_status()}")
    
    def _provide_audio_feedback(self):
        """Provide audio feedback for detection."""
        if self.audio_feedback:
            try:
                logger.info("🔊 Providing audio feedback with beep...")
                success = self.audio_feedback.play_wake_word_detected()
                if success:
                    logger.info("✅ Audio feedback completed successfully")
                else:
                    logger.warning("⚠️ Audio feedback failed (but system continues)")
            except Exception as e:
                logger.error(f"❌ Audio feedback failed: {e}")
    
    def _capture_postroll_audio(self):
        """Capture post-roll audio after detection."""
        logger.info("📦 Starting post-roll capture...")
        self.audio_buffer.start_postroll_capture()
        
        # Wait for post-roll capture to complete
        postroll_chunks = 0
        while self.audio_buffer.is_capturing_postroll():
            audio_chunk = self.stream.read(self.wake_detector.get_frame_length(), exception_on_overflow=False)
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            self.audio_buffer.add_audio(audio_data)
            self.chunk_count += 1
            postroll_chunks += 1
            
            if postroll_chunks % 10 == 0:
                logger.debug(f"📦 Post-roll capture: {postroll_chunks} chunks captured")
        
        logger.info(f"📦 Post-roll capture completed: {postroll_chunks} chunks")
    
    def _save_audio_clip(self):
        """Save the complete audio clip."""
        logger.info("🎵 Retrieving complete audio clip...")
        complete_audio = self.audio_buffer.get_complete_audio_clip()
        if complete_audio is not None:
            clip_filename = self._get_audio_clip_filename()
            logger.info(f"💾 Saving audio clip to: {clip_filename}")
            
            if self.audio_buffer.save_audio_clip(clip_filename):
                logger.info(f"✅ Audio clip saved successfully")
                logger.info(f"📦 Audio clip details:")
                logger.info(f"   Duration: {len(complete_audio)/self.audio_buffer.sample_rate:.2f}s")
                logger.info(f"   Samples: {len(complete_audio)}")
                logger.info(f"   RMS level: {np.sqrt(np.mean(complete_audio**2)):.4f}")
                logger.info(f"   Max level: {np.max(np.abs(complete_audio)):.4f}")
            else:
                logger.error("❌ Failed to save audio clip")
        else:
            logger.warning("⚠️ No complete audio clip available")
        
        logger.info("📡 Audio capture completed - ready for streaming to Jetson")
        logger.info("🔄 Resuming monitoring...")
    
    @abstractmethod
    def _get_audio_clip_filename(self) -> str:
        """
        Get the filename for the audio clip.
        Must be implemented by subclasses.
        
        Returns:
            Filename for the audio clip
        """
        pass
    
    def _log_progress(self):
        """Log progress information."""
        if self.chunk_count % 1000 == 0:
            buffer_status = self.audio_buffer.get_buffer_status()
            logger.info(f"📊 Progress Report:")
            logger.info(f"   Processed chunks: {self.chunk_count}")
            logger.info(f"   Detections: {self.detection_count}")
            logger.info(f"   Buffer pre-roll samples: {buffer_status['preroll_samples']}")
            logger.info(f"   Buffer post-roll samples: {buffer_status['postroll_samples']}")
            logger.info(f"   Runtime: {self.chunk_count * self.wake_detector.get_frame_length() / self.wake_detector.get_sample_rate():.1f}s")
    
    def _log_audio_levels(self, audio_data: np.ndarray):
        """Log audio level information."""
        if self.chunk_count % 100 == 0:
            rms_level = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
            max_level = np.max(np.abs(audio_data))
            logger.info(f"📊 Chunk {self.chunk_count}: RMS={rms_level:.2f}, Max={max_level}, Samples={len(audio_data)}")
    
    def run(self) -> int:
        """
        Run the monitoring loop.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        logger.info("🎯 Starting continuous monitoring...")
        logger.info("📊 Debug info will be logged every 100 chunks")
        
        try:
            while True:
                try:
                    # DEBUG: Log every 25 chunks to track if we're reading audio
                    if self.chunk_count % 25 == 0:
                        logger.info(f"🔍 DEBUG: About to read chunk {self.chunk_count + 1}...")
                    
                    # Read audio chunk
                    audio_chunk = self.stream.read(self.wake_detector.get_frame_length(), exception_on_overflow=False)
                    self.chunk_count += 1
                    
                    # DEBUG: Log every 25 chunks to track audio data
                    if self.chunk_count % 25 == 0:
                        logger.info(f"🔍 DEBUG: Read chunk {self.chunk_count} - Audio chunk length: {len(audio_chunk)}")
                    
                    # Convert to numpy array
                    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                    
                    # DEBUG: Log every 25 chunks to track numpy conversion
                    if self.chunk_count % 25 == 0:
                        logger.info(f"🔍 DEBUG: Converted chunk {self.chunk_count} - Numpy array length: {len(audio_data)}")
                    
                    # Log audio levels
                    self._log_audio_levels(audio_data)
                    
                    # Process audio chunk
                    self._process_audio_chunk(audio_data)
                    
                    # Log progress
                    self._log_progress()
                    
                except Exception as e:
                    logger.error(f"❌ Error processing audio chunk {self.chunk_count}: {e}")
                    logger.error(f"   Audio data length: {len(audio_data) if 'audio_data' in locals() else 'N/A'}")
                    logger.error(f"   Stream active: {self.stream.is_active() if self.stream else False}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
            logger.error(f"   Last chunk processed: {self.chunk_count}")
            return 1
        finally:
            self._cleanup()
        
        return 0
    
    def _cleanup(self):
        """Clean up resources."""
        logger.info("🧹 Starting cleanup...")
        try:
            if self.stream:
                logger.info("🛑 Stopping audio stream...")
                self.stream.stop_stream()
                self.stream.close()
                logger.info("✅ Audio stream closed")
            
            logger.info("🛑 Stopping audio manager...")
            self.audio_manager.stop_recording()
            logger.info("✅ Audio manager stopped")
            
            logger.info("🛑 Cleaning up wake detector...")
            self.wake_detector.cleanup()
            logger.info("✅ Wake detector cleaned up")
            
            logger.info("✅ All cleanup completed successfully")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}") 