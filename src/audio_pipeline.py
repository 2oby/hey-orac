#!/usr/bin/env python3
"""
Audio Pipeline - Optimized wake word detection with volume monitoring
Implements: USB Mic → Volume Check (RMS) → Wake Word → Stream to Orin
"""

import logging
import time
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
from audio_utils import AudioManager
from wake_word_interface import WakeWordDetector
from audio_buffer import AudioBuffer
from audio_feedback import create_audio_feedback

logger = logging.getLogger(__name__)


class AudioPipeline:
    """
    Optimized audio pipeline with volume monitoring.
    
    Architecture: USB Mic → Volume Check (RMS) → Wake Word → Stream to Orin
    """
    
    def __init__(self, config: dict, usb_device, audio_manager: AudioManager):
        """
        Initialize the audio pipeline.
        
        Args:
            config: Configuration dictionary
            usb_device: USB audio device
            audio_manager: Audio manager instance
        """
        self.config = config
        self.usb_device = usb_device
        self.audio_manager = audio_manager
        
        # Volume monitoring parameters
        self.silence_threshold = config.get('volume_monitoring', {}).get('silence_threshold', 100)
        self.volume_window_size = config.get('volume_monitoring', {}).get('window_size', 10)
        self.volume_history = []
        
        # Wake word detection
        self.wake_detector = None
        self.initialize_wake_detector()
        
        # Audio buffer for pre/post-roll capture
        self.audio_buffer = None
        self.initialize_audio_buffer()
        
        # Audio feedback
        self.audio_feedback = create_audio_feedback()
        
        # Statistics
        self.chunk_count = 0
        self.detection_count = 0
        self.silence_chunks = 0
        self.volume_checks = 0
        self.wake_word_checks = 0
        
        # Performance monitoring
        self.last_stats_time = time.time()
        self.stats_interval = 10.0  # seconds
        
        logger.info("🎯 Audio Pipeline initialized")
        logger.info(f"   Silence threshold: {self.silence_threshold}")
        logger.info(f"   Volume window size: {self.volume_window_size}")
    
    def initialize_wake_detector(self):
        """Initialize the wake word detector."""
        try:
            self.wake_detector = WakeWordDetector()
            if not self.wake_detector.initialize(self.config):
                raise Exception("Failed to initialize wake word detector")
            
            logger.info(f"✅ Wake word detector initialized: {self.wake_detector.get_wake_word_name()}")
            logger.info(f"   Sample rate: {self.wake_detector.get_sample_rate()}")
            logger.info(f"   Frame length: {self.wake_detector.get_frame_length()}")
            logger.info(f"   Threshold: {self.config['wake_word'].get('threshold', 0.1)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize wake word detector: {e}")
            raise
    
    def initialize_audio_buffer(self):
        """Initialize the audio buffer for pre/post-roll capture."""
        try:
            self.audio_buffer = AudioBuffer(
                sample_rate=self.wake_detector.get_sample_rate(),
                preroll_seconds=self.config['buffer'].get('preroll_seconds', 1.0),
                postroll_seconds=self.config['buffer'].get('postroll_seconds', 2.0)
            )
            logger.info("✅ Audio buffer initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize audio buffer: {e}")
            raise
    
    def calculate_volume_level(self, audio_data: np.ndarray) -> float:
        """
        Calculate RMS volume level from audio data.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            RMS volume level
        """
        # Convert to float32 for accurate RMS calculation
        audio_float = audio_data.astype(np.float32)
        
        # Calculate RMS (Root Mean Square) - standard measure of audio level
        rms_level = np.sqrt(np.mean(audio_float ** 2))
        
        return rms_level
    
    def is_audio_above_threshold(self, volume_level: float) -> bool:
        """
        Check if audio level is above silence threshold.
        
        Args:
            volume_level: RMS volume level
            
        Returns:
            True if audio is above threshold, False if silence
        """
        return volume_level > self.silence_threshold
    
    def update_volume_history(self, volume_level: float):
        """Update rolling volume history for trend analysis."""
        self.volume_history.append(volume_level)
        
        # Keep only the last N samples
        if len(self.volume_history) > self.volume_window_size:
            self.volume_history.pop(0)
    
    def get_volume_trend(self) -> str:
        """
        Analyze volume trend over the history window.
        
        Returns:
            Trend description: "increasing", "decreasing", "stable", or "unknown"
        """
        if len(self.volume_history) < 3:
            return "unknown"
        
        # Calculate trend using linear regression
        x = np.arange(len(self.volume_history))
        y = np.array(self.volume_history)
        
        # Simple linear trend calculation
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 10:  # Significant increase
            return "increasing"
        elif slope < -10:  # Significant decrease
            return "decreasing"
        else:
            return "stable"
    
    def process_audio_chunk(self, audio_data: np.ndarray) -> bool:
        """
        Process a single audio chunk through the pipeline.
        
        Architecture: Volume Check → Wake Word Detection
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            True if wake word detected, False otherwise
        """
        self.chunk_count += 1
        
        # STEP 1: Volume Check (RMS)
        volume_level = self.calculate_volume_level(audio_data)
        self.volume_checks += 1
        self.update_volume_history(volume_level)
        
        # Check if audio is above silence threshold
        if not self.is_audio_above_threshold(volume_level):
            self.silence_chunks += 1
            return False
        
        # STEP 2: Wake Word Detection (only if volume is above threshold)
        self.wake_word_checks += 1
        detection_result = self.wake_detector.process_audio(audio_data)
        
        return detection_result
    
    def handle_wake_word_detection(self, audio_data: np.ndarray):
        """
        Handle wake word detection event.
        
        Args:
            audio_data: Audio data that triggered the detection
        """
        self.detection_count += 1
        
        # PROMINENT DETECTION LOG
        logger.info("🎯🎯🎯 WAKE WORD DETECTED! 🎯🎯🎯")
        logger.info(f"🎯 DETECTION #{self.detection_count} - {self.wake_detector.get_wake_word_name()} detected!")
        
        # Detection log with timestamp
        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detection_log_line = f"[{detection_time}] WAKE WORD DETECTED: {self.wake_detector.get_wake_word_name()} (Detection #{self.detection_count})"
        logger.info(detection_log_line)
        
        # Write to dedicated detection log file
        try:
            with open("/app/logs/pipeline_detections.log", "a") as f:
                f.write(f"{detection_log_line}\n")
        except Exception as e:
            logger.warning(f"⚠️ Could not write to detection log: {e}")
        
        # Log detection details
        volume_level = self.calculate_volume_level(audio_data)
        volume_trend = self.get_volume_trend()
        
        logger.info(f"📊 Detection details:")
        logger.info(f"   Chunk number: {self.chunk_count}")
        logger.info(f"   Volume level: {volume_level:.4f}")
        logger.info(f"   Volume trend: {volume_trend}")
        logger.info(f"   Audio max level: {np.max(np.abs(audio_data))}")
        logger.info(f"   Buffer status: {self.audio_buffer.get_buffer_status()}")
        
        # Audio feedback
        if self.audio_feedback:
            try:
                logger.info("🔊 Providing audio feedback...")
                success = self.audio_feedback.play_wake_word_detected()
                if success:
                    logger.info("✅ Audio feedback completed successfully")
                else:
                    logger.warning("⚠️ Audio feedback failed (but system continues)")
            except Exception as e:
                logger.error(f"❌ Audio feedback failed: {e}")
        
        # Start post-roll capture
        logger.info("📦 Starting post-roll capture...")
        self.audio_buffer.start_postroll_capture()
        
        # TODO: Stream audio to Orin Nano
        logger.info("📡 Ready to stream audio to Orin Nano")
        
        # Get complete audio clip
        logger.info("🎵 Retrieving complete audio clip...")
        complete_audio = self.audio_buffer.get_complete_audio_clip()
        if complete_audio is not None:
            clip_filename = f"/tmp/pipeline_wake_word_detection_{self.detection_count}.wav"
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
        
        logger.info("🔄 Resuming audio pipeline monitoring...")
    
    def log_performance_stats(self):
        """Log performance statistics."""
        current_time = time.time()
        if current_time - self.last_stats_time >= self.stats_interval:
            runtime = current_time - self.last_stats_time
            
            # Calculate processing rates
            chunks_per_second = self.chunk_count / runtime if runtime > 0 else 0
            volume_check_rate = self.volume_checks / runtime if runtime > 0 else 0
            wake_word_check_rate = self.wake_word_checks / runtime if runtime > 0 else 0
            
            # Calculate efficiency metrics
            silence_percentage = (self.silence_chunks / max(self.chunk_count, 1)) * 100
            efficiency_gain = (self.chunk_count - self.wake_word_checks) / max(self.chunk_count, 1) * 100
            
            logger.info(f"📊 Performance Stats (last {runtime:.1f}s):")
            logger.info(f"   Chunks processed: {self.chunk_count}")
            logger.info(f"   Volume checks: {self.volume_checks} ({volume_check_rate:.1f}/s)")
            logger.info(f"   Wake word checks: {self.wake_word_checks} ({wake_word_check_rate:.1f}/s)")
            logger.info(f"   Silence chunks: {self.silence_chunks} ({silence_percentage:.1f}%)")
            logger.info(f"   Efficiency gain: {efficiency_gain:.1f}% (skipped wake word checks)")
            logger.info(f"   Detections: {self.detection_count}")
            
            # Reset counters for next interval
            self.chunk_count = 0
            self.volume_checks = 0
            self.wake_word_checks = 0
            self.silence_chunks = 0
            self.last_stats_time = current_time
    
    def run_continuous_monitoring(self) -> int:
        """
        Run continuous audio monitoring with optimized pipeline.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        logger.info("🎯 Starting optimized audio pipeline monitoring...")
        logger.info(f"🎤 Using USB microphone: {self.usb_device.name}")
        logger.info(f"⚙️ Stream parameters: {self.wake_detector.get_sample_rate()}Hz, 1 channel, {self.wake_detector.get_frame_length()} samples/chunk")
        
        # Start audio stream
        stream = self.audio_manager.start_stream(
            device_index=self.usb_device.index,
            sample_rate=self.wake_detector.get_sample_rate(),
            channels=1,
            chunk_size=self.wake_detector.get_frame_length()
        )
        
        if not stream:
            logger.error("❌ Failed to start audio stream")
            return 1
        
        logger.info("✅ Audio stream started successfully")
        logger.info("📊 Performance stats will be logged every 10 seconds")
        
        try:
            while True:
                try:
                    # Read audio chunk
                    audio_chunk = stream.read(self.wake_detector.get_frame_length(), exception_on_overflow=False)
                    
                    # Convert to numpy array
                    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                    
                    # Add to audio buffer
                    self.audio_buffer.add_audio(audio_data)
                    
                    # Process through pipeline: Volume Check → Wake Word Detection
                    detection_result = self.process_audio_chunk(audio_data)
                    
                    if detection_result:
                        self.handle_wake_word_detection(audio_data)
                    
                    # Log performance stats periodically
                    self.log_performance_stats()
                    
                except Exception as e:
                    logger.error(f"❌ Error processing audio chunk: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("🛑 Audio pipeline monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Audio pipeline monitoring error: {e}")
            return 1
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            logger.info("✅ Audio stream closed")
        
        return 0


def run_audio_pipeline(config: dict, usb_device, audio_manager: AudioManager) -> int:
    """
    Run the optimized audio pipeline.
    
    Args:
        config: Configuration dictionary
        usb_device: USB audio device
        audio_manager: Audio manager instance
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        pipeline = AudioPipeline(config, usb_device, audio_manager)
        return pipeline.run_continuous_monitoring()
    except Exception as e:
        logger.error(f"❌ Failed to initialize audio pipeline: {e}")
        return 1 