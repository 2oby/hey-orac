#!/usr/bin/env python3
"""
Monitor Default Wake Word Models
Handles monitoring of pre-trained wake word models (Hey Jarvis, etc.)
"""

import logging
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from audio_utils import AudioManager
from wake_word_interface import WakeWordDetector
from audio_buffer import AudioBuffer
from audio_feedback import create_audio_feedback

logger = logging.getLogger(__name__)


def monitor_default_models(config: dict, usb_device, audio_manager: AudioManager) -> int:
    """
    Monitor default/pre-trained wake word models.
    
    Args:
        config: Configuration dictionary
        usb_device: USB audio device to use
        audio_manager: Initialized audio manager
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("🎯 Starting default model monitoring...")
    
    # Initialize wake word detector for default models
    wake_detector = WakeWordDetector()
    if not wake_detector.initialize(config):
        logger.error("❌ Failed to initialize wake word detector for default models")
        return 1
    
    logger.info(f"✅ Default wake word detector initialized: {wake_detector.get_wake_word_name()}")
    
    # Initialize audio feedback
    audio_feedback = create_audio_feedback()
    if audio_feedback:
        logger.info("✅ Audio feedback system initialized")
    else:
        logger.warning("⚠️ Audio feedback system not available")
    
    # Initialize audio buffer
    audio_buffer = AudioBuffer(
        sample_rate=wake_detector.get_sample_rate(),
        buffer_duration=config['audio'].get('buffer_duration', 3.0),
        postroll_duration=config['audio'].get('postroll_duration', 1.0)
    )
    
    # Start continuous audio stream
    logger.info(f"🎤 Starting audio stream on device {usb_device.index} ({usb_device.name})")
    logger.info(f"⚙️ Stream parameters: {wake_detector.get_sample_rate()}Hz, 1 channel, {wake_detector.get_frame_length()} samples/chunk")
    
    stream = audio_manager.start_stream(
        device_index=usb_device.index,
        sample_rate=wake_detector.get_sample_rate(),
        channels=1,
        chunk_size=wake_detector.get_frame_length()
    )
    
    if not stream:
        logger.error("❌ Failed to start audio stream")
        return 1
    
    logger.info("✅ Audio stream started successfully")
    
    # Main monitoring loop for default models
    try:
        detection_count = 0
        chunk_count = 0
        
        logger.info("🎯 Starting continuous default model monitoring...")
        logger.info("📊 Debug info will be logged every 100 chunks")
        
        while True:
            try:
                # Read audio chunk
                audio_chunk = stream.read(wake_detector.get_frame_length(), exception_on_overflow=False)
                chunk_count += 1
                
                # Convert to numpy array
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # Enhanced audio level monitoring
                if chunk_count % 100 == 0:
                    rms_level = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                    max_level = np.max(np.abs(audio_data))
                    logger.info(f"📊 Chunk {chunk_count}: RMS={rms_level:.2f}, Max={max_level}, Samples={len(audio_data)}")
                
                # Add to audio buffer
                audio_buffer.add_audio(audio_data)
                
                # Process audio for wake-word detection
                detection_result = wake_detector.process_audio(audio_data)
                
                if detection_result:
                    detection_count += 1
                    
                    # PROMINENT DETECTION LOG
                    logger.info("🎯🎯🎯 DEFAULT WAKE WORD DETECTED! 🎯🎯🎯")
                    logger.info(f"🎯 DETECTION #{detection_count} - {wake_detector.get_wake_word_name()} detected!")
                    
                    # Detection log with timestamp
                    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    detection_log_line = f"[{detection_time}] DEFAULT WAKE WORD DETECTED: {wake_detector.get_wake_word_name()} (Detection #{detection_count})"
                    logger.info(detection_log_line)
                    
                    # Write to dedicated detection log file
                    try:
                        with open("/app/logs/default_detections.log", "a") as f:
                            f.write(f"{detection_log_line}\n")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not write to detection log: {e}")
                    
                    # Log detection details
                    logger.info(f"📊 Detection details:")
                    logger.info(f"   Chunk number: {chunk_count}")
                    logger.info(f"   Audio RMS level: {np.sqrt(np.mean(audio_data.astype(np.float32)**2)):.4f}")
                    logger.info(f"   Audio max level: {np.max(np.abs(audio_data))}")
                    logger.info(f"   Buffer status: {audio_buffer.get_buffer_status()}")
                    
                    # Audio feedback
                    if audio_feedback:
                        try:
                            logger.info("🔊 Providing audio feedback with beep...")
                            success = audio_feedback.play_wake_word_detected()
                            if success:
                                logger.info("✅ Audio feedback completed successfully")
                            else:
                                logger.warning("⚠️ Audio feedback failed (but system continues)")
                        except Exception as e:
                            logger.error(f"❌ Audio feedback failed: {e}")
                    
                    # Start post-roll capture
                    logger.info("📦 Starting post-roll capture...")
                    audio_buffer.start_postroll_capture()
                    
                    # Wait for post-roll capture to complete
                    postroll_chunks = 0
                    while audio_buffer.is_capturing_postroll():
                        audio_chunk = stream.read(wake_detector.get_frame_length(), exception_on_overflow=False)
                        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                        audio_buffer.add_audio(audio_data)
                        chunk_count += 1
                        postroll_chunks += 1
                        
                        if postroll_chunks % 10 == 0:
                            logger.debug(f"📦 Post-roll capture: {postroll_chunks} chunks captured")
                    
                    logger.info(f"📦 Post-roll capture completed: {postroll_chunks} chunks")
                    
                    # Get complete audio clip
                    logger.info("🎵 Retrieving complete audio clip...")
                    complete_audio = audio_buffer.get_complete_audio_clip()
                    if complete_audio is not None:
                        clip_filename = f"/tmp/default_wake_word_detection_{detection_count}.wav"
                        logger.info(f"💾 Saving audio clip to: {clip_filename}")
                        
                        if audio_buffer.save_audio_clip(clip_filename):
                            logger.info(f"✅ Audio clip saved successfully")
                            logger.info(f"📦 Audio clip details:")
                            logger.info(f"   Duration: {len(complete_audio)/audio_buffer.sample_rate:.2f}s")
                            logger.info(f"   Samples: {len(complete_audio)}")
                            logger.info(f"   RMS level: {np.sqrt(np.mean(complete_audio**2)):.4f}")
                            logger.info(f"   Max level: {np.max(np.abs(complete_audio)):.4f}")
                        else:
                            logger.error("❌ Failed to save audio clip")
                    else:
                        logger.warning("⚠️ No complete audio clip available")
                    
                    logger.info("📡 Audio capture completed - ready for streaming to Jetson")
                    logger.info("🔄 Resuming default model monitoring...")
                
                # Progress logging
                if chunk_count % 1000 == 0:
                    buffer_status = audio_buffer.get_buffer_status()
                    logger.info(f"📊 Progress Report:")
                    logger.info(f"   Processed chunks: {chunk_count}")
                    logger.info(f"   Detections: {detection_count}")
                    logger.info(f"   Buffer pre-roll samples: {buffer_status['preroll_samples']}")
                    logger.info(f"   Buffer post-roll samples: {buffer_status['postroll_samples']}")
                    logger.info(f"   Runtime: {chunk_count * wake_detector.get_frame_length() / wake_detector.get_sample_rate():.1f}s")
                
            except Exception as e:
                logger.error(f"❌ Error processing audio chunk {chunk_count}: {e}")
                logger.error(f"   Audio data length: {len(audio_data) if 'audio_data' in locals() else 'N/A'}")
                logger.error(f"   Stream active: {stream.is_active() if stream else False}")
                continue
                
    except KeyboardInterrupt:
        logger.info("🛑 Default model monitoring stopped by user")
    except Exception as e:
        logger.error(f"❌ Default model monitoring error: {e}")
        logger.error(f"   Last chunk processed: {chunk_count}")
        return 1
    finally:
        # Cleanup
        logger.info("🧹 Starting cleanup for default model monitoring...")
        try:
            if stream:
                logger.info("🛑 Stopping audio stream...")
                stream.stop_stream()
                stream.close()
                logger.info("✅ Audio stream closed")
            
            logger.info("🛑 Stopping audio manager...")
            audio_manager.stop_recording()
            logger.info("✅ Audio manager stopped")
            
            logger.info("🛑 Cleaning up wake detector...")
            wake_detector.cleanup()
            logger.info("✅ Wake detector cleaned up")
            
            logger.info("✅ All cleanup completed successfully")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    return 0


if __name__ == "__main__":
    # This file is designed to be imported and used by main.py
    # It can also be run independently for testing
    logger.info("🎯 Default model monitor module loaded") 