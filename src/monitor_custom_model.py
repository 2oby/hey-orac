#!/usr/bin/env python3
"""
Monitor Custom Wake Word Models
Handles monitoring of custom wake word models (Hey Computer, etc.)
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
from rms_monitor import rms_monitor

logger = logging.getLogger(__name__)


def monitor_custom_models(config: dict, usb_device, audio_manager: AudioManager, custom_model_path: str = None) -> int:
    """
    Monitor custom wake word models.
    
    Args:
        config: Configuration dictionary
        usb_device: USB audio device to use
        audio_manager: Initialized audio manager
        custom_model_path: Path to custom model file (optional)
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("🎯 Starting custom model monitoring...")
    
    # Create custom config for custom models
    custom_config = config.copy()
    if custom_model_path:
        custom_config['wake_word']['custom_model_path'] = custom_model_path
        logger.info(f"📁 Using custom model: {custom_model_path}")
    
    # Initialize wake word detector for custom models
    wake_detector = WakeWordDetector()
    if not wake_detector.initialize(custom_config):
        logger.error("❌ Failed to initialize wake word detector for custom models")
        return 1
    
    logger.info(f"✅ Custom wake word detector initialized: {wake_detector.get_wake_word_name()}")
    
    # Initialize audio feedback
    audio_feedback = create_audio_feedback()
    if audio_feedback:
        logger.info("✅ Audio feedback system initialized")
    else:
        logger.warning("⚠️ Audio feedback system not available")
    
    # Initialize audio buffer
    audio_buffer = AudioBuffer(
        sample_rate=wake_detector.get_sample_rate(),
        preroll_seconds=config['buffer'].get('preroll_seconds', 1.0),
        postroll_seconds=config['buffer'].get('postroll_seconds', 2.0)
    )
    
    # Detection debouncing and cooldown
    last_detection_time = 0  # Initialize to 0 so first detection can get through
    detection_cooldown_seconds = 1.5  # Reduced from 3.0s to 1.5s - minimum time between detections
    detection_debounce_chunks = int((wake_detector.get_sample_rate() * 0.2) / wake_detector.get_frame_length())  # 0.2s debounce in chunks
    last_detection_chunk = -detection_debounce_chunks  # Initialize to negative so first detection can get through
    
    # Volume filtering for efficiency
    silence_threshold = 0.1  # RMS threshold for silence detection
    volume_window_size = 10  # Number of chunks to average for volume calculation
    volume_history = []
    
    # Start continuous audio stream
    logger.info(f"🎤 Starting audio stream on device {usb_device.index} ({usb_device.name})")
    logger.info(f"⚙️ Stream parameters: {wake_detector.get_sample_rate()}Hz, 1 channel, {wake_detector.get_frame_length()} samples/chunk")
    logger.info(f"🛡️ Detection cooldown: {detection_cooldown_seconds}s, Debounce: {detection_debounce_chunks} chunks")
    
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
    
    # Main monitoring loop for custom models
    try:
        detection_count = 0
        chunk_count = 0
        confidence_log_count = 0
        
        logger.info("🎯 Starting continuous custom model monitoring...")
        logger.info("📊 Debug info will be logged every 100 chunks")
        logger.info("🔍 Custom model confidence will be logged every 50 chunks")
        
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
                
                # Update RMS monitor for web interface (every chunk)
                rms_level = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                rms_monitor.update_rms(rms_level)
                
                # Volume filtering - skip silent audio
                volume_history.append(rms_level)
                if len(volume_history) > volume_window_size:
                    volume_history.pop(0)
                
                avg_volume = np.mean(volume_history) if volume_history else 0
                
                # Skip processing if audio is too quiet
                if avg_volume < silence_threshold:
                    if chunk_count % 100 == 0:  # Log every 100 chunks to avoid spam
                        logger.debug(f"🔇 Skipping silent audio: avg_volume={avg_volume:.3f} < threshold={silence_threshold}")
                    continue
                
                # Add to audio buffer
                audio_buffer.add_audio(audio_data)
                
                # Enhanced custom model debugging - log confidence scores periodically
                if chunk_count % 50 == 0:
                    confidence_log_count += 1
                    try:
                        # Try to get detailed confidence scores for debugging
                        if hasattr(wake_detector, 'engine') and wake_detector.engine:
                            # Convert audio to float for model prediction
                            audio_float = audio_data.astype(np.float32) / 32768.0
                            
                            # Get raw predictions from the model
                            predictions = wake_detector.engine.model.predict(audio_float)
                            
                            if isinstance(predictions, dict):
                                # Log all prediction values
                                for model_name, confidence in predictions.items():
                                    logger.info(f"🔍 Custom Model Confidence #{confidence_log_count}: {model_name} = {confidence:.6f}")
                            else:
                                logger.info(f"🔍 Custom Model Confidence #{confidence_log_count}: {predictions}")
                    except Exception as e:
                        logger.debug(f"⚠️ Could not get detailed confidence scores: {e}")
                
                # Log confidence scores periodically
                if chunk_count % 50 == 0:
                    confidence = wake_detector.engine.get_latest_confidence()
                    audio_rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                    audio_max = np.max(np.abs(audio_data))
                    logger.info(f"🔍 Custom Model Confidence #{chunk_count//50}: {wake_detector.get_wake_word_name()} = {confidence:.6f}")
                    logger.info(f"📊 Audio levels - RMS: {audio_rms:.4f}, Max: {audio_max}")
                
                # Check if we're in cooldown period
                current_time = time.time()
                time_since_last_detection = current_time - last_detection_time
                
                # Check if we're too close to the last detection (debouncing)
                chunks_since_last_detection = chunk_count - last_detection_chunk
                
                # Process audio for wake-word detection (ALWAYS process for detector state)
                detection_result = wake_detector.process_audio(audio_data)
                
                if detection_result:
                    # Check if we should allow this detection (cooldown and debounce)
                    should_allow = (time_since_last_detection >= detection_cooldown_seconds and 
                                  chunks_since_last_detection >= detection_debounce_chunks and
                                  not audio_buffer.is_capturing_postroll())
                    
                    if should_allow:
                        
                        detection_count += 1
                        last_detection_time = current_time
                        last_detection_chunk = chunk_count
                        
                        # PROMINENT DETECTION LOG
                        logger.info("🎯🎯🎯 CUSTOM WAKE WORD DETECTED! 🎯🎯🎯")
                        logger.info(f"🎯 DETECTION #{detection_count} - {wake_detector.get_wake_word_name()} detected!")
                        logger.info(f"⏱️ Time since last detection: {time_since_last_detection:.2f}s")
                        logger.info(f"📦 Chunks since last detection: {chunks_since_last_detection}")
                        
                        # Detection log with timestamp
                        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        detection_log_line = f"[{detection_time}] CUSTOM WAKE WORD DETECTED: {wake_detector.get_wake_word_name()} (Detection #{detection_count})"
                        logger.info(detection_log_line)
                        
                        # Write to dedicated detection log file
                        try:
                            with open("/tmp/custom_detections.log", "a") as f:
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
                        
                        # Notify web interface of detection
                        try:
                            import json
                            import os
                            
                            # Record detection to a file for web interface to read
                            detection_data = {
                                'model_name': wake_detector.get_wake_word_name(),
                                'confidence': wake_detector.engine.get_latest_confidence() if hasattr(wake_detector, 'engine') else 0.0,
                                'timestamp': time.time()
                            }
                            
                            # Write to detection log file
                            detection_file = '/tmp/recent_detections.json'
                            detections = []
                            
                            # Read existing detections if file exists
                            if os.path.exists(detection_file):
                                try:
                                    with open(detection_file, 'r') as f:
                                        detections = json.load(f)
                                except:
                                    detections = []
                            
                            # Add new detection
                            detections.append(detection_data)
                            
                            # Keep only last 50 detections
                            if len(detections) > 50:
                                detections = detections[-50:]
                            
                            # Write back to file
                            with open(detection_file, 'w') as f:
                                json.dump(detections, f)
                            
                            logger.info(f"🌐 Detection recorded to file: {wake_detector.get_wake_word_name()}")
                        except Exception as e:
                            logger.debug(f"⚠️ Could not record detection to file: {e}")
                        
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
                            clip_filename = f"/tmp/custom_wake_word_detection_{detection_count}.wav"
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
                        logger.info("🔄 Resuming custom model monitoring...")
                        logger.info(f"🛡️ Cooldown active for {detection_cooldown_seconds}s...")
                    else:
                        # Log that detection was blocked by cooldown/debounce
                        logger.info(f"🛡️ Detection blocked: time_since={time_since_last_detection:.2f}s, chunks_since={chunks_since_last_detection}, cooldown={detection_cooldown_seconds}s, debounce={detection_debounce_chunks}")
                        if time_since_last_detection < detection_cooldown_seconds:
                            logger.info(f"   Reason: Cooldown period active (need {detection_cooldown_seconds - time_since_last_detection:.1f}s more)")
                        elif chunks_since_last_detection < detection_debounce_chunks:
                            logger.info(f"   Reason: Debounce period active (need {detection_debounce_chunks - chunks_since_last_detection} more chunks)")
                        elif audio_buffer.is_capturing_postroll():
                            logger.info(f"   Reason: Post-roll capture in progress")
                
                # Progress logging
                if chunk_count % 1000 == 0:
                    buffer_status = audio_buffer.get_buffer_status()
                    time_since_last = time.time() - last_detection_time
                    logger.info(f"📊 Progress Report:")
                    logger.info(f"   Processed chunks: {chunk_count}")
                    logger.info(f"   Detections: {detection_count}")
                    logger.info(f"   Time since last detection: {time_since_last:.1f}s")
                    logger.info(f"   Cooldown remaining: {max(0, detection_cooldown_seconds - time_since_last):.1f}s")
                    logger.info(f"   Buffer pre-roll samples: {buffer_status['preroll_samples']}")
                    logger.info(f"   Buffer post-roll samples: {buffer_status['postroll_samples']}")
                    logger.info(f"   Runtime: {chunk_count * wake_detector.get_frame_length() / wake_detector.get_sample_rate():.1f}s")
                
            except Exception as e:
                logger.error(f"❌ Error processing audio chunk {chunk_count}: {e}")
                logger.error(f"   Audio data length: {len(audio_data) if 'audio_data' in locals() else 'N/A'}")
                logger.error(f"   Stream active: {stream.is_active() if stream else False}")
                continue
                
    except KeyboardInterrupt:
        logger.info("🛑 Custom model monitoring stopped by user")
    except Exception as e:
        logger.error(f"❌ Custom model monitoring error: {e}")
        logger.error(f"   Last chunk processed: {chunk_count}")
        return 1
    finally:
        # Cleanup
        logger.info("🧹 Starting cleanup for custom model monitoring...")
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


def test_custom_model_with_speech(config: dict, usb_device, audio_manager: AudioManager, 
                                custom_model_path: str, duration: int = 30) -> int:
    """
    Test a specific custom model with actual speech input.
    
    Args:
        config: Configuration dictionary
        usb_device: USB audio device to use
        audio_manager: Initialized audio manager
        custom_model_path: Path to custom model file
        duration: Test duration in seconds
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info(f"🧪 Testing custom model: {custom_model_path}")
    logger.info(f"⏱️  Test duration: {duration} seconds")
    logger.info(f"🎤 Say 'Hey Computer' into the microphone...")
    
    # Create custom config for this specific model
    custom_config = config.copy()
    custom_config['wake_word']['custom_model_path'] = custom_model_path
    
    # Initialize wake word detector for this custom model
    wake_detector = WakeWordDetector()
    if not wake_detector.initialize(custom_config):
        logger.error(f"❌ Failed to initialize wake word detector for {custom_model_path}")
        return 1
    
    logger.info(f"✅ Custom wake word detector initialized: {wake_detector.get_wake_word_name()}")
    
    # Start audio stream
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
    
    # Test loop
    try:
        detection_count = 0
        chunk_count = 0
        start_time = time.time()
        
        logger.info("🎯 Starting custom model test...")
        
        while time.time() - start_time < duration:
            try:
                # Read audio chunk
                audio_chunk = stream.read(wake_detector.get_frame_length(), exception_on_overflow=False)
                chunk_count += 1
                
                # Convert to numpy array
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # Log progress every 5 seconds
                elapsed = time.time() - start_time
                if chunk_count % 62 == 0:  # ~5 seconds at 80ms chunks
                    rms_level = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                    logger.info(f"⏱️  {elapsed:.1f}s elapsed - {chunk_count} chunks processed - {detection_count} detections - RMS: {rms_level:.2f}")
                
                # Process audio for wake-word detection
                detection_result = wake_detector.process_audio(audio_data)
                
                if detection_result:
                    detection_count += 1
                    logger.info(f"🎯 DETECTION #{detection_count} - {wake_detector.get_wake_word_name()} detected!")
                
            except Exception as e:
                logger.error(f"❌ Error processing audio chunk {chunk_count}: {e}")
                continue
        
        # Test results
        elapsed = time.time() - start_time
        logger.info(f"📊 Test Results for {custom_model_path}:")
        logger.info(f"   Duration: {elapsed:.1f} seconds")
        logger.info(f"   Chunks processed: {chunk_count}")
        logger.info(f"   Detections: {detection_count}")
        logger.info(f"   Detection rate: {detection_count/elapsed:.2f} detections/second")
        
        if detection_count > 0:
            logger.info("✅ Custom model detected speech!")
            return 0
        else:
            logger.warning("⚠️ Custom model did not detect any speech")
            return 1
                
    except KeyboardInterrupt:
        logger.info("⏹️  Test interrupted by user")
        return 1
    finally:
        # Cleanup
        try:
            if stream:
                stream.stop_stream()
                stream.close()
            wake_detector.cleanup()
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    return 0


if __name__ == "__main__":
    # This file is designed to be imported and used by main.py
    # It can also be run independently for testing
    logger.info("🎯 Custom model monitor module loaded") 