#!/usr/bin/env python3
"""
Audio Pipeline Module
Handles continuous audio processing and wake word detection
"""

import logging
import time
import numpy as np
from datetime import datetime
from audio_buffer import AudioBuffer
from audio_feedback import AudioFeedback
from shared_memory_ipc import shared_memory_ipc
from wake_word_interface import WakeWordDetector

logger = logging.getLogger(__name__)

def initialize_wake_detector(config: dict) -> WakeWordDetector:
    """Initialize wake word detector with error handling."""
    try:
        wake_detector = WakeWordDetector()
        if wake_detector.initialize(config):
            logger.info(f"✅ Wake word detector initialized: {wake_detector.get_wake_word_name()}")
            return wake_detector
        else:
            logger.error("❌ Failed to initialize wake word detector")
            return None
    except Exception as e:
        logger.error(f"❌ Error initializing wake word detector: {e}")
        return None


def run_audio_pipeline(config: dict, usb_device, audio_manager) -> int:
    """Run the main audio processing pipeline with enhanced debugging."""
    logger.info("🎯 Starting audio pipeline with enhanced debugging...")
    
    # Initialize components
    wake_detector = initialize_wake_detector(config)
    if not wake_detector:
        logger.error("❌ Failed to initialize wake detector")
        return 1
    
    audio_buffer = AudioBuffer(
        sample_rate=wake_detector.get_sample_rate(),
        preroll_seconds=config.get('buffer', {}).get('preroll_seconds', 1.0),
        postroll_seconds=config.get('buffer', {}).get('postroll_seconds', 2.0)
    )
    
    audio_feedback = AudioFeedback()
    
    # Audio processing parameters
    silence_threshold = config.get('volume_monitoring', {}).get('silence_threshold', 0.5)
    volume_window_size = config.get('volume_monitoring', {}).get('volume_window_size', 10)
    volume_history = []
    
    # Enhanced debugging variables
    detection_debug_count = 0
    last_detection_time = 0
    last_detection_chunk = 0
    detection_cooldown_seconds = 1.5
    debounce_seconds = 0.2
    detection_debounce_chunks = int(debounce_seconds * wake_detector.get_sample_rate() / wake_detector.get_frame_length())
    
    logger.info(f"🔧 DEBUG: Detection timing controls:")
    logger.info(f"   Cooldown: {detection_cooldown_seconds}s")
    logger.info(f"   Debounce: {debounce_seconds}s ({detection_debounce_chunks} chunks)")
    logger.info(f"   Silence threshold: {silence_threshold}")
    logger.info(f"   Volume window size: {volume_window_size}")
    
    # Start audio stream
    stream = audio_manager.start_stream(
        device_index=usb_device.index,
        sample_rate=wake_detector.get_sample_rate(),
        channels=wake_detector.get_channels(),
        chunk_size=wake_detector.get_frame_length()
    )
    
    if not stream:
        logger.error("❌ Failed to start audio stream")
        return 1
    
    logger.info("✅ Audio stream started successfully")
    
    # Main pipeline loop
    try:
        detection_count = 0
        chunk_count = 0
        silent_chunks = 0
        active_chunks = 0
        
        logger.info("🎯 Starting audio pipeline monitoring with enhanced debugging...")
        
        while True:
            try:
                # Read audio chunk
                audio_chunk = stream.read(wake_detector.get_frame_length(), exception_on_overflow=False)
                chunk_count += 1
                
                # Convert to numpy array
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # Calculate RMS volume level
                rms_level = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                volume_history.append(rms_level)
                
                # Update shared memory IPC for web interface
                shared_memory_ipc.update_audio_state(rms_level)
                
                # Keep only recent volume history
                if len(volume_history) > volume_window_size:
                    volume_history.pop(0)
                
                # Calculate average volume over window
                avg_volume = np.mean(volume_history) if volume_history else 0
                
                # Volume monitoring - skip silent audio
                if avg_volume < silence_threshold:
                    silent_chunks += 1
                    if silent_chunks % 100 == 0:
                        logger.debug(f"🔇 Silent audio: {silent_chunks} chunks, avg_volume={avg_volume:.2f}")
                    continue
                else:
                    active_chunks += 1
                    silent_chunks = 0
                
                # Add to audio buffer
                audio_buffer.add_audio(audio_data)
                
                # ENHANCED DEBUGGING: Check timing controls before processing
                current_time = time.time()
                time_since_last = current_time - last_detection_time
                chunks_since_last = chunk_count - last_detection_chunk
                
                # Log timing status every 50 chunks
                if chunk_count % 50 == 0:
                    logger.info(f"🔍 DEBUG: Timing status at chunk {chunk_count}:")
                    logger.info(f"   Time since last detection: {time_since_last:.2f}s (cooldown: {detection_cooldown_seconds}s)")
                    logger.info(f"   Chunks since last detection: {chunks_since_last} (debounce: {detection_debounce_chunks})")
                    logger.info(f"   Cooldown active: {time_since_last < detection_cooldown_seconds}")
                    logger.info(f"   Debounce active: {chunks_since_last < detection_debounce_chunks}")
                    logger.info(f"   Audio RMS: {rms_level:.4f}, Avg: {avg_volume:.4f}")
                    logger.info(f"   Buffer capturing postroll: {audio_buffer.is_capturing_postroll()}")
                
                # Process audio for wake-word detection
                detection_result = wake_detector.process_audio(audio_data)
                
                if detection_result:
                    detection_debug_count += 1
                    logger.info(f"🎯🎯🎯 WAKE WORD DETECTED! (Debug #{detection_debug_count}) 🎯🎯🎯")
                    logger.info(f"🎯 DETECTION #{detection_count + 1} - {wake_detector.get_wake_word_name()} detected!")
                    logger.info(f"🔊 Audio volume: {avg_volume:.2f} (threshold: {silence_threshold})")
                    
                    # ENHANCED DEBUGGING: Check all blocking conditions
                    cooldown_blocked = time_since_last < detection_cooldown_seconds
                    debounce_blocked = chunks_since_last < detection_debounce_chunks
                    buffer_blocked = audio_buffer.is_capturing_postroll()
                    volume_blocked = avg_volume < silence_threshold
                    
                    logger.info(f"🔍 DEBUG: Detection blocking analysis:")
                    logger.info(f"   Cooldown blocked: {cooldown_blocked} ({time_since_last:.2f}s < {detection_cooldown_seconds}s)")
                    logger.info(f"   Debounce blocked: {debounce_blocked} ({chunks_since_last} < {detection_debounce_chunks})")
                    logger.info(f"   Buffer blocked: {buffer_blocked}")
                    logger.info(f"   Volume blocked: {volume_blocked} ({avg_volume:.4f} < {silence_threshold})")
                    
                    # Check if detection should be allowed
                    should_allow = (not cooldown_blocked and 
                                  not debounce_blocked and 
                                  not buffer_blocked and 
                                  not volume_blocked)
                    
                    logger.info(f"🔍 DEBUG: Should allow detection: {should_allow}")
                    
                    if should_allow:
                        detection_count += 1
                        
                        # Update timing controls
                        last_detection_time = current_time
                        last_detection_chunk = chunk_count
                        
                        logger.info(f"✅ DETECTION ALLOWED - Updating timing controls:")
                        logger.info(f"   Last detection time: {last_detection_time}")
                        logger.info(f"   Last detection chunk: {last_detection_chunk}")
                        
                        # Update activation state in shared memory
                        try:
                            model_name = wake_detector.get_wake_word_name()
                            confidence = wake_detector.engine.get_latest_confidence() if hasattr(wake_detector, 'engine') else 0.0
                            shared_memory_ipc.update_activation_state(True, model_name, confidence)
                            logger.info(f"🌐 ACTIVATION: Updated shared memory - Listening: True, Model: {model_name}, Confidence: {confidence:.3f}")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not update activation state: {e}")
                        
                        # Detection log with timestamp
                        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        detection_log_line = f"[{detection_time}] WAKE WORD DETECTED: {wake_detector.get_wake_word_name()} (Detection #{detection_count})"
                        logger.info(detection_log_line)
                        
                        # Write to dedicated detection log file
                        try:
                            with open("/tmp/pipeline_detections.log", "a") as f:
                                f.write(f"{detection_log_line}\n")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not write to detection log: {e}")
                        
                        # Log detection details
                        logger.info(f"📊 Detection details:")
                        logger.info(f"   Chunk number: {chunk_count}")
                        logger.info(f"   Audio RMS level: {rms_level:.4f}")
                        logger.info(f"   Audio max level: {np.max(np.abs(audio_data))}")
                        logger.info(f"   Average volume: {avg_volume:.2f}")
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
                            clip_filename = f"/tmp/pipeline_wake_word_detection_{detection_count}.wav"
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
                        
                        # Reset activation state after processing
                        try:
                            shared_memory_ipc.update_activation_state(False)
                            logger.info("🌐 ACTIVATION: Reset to not listening after processing")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not reset activation state: {e}")
                        
                        # TODO: Stream audio to Jetson Orin
                        logger.info("📡 Audio capture completed - ready for streaming to Jetson")
                        logger.info("🔄 Resuming audio pipeline monitoring...")
                    else:
                        logger.info(f"🚫 DETECTION BLOCKED - Reason:")
                        if cooldown_blocked:
                            logger.info(f"   Cooldown period active ({detection_cooldown_seconds - time_since_last:.2f}s remaining)")
                        if debounce_blocked:
                            logger.info(f"   Debounce period active ({detection_debounce_chunks - chunks_since_last} chunks remaining)")
                        if buffer_blocked:
                            logger.info(f"   Audio buffer capturing post-roll")
                        if volume_blocked:
                            logger.info(f"   Audio volume too low ({avg_volume:.4f} < {silence_threshold})")
                
                # Progress logging
                if chunk_count % 1000 == 0:
                    buffer_status = audio_buffer.get_buffer_status()
                    logger.info(f"📊 Pipeline Progress Report:")
                    logger.info(f"   Processed chunks: {chunk_count}")
                    logger.info(f"   Active chunks: {active_chunks}")
                    logger.info(f"   Silent chunks: {silent_chunks}")
                    logger.info(f"   Detections: {detection_count}")
                    logger.info(f"   Detection attempts: {detection_debug_count}")
                    logger.info(f"   Current volume: {avg_volume:.2f} (threshold: {silence_threshold})")
                    logger.info(f"   Buffer pre-roll samples: {buffer_status['preroll_samples']}")
                    logger.info(f"   Buffer post-roll samples: {buffer_status['postroll_samples']}")
                    logger.info(f"   Runtime: {chunk_count * wake_detector.get_frame_length() / wake_detector.get_sample_rate():.1f}s")
                
            except Exception as e:
                logger.error(f"❌ Error processing audio chunk {chunk_count}: {e}")
                logger.error(f"   Audio data length: {len(audio_data) if 'audio_data' in locals() else 'N/A'}")
                logger.error(f"   Stream active: {stream.is_active() if stream else False}")
                continue
                
    except KeyboardInterrupt:
        logger.info("🛑 Audio pipeline stopped by user")
    except Exception as e:
        logger.error(f"❌ Audio pipeline error: {e}")
        logger.error(f"   Last chunk processed: {chunk_count}")
        return 1
    finally:
        # Cleanup
        logger.info("🧹 Starting cleanup for audio pipeline...")
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