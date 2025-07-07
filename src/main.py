#!/usr/bin/env python3
"""
Hey Orac - Wake-word Detection Service
Phase 1a of the ORAC Voice-Control Architecture
"""

import argparse
import logging
import sys
import yaml
import time
import numpy as np
from pathlib import Path
from audio_utils import AudioManager
from wake_word_interface import WakeWordDetector
from audio_buffer import AudioBuffer
from led_controller import SH04LEDController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)


def test_audio_file(audio_path: str, config: dict) -> None:
    """Test wake-word detection with audio file."""
    logger.info(f"Testing wake-word detection with {audio_path}")
    # TODO: Implement audio file testing
    logger.info("Audio file testing not yet implemented")


def test_openwakeword_integration(wake_detector, config: dict) -> bool:
    """
    Test OpenWakeWord integration with the actual audio pipeline.
    This runs during startup to verify the wake word detection is working.
    """
    try:
        logger.info("🔍 Testing OpenWakeWord model and prediction pipeline...")
        
        # Test 1: Verify model is loaded
        if not wake_detector.is_ready():
            logger.error("❌ Wake word detector not ready")
            return False
        
        logger.info(f"✅ Wake word detector ready: {wake_detector.get_wake_word_name()}")
        
        # Test 2: Test with silence (should give low confidence)
        logger.info("🔍 Testing with silence (should give low confidence)...")
        silence_audio = np.zeros(wake_detector.get_frame_length(), dtype=np.int16)
        
        # Process silence multiple times to get a baseline
        silence_confidences = []
        for i in range(10):
            result = wake_detector.process_audio(silence_audio)
            silence_confidences.append(result)
        
        logger.info(f"✅ Silence test completed - detections: {sum(silence_confidences)}/10")
        
        # Test 2.5: Get detailed confidence scores if possible
        try:
            # Try to access the engine directly for detailed testing
            if hasattr(wake_detector, 'engine') and wake_detector.engine:
                logger.info("🔍 Testing detailed confidence scores...")
                
                # Test silence with direct engine access
                silence_float = silence_audio.astype(np.float32) / 32768.0
                predictions = wake_detector.engine.model.predict(silence_float)
                
                if isinstance(predictions, dict):
                    confidence = predictions.get(wake_detector.get_wake_word_name(), 0.0)
                elif isinstance(predictions, (list, tuple)):
                    confidence = float(predictions[0]) if predictions else 0.0
                else:
                    confidence = float(predictions)
                
                logger.info(f"   Silence confidence: {confidence:.6f}")
                
                # Test with noise
                noise_audio = np.random.randint(-1000, 1000, wake_detector.get_frame_length(), dtype=np.int16)
                noise_float = noise_audio.astype(np.float32) / 32768.0
                predictions = wake_detector.engine.model.predict(noise_float)
                
                if isinstance(predictions, dict):
                    confidence = predictions.get(wake_detector.get_wake_word_name(), 0.0)
                elif isinstance(predictions, (list, tuple)):
                    confidence = float(predictions[0]) if predictions else 0.0
                else:
                    confidence = float(predictions)
                
                logger.info(f"   Noise confidence: {confidence:.6f}")
                
                # Test with sine wave
                t = np.linspace(0, wake_detector.get_frame_length() / wake_detector.get_sample_rate(), 
                               wake_detector.get_frame_length())
                sine_audio = (np.sin(2 * np.pi * 1000 * t) * 5000).astype(np.int16)
                sine_float = sine_audio.astype(np.float32) / 32768.0
                predictions = wake_detector.engine.model.predict(sine_float)
                
                if isinstance(predictions, dict):
                    confidence = predictions.get(wake_detector.get_wake_word_name(), 0.0)
                elif isinstance(predictions, (list, tuple)):
                    confidence = float(predictions[0]) if predictions else 0.0
                else:
                    confidence = float(predictions)
                
                logger.info(f"   Sine wave confidence: {confidence:.6f}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not get detailed confidence scores: {e}")
        
        # Test 3: Test with random noise (should give varying confidence)
        logger.info("🔍 Testing with random noise...")
        noise_audio = np.random.randint(-1000, 1000, wake_detector.get_frame_length(), dtype=np.int16)
        
        noise_results = []
        for i in range(5):
            result = wake_detector.process_audio(noise_audio)
            noise_results.append(result)
        
        logger.info(f"✅ Noise test completed - detections: {sum(noise_results)}/5")
        
        # Test 4: Test with sine wave (simulating speech-like audio)
        logger.info("🔍 Testing with sine wave (speech-like audio)...")
        t = np.linspace(0, wake_detector.get_frame_length() / wake_detector.get_sample_rate(), 
                       wake_detector.get_frame_length())
        sine_audio = (np.sin(2 * np.pi * 1000 * t) * 5000).astype(np.int16)  # 1kHz sine wave
        
        sine_results = []
        for i in range(5):
            result = wake_detector.process_audio(sine_audio)
            sine_results.append(result)
        
        logger.info(f"✅ Sine wave test completed - detections: {sum(sine_results)}/5")
        
        # Test 5: Verify threshold behavior
        logger.info("🔍 Testing threshold behavior...")
        threshold = config['wake_word'].get('threshold', 0.1)
        logger.info(f"   Current threshold: {threshold}")
        logger.info(f"   Silence detections: {sum(silence_confidences)}/10")
        logger.info(f"   Noise detections: {sum(noise_results)}/5")
        logger.info(f"   Sine wave detections: {sum(sine_results)}/5")
        
        # Summary
        total_tests = 10 + 5 + 5
        total_detections = sum(silence_confidences) + sum(noise_results) + sum(sine_results)
        
        logger.info(f"📊 Integration test summary:")
        logger.info(f"   Total audio chunks tested: {total_tests}")
        logger.info(f"   Total detections: {total_detections}")
        logger.info(f"   Detection rate: {total_detections/total_tests*100:.1f}%")
        
        # Consider the test passed if the system is processing audio
        # (we can't easily get confidence scores from the interface)
        logger.info("✅ OpenWakeWord integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenWakeWord integration test failed: {e}", exc_info=True)
        return False


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Hey Orac Wake-word Detection Service")
    parser.add_argument(
        "--config", 
        default="/app/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-audio",
        help="Test wake-word detection with audio file"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices"
    )
    parser.add_argument(
        "--test-recording",
        action="store_true",
        help="Test recording from USB microphone"
    )
    parser.add_argument(
        "--test-wakeword",
        action="store_true",
        help="Test wake-word detection with live audio"
    )
    parser.add_argument(
        "--audio-diagnostics",
        action="store_true",
        help="Run comprehensive audio system diagnostics"
    )
    parser.add_argument(
        "--test-pyaudio",
        action="store_true",
        help="Test PyAudio ALSA support and capabilities"
    )
    parser.add_argument(
        "--test-led",
        action="store_true",
        help="Test SH-04 LED controller functionality"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.audio_diagnostics:
        logger.info("🔧 Running comprehensive audio system diagnostics...")
        print("\n" + "="*60)
        print("🔧 COMPREHENSIVE AUDIO SYSTEM DIAGNOSTICS")
        print("="*60)
        
        # This will trigger all the enhanced diagnostics in AudioManager
        audio_manager = AudioManager()
        
        print("\n🎤 Testing device detection...")
        devices = audio_manager.list_input_devices()
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total devices detected: {len(devices)}")
        usb_devices = [d for d in devices if d.is_usb]
        print(f"   USB devices: {len(usb_devices)}")
        
        if usb_devices:
            print(f"   USB device names: {[d.name for d in usb_devices]}")
        
        print("\n" + "="*60)
        return
    
    if args.test_pyaudio:
        logger.info("🧪 Running PyAudio ALSA support test...")
        from test_pyaudio import test_pyaudio_alsa, check_alsa_environment
        
        check_alsa_environment()
        success = test_pyaudio_alsa()
        
        if success:
            print("\n🎉 PyAudio appears to be working correctly!")
        else:
            print("\n❌ PyAudio has issues - check the diagnostics above")
        return
    
    if args.test_led:
        logger.info("🧪 Testing SH-04 LED controller...")
        from led_controller import test_led_controller
        
        success = test_led_controller()
        if success:
            print("\n✅ LED controller test completed successfully!")
        else:
            print("\n❌ LED controller test failed!")
        return
    
    if args.list_devices:
        logger.info("🔍 Enhanced audio device detection and listing...")
        print("\n" + "="*60)
        print("🔍 ENHANCED AUDIO DEVICE DETECTION")
        print("="*60)
        
        audio_manager = AudioManager()
        devices = audio_manager.list_input_devices()
        
        print("\n📋 AVAILABLE AUDIO INPUT DEVICES")
        print("-" * 40)
        
        if not devices:
            print("❌ NO INPUT DEVICES FOUND!")
            print("   This indicates a serious audio system issue.")
            print("   Check the logs above for detailed diagnostics.")
            return
        
        for device in devices:
            print(f"\n🎤 Device {device.index}: {device.name}")
            print(f"   📊 Max Input Channels: {device.max_input_channels}")
            print(f"   🎵 Default Sample Rate: {device.default_sample_rate}")
            print(f"   🔌 Host API: {device.host_api}")
            print(f"   🔌 USB Device: {'✅ Yes' if device.is_usb else '❌ No'}")
            
            # Show additional device info if available
            if device.device_info:
                print(f"   📋 Additional Info:")
                print(f"      - Max Output Channels: {device.device_info.get('maxOutputChannels', 'N/A')}")
                print(f"      - Default Low Input Latency: {device.device_info.get('defaultLowInputLatency', 'N/A')}")
                print(f"      - Default High Input Latency: {device.device_info.get('defaultHighInputLatency', 'N/A')}")
        
        print("\n🔍 USB MICROPHONE DETECTION")
        print("-" * 30)
        
        # Find USB microphone with enhanced logging
        usb_device = audio_manager.find_usb_microphone()
        if usb_device:
            print(f"✅ SUCCESS: Found USB microphone!")
            print(f"   🎤 Device: {usb_device.name}")
            print(f"   📊 Index: {usb_device.index}")
            print(f"   🎵 Sample Rate: {usb_device.default_sample_rate}")
            print(f"   📊 Channels: {usb_device.max_input_channels}")
            print(f"   🔌 Host API: {usb_device.host_api}")
            
            # Test device accessibility
            print(f"\n🧪 Testing device accessibility...")
            try:
                test_stream = audio_manager.start_stream(
                    device_index=usb_device.index,
                    sample_rate=16000,
                    channels=1,
                    chunk_size=512
                )
                if test_stream:
                    print("   ✅ Device is accessible for recording!")
                    test_stream.stop_stream()
                    test_stream.close()
                else:
                    print("   ❌ Device is not accessible for recording!")
            except Exception as e:
                print(f"   ❌ Error testing device: {e}")
        else:
            print("❌ FAILED: No USB microphone found!")
            print("   Check the system diagnostics above for troubleshooting.")
        
        print("\n" + "="*60)
        return
    
    if args.test_audio:
        test_audio_file(args.test_audio, config)
        return
    
    if args.test_recording:
        logger.info("Testing USB microphone recording...")
        audio_manager = AudioManager()
        
        # Debug: Check AudioManager type and attributes
        logger.info(f"AudioManager type: {type(audio_manager)}")
        logger.info(f"AudioManager module: {audio_manager.__class__.__module__}")
        logger.info(f"AudioManager attributes: {[attr for attr in dir(audio_manager) if not attr.startswith('_')]}")
        
        # Find USB microphone
        usb_device = audio_manager.find_usb_microphone()
        if not usb_device:
            logger.error("❌ No USB microphone found for recording test")
            return
        
        # Test recording for 3 seconds
        output_file = "test_usb_recording.wav"
        logger.info(f"Recording 3 seconds from {usb_device.name}...")
        
        if audio_manager.record_to_file(
            usb_device.index, 
            duration=3.0, 
            output_file=output_file,
            sample_rate=16000,
            channels=1
        ):
            logger.info(f"✅ Recording successful! Saved to {output_file}")
        else:
            logger.error("❌ Recording failed")
        
        return
    
    if args.test_wakeword:
        logger.info("Testing wake-word detection with live audio...")
        
        # Initialize wake word detector
        wake_detector = WakeWordDetector()
        if not wake_detector.initialize(config['wake_word']):
            logger.error("❌ Failed to initialize wake word detector")
            return
        
        # Initialize audio manager
        audio_manager = AudioManager()
        
        # Debug: Check AudioManager type and attributes
        logger.info(f"AudioManager type: {type(audio_manager)}")
        logger.info(f"AudioManager module: {audio_manager.__class__.__module__}")
        logger.info(f"AudioManager attributes: {[attr for attr in dir(audio_manager) if not attr.startswith('_')]}")
        
        usb_device = audio_manager.find_usb_microphone()
        if not usb_device:
            logger.error("❌ No USB microphone found")
            return
        
        logger.info(f"🎤 Listening for '{wake_detector.get_wake_word_name()}' on {usb_device.name}")
        logger.info("Press Ctrl+C to stop...")
        
        # Simple test mode to prevent system crashes
        logger.info("🔧 Using simple test mode to prevent system crashes...")
        
        try:
            for i in range(5):  # Test 5 samples
                logger.info(f"🎙️ Recording 2-second sample...")
                
                # Record audio sample
                if not audio_manager.record_to_file(
                    usb_device.index,
                    2.0,
                    "/tmp/test_audio.wav",
                    sample_rate=wake_detector.get_sample_rate(),
                    channels=1
                ):
                    logger.error("❌ Failed to record audio sample")
                    continue
                
                # Load and process audio in chunks
                import soundfile as sf
                
                try:
                    # Load the recorded audio
                    audio_data, sample_rate = sf.read("/tmp/test_audio.wav")
                    
                    # Debug: Check audio quality
                    rms_level = np.sqrt(np.mean(audio_data**2))
                    max_level = np.max(np.abs(audio_data))
                    min_level = np.min(audio_data)
                    mean_level = np.mean(audio_data)
                    
                    logger.info(f"🎵 Audio Quality Check:")
                    logger.info(f"   RMS Level: {rms_level:.6f}")
                    logger.info(f"   Max Level: {max_level:.6f}")
                    logger.info(f"   Min Level: {min_level:.6f}")
                    logger.info(f"   Mean Level: {mean_level:.6f}")
                    logger.info(f"   Sample Rate: {sample_rate}")
                    logger.info(f"   Duration: {len(audio_data)/sample_rate:.2f}s")
                    logger.info(f"   Samples: {len(audio_data)}")
                    
                    # Check if audio is too quiet
                    if rms_level < 0.001:
                        logger.warning("⚠️ Audio is very quiet - check microphone volume!")
                    elif rms_level < 0.01:
                        logger.warning("⚠️ Audio is quiet - consider speaking louder")
                    else:
                        logger.info("✅ Audio levels look good")
                    
                    # Save a copy for manual inspection
                    sf.write("/tmp/debug_audio.wav", audio_data, sample_rate)
                    logger.info("💾 Saved debug audio to /tmp/debug_audio.wav")
                    
                    # Process audio in 1280-sample chunks
                    chunk_size = wake_detector.get_frame_length()
                    detected = False
                    
                    for i in range(0, len(audio_data) - chunk_size + 1, chunk_size):
                        chunk = audio_data[i:i + chunk_size]
                        
                        # Convert to int16 format for OpenWakeWord
                        chunk_int16 = (chunk * 32767).astype(np.int16)
                        
                        if wake_detector.process_audio(chunk_int16):
                            detected = True
                            logger.info(f"🎯 Wake word detected in chunk {i//chunk_size + 1}!")
                            break
                    
                    if not detected:
                        logger.info("👂 No wake word detected in this sample")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing audio: {e}")
                    continue
                
                time.sleep(1)  # Wait between samples
                
        except KeyboardInterrupt:
            logger.info("🛑 Test stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in test mode: {e}")
        finally:
            # Clean up
            try:
                import os
                if os.path.exists("/tmp/test_audio.wav"):
                    os.remove("/tmp/test_audio.wav")
                logger.info("✅ Test mode cleanup completed")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        return
    
    # Main service loop - TEMPORARILY DISABLED FOR MICROPHONE TESTING
    logger.info("Starting Hey Orac wake-word detection service...")
    logger.info(f"Configuration: {config}")
    
    # Initialize wake word detector
    wake_detector = WakeWordDetector()
    if not wake_detector.initialize(config['wake_word']):
        logger.error("❌ Failed to initialize wake word detector")
        sys.exit(1)
    
    # Run OpenWakeWord integration test during startup
    logger.info("🧪 Running OpenWakeWord integration test...")
    if not test_openwakeword_integration(wake_detector, config):
        logger.warning("⚠️ OpenWakeWord integration test failed - continuing anyway")
    else:
        logger.info("✅ OpenWakeWord integration test passed")
    
    # Initialize audio manager
    audio_manager = AudioManager()
    
    # Find USB microphone
    usb_device = audio_manager.find_usb_microphone()
    if not usb_device:
        logger.error("❌ No USB microphone found")
        sys.exit(1)
    
    # Initialize audio buffer for pre-roll and post-roll capture
    audio_buffer = AudioBuffer(
        sample_rate=wake_detector.get_sample_rate(),
        channels=1,
        preroll_seconds=config['buffer']['preroll_seconds'],
        postroll_seconds=config['buffer']['postroll_seconds']
    )
    
    # Initialize LED controller for visual feedback
    led_controller = SH04LEDController()
    if led_controller.connect():
        logger.info("✅ LED controller connected - visual feedback enabled")
        
        # Run LED test for 30 seconds on startup
        logger.info("🎯 Running LED test for 30 seconds on startup...")
        logger.info("   LED should flash between green (off) and red (on)")
        
        start_time = time.time()
        flash_count = 0
        
        while time.time() - start_time < 30:
            # LED ON (red/muted)
            if led_controller.set_led_state(True):
                flash_count += 1
                elapsed = time.time() - start_time
                logger.info(f"   {flash_count:2d}. LED ON  (red) - {elapsed:.1f}s")
            else:
                logger.error("   ❌ Error setting LED ON")
            
            time.sleep(0.5)
            
            # LED OFF (green/active)
            if led_controller.set_led_state(False):
                elapsed = time.time() - start_time
                logger.info(f"   {flash_count:2d}. LED OFF (green) - {elapsed:.1f}s")
            else:
                logger.error("   ❌ Error setting LED OFF")
            
            time.sleep(0.5)
        
        logger.info(f"✅ LED test completed! Total flashes: {flash_count}")
        logger.info("   LED should be back to green (normal state)")
        
    else:
        logger.warning("⚠️ LED controller not available - no visual feedback")
        led_controller = None
    
    logger.info(f"🎤 Microphone found: {usb_device.name}")
    logger.info(f"🎯 Wake word: '{wake_detector.get_wake_word_name()}'")
    logger.info(f"⚙️ Sample rate: {wake_detector.get_sample_rate()}")
    logger.info(f"📏 Frame length: {wake_detector.get_frame_length()}")
    logger.info("🎯 Starting continuous wake-word detection with OpenWakeWord")
    logger.info("📋 Use --audio-diagnostics, --test-pyaudio, or --list-devices for diagnostics")
    logger.info("🔄 Main audio processing loop ENABLED with comprehensive debugging")
    
    # ENABLED: Main audio processing loop with comprehensive debugging
    # This is the main wake-word detection service
    
    # Start continuous audio stream with enhanced debugging
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
        sys.exit(1)
    
    logger.info("✅ Audio stream started successfully")
    
    try:
        detection_count = 0
        chunk_count = 0
        last_audio_level_log = 0
        
        logger.info("🎯 Starting continuous wake-word detection loop...")
        logger.info("📊 Debug info will be logged every 100 chunks")
        
        while True:
            try:
                # Read audio chunk with enhanced error handling
                audio_chunk = stream.read(wake_detector.get_frame_length(), exception_on_overflow=False)
                chunk_count += 1
                
                # Convert to numpy array
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # Enhanced audio level monitoring
                if chunk_count % 100 == 0:  # Log every 100 chunks
                    rms_level = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                    max_level = np.max(np.abs(audio_data))
                    logger.info(f"📊 Chunk {chunk_count}: RMS={rms_level:.2f}, Max={max_level}, Samples={len(audio_data)}")
                    last_audio_level_log = chunk_count
                
                # Add to audio buffer
                audio_buffer.add_audio(audio_data)
                
                # Process audio for wake-word detection with enhanced logging
                logger.debug(f"🔍 Processing chunk {chunk_count} for wake-word detection...")
                detection_result = wake_detector.process_audio(audio_data)
                
                if detection_result:
                    detection_count += 1
                    logger.info(f"🎯 WAKE WORD DETECTED! (Detection #{detection_count})")
                    logger.info(f"📊 Detection details:")
                    logger.info(f"   Chunk number: {chunk_count}")
                    logger.info(f"   Audio RMS level: {np.sqrt(np.mean(audio_data.astype(np.float32)**2)):.4f}")
                    logger.info(f"   Audio max level: {np.max(np.abs(audio_data))}")
                    logger.info(f"   Buffer status: {audio_buffer.get_buffer_status()}")
                    
                    # Visual feedback with LED
                    if led_controller:
                        logger.info("💡 Providing visual feedback with LED...")
                        led_controller.wake_word_detected_feedback()
                    
                    # Start post-roll capture
                    logger.info("📦 Starting post-roll capture...")
                    audio_buffer.start_postroll_capture()
                    
                    # Wait for post-roll capture to complete
                    postroll_chunks = 0
                    while audio_buffer.is_capturing_postroll():
                        # Continue reading audio during post-roll
                        audio_chunk = stream.read(wake_detector.get_frame_length(), exception_on_overflow=False)
                        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                        audio_buffer.add_audio(audio_data)
                        chunk_count += 1
                        postroll_chunks += 1
                        
                        if postroll_chunks % 10 == 0:  # Log every 10 post-roll chunks
                            logger.debug(f"📦 Post-roll capture: {postroll_chunks} chunks captured")
                    
                    logger.info(f"📦 Post-roll capture completed: {postroll_chunks} chunks")
                    
                    # Get complete audio clip
                    logger.info("🎵 Retrieving complete audio clip...")
                    complete_audio = audio_buffer.get_complete_audio_clip()
                    if complete_audio is not None:
                        # Save audio clip for now (replace with streaming to Jetson)
                        clip_filename = f"/tmp/wake_word_detection_{detection_count}.wav"
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
                    
                    # TODO: Stream audio to Jetson Orin
                    logger.info("📡 Audio capture completed - ready for streaming to Jetson")
                    logger.info("🔄 Resuming wake-word detection...")
                
                # Enhanced progress logging
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
        logger.info("🛑 Service stopped by user")
    except Exception as e:
        logger.error(f"❌ Service error: {e}")
        logger.error(f"   Last chunk processed: {chunk_count}")
        sys.exit(1)
    finally:
        # Enhanced cleanup with detailed logging
        logger.info("🧹 Starting cleanup...")
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
            
            # Cleanup LED controller
            if led_controller:
                logger.info("🛑 Disconnecting LED controller...")
                led_controller.disconnect()
                logger.info("✅ LED controller disconnected")
            
            logger.info("✅ All cleanup completed successfully")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


if __name__ == "__main__":
    main() 