#!/usr/bin/env python3
"""
OpenWakeWord script for testing custom TFLite models (Hay--compUta_v_lrg.tflite) 
from a USB microphone on a Raspberry Pi in a Docker container.
"""

import sys
import os
import argparse
import time
import wave
import json
from datetime import datetime
# Use environment variable for unbuffered output instead
os.environ['PYTHONUNBUFFERED'] = '1'

import openwakeword
from openwakeword.model import Model
import logging
import numpy as np
from audio_utils import AudioManager  # Import the AudioManager class

# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='OpenWakeWord test script with custom models')
    parser.add_argument('-record_test', '-rt', action='store_true', 
                       help='Record 10 seconds of audio for testing')
    parser.add_argument('-test_pipeline', '-tp', action='store_true',
                       help='Test pipeline with recorded audio file')
    parser.add_argument('-audio_file', default=None,
                       help='Audio file to use for testing (default: auto-generate timestamp)')
    parser.add_argument('-custom_model', default='/app/models/Hay--compUta_v_lrg.tflite',
                       help='Path to custom TFLite model')
    parser.add_argument('-load_all_models', action='store_true',
                       help='Load all custom models for comparison')
    return parser.parse_args()

def generate_timestamp_filename():
    """Generate filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"/app/recordings/custom_model_test_{timestamp}.wav"

def record_test_audio(audio_manager, usb_mic, model, filename='test_recording.wav'):
    """Record 10 seconds of test audio with countdown and metadata generation."""
    logger.info("🎤 RECORDING MODE: Recording 10 seconds of test audio...")
    
    # Metadata collection
    metadata = {
        "recording_info": {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "duration_seconds": 10,
            "sample_rate": 16000,
            "channels": 2,
            "chunk_size": 1280,
            "microphone_info": {
                "name": usb_mic.name,
                "index": usb_mic.index
            }
        },
        "rms_data": [],
        "detection_results": {
            "wake_words_detected": [],
            "total_detections": 0,
            "confidence_scores": []
        }
    }
    
    # Start audio stream for recording
    stream = audio_manager.start_stream(
        device_index=usb_mic.index,
        sample_rate=16000,
        channels=2,
        chunk_size=1280
    )
    
    if not stream:
        logger.error("Failed to start audio stream for recording")
        return False, None
    
    # Countdown
    logger.info("📡 Starting countdown...")
    for i in range(5, 0, -1):
        logger.info(f"   {i}...")
        time.sleep(1)
    
    logger.info("🔴 RECORDING...")
    sys.stdout.flush()
    
    # Record audio data and process in real-time
    frames = []
    chunks_per_second = 16000 // 1280  # ~12.5 chunks per second
    total_chunks = chunks_per_second * 10  # 10 seconds
    
    for i in range(total_chunks):
        try:
            data = stream.read(1280, exception_on_overflow=False)
            if data:
                frames.append(data)
                
                # Convert to audio data for real-time processing
                audio_array = np.frombuffer(data, dtype=np.int16)
                if len(audio_array) > 1280:  # Stereo
                    stereo_data = audio_array.reshape(-1, 2)
                    audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
                else:
                    audio_data = audio_array.astype(np.float32)
                
                # Calculate RMS
                rms = np.sqrt(np.mean(audio_data**2))
                timestamp = i / chunks_per_second
                
                # Store RMS data
                metadata["rms_data"].append({
                    "timestamp": timestamp,
                    "rms": float(rms)
                })
                
                # Process through model for real-time detection
                prediction = model.predict(audio_data)
                
                # Find highest confidence
                max_confidence = 0.0
                best_model = None
                
                for wakeword, score in prediction.items():
                    if score > max_confidence:
                        max_confidence = score
                        best_model = wakeword
                
                # Store confidence scores
                confidence_entry = {
                    "timestamp": timestamp,
                    "scores": {k: float(v) for k, v in prediction.items()},
                    "best_model": best_model,
                    "max_confidence": float(max_confidence)
                }
                metadata["detection_results"]["confidence_scores"].append(confidence_entry)
                
                # Check for wake word detection
                detection_threshold = 0.3
                if max_confidence >= detection_threshold:
                    detection = {
                        "timestamp": timestamp,
                        "wake_word": best_model,
                        "confidence": float(max_confidence),
                        "all_scores": {k: float(v) for k, v in prediction.items()}
                    }
                    metadata["detection_results"]["wake_words_detected"].append(detection)
                    logger.info(f"🎯 DETECTED during recording at {timestamp:.2f}s: {best_model} = {max_confidence:.6f}")
                
                # Progress indicator
                if i % (chunks_per_second * 2) == 0:  # Every 2 seconds
                    seconds_elapsed = i // chunks_per_second
                    logger.info(f"   Recording... {seconds_elapsed}/10 seconds (RMS: {rms:.4f})")
                
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            break
    
    # Stop recording
    stream.stop_stream()
    stream.close()
    
    # Update metadata with final results
    metadata["detection_results"]["total_detections"] = len(metadata["detection_results"]["wake_words_detected"])
    
    # Save to WAV file
    logger.info(f"💾 Saving recording to {filename}...")
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(2)  # Stereo
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(b''.join(frames))
        
        logger.info(f"✅ Recording saved successfully to {filename}")
        
        # Save metadata file
        metadata_filename = filename.replace('.wav', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📊 Metadata saved to {metadata_filename}")
        
        # Log summary
        total_detections = metadata["detection_results"]["total_detections"]
        avg_rms = np.mean([entry["rms"] for entry in metadata["rms_data"]])
        
        logger.info(f"📈 Recording Summary:")
        logger.info(f"   Duration: 10 seconds")
        logger.info(f"   Average RMS: {avg_rms:.4f}")
        logger.info(f"   Wake words detected: {total_detections}")
        
        if total_detections > 0:
            logger.info(f"   Detections:")
            for detection in metadata["detection_results"]["wake_words_detected"]:
                logger.info(f"     {detection['timestamp']:.2f}s: {detection['wake_word']} (confidence: {detection['confidence']:.6f})")
        else:
            logger.info(f"   No wake words detected during recording")
        
        return True, metadata
        
    except Exception as e:
        logger.error(f"❌ Error saving recording: {e}")
        return False, None

def load_test_audio(filename):
    """Load recorded audio file and return as audio data for pipeline."""
    logger.info(f"📂 Loading test audio from {filename}...")
    
    try:
        with wave.open(filename, 'rb') as wf:
            # Verify audio format
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            
            logger.info(f"   Audio format: {channels} channels, {sample_width} bytes/sample, {framerate} Hz")
            
            if framerate != 16000:
                logger.warning(f"⚠️  Sample rate is {framerate} Hz, expected 16000 Hz")
            
            # Read all frames
            frames = wf.readframes(wf.getnframes())
            
            # Convert to numpy array
            audio_array = np.frombuffer(frames, dtype=np.int16)
            
            # Convert stereo to mono (same as live pipeline)
            if channels == 2:
                stereo_data = audio_array.reshape(-1, 2)
                audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
                logger.info(f"   Converted stereo to mono: {len(audio_array)} -> {len(audio_data)} samples")
            else:
                audio_data = audio_array.astype(np.float32)
                logger.info(f"   Mono audio: {len(audio_data)} samples")
                
            duration = len(audio_data) / 16000
            logger.info(f"✅ Loaded {duration:.2f} seconds of audio data")
            
            return audio_data
            
    except Exception as e:
        logger.error(f"❌ Error loading audio file: {e}")
        return None

def test_pipeline_with_audio(model, audio_data):
    """Test the pipeline with recorded audio, processing in chunks like live audio."""
    logger.info("🧪 TESTING PIPELINE with recorded audio...")
    
    chunk_size = 1280
    total_chunks = len(audio_data) // chunk_size
    
    logger.info(f"   Processing {total_chunks} chunks of {chunk_size} samples each")
    
    detected_words = []
    
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        # Extract chunk (same size as live audio)
        chunk = audio_data[start_idx:end_idx]
        
        # Calculate RMS for this chunk
        rms = np.sqrt(np.mean(chunk**2))
        
        # Process through model (exactly like live pipeline)
        prediction = model.predict(chunk)
        
        # Find highest confidence
        max_confidence = 0.0
        best_model = None
        
        for wakeword, score in prediction.items():
            if score > max_confidence:
                max_confidence = score
                best_model = wakeword
        
        # Log RMS and confidence every 25 chunks (~2 seconds)
        if i % 25 == 0:
            timestamp = (i * chunk_size) / 16000
            logger.info(f"   📊 Time: {timestamp:.2f}s, RMS: {rms:.4f}, Best: {best_model} = {max_confidence:.6f}")
        
        # Detection logic (same threshold as live)
        detection_threshold = 0.3
        if max_confidence >= detection_threshold:
            timestamp = (i * chunk_size) / 16000
            logger.info(f"🎯 WAKE WORD DETECTED at {timestamp:.2f}s! Confidence: {max_confidence:.6f} - Source: {best_model}")
            detected_words.append((timestamp, best_model, max_confidence))
            
        # Also log moderate confidence like live pipeline
        elif max_confidence > 0.1:
            timestamp = (i * chunk_size) / 16000
            logger.info(f"🔍 Moderate confidence at {timestamp:.2f}s: {best_model} = {max_confidence:.6f}")
    
    # Summary
    logger.info(f"🏁 Pipeline test completed. Detected {len(detected_words)} wake words:")
    for timestamp, word, confidence in detected_words:
        logger.info(f"   {timestamp:.2f}s: {word} (confidence: {confidence:.6f})")
    
    return detected_words

def main():
    """Main function to run the custom wake word detection system."""
    # Parse command line arguments
    args = parse_arguments()

    logger.info("🚀 Starting Custom Wake Word Detection with TFLite optimization")
    logger.info(f"🎯 Target model: {args.custom_model}")

    try:
        # Handle test pipeline mode FIRST - don't need audio manager for testing
        if args.test_pipeline:
            # Use default filename if not provided
            if args.audio_file is None:
                logger.error("❌ No audio file specified for testing. Use -audio_file parameter.")
                return
            
            # Load the recorded audio
            audio_data = load_test_audio(args.audio_file)
            if audio_data is None:
                logger.error("❌ Failed to load test audio. Exiting.")
                return
            
            # Initialize the OpenWakeWord model for testing with custom model(s)
            if args.load_all_models:
                logger.info("Creating Model for pipeline testing with ALL custom TFLite models...")
                custom_models = [
                    '/app/models/Hay--compUta_v_lrg.tflite',
                    '/app/models/Hey_computer.tflite',
                    '/app/models/hey-CompUter_lrg.tflite'
                ]
                
                # Check which models exist
                available_models = []
                for model_path in custom_models:
                    if os.path.exists(model_path):
                        available_models.append(model_path)
                        logger.info(f"📁 Found model: {os.path.basename(model_path)}")
                    else:
                        logger.warning(f"⚠️ Model not found: {model_path}")
                
                if not available_models:
                    logger.error("❌ No custom models found")
                    return
                
                model = openwakeword.Model(
                    wakeword_models=available_models,
                    vad_threshold=0.5,
                    enable_speex_noise_suppression=False
                )
                logger.info(f"✅ Loaded {len(available_models)} custom models")
            else:
                logger.info("Creating Model for pipeline testing with single custom TFLite model...")
                logger.info(f"📁 Loading custom model: {args.custom_model}")
                
                # Check if custom model exists
                if not os.path.exists(args.custom_model):
                    logger.error(f"❌ Custom model not found: {args.custom_model}")
                    return
                
                model = openwakeword.Model(
                    wakeword_models=[args.custom_model],
                    vad_threshold=0.5,
                    enable_speex_noise_suppression=False
                )
            
            # Run pipeline test
            detected_words = test_pipeline_with_audio(model, audio_data)
            
            if detected_words:
                logger.info(f"✅ Pipeline test completed. Found {len(detected_words)} wake word detections.")
            else:
                logger.info("ℹ️  Pipeline test completed. No wake words detected.")
            
            return

        # Initialize AudioManager for audio device handling (only if not testing)
        audio_manager = AudioManager()
        logger.info("AudioManager initialized")

        # Find the USB microphone
        usb_mic = audio_manager.find_usb_microphone()
        if not usb_mic:
            logger.error("No USB microphone found. Exiting.")
            raise RuntimeError("No USB microphone detected")

        logger.info(f"Using USB microphone: {usb_mic.name} (index {usb_mic.index})")
        
        # Handle recording mode
        if args.record_test:
            # Generate timestamp filename if not provided
            if args.audio_file is None:
                audio_filename = generate_timestamp_filename()
            else:
                audio_filename = args.audio_file
            
            # Initialize model for recording (needed for real-time detection)
            logger.info("Creating Model for recording with custom TFLite model...")
            logger.info(f"📁 Loading custom model: {args.custom_model}")
            
            # Check if custom model exists
            if not os.path.exists(args.custom_model):
                logger.error(f"❌ Custom model not found: {args.custom_model}")
                return
            
            model = openwakeword.Model(
                wakeword_models=[args.custom_model],
                vad_threshold=0.5,
                enable_speex_noise_suppression=False
            )
            
            success, metadata = record_test_audio(audio_manager, usb_mic, model, audio_filename)
            if success:
                logger.info("✅ Recording completed successfully. Exiting.")
            else:
                logger.error("❌ Recording failed. Exiting.")
            return

        # Start audio stream with parameters suitable for OpenWakeWord
        stream = audio_manager.start_stream(
            device_index=usb_mic.index,
            sample_rate=16000,
            channels=2,
            chunk_size=1280
        )
        if not stream:
            logger.error("Failed to start audio stream. Exiting.")
            raise RuntimeError("Failed to start audio stream")

        # Initialize the OpenWakeWord model with custom TFLite model
        logger.info("🔧 Creating Model with custom TFLite model...")
        logger.info(f"📁 Loading custom model: {args.custom_model}")
        
        # Check if custom model exists
        if not os.path.exists(args.custom_model):
            logger.error(f"❌ Custom model not found: {args.custom_model}")
            return
        
        try:
            model = openwakeword.Model(
                wakeword_models=[args.custom_model],
                vad_threshold=0.5,
                enable_speex_noise_suppression=False
            )
            logger.info("✅ Custom TFLite model loaded successfully")
            
            # Test model with dummy audio
            logger.info("🔍 Testing custom model with dummy audio...")
            test_audio = np.zeros(1280, dtype=np.float32)
            try:
                test_predictions = model.predict(test_audio)
                logger.info(f"✅ Model test successful - prediction keys: {list(test_predictions.keys())}")
                
                # Log available models
                for key in test_predictions.keys():
                    logger.info(f"   Available model: '{key}'")
                
            except Exception as e:
                logger.error(f"❌ Error testing custom model: {e}")
                raise
            
        except Exception as e:
            logger.error(f"❌ Model creation failed: {e}")
            raise

        # Test audio stream
        logger.info("🧪 Testing audio stream...")
        try:
            test_data = stream.read(1280, exception_on_overflow=False)
            logger.info(f"✅ Audio stream test successful, read {len(test_data)} bytes")
        except Exception as e:
            logger.error(f"❌ Audio stream test failed: {e}")
            raise

        # Continuously listen to the audio stream and detect wake words
        logger.info("🎤 Starting custom wake word detection loop...")
        chunk_count = 0
        detection_count = 0
        
        while True:
            try:
                # Read one chunk of audio data (1280 samples)
                data = stream.read(1280, exception_on_overflow=False)
                if data is None or len(data) == 0:
                    logger.warning("No audio data read from stream")
                    continue

                # Convert bytes to numpy array - handling stereo input
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # Convert stereo to mono by averaging the channels
                if len(audio_array) > 1280:  # Stereo data
                    stereo_data = audio_array.reshape(-1, 2)
                    audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
                else:
                    audio_data = audio_array.astype(np.float32)

                # Log every 100 chunks to show we're processing audio
                chunk_count += 1
                if chunk_count % 100 == 0:
                    audio_volume = np.abs(audio_data).mean()
                    logger.info(f"📊 Processed {chunk_count} audio chunks, {detection_count} detections")
                    logger.info(f"   Audio volume: {audio_volume:.4f}")

                # Pass the audio data to the model for wake word prediction
                prediction = model.predict(audio_data)
                
                # Enhanced wake word detection logic
                max_confidence = 0.0
                best_model = None
                
                # Find the highest confidence score
                for wakeword, score in prediction.items():
                    if score > max_confidence:
                        max_confidence = score
                        best_model = wakeword
                
                # Use threshold for detection
                detection_threshold = 0.3
                if max_confidence >= detection_threshold:
                    detection_count += 1
                    logger.info(f"🎯 CUSTOM WAKE WORD DETECTED! '{best_model}' with confidence {max_confidence:.6f}")
                    logger.info(f"   Threshold: {detection_threshold:.3f}")
                    logger.info(f"   All scores: {[f'{k}: {v:.6f}' for k, v in prediction.items()]}")
                else:
                    # Log moderate confidence for debugging
                    if max_confidence > 0.1:
                        logger.info(f"🔍 Moderate confidence: {best_model} = {max_confidence:.6f}")
                    elif chunk_count % 50 == 0:  # Periodic status
                        logger.debug(f"🔍 Best confidence: {best_model} = {max_confidence:.6f}")

            except Exception as e:
                logger.error(f"Error processing audio data: {e}")
                continue

    except KeyboardInterrupt:
        logger.info("Stopping audio stream and terminating...")
        if stream:
            stream.stop_stream()
            stream.close()
        if 'audio_manager' in locals():
            audio_manager.__del__()

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        if 'stream' in locals() and stream:
            stream.stop_stream()
            stream.close()
        if 'audio_manager' in locals():
            audio_manager.__del__()

if __name__ == "__main__":
    main()