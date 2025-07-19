#!/usr/bin/env python3
"""
OpenWakeWord script for detecting wake words from a USB microphone on a Raspberry Pi in a Docker container.
Uses AudioManager for robust audio device handling.
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
import logging
import numpy as np
from hey_orac.audio.utils import AudioManager  # Import the AudioManager class

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
    parser = argparse.ArgumentParser(description='OpenWakeWord test script')
    parser.add_argument('-record_test', '-rt', action='store_true', 
                       help='Record 10 seconds of audio for testing')
    parser.add_argument('-test_pipeline', '-tp', action='store_true',
                       help='Test pipeline with recorded audio file')
    parser.add_argument('-audio_file', default=None,
                       help='Audio file to use for testing (default: auto-generate timestamp)')
    parser.add_argument('--input-wav', '--input_wav', dest='input_wav', default=None,
                       help='Use WAV file as input instead of microphone for live detection')
    return parser.parse_args()

def generate_timestamp_filename():
    """Generate filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"/app/recordings/wake_word_test_{timestamp}.wav"

class WavFileStream:
    """A class that mimics audio stream interface but reads from WAV file."""
    def __init__(self, filename, chunk_size=1280):
        self.chunk_size = chunk_size
        self.filename = filename
        self.audio_data = None
        self.position = 0
        self.channels = 1
        self._load_wav_file()
    
    def _load_wav_file(self):
        """Load WAV file and prepare audio data."""
        logger.info(f"📂 Loading WAV file: {self.filename}")
        try:
            with wave.open(self.filename, 'rb') as wf:
                self.channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                
                logger.info(f"   Audio format: {self.channels} channels, {sample_width} bytes/sample, {framerate} Hz")
                
                if framerate != 16000:
                    logger.warning(f"⚠️  Sample rate is {framerate} Hz, expected 16000 Hz")
                
                # Read all frames
                frames = wf.readframes(wf.getnframes())
                
                # Convert to numpy array (keep as int16 for now)
                self.audio_data = np.frombuffer(frames, dtype=np.int16)
                
                duration = len(self.audio_data) / (framerate * self.channels)
                logger.info(f"✅ Loaded {duration:.2f} seconds of audio data")
                
        except Exception as e:
            logger.error(f"❌ Error loading WAV file: {e}")
            raise
    
    def read(self, chunk_size, exception_on_overflow=False):
        """Read a chunk of audio data, mimicking stream.read() interface."""
        if self.position >= len(self.audio_data):
            # Loop back to beginning when we reach the end
            logger.info("🔄 Reached end of WAV file, looping back to beginning")
            self.position = 0
        
        # Calculate how many samples to read
        samples_to_read = chunk_size
        if self.channels == 2:
            samples_to_read = chunk_size * 2  # Need twice as many samples for stereo
        
        # Get the chunk
        end_pos = min(self.position + samples_to_read, len(self.audio_data))
        chunk = self.audio_data[self.position:end_pos]
        self.position = end_pos
        
        # If chunk is smaller than requested (at end of file), pad with zeros
        if len(chunk) < samples_to_read:
            chunk = np.pad(chunk, (0, samples_to_read - len(chunk)), 'constant')
        
        # Convert back to bytes to match stream interface
        return chunk.tobytes()
    
    def stop_stream(self):
        """Compatibility method - does nothing for WAV file."""
        pass
    
    def close(self):
        """Compatibility method - does nothing for WAV file."""
        pass

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
        
        # Detection logic (very low threshold for custom model testing)
        detection_threshold = 0.05
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

# Download pre-trained OpenWakeWord models if not already present
# This ensures models like "alexa", "hey jarvis", etc., are available
openwakeword.utils.download_models()

def main():
    """Main function to run the wake word detection system."""
    # Parse command line arguments
    args = parse_arguments()

    # Try to explicitly load specific models
    logger.info("Attempting to load pre-trained models...")

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
            
            # Test all three custom models individually
            custom_models = [
                '/app/models/openwakeword/Hay--compUta_v_lrg.tflite',
                '/app/models/openwakeword/hey-CompUter_lrg.tflite', 
                '/app/models/openwakeword/Hey_computer.tflite'
            ]
            
            all_detected_words = []
            
            for i, custom_model_path in enumerate(custom_models):
                logger.info(f"🔧 Testing custom model {i+1}/{len(custom_models)}: {custom_model_path}")
                
                # Check if model file exists
                if os.path.exists(custom_model_path):
                    logger.info(f"✅ Model file found at: {custom_model_path}")
                else:
                    logger.error(f"❌ Model file NOT found at: {custom_model_path}")
                    continue
                
                # Load model without class mapping to see basename detection
                logger.info("Creating Model for pipeline testing...")
                try:
                    model = openwakeword.Model(
                        wakeword_models=[custom_model_path],
                        vad_threshold=0.5,
                        enable_speex_noise_suppression=False
                    )
                    
                    # Run pipeline test for this model
                    logger.info(f"🧪 TESTING {os.path.basename(custom_model_path)} with recorded audio...")
                    detected_words = test_pipeline_with_audio(model, audio_data)
                    
                    if detected_words:
                        logger.info(f"✅ Model {os.path.basename(custom_model_path)} detected {len(detected_words)} wake words.")
                        all_detected_words.extend([(os.path.basename(custom_model_path), *word) for word in detected_words])
                    else:
                        logger.info(f"ℹ️  Model {os.path.basename(custom_model_path)} detected no wake words.")
                    
                except Exception as e:
                    logger.error(f"❌ Error testing model {custom_model_path}: {e}")
                
                logger.info("=" * 60)
            
            detected_words = all_detected_words
            
            if detected_words:
                logger.info(f"✅ Pipeline test completed. Found {len(detected_words)} wake word detections.")
            else:
                logger.info("ℹ️  Pipeline test completed. No wake words detected.")
            
            return

        # Initialize audio source - either WAV file or microphone
        stream = None
        audio_manager = None
        usb_mic = None
        
        if args.input_wav:
            # Use WAV file as input
            logger.info(f"🎵 Using WAV file as input: {args.input_wav}")
            if not os.path.exists(args.input_wav):
                logger.error(f"❌ WAV file not found: {args.input_wav}")
                return
            stream = WavFileStream(args.input_wav)
        else:
            # Initialize AudioManager for audio device handling
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
            logger.info("Creating Model for recording with real-time detection...")
            model = openwakeword.Model(
                wakeword_models=['hey_jarvis', 'alexa', 'hey_mycroft']
            )
            
            success, metadata = record_test_audio(audio_manager, usb_mic, model, audio_filename)
            if success:
                logger.info("✅ Recording completed successfully. Exiting.")
            else:
                logger.error("❌ Recording failed. Exiting.")
            return

        # Start audio stream if using microphone (skip if using WAV file)
        if not args.input_wav:
            # Start audio stream with parameters suitable for OpenWakeWord
            # - Sample rate: 16000 Hz (required by OpenWakeWord)  
            # - Channels: 2 (stereo) to match microphone capabilities, then convert to mono
            # - Chunk size: 1280 samples (80 ms at 16000 Hz) for optimal efficiency
            # Note: Microphone has 2 input channels, so read as stereo then process to mono
            stream = audio_manager.start_stream(
                device_index=usb_mic.index,
                sample_rate=16000,
                channels=2,  # Read as stereo to match microphone capabilities
                chunk_size=1280
            )
            if not stream:
                logger.error("Failed to start audio stream. Exiting.")
                raise RuntimeError("Failed to start audio stream")

        # Initialize the OpenWakeWord model
        print("DEBUG: About to create Model()", flush=True)
        try:
            # Always use custom model (Hay--compUta_v_lrg.tflite)
            custom_model_path = '/app/models/openwakeword/Hay--compUta_v_lrg.tflite'
            logger.info(f"Creating Model with custom model: {custom_model_path}")
            
            # Check if model file exists
            if os.path.exists(custom_model_path):
                logger.info(f"✅ Custom model file found at: {custom_model_path}")
            else:
                logger.error(f"❌ Custom model file NOT found at: {custom_model_path}")
                raise FileNotFoundError(f"Custom model not found: {custom_model_path}")
            
            model = openwakeword.Model(
                wakeword_models=[custom_model_path],
                vad_threshold=0.5,
                enable_speex_noise_suppression=False
            )
            # Set detection threshold
            detection_threshold = 0.1
            logger.info(f"Using custom model with detection threshold: {detection_threshold}")
            print("DEBUG: Model created successfully", flush=True)
            logger.info("OpenWakeWord model initialized")
            
            # Check what models are actually loaded
            if hasattr(model, 'models'):
                logger.info(f"Loaded models: {list(model.models.keys()) if model.models else 'None'}")
            logger.info(f"Prediction buffer keys: {list(model.prediction_buffer.keys())}")
            
            # Enhanced model verification - test with dummy audio like old working code
            logger.info("🔍 Testing model with dummy audio to verify initialization...")
            test_audio = np.zeros(1280, dtype=np.float32)
            try:
                test_predictions = model.predict(test_audio)
                logger.info(f"✅ Model test successful - prediction type: {type(test_predictions)}")
                logger.info(f"   Test prediction content: {test_predictions}")
                logger.info(f"   Test prediction keys: {list(test_predictions.keys())}")
                
                # Check prediction_buffer after first prediction (like old working code)
                if hasattr(model, 'prediction_buffer'):
                    logger.info(f"✅ Prediction buffer populated after test prediction")
                    logger.info(f"   Prediction buffer keys: {list(model.prediction_buffer.keys())}")
                    for key, scores in model.prediction_buffer.items():
                        latest_score = scores[-1] if scores else 'N/A'
                        logger.info(f"     Model '{key}': {len(scores)} scores, latest: {latest_score}")
                else:
                    logger.warning("⚠️ Prediction buffer not available after test prediction")
                    
            except Exception as e:
                logger.error(f"❌ Error testing model after creation: {e}")
                raise
            
            print("DEBUG: After model initialized log", flush=True)
        except Exception as e:
            print(f"ERROR: Model creation failed: {e}", flush=True)
            raise
        
        # Force log flush
        sys.stdout.flush()
        print("DEBUG: After sys.stdout.flush()", flush=True)

        # Test audio stream first
        print("DEBUG: About to test audio stream", flush=True)
        logger.info("🧪 Testing audio stream...")
        print("DEBUG: After audio stream test log", flush=True)
        sys.stdout.flush()
        try:
            test_data = stream.read(1280, exception_on_overflow=False)
            logger.info(f"✅ Audio stream test successful, read {len(test_data)} bytes")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"❌ Audio stream test failed: {e}")
            sys.stdout.flush()
            raise

        # Continuously listen to the audio stream and detect wake words
        logger.info("🎤 Starting wake word detection loop...")
        sys.stdout.flush()
        chunk_count = 0
        while True:
            try:
                # Read one chunk of audio data (1280 samples)
                data = stream.read(1280, exception_on_overflow=False)
                if data is None or len(data) == 0:
                    logger.warning("No audio data read from stream")
                    continue

                # Convert bytes to numpy array - now handling stereo input
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # Handle channel conversion based on input source
                if args.input_wav and hasattr(stream, 'channels'):
                    # For WAV files, check the stream's channel count
                    if stream.channels == 2 and len(audio_array) > 1280:
                        # Stereo WAV file - convert to mono
                        stereo_data = audio_array.reshape(-1, 2)
                        audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
                    else:
                        # Mono WAV file
                        audio_data = audio_array.astype(np.float32)
                else:
                    # Microphone input - use original logic
                    if len(audio_array) > 1280:  # If we got stereo data (2560 samples for stereo vs 1280 for mono)
                        # Reshape to separate left and right channels, then average
                        stereo_data = audio_array.reshape(-1, 2)
                        # CRITICAL FIX: OpenWakeWord expects raw int16 values as float32, NOT normalized!
                        audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
                    else:
                        # Already mono - CRITICAL FIX: no normalization!
                        audio_data = audio_array.astype(np.float32)

                # Log every 100 chunks to show we're processing audio
                chunk_count += 1
                if chunk_count % 100 == 0:
                    audio_volume = np.abs(audio_data).mean()
                    logger.info(f"📊 Processed {chunk_count} audio chunks")
                    logger.info(f"   Audio data shape: {audio_data.shape}, volume: {audio_volume:.4f}")
                    logger.info(f"   Raw data size: {len(data)} bytes, samples: {len(audio_array)}")
                    if len(audio_array) > 1280:
                        logger.info(f"   ✅ Stereo→Mono conversion active")

                # Pass the audio data to the model for wake word prediction
                prediction = model.predict(audio_data)
                
                # Log ALL confidence scores after each processed chunk
                if chunk_count % 100 == 0:
                    all_scores = {word: f"{score:.6f}" for word, score in prediction.items()}
                    logger.debug(f"🎯 All confidence scores: {all_scores}")
            except Exception as e:
                logger.error(f"Error processing audio data: {e}")
                continue

            # Enhanced wake word detection logic like old working code
            max_confidence = 0.0
            best_model = None
            
            # Find the highest confidence score
            for wakeword, score in prediction.items():
                if score > max_confidence:
                    max_confidence = score
                    best_model = wakeword
            
            # Use the threshold set during model initialization
            if max_confidence >= detection_threshold:
                logger.info(f"🎯 WAKE WORD DETECTED! Confidence: {max_confidence:.6f} (threshold: {detection_threshold:.6f}) - Source: {best_model}")
                logger.info(f"   All model scores: {[f'{k}: {v:.6f}' for k, v in prediction.items()]}")
            else:
                # Enhanced debugging - log more frequent confidence updates
                if chunk_count % 50 == 0:  # Every 50 chunks instead of 100
                    logger.debug(f"🎯 Best confidence: {max_confidence:.6f} from '{best_model}' (threshold: {detection_threshold:.6f})")
                    logger.debug(f"   All scores: {[f'{k}: {v:.6f}' for k, v in prediction.items()]}")
                
                # Also check for moderate confidence levels for debugging
                if max_confidence > 0.1:
                    logger.info(f"🔍 Moderate confidence detected: {best_model} = {max_confidence:.6f}")
                elif max_confidence > 0.05:
                    logger.debug(f"🔍 Weak signal: {best_model} = {max_confidence:.6f}")
                elif max_confidence > 0.01:
                    logger.debug(f"🔍 Very weak signal: {best_model} = {max_confidence:.6f}")

    except KeyboardInterrupt:
        # Handle graceful shutdown on Ctrl+C
        logger.info("Stopping audio stream and terminating AudioManager...")
        if stream:
            stream.stop_stream()
            stream.close()
        if audio_manager:
            audio_manager.__del__()  # Explicitly clean up AudioManager

    except Exception as e:
        # Log any other errors and clean up
        import traceback
        logger.error(f"Error during execution: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        if stream:
            stream.stop_stream()
            stream.close()
        if audio_manager:
            audio_manager.__del__()

if __name__ == "__main__":
    main()