#!/usr/bin/env python3
"""
Main Test Module for OpenWakeWord Detection
A self-contained test file for OpenWakeWord functionality with USB microphone access
"""

import logging
import time
import sys
import numpy as np
import pyaudio
from datetime import datetime

# OpenWakeWord imports
try:
    import openwakeword
    OPENWAKEWORD_AVAILABLE = True
    logger_init = logging.getLogger(__name__)
    logger_init.info("✅ OpenWakeWord library imported successfully")
except ImportError as e:
    OPENWAKEWORD_AVAILABLE = False
    logger_init = logging.getLogger(__name__)
    logger_init.error(f"❌ OpenWakeWord not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class USBMicrophoneTest:
    """Test class for USB microphone access and audio processing"""
    
    def __init__(self):
        self.audio = None
        self.stream = None
        self.sample_rate = 16000  # OpenWakeWord requirement
        self.channels = 1  # Mono
        self.chunk_size = 1280  # 80ms at 16kHz (OpenWakeWord requirement)
        self.format = pyaudio.paInt16
        
    def initialize_audio(self):
        """Initialize PyAudio and find USB microphone"""
        try:
            logger.info("🎤 Initializing PyAudio...")
            self.audio = pyaudio.PyAudio()
            
            # List available devices
            logger.info("📋 Available audio devices:")
            device_count = self.audio.get_device_count()
            usb_devices = []
            
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    logger.info(f"  Device {i}: {device_info['name']} - Max input channels: {device_info['maxInputChannels']}")
                    if 'usb' in device_info['name'].lower() or 'microphone' in device_info['name'].lower():
                        usb_devices.append((i, device_info))
            
            # Use default input device or first USB device found
            device_index = None
            if usb_devices:
                device_index = usb_devices[0][0]
                logger.info(f"🎯 Using USB device: {usb_devices[0][1]['name']}")
            else:
                # Use default input device
                try:
                    default_device = self.audio.get_default_input_device_info()
                    device_index = default_device['index']
                    logger.info(f"🎯 Using default device: {default_device['name']}")
                except:
                    logger.error("❌ No suitable audio input device found")
                    return False
            
            return self._test_device_access(device_index)
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize audio: {e}")
            return False
    
    def _test_device_access(self, device_index):
        """Test access to the specified audio device"""
        try:
            logger.info(f"🧪 Testing access to device {device_index}...")
            
            # Try to open the audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                start=False
            )
            
            logger.info("✅ Audio device access successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to access audio device: {e}")
            return False
    
    def test_audio_signal(self, duration_seconds=10):
        """Test audio signal capture and calculate RMS"""
        if not self.stream:
            logger.error("❌ Audio stream not initialized")
            return False
        
        try:
            logger.info(f"🎵 Starting audio signal test for {duration_seconds} seconds...")
            self.stream.start_stream()
            
            chunks_to_capture = int(duration_seconds * self.sample_rate / self.chunk_size)
            rms_values = []
            
            for chunk_num in range(chunks_to_capture):
                try:
                    # Read audio chunk
                    audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Convert to numpy array
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Calculate RMS
                    rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
                    rms_values.append(rms)
                    
                    # Log every second
                    if chunk_num % int(self.sample_rate / self.chunk_size) == 0:
                        seconds_elapsed = chunk_num * self.chunk_size / self.sample_rate
                        logger.info(f"📊 {seconds_elapsed:.1f}s - RMS: {rms:.4f}, Max: {np.max(np.abs(audio_np))}, Min: {np.min(audio_np)}")
                
                except Exception as e:
                    logger.error(f"❌ Error reading audio chunk {chunk_num}: {e}")
                    continue
            
            # Calculate statistics
            if rms_values:
                avg_rms = np.mean(rms_values)
                max_rms = np.max(rms_values)
                min_rms = np.min(rms_values)
                
                logger.info("📈 Audio Signal Statistics:")
                logger.info(f"  Average RMS: {avg_rms:.4f}")
                logger.info(f"  Maximum RMS: {max_rms:.4f}")
                logger.info(f"  Minimum RMS: {min_rms:.4f}")
                logger.info(f"  Total chunks: {len(rms_values)}")
                
                # Determine if we're getting a signal
                if avg_rms > 1.0:
                    logger.info("✅ Audio signal detected - microphone is working")
                    return True
                else:
                    logger.warning(f"⚠️ Low audio signal (avg RMS: {avg_rms:.4f}) - check microphone")
                    return True  # Still consider it working, just quiet
            else:
                logger.error("❌ No audio data captured")
                return False
                
        except Exception as e:
            logger.error(f"❌ Audio signal test failed: {e}")
            return False
        finally:
            if self.stream and self.stream.is_active():
                self.stream.stop_stream()
    
    def cleanup(self):
        """Clean up audio resources"""
        try:
            if self.stream:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                logger.info("✅ Audio stream closed")
            
            if self.audio:
                self.audio.terminate()
                logger.info("✅ PyAudio terminated")
                
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

class OpenWakeWordTest:
    """Test class for OpenWakeWord functionality"""
    
    def __init__(self):
        self.model = None
        self.threshold = 0.3
        self.sample_rate = 16000
        self.chunk_size = 1280
        self.is_initialized = False
        
    def initialize_openwakeword(self):
        """Initialize OpenWakeWord with default models"""
        try:
            if not OPENWAKEWORD_AVAILABLE:
                logger.error("❌ OpenWakeWord not available")
                return False
                
            logger.info("🧠 Initializing OpenWakeWord with default models...")
            
            # Initialize with default pre-trained models
            self.model = openwakeword.Model(
                vad_threshold=0.0,
                enable_speex_noise_suppression=False
            )
            
            # Test with silence to validate
            silence = np.zeros(self.chunk_size, dtype=np.float32)
            predictions = self.model.predict(silence)
            
            logger.info("📊 Available wake words:")
            if isinstance(predictions, dict):
                for wake_word in predictions.keys():
                    logger.info(f"  - {wake_word}")
            
            self.is_initialized = True
            logger.info("✅ OpenWakeWord initialized successfully")
            logger.info("💡 Try saying: 'Hey Jarvis', 'Hey Mycroft', or 'Alexa'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenWakeWord: {e}")
            return False
    
    def process_audio_chunk(self, audio_np):
        """Process audio chunk through OpenWakeWord"""
        try:
            if not self.is_initialized:
                return False, None, 0.0
            
            # Convert int16 to float32 and normalize
            if audio_np.dtype == np.int16:
                audio_np = audio_np.astype(np.float32) / 32768.0
            
            # Get predictions
            predictions = self.model.predict(audio_np)
            
            # Check for detections
            if isinstance(predictions, dict):
                for wake_word, confidence in predictions.items():
                    if confidence > self.threshold:
                        return True, wake_word, confidence
            
            return False, None, 0.0
            
        except Exception as e:
            logger.error(f"❌ Error processing audio: {e}")
            return False, None, 0.0
    
    def get_confidence_summary(self, audio_np):
        """Get confidence summary for logging"""
        try:
            if not self.is_initialized:
                return "Not initialized"
            
            # Convert int16 to float32 and normalize
            if audio_np.dtype == np.int16:
                audio_np = audio_np.astype(np.float32) / 32768.0
            
            predictions = self.model.predict(audio_np)
            
            if isinstance(predictions, dict):
                max_conf = max(predictions.values()) if predictions else 0.0
                summary = f"Max: {max_conf:.4f}"
                for wake_word, confidence in predictions.items():
                    if confidence > 0.01:  # Only show non-zero confidences
                        summary += f", {wake_word}: {confidence:.4f}"
                return summary
            
            return str(predictions)
            
        except Exception as e:
            return f"Error: {e}"

class IntegratedTest:
    """Integrated test combining microphone and OpenWakeWord"""
    
    def __init__(self):
        self.mic_test = USBMicrophoneTest()
        self.wake_test = OpenWakeWordTest()
        
    def run_integrated_test(self, duration_seconds=60):
        """Run integrated test with live wake word detection"""
        try:
            logger.info("🎯 Starting integrated wake word detection test...")
            logger.info(f"📅 Test duration: {duration_seconds} seconds")
            
            if not self.mic_test.stream:
                logger.error("❌ Audio stream not initialized")
                return False
            
            if not self.wake_test.is_initialized:
                logger.error("❌ OpenWakeWord not initialized")
                return False
            
            self.mic_test.stream.start_stream()
            
            chunks_to_capture = int(duration_seconds * self.mic_test.sample_rate / self.mic_test.chunk_size)
            detection_count = 0
            chunk_count = 0
            
            logger.info("🎤 Listening for wake words...")
            logger.info("💡 Try saying: 'Hey Jarvis', 'Hey Mycroft', or 'Alexa'")
            
            for chunk_num in range(chunks_to_capture):
                try:
                    # Read audio chunk
                    audio_data = self.mic_test.stream.read(self.mic_test.chunk_size, exception_on_overflow=False)
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    chunk_count += 1
                    
                    # Calculate RMS for monitoring
                    rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
                    
                    # Process through OpenWakeWord
                    detected, wake_word, confidence = self.wake_test.process_audio_chunk(audio_np)
                    
                    if detected:
                        detection_count += 1
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        logger.info(f"🎯 WAKE WORD DETECTED! [{timestamp}]")
                        logger.info(f"   Word: {wake_word}")
                        logger.info(f"   Confidence: {confidence:.4f}")
                        logger.info(f"   Detection #: {detection_count}")
                        logger.info(f"   Audio RMS: {rms:.4f}")
                    
                    # Progress updates every 10 seconds
                    if chunk_num % int(10 * self.mic_test.sample_rate / self.mic_test.chunk_size) == 0:
                        seconds_elapsed = chunk_num * self.mic_test.chunk_size / self.mic_test.sample_rate
                        conf_summary = self.wake_test.get_confidence_summary(audio_np)
                        logger.info(f"📊 {seconds_elapsed:.0f}s - RMS: {rms:.4f}, Detections: {detection_count}, {conf_summary}")
                
                except Exception as e:
                    logger.error(f"❌ Error processing chunk {chunk_num}: {e}")
                    continue
            
            logger.info("📈 Test Results:")
            logger.info(f"  Total chunks processed: {chunk_count}")
            logger.info(f"  Total detections: {detection_count}")
            logger.info(f"  Detection rate: {detection_count/duration_seconds*60:.1f} per minute")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Integrated test failed: {e}")
            return False
        finally:
            if self.mic_test.stream and self.mic_test.stream.is_active():
                self.mic_test.stream.stop_stream()

def main():
    """Main test function"""
    logger.info("🚀 Starting OpenWakeWord Test")
    logger.info("=" * 50)
    
    # Initialize test components
    integrated_test = IntegratedTest()
    
    try:
        # Stage 1: Test USB microphone access
        logger.info("📋 Stage 1: Testing USB microphone access...")
        
        if not integrated_test.mic_test.initialize_audio():
            logger.error("❌ Failed to initialize audio system")
            return 1
        
        if not integrated_test.mic_test.test_audio_signal(duration_seconds=5):
            logger.error("❌ Failed to capture audio signal")
            return 1
        
        logger.info("✅ Stage 1 completed successfully")
        
        # Stage 2: Initialize OpenWakeWord
        logger.info("📋 Stage 2: Initializing OpenWakeWord...")
        
        if not integrated_test.wake_test.initialize_openwakeword():
            logger.error("❌ Failed to initialize OpenWakeWord")
            return 1
        
        logger.info("✅ Stage 2 completed successfully")
        
        # Stage 3: Integrated test
        logger.info("📋 Stage 3: Running integrated wake word detection test...")
        
        if not integrated_test.run_integrated_test(duration_seconds=30):
            logger.error("❌ Integrated test failed")
            return 1
        
        logger.info("✅ Stage 3 completed successfully")
        logger.info("🎉 All tests completed successfully!")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return 1
    finally:
        integrated_test.mic_test.cleanup()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)