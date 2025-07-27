"""
Speech recorder that captures audio after wake word detection.
"""

import logging
import numpy as np
import threading
import time
from typing import Optional, Tuple, Dict, Any
from .ring_buffer import RingBuffer
from .endpointing import SpeechEndpointer, EndpointConfig
from ..transport.stt_client import STTClient

logger = logging.getLogger(__name__)


class SpeechRecorder:
    """
    Records speech after wake word detection and sends to STT.
    
    Combines pre-roll from ring buffer with actively recorded speech
    until endpoint is detected or timeout occurs.
    """
    
    def __init__(self,
                 ring_buffer: RingBuffer,
                 stt_client: STTClient,
                 endpoint_config: Optional[EndpointConfig] = None):
        """
        Initialize speech recorder.
        
        Args:
            ring_buffer: Ring buffer containing recent audio
            stt_client: STT client for transcription
            endpoint_config: Configuration for speech endpointing
        """
        self.ring_buffer = ring_buffer
        self.stt_client = stt_client
        self.endpoint_config = endpoint_config or EndpointConfig()
        
        # Recording state
        self.is_recording = False
        self.recording_thread = None
        self.recording_lock = threading.Lock()
        
        logger.info("Speech recorder initialized")
    
    def start_recording(self, 
                       audio_stream,
                       wake_word: str,
                       confidence: float,
                       language: Optional[str] = None) -> None:
        """
        Start recording speech in a separate thread.
        
        Args:
            audio_stream: Audio stream to read from
            wake_word: The detected wake word
            confidence: Detection confidence
            language: Language code for STT
        """
        with self.recording_lock:
            if self.is_recording:
                logger.warning("Recording already in progress, ignoring new request")
                return
            
            self.is_recording = True
            
        logger.info(f"🎤 Starting speech recording thread for wake word '{wake_word}'")
        logger.debug(f"Recording parameters: confidence={confidence:.3f}, language={language}")
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(
            target=self._record_and_transcribe,
            args=(audio_stream, wake_word, confidence, language),
            daemon=True
        )
        self.recording_thread.start()
        logger.debug("Recording thread started successfully")
    
    def _record_and_transcribe(self,
                              audio_stream,
                              wake_word: str,
                              confidence: float,
                              language: Optional[str] = None) -> None:
        """
        Record speech and send to STT (runs in separate thread).
        
        Args:
            audio_stream: Audio stream to read from
            wake_word: The detected wake word
            confidence: Detection confidence
            language: Language code for STT
        """
        try:
            logger.info(f"🎙️ Starting speech recording after '{wake_word}' (confidence: {confidence:.3f})")
            
            # Get pre-roll audio from ring buffer
            logger.debug(f"Requesting {self.endpoint_config.pre_roll}s of pre-roll audio from ring buffer")
            pre_roll_audio = self.ring_buffer.read_last(self.endpoint_config.pre_roll)
            pre_roll_duration = len(pre_roll_audio)/16000 if len(pre_roll_audio) > 0 else 0
            logger.info(f"Retrieved {pre_roll_duration:.2f}s of pre-roll audio ({len(pre_roll_audio)} samples)")
            logger.debug(f"Ring buffer state: has data={len(pre_roll_audio) > 0}")
            
            # Initialize endpointer
            endpointer = SpeechEndpointer(self.endpoint_config)
            
            # Collect audio chunks
            audio_chunks = [pre_roll_audio] if len(pre_roll_audio) > 0 else []
            
            # Record until endpoint detected
            start_time = time.time()
            chunk_size = 1280  # Same as main detection loop
            
            while True:
                try:
                    # Read audio chunk
                    data = audio_stream.read(chunk_size, exception_on_overflow=False)
                    if data is None or len(data) == 0:
                        logger.warning("No audio data from stream")
                        break
                    
                    # Convert to numpy array
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    
                    # Handle stereo to mono conversion if needed
                    if len(audio_array) > chunk_size:
                        # Stereo data - convert to mono
                        stereo_data = audio_array.reshape(-1, 2)
                        audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
                    else:
                        # Already mono
                        audio_data = audio_array.astype(np.float32)
                    
                    # Store in buffer
                    audio_chunks.append(audio_data)
                    
                    # Process through endpointer
                    is_speech, should_end = endpointer.process_audio(audio_data)
                    
                    if is_speech and not hasattr(self, '_speech_started'):
                        self._speech_started = True
                        logger.debug("Speech activity detected")
                    
                    if should_end:
                        duration = endpointer.get_speech_duration()
                        logger.info(f"🔚 Speech endpoint detected after {duration:.2f}s")
                        logger.debug(f"Endpoint reason: silence detected for {self.endpoint_config.silence_duration}s + grace period {self.endpoint_config.grace_period}s")
                        break
                    
                    # Check timeout
                    if time.time() - start_time > self.endpoint_config.max_duration:
                        logger.warning("Recording timeout reached")
                        break
                        
                except Exception as e:
                    logger.error(f"Error reading audio during recording: {e}")
                    break
            
            # Combine all audio chunks
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                duration = len(full_audio) / 16000
                logger.info(f"Recorded {duration:.2f}s of audio, sending to STT...")
                
                # Send to STT
                logger.info(f"📤 Sending {duration:.2f}s audio to STT service...")
                logger.debug(f"Audio format: 16kHz, mono, {len(full_audio)} samples")
                
                success, result = self.stt_client.transcribe(
                    full_audio,
                    language=language
                )
                logger.debug(f"STT response received: success={success}")
                
                if success:
                    transcription = result.get('text', '')
                    confidence_score = result.get('confidence', 0.0)
                    processing_time = result.get('processing_time', 0.0)
                    
                    logger.info(f"✅ STT transcription successful:")
                    logger.info(f"   Wake word: {wake_word}")
                    logger.info(f"   📝 TRANSCRIPTION: '{transcription}'")
                    logger.info(f"   Confidence: {confidence_score:.2f}")
                    logger.info(f"   Processing time: {processing_time:.3f}s")
                    logger.debug(f"Full STT response: {result}")
                    
                    # TODO: Here we could emit an event or call a callback
                    # with the transcription result for further processing
                    
                else:
                    error = result.get('error', 'Unknown error')
                    logger.error(f"❌ STT transcription failed: {error}")
            else:
                logger.warning("No audio data collected for transcription")
                
        except Exception as e:
            logger.error(f"Error during recording/transcription: {e}", exc_info=True)
            
        finally:
            with self.recording_lock:
                self.is_recording = False
            if hasattr(self, '_speech_started'):
                delattr(self, '_speech_started')
            logger.info("🏁 Speech recording completed")
            logger.debug("Recording thread finished, ready for next detection")
    
    def is_busy(self) -> bool:
        """Check if currently recording."""
        with self.recording_lock:
            return self.is_recording
    
    def stop(self) -> None:
        """Stop any ongoing recording."""
        with self.recording_lock:
            self.is_recording = False
        
        if self.recording_thread and self.recording_thread.is_alive():
            logger.info("Waiting for recording thread to finish...")
            self.recording_thread.join(timeout=2.0)