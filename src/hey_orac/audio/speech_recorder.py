"""
Speech recorder that captures audio after wake word detection.

Supports two modes:
- HTTP (bulk): Record all audio, then send complete WAV file
- WebSocket (streaming): Stream audio chunks as they're recorded
"""

import logging
import numpy as np
import threading
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Generator
from .ring_buffer import RingBuffer
from .endpointing import SpeechEndpointer, EndpointConfig
from ..transport.stt_client import STTClient, WebSocketSTTClient

logger = logging.getLogger(__name__)


class SpeechRecorder:
    """
    Records speech after wake word detection and sends to STT.

    Combines pre-roll from ring buffer with actively recorded speech
    until endpoint is detected or timeout occurs.

    Supports both HTTP (bulk) and WebSocket (streaming) modes.
    """

    def __init__(self,
                 ring_buffer: RingBuffer,
                 stt_client: STTClient,
                 endpoint_config: Optional[EndpointConfig] = None,
                 streaming_client: Optional[WebSocketSTTClient] = None,
                 use_streaming: bool = False,
                 fallback_to_http: bool = True):
        """
        Initialize speech recorder.

        Args:
            ring_buffer: Ring buffer containing recent audio
            stt_client: STT client for HTTP transcription
            endpoint_config: Configuration for speech endpointing
            streaming_client: Optional WebSocket client for streaming mode
            use_streaming: Whether to use streaming mode (requires streaming_client)
            fallback_to_http: If streaming fails, fall back to HTTP mode
        """
        self.ring_buffer = ring_buffer
        self.stt_client = stt_client
        self.endpoint_config = endpoint_config or EndpointConfig()
        self.streaming_client = streaming_client
        self.use_streaming = use_streaming and streaming_client is not None
        self.fallback_to_http = fallback_to_http

        # Recording state
        self.is_recording = False
        self.recording_thread = None
        self.recording_lock = threading.Lock()

        mode = "streaming" if self.use_streaming else "HTTP"
        logger.info(f"Speech recorder initialized (mode: {mode})")
    
    def start_recording(self,
                       audio_stream,
                       wake_word: str,
                       confidence: float,
                       language: Optional[str] = None,
                       webhook_url: Optional[str] = None,
                       topic: str = "general",
                       wake_word_time: Optional[datetime] = None) -> None:
        """
        Start recording speech in a separate thread.

        Args:
            audio_stream: Audio stream to read from
            wake_word: The detected wake word
            confidence: Detection confidence
            language: Language code for STT
            webhook_url: Optional webhook URL for STT service
            topic: Topic ID for routing
            wake_word_time: Timestamp when wake word was detected (for end-to-end timing)
        """
        with self.recording_lock:
            if self.is_recording:
                logger.warning("Recording already in progress, ignoring new request")
                return

            self.is_recording = True

        # Use current time if wake_word_time not provided
        if wake_word_time is None:
            wake_word_time = datetime.now()

        logger.info(f"🎤 Starting speech recording thread for wake word '{wake_word}'")
        logger.debug(f"Recording parameters: confidence={confidence:.3f}, language={language}, wake_word_time={wake_word_time.isoformat()}")

        # Start recording in a separate thread
        self.recording_thread = threading.Thread(
            target=self._record_and_transcribe,
            args=(audio_stream, wake_word, confidence, language, webhook_url, topic, wake_word_time),
            daemon=True
        )
        self.recording_thread.start()
        logger.debug("Recording thread started successfully")
    
    def _record_and_transcribe(self,
                              audio_stream,
                              wake_word: str,
                              confidence: float,
                              language: Optional[str] = None,
                              webhook_url: Optional[str] = None,
                              topic: str = "general",
                              wake_word_time: Optional[datetime] = None) -> None:
        """
        Record speech and send to STT (runs in separate thread).

        Args:
            audio_stream: Audio stream to read from
            wake_word: The detected wake word
            confidence: Detection confidence
            language: Language code for STT
            webhook_url: Optional webhook URL for STT service
            topic: Topic ID for routing
            wake_word_time: Timestamp when wake word was detected (for end-to-end timing)
        """
        recorder_queue = None

        try:
            logger.info(f"🎙️ Starting speech recording after '{wake_word}' (confidence: {confidence:.3f})")

            # Register as consumer if using AudioReaderThread
            if hasattr(audio_stream, 'register_consumer'):
                recorder_queue = audio_stream.register_consumer("speech_recorder")
                logger.info("✅ Speech recorder registered as audio consumer")

            # Choose transcription mode
            if self.use_streaming:
                success, result = self._record_and_transcribe_streaming(
                    audio_stream, recorder_queue, topic, wake_word_time
                )

                # Fall back to HTTP if streaming failed and fallback enabled
                if not success and self.fallback_to_http:
                    logger.warning("⚠️ Streaming failed, falling back to HTTP mode")
                    success, result = self._record_and_transcribe_http(
                        audio_stream, recorder_queue, language, webhook_url, topic, wake_word_time
                    )
            else:
                success, result = self._record_and_transcribe_http(
                    audio_stream, recorder_queue, language, webhook_url, topic, wake_word_time
                )

            # Log result
            if success:
                transcription = result.get('text', '')
                confidence_score = result.get('confidence', 0.0)
                processing_time = result.get('processing_time', 0.0)
                mode = "streaming" if result.get('streaming') else "HTTP"

                logger.info(f"✅ STT transcription successful ({mode}):")
                logger.info(f"   Wake word: {wake_word}")
                logger.info(f"   📝 TRANSCRIPTION: '{transcription}'")
                logger.info(f"   Confidence: {confidence_score:.2f}")
                logger.info(f"   Processing time: {processing_time:.3f}s")
            else:
                error = result.get('error', 'Unknown error')
                logger.error(f"❌ STT transcription failed: {error}")

        except Exception as e:
            logger.error(f"Error during recording/transcription: {e}", exc_info=True)

        finally:
            # Unregister as consumer if we registered
            if recorder_queue is not None and hasattr(audio_stream, 'unregister_consumer'):
                audio_stream.unregister_consumer("speech_recorder")
                logger.info("🚮 Speech recorder unregistered as audio consumer")

            with self.recording_lock:
                self.is_recording = False
            if hasattr(self, '_speech_started'):
                delattr(self, '_speech_started')
            logger.info("🏁 Speech recording completed")

    def _record_and_transcribe_streaming(self,
                                         audio_stream,
                                         recorder_queue,
                                         topic: str,
                                         wake_word_time: Optional[datetime]) -> Tuple[bool, Dict[str, Any]]:
        """
        Record and transcribe using WebSocket streaming.

        Streams audio chunks to STT as they're recorded, rather than
        waiting for the complete recording.

        Returns:
            Tuple of (success, result_dict)
        """
        logger.info("🌊 Using WebSocket streaming mode")

        # Create audio generator that yields chunks and endpoint status
        def audio_generator():
            # Get pre-roll audio from ring buffer
            pre_roll_audio = self.ring_buffer.read_last(self.endpoint_config.pre_roll)
            if len(pre_roll_audio) > 0:
                pre_roll_duration = len(pre_roll_audio) / 16000
                logger.info(f"📼 Sending {pre_roll_duration:.2f}s pre-roll audio")
                # Yield pre-roll in chunks to avoid large single send
                chunk_samples = 1280
                for i in range(0, len(pre_roll_audio), chunk_samples):
                    chunk = pre_roll_audio[i:i + chunk_samples]
                    yield chunk, False

            # Initialize endpointer
            endpointer = SpeechEndpointer(self.endpoint_config)
            start_time = time.time()
            chunk_size = 1280

            while True:
                try:
                    # Read audio chunk
                    if recorder_queue is not None:
                        try:
                            data = recorder_queue.get(timeout=2.0)
                        except:
                            data = None
                    elif hasattr(audio_stream, 'get_audio'):
                        data = audio_stream.get_audio(timeout=2.0)
                    else:
                        data = audio_stream.read(chunk_size, exception_on_overflow=False)

                    if data is None or len(data) == 0:
                        logger.warning("No audio data from stream/queue")
                        yield None, True
                        break

                    # Convert to numpy array
                    audio_array = np.frombuffer(data, dtype=np.int16)

                    # Handle stereo to mono conversion
                    if len(audio_array) > chunk_size:
                        stereo_data = audio_array.reshape(-1, 2)
                        audio_data = np.mean(stereo_data, axis=1).astype(np.float32) / 32768.0
                    else:
                        audio_data = audio_array.astype(np.float32) / 32768.0

                    # Process through endpointer
                    is_speech, should_end = endpointer.process_audio(audio_data)

                    if is_speech and not hasattr(self, '_speech_started'):
                        self._speech_started = True
                        logger.debug("Speech activity detected")

                    # Yield chunk (as float32, will be converted to int16 by client)
                    yield audio_data, should_end

                    if should_end:
                        duration = endpointer.get_speech_duration()
                        logger.info(f"🔚 Speech endpoint detected after {duration:.2f}s")
                        break

                    # Check timeout
                    if time.time() - start_time > self.endpoint_config.max_duration:
                        logger.warning("Recording timeout reached")
                        yield None, True
                        break

                except Exception as e:
                    logger.error(f"Error reading audio during streaming: {e}")
                    yield None, True
                    break

        # Stream transcribe
        return self.streaming_client.transcribe_sync(
            audio_generator(),
            topic=topic,
            wake_word_time=wake_word_time
        )

    def _record_and_transcribe_http(self,
                                    audio_stream,
                                    recorder_queue,
                                    language: Optional[str],
                                    webhook_url: Optional[str],
                                    topic: str,
                                    wake_word_time: Optional[datetime]) -> Tuple[bool, Dict[str, Any]]:
        """
        Record and transcribe using HTTP bulk mode (original behavior).

        Collects all audio, then sends complete WAV file.

        Returns:
            Tuple of (success, result_dict)
        """
        logger.info("📦 Using HTTP bulk mode")

        # Get pre-roll audio from ring buffer
        pre_roll_audio = self.ring_buffer.read_last(self.endpoint_config.pre_roll)
        pre_roll_duration = len(pre_roll_audio) / 16000 if len(pre_roll_audio) > 0 else 0
        logger.info(f"Retrieved {pre_roll_duration:.2f}s of pre-roll audio")

        # Initialize endpointer
        endpointer = SpeechEndpointer(self.endpoint_config)

        # Collect audio chunks
        audio_chunks = []
        if len(pre_roll_audio) > 0:
            audio_chunks.append(pre_roll_audio)

        # Record until endpoint detected
        start_time = time.time()
        chunk_size = 1280

        while True:
            try:
                # Read audio chunk
                if recorder_queue is not None:
                    try:
                        data = recorder_queue.get(timeout=2.0)
                    except:
                        data = None
                elif hasattr(audio_stream, 'get_audio'):
                    data = audio_stream.get_audio(timeout=2.0)
                else:
                    data = audio_stream.read(chunk_size, exception_on_overflow=False)

                if data is None or len(data) == 0:
                    logger.warning("No audio data from stream/queue")
                    break

                # Convert to numpy array
                audio_array = np.frombuffer(data, dtype=np.int16)

                # Handle stereo to mono conversion
                if len(audio_array) > chunk_size:
                    stereo_data = audio_array.reshape(-1, 2)
                    audio_data = np.mean(stereo_data, axis=1).astype(np.float32) / 32768.0
                else:
                    audio_data = audio_array.astype(np.float32) / 32768.0

                audio_chunks.append(audio_data)

                # Process through endpointer
                is_speech, should_end = endpointer.process_audio(audio_data)

                if is_speech and not hasattr(self, '_speech_started'):
                    self._speech_started = True
                    logger.debug("Speech activity detected")

                if should_end:
                    duration = endpointer.get_speech_duration()
                    logger.info(f"🔚 Speech endpoint detected after {duration:.2f}s")
                    break

                # Check timeout
                if time.time() - start_time > self.endpoint_config.max_duration:
                    logger.warning("Recording timeout reached")
                    break

            except Exception as e:
                logger.error(f"Error reading audio during recording: {e}")
                break

        # Send to STT
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            duration = len(full_audio) / 16000
            recording_end_time = datetime.now()

            logger.info(f"📤 Sending {duration:.2f}s audio to STT service...")
            if wake_word_time:
                elapsed_since_wake = (recording_end_time - wake_word_time).total_seconds()
                logger.info(f"⏱️ Time from wake word to recording end: {elapsed_since_wake:.2f}s")

            return self.stt_client.transcribe(
                full_audio,
                language=language,
                webhook_url=webhook_url,
                topic=topic,
                wake_word_time=wake_word_time,
                recording_end_time=recording_end_time
            )
        else:
            logger.warning("No audio data collected for transcription")
            return False, {'error': 'No audio data collected'}
    
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