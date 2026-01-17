"""
STT (Speech-to-Text) client for sending audio to transcription service.

Supports two modes:
- HTTP (bulk): Record all audio, then send complete WAV file
- WebSocket (streaming): Stream audio chunks as they're recorded
"""

import asyncio
import io
import json
import wave
import logging
import requests
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class WebSocketSTTClient:
    """WebSocket client for streaming audio to STT service."""

    def __init__(self, base_url: str = "http://192.168.8.192:7272",
                 ws_path: str = "/stt/v1/ws/stream",
                 timeout: int = 30):
        """
        Initialize WebSocket STT client.

        Args:
            base_url: Base URL of the STT service (http:// will be converted to ws://)
            ws_path: WebSocket endpoint path
            timeout: Connection timeout in seconds
        """
        # Convert HTTP URL to WebSocket URL
        parsed = urlparse(base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        self.ws_base_url = f"{ws_scheme}://{parsed.netloc}"
        self.ws_path = ws_path
        self.timeout = timeout

        logger.info(f"WebSocket STT client initialized: {self.ws_base_url}{self.ws_path}")

    def get_ws_url(self, topic: str) -> str:
        """Get full WebSocket URL for a topic."""
        return f"{self.ws_base_url}{self.ws_path}/{topic}"

    async def stream_transcribe(self,
                                audio_generator,
                                topic: str = "general",
                                wake_word_time: Optional[datetime] = None,
                                on_chunk_sent: Optional[callable] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Stream audio chunks and get transcription.

        Args:
            audio_generator: Generator yielding (audio_chunk, is_end) tuples
                           audio_chunk: numpy array of int16 samples
                           is_end: True when speech endpoint detected
            topic: Topic ID for ORAC Core routing
            wake_word_time: Timestamp when wake word was detected
            on_chunk_sent: Optional callback(chunk_count, total_samples) after each chunk

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed. Install with: pip install websockets")
            return False, {'error': 'websockets package not installed'}

        ws_url = self.get_ws_url(topic)
        logger.info(f"🔌 Connecting to WebSocket: {ws_url}")

        start_time = datetime.now()
        chunk_count = 0
        total_samples = 0

        try:
            async with websockets.connect(ws_url, close_timeout=self.timeout) as ws:
                logger.info("✅ WebSocket connected")

                # Send config with timing info
                config_msg = {"type": "config"}
                if wake_word_time:
                    config_msg["wake_word_time"] = wake_word_time.isoformat()
                await ws.send(json.dumps(config_msg))
                logger.debug(f"Sent config: {config_msg}")

                # Stream audio chunks
                for audio_chunk, is_end in audio_generator:
                    if audio_chunk is not None and len(audio_chunk) > 0:
                        # Convert to int16 bytes if needed
                        if audio_chunk.dtype == np.float32:
                            audio_int16 = np.clip(audio_chunk * 32767, -32768, 32767).astype(np.int16)
                        elif audio_chunk.dtype == np.int16:
                            audio_int16 = audio_chunk
                        else:
                            audio_int16 = audio_chunk.astype(np.int16)

                        # Send binary audio data
                        await ws.send(audio_int16.tobytes())
                        chunk_count += 1
                        total_samples += len(audio_int16)

                        if on_chunk_sent:
                            on_chunk_sent(chunk_count, total_samples)

                        if chunk_count % 25 == 0:  # Log every ~2 seconds
                            duration = total_samples / 16000
                            logger.debug(f"Streamed {chunk_count} chunks ({duration:.1f}s)")

                    if is_end:
                        break

                # Send end signal
                end_signal = {"type": "end"}
                await ws.send(json.dumps(end_signal))
                stream_duration = (datetime.now() - start_time).total_seconds()
                audio_duration = total_samples / 16000
                logger.info(f"📤 Sent end signal. Streamed {audio_duration:.2f}s audio in {stream_duration:.2f}s ({chunk_count} chunks)")

                # Wait for transcription result
                logger.info("⏳ Waiting for transcription...")
                response = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                result = json.loads(response)

                total_time = (datetime.now() - start_time).total_seconds()
                transcription = result.get('text', '')

                logger.info(f"✅ WebSocket transcription successful: '{transcription}'")
                logger.info(f"⏱️ Total WebSocket session: {total_time:.2f}s")

                return True, {
                    'text': transcription,
                    'confidence': result.get('confidence', 0.0),
                    'language': result.get('language'),
                    'duration': result.get('duration', audio_duration),
                    'processing_time': result.get('processing_time', 0.0),
                    'streaming': True,
                    'chunks_sent': chunk_count
                }

        except asyncio.TimeoutError:
            logger.error(f"WebSocket transcription timed out after {self.timeout}s")
            return False, {'error': 'WebSocket transcription timed out'}

        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
            return False, {'error': str(e)}

    def transcribe_sync(self,
                        audio_generator,
                        topic: str = "general",
                        wake_word_time: Optional[datetime] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Synchronous wrapper for stream_transcribe.

        Args:
            audio_generator: Generator yielding (audio_chunk, is_end) tuples
            topic: Topic ID for ORAC Core routing
            wake_word_time: Timestamp when wake word was detected

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(
                self.stream_transcribe(audio_generator, topic, wake_word_time)
            )
        except Exception as e:
            logger.error(f"Error in synchronous WebSocket transcribe: {e}")
            return False, {'error': str(e)}


class STTClient:
    """Client for interacting with the STT API service."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize STT client.
        
        Args:
            base_url: Base URL of the STT service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        logger.info(f"STT client initialized with base URL: {self.base_url}")
        logger.debug(f"STT client configuration: timeout={timeout}s")
    
    def create_wav_file(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bytes:
        """
        Convert audio data to WAV format in memory.
        
        Args:
            audio_data: Audio samples as numpy array (float32 or int16)
            sample_rate: Sample rate in Hz
            
        Returns:
            WAV file data as bytes
        """
        # Ensure audio is int16
        logger.debug(f"Converting audio: dtype={audio_data.dtype}, shape={audio_data.shape}")
        if audio_data.dtype == np.float32:
            # Convert float32 to int16 (assuming normalized range -1.0 to 1.0)
            audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            logger.debug("Converted float32 audio to int16")
        elif audio_data.dtype == np.int16:
            audio_int16 = audio_data
        else:
            audio_int16 = audio_data.astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    def transcribe(self,
                   audio_data: np.ndarray,
                   language: Optional[str] = None,
                   task: str = "transcribe",
                   webhook_url: Optional[str] = None,
                   topic: str = "general",
                   wake_word_time: Optional[datetime] = None,
                   recording_end_time: Optional[datetime] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Send audio to STT service for transcription.

        Args:
            audio_data: Audio samples as numpy array
            language: Language code (e.g., "en", "es", "fr")
            task: "transcribe" or "translate"
            webhook_url: Override base URL with specific webhook URL
            topic: Topic ID for ORAC Core routing (default: "general")
            wake_word_time: Timestamp when wake word was detected (for end-to-end timing)
            recording_end_time: Timestamp when recording ended (for end-to-end timing)

        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Convert audio to WAV format
            wav_data = self.create_wav_file(audio_data)
            
            # Calculate audio duration
            duration = len(audio_data) / 16000
            wav_size = len(wav_data)
            logger.info(f"📤 Sending {duration:.2f}s of audio to STT service ({wav_size/1024:.1f} KB)")
            
            # Prepare request
            files = {'file': ('audio.wav', wav_data, 'audio/wav')}
            data = {}
            if language:
                data['language'] = language
            data['task'] = task
            
            logger.debug(f"Request parameters: language={language}, task={task}")
            
            # Make request - use webhook_url if provided, otherwise use base_url
            # Include topic in the URL path
            if webhook_url:
                # Extract just the base URL from webhook URL (remove any path)
                from urllib.parse import urlparse
                parsed = urlparse(webhook_url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                url = f"{base_url}/stt/v1/stream/{topic}"
                logger.debug(f"Using webhook URL for STT: {url} (from {webhook_url})")
            else:
                url = f"{self.base_url}/stt/v1/stream/{topic}"
            logger.debug(f"Making POST request to: {url} with topic: {topic}")

            # Build headers with timing information for end-to-end tracking
            headers = {}
            if wake_word_time:
                headers['X-Wake-Word-Time'] = wake_word_time.isoformat()
                logger.debug(f"Adding wake word time header: {wake_word_time.isoformat()}")
            if recording_end_time:
                headers['X-Recording-End-Time'] = recording_end_time.isoformat()
                logger.debug(f"Adding recording end time header: {recording_end_time.isoformat()}")

            start_time = datetime.now()
            response = self.session.post(
                url,
                files=files,
                data=data,
                headers=headers if headers else None,
                timeout=self.timeout
            )
            request_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Request completed in {request_time:.3f}s, status={response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                transcription = result.get('text', '')
                logger.info(f"✅ STT transcription successful: '{transcription}'")
                logger.debug(f"Full STT response: {result}")
                logger.info(f"📝 TRANSCRIPTION LOG: '{transcription}'")
                return True, result
            else:
                error_msg = f"STT request failed with status {response.status_code}"
                try:
                    error_detail = response.json().get('detail', '')
                    if error_detail:
                        error_msg += f": {error_detail}"
                except:
                    error_msg += f": {response.text}"
                
                logger.error(error_msg)
                return False, {'error': error_msg}
                
        except requests.exceptions.Timeout:
            error_msg = "STT request timed out"
            logger.error(error_msg)
            return False, {'error': error_msg}
            
        except requests.exceptions.ConnectionError:
            error_msg = f"Failed to connect to STT service at {self.base_url}"
            logger.error(error_msg)
            return False, {'error': error_msg}
            
        except Exception as e:
            error_msg = f"Unexpected error during STT request: {str(e)}"
            logger.error(error_msg)
            return False, {'error': error_msg}
    
    def health_check(self, webhook_url: Optional[str] = None) -> bool:
        """
        Check if STT service is healthy.
        
        Args:
            webhook_url: Optional webhook URL to check health for
            
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            if webhook_url:
                # Extract base URL from webhook URL
                from urllib.parse import urlparse
                parsed = urlparse(webhook_url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
                health_url = f"{base_url}/stt/v1/health"
                logger.debug(f"Checking STT health at webhook URL: {health_url}")
            else:
                health_url = f"{self.base_url}/stt/v1/health"
                logger.debug(f"Checking STT health at: {health_url}")
            
            response = self.session.get(
                health_url,
                timeout=5
            )
            
            if response.status_code == 200:
                health = response.json()
                logger.debug(f"Health check response: {health}")
                # Consider service healthy if it responds, even if still initializing
                status = health.get('status', 'unknown')
                if status in ['healthy', 'ready', 'initializing']:
                    logger.info(f"✅ STT service accessible: status={status}, model={health.get('model_name')}, backend={health.get('backend')}, device={health.get('device')}")
                    return True
                else:
                    logger.warning(f"⚠️ STT service unhealthy: {health}")
                    return False
            else:
                logger.warning(f"STT health check failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"STT health check failed: {e}")
            return False
    
    def preload_model(self) -> bool:
        """
        Request STT service to preload its model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.post(
                f"{self.base_url}/stt/v1/preload",
                timeout=60  # Model loading can take time
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"STT model preload: {result.get('message', 'Success')}")
                return True
            else:
                logger.error(f"STT model preload failed with status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"STT model preload failed: {e}")
            return False
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()