"""
STT (Speech-to-Text) client for sending audio to transcription service.
"""

import io
import wave
import logging
import requests
import numpy as np
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


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
        if audio_data.dtype == np.float32:
            # Convert float32 to int16 (assuming range -1.0 to 1.0)
            audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
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
                   task: str = "transcribe") -> Tuple[bool, Dict[str, Any]]:
        """
        Send audio to STT service for transcription.
        
        Args:
            audio_data: Audio samples as numpy array
            language: Language code (e.g., "en", "es", "fr")
            task: "transcribe" or "translate"
            
        Returns:
            Tuple of (success, result_dict)
        """
        try:
            # Convert audio to WAV format
            wav_data = self.create_wav_file(audio_data)
            
            # Calculate audio duration
            duration = len(audio_data) / 16000
            logger.info(f"Sending {duration:.2f}s of audio to STT service")
            
            # Prepare request
            files = {'file': ('audio.wav', wav_data, 'audio/wav')}
            data = {}
            if language:
                data['language'] = language
            data['task'] = task
            
            # Make request
            url = f"{self.base_url}/stt/v1/stream"
            response = self.session.post(
                url,
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"STT transcription successful: '{result.get('text', '')}'")
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
    
    def health_check(self) -> bool:
        """
        Check if STT service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/stt/v1/health",
                timeout=5
            )
            
            if response.status_code == 200:
                health = response.json()
                if health.get('status') == 'healthy' and health.get('model_loaded'):
                    logger.info(f"STT service healthy: model={health.get('model_name')}")
                    return True
                else:
                    logger.warning(f"STT service unhealthy: {health}")
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