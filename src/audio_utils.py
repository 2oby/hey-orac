#!/usr/bin/env python3
"""
Audio utilities for Hey Orac wake-word detection service
"""

import pyaudio
import wave
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AudioDevice:
    """Audio device information."""
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: float
    host_api: str
    is_usb: bool = False

class AudioManager:
    """Manages audio devices and recording."""
    
    def __init__(self):
        self.pyaudio = pyaudio.PyAudio()
        self.current_stream = None
        
    def __del__(self):
        """Cleanup PyAudio on destruction."""
        if hasattr(self, 'pyaudio'):
            self.pyaudio.terminate()
    
    def list_input_devices(self) -> List[AudioDevice]:
        """List all available audio input devices."""
        devices = []
        
        for i in range(self.pyaudio.get_device_count()):
            device_info = self.pyaudio.get_device_info_by_index(i)
            
            if device_info['maxInputChannels'] > 0:  # Input device
                device_name = device_info['name'].lower()
                is_usb = ('usb' in device_name or 
                         'hw:' in device_info['name'] or  # Hardware device
                         'card' in device_name)  # ALSA card device
                
                device = AudioDevice(
                    index=i,
                    name=device_info['name'],
                    max_input_channels=device_info['maxInputChannels'],
                    default_sample_rate=device_info['defaultSampleRate'],
                    host_api=device_info['hostApi'],
                    is_usb=is_usb
                )
                devices.append(device)
        
        return devices
    
    def find_usb_microphone(self) -> Optional[AudioDevice]:
        """Find the first USB microphone device."""
        devices = self.list_input_devices()
        
        for device in devices:
            if device.is_usb:
                logger.info(f"Found USB microphone: {device.name} (index {device.index})")
                return device
        
        logger.warning("No USB microphone found")
        return None
    
    def get_default_input_device(self) -> Optional[AudioDevice]:
        """Get the default input device."""
        try:
            default_index = self.pyaudio.get_default_input_device_info()['index']
            device_info = self.pyaudio.get_device_info_by_index(default_index)
            
            device_name = device_info['name'].lower()
            is_usb = ('usb' in device_name or 
                     'hw:' in device_info['name'] or  # Hardware device
                     'card' in device_name)  # ALSA card device
            
            return AudioDevice(
                index=default_index,
                name=device_info['name'],
                max_input_channels=device_info['maxInputChannels'],
                default_sample_rate=device_info['defaultSampleRate'],
                host_api=device_info['hostApi'],
                is_usb=is_usb
            )
        except Exception as e:
            logger.error(f"Error getting default input device: {e}")
            return None
    
    def start_recording(self, device_index: int, sample_rate: int = 16000, 
                       channels: int = 1, chunk_size: int = 512) -> bool:
        """Start recording from specified device."""
        try:
            if self.current_stream:
                self.stop_recording()
            
            self.current_stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk_size
            )
            
            logger.info(f"Started recording from device {device_index}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting recording from device {device_index}: {e}")
            return False
    
    def start_stream(self, device_index: int, sample_rate: int = 16000, 
                    channels: int = 1, chunk_size: int = 512):
        """Start audio stream and return the stream object for real-time processing."""
        try:
            stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk_size
            )
            
            logger.info(f"Started audio stream from device {device_index}")
            return stream
            
        except Exception as e:
            logger.error(f"Error starting audio stream from device {device_index}: {e}")
            return None
    
    def read_audio_chunk(self) -> Optional[bytes]:
        """Read one chunk of audio data."""
        if not self.current_stream:
            return None
        
        try:
            return self.current_stream.read(512, exception_on_overflow=False)
        except Exception as e:
            logger.error(f"Error reading audio chunk: {e}")
            return None
    
    def stop_recording(self):
        """Stop recording and close stream."""
        if self.current_stream:
            self.current_stream.stop_stream()
            self.current_stream.close()
            self.current_stream = None
            logger.info("Stopped recording")
    
    def record_to_file(self, device_index: int, duration: float, 
                      output_file: str, sample_rate: int = 16000, 
                      channels: int = 1) -> bool:
        """Record audio to a WAV file."""
        try:
            # Start recording
            if not self.start_recording(device_index, sample_rate, channels):
                return False
            
            frames = []
            chunks_needed = int(sample_rate / 512 * duration)
            
            logger.info(f"Recording {duration}s to {output_file}")
            
            for i in range(chunks_needed):
                chunk = self.read_audio_chunk()
                if chunk:
                    frames.append(chunk)
                else:
                    logger.error("Failed to read audio chunk")
                    return False
            
            # Stop recording
            self.stop_recording()
            
            # Save to file
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(self.pyaudio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(sample_rate)
                wf.writeframes(b''.join(frames))
            
            logger.info(f"Saved recording to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording to file: {e}")
            self.stop_recording()
            return False 