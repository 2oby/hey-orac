#!/usr/bin/env python3
"""
Audio utilities for Hey Orac wake-word detection service
"""

import pyaudio
import wave
import logging
import subprocess
import os
import sys
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
    # Enhanced device info
    device_info: Dict = None

class AudioManager:
    """Manages audio devices and recording."""
    
    def __init__(self):
        logger.info("🔍 Initializing AudioManager with enhanced device detection...")
        self.pyaudio = pyaudio.PyAudio()
        self.current_stream = None
        
        # Enhanced system diagnostics
        self._run_system_audio_diagnostics()
        
    def __del__(self):
        """Cleanup PyAudio on destruction."""
        if hasattr(self, 'pyaudio'):
            self.pyaudio.terminate()
    
    def _run_system_audio_diagnostics(self):
        """Run comprehensive system audio diagnostics."""
        logger.info("🔧 Running system audio diagnostics...")
        
        # Check ALSA system
        try:
            logger.info("📋 ALSA System Information:")
            result = subprocess.run(['cat', '/proc/asound/version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"   ALSA Version: {result.stdout.strip()}")
            else:
                logger.warning("   Could not read ALSA version")
        except Exception as e:
            logger.warning(f"   Error reading ALSA version: {e}")
        
        # Check ALSA cards
        try:
            logger.info("🎵 ALSA Cards:")
            result = subprocess.run(['cat', '/proc/asound/cards'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"   {line}")
            else:
                logger.warning("   Could not read ALSA cards")
        except Exception as e:
            logger.warning(f"   Error reading ALSA cards: {e}")
        
        # Check USB devices
        try:
            logger.info("🔌 USB Audio Devices:")
            result = subprocess.run(['lsusb'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if 'audio' in line.lower() or 'mic' in line.lower() or 'sh-04' in line.lower():
                        logger.info(f"   {line}")
            else:
                logger.warning("   Could not list USB devices")
        except Exception as e:
            logger.warning(f"   Error listing USB devices: {e}")
        
        # Check ALSA devices
        try:
            logger.info("🎤 ALSA Devices:")
            result = subprocess.run(['arecord', '-l'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"   {line}")
            else:
                logger.warning("   Could not list ALSA devices")
        except Exception as e:
            logger.warning(f"   Error listing ALSA devices: {e}")
        
        # Check environment variables
        logger.info("🌍 Audio Environment Variables:")
        audio_env_vars = ['ALSA_CARD', 'ALSA_DEVICE', 'AUDIODEV', 'AUDIO_DEVICE']
        for var in audio_env_vars:
            value = os.environ.get(var)
            if value:
                logger.info(f"   {var}: {value}")
            else:
                logger.info(f"   {var}: Not set")
        
        # Check ALSA configuration
        logger.info("⚙️ ALSA Configuration:")
        alsa_configs = ['/etc/asound.conf', '/etc/asoundrc', '~/.asoundrc', '/app/.asoundrc']
        for config_path in alsa_configs:
            expanded_path = os.path.expanduser(config_path)
            if os.path.exists(expanded_path):
                logger.info(f"   Found ALSA config: {expanded_path}")
                try:
                    with open(expanded_path, 'r') as f:
                        content = f.read()
                        logger.info(f"   Config content:\n{content}")
                except Exception as e:
                    logger.warning(f"   Error reading config {expanded_path}: {e}")
            else:
                logger.info(f"   No ALSA config found at: {expanded_path}")
    
    def list_input_devices(self) -> List[AudioDevice]:
        """List all available audio input devices with enhanced logging."""
        devices = []
        
        total_devices = self.pyaudio.get_device_count()
        logger.info(f"🔍 PyAudio Device Detection:")
        logger.info(f"   Total devices found: {total_devices}")
        
        if total_devices == 0:
            logger.error("❌ No audio devices detected by PyAudio!")
            logger.error("   This may indicate:")
            logger.error("   - ALSA not properly configured")
            logger.error("   - Audio devices not accessible in container")
            logger.error("   - Missing audio permissions")
            logger.error("   - PyAudio not compiled with ALSA support")
            return devices
        
        for i in range(total_devices):
            try:
                device_info = self.pyaudio.get_device_info_by_index(i)
                logger.info(f"📋 Device {i} Details:")
                logger.info(f"   Name: {device_info['name']}")
                logger.info(f"   Host API: {device_info['hostApi']}")
                logger.info(f"   Max Input Channels: {device_info['maxInputChannels']}")
                logger.info(f"   Max Output Channels: {device_info['maxOutputChannels']}")
                logger.info(f"   Default Sample Rate: {device_info['defaultSampleRate']}")
                logger.info(f"   Default Low Input Latency: {device_info.get('defaultLowInputLatency', 'N/A')}")
                logger.info(f"   Default High Input Latency: {device_info.get('defaultHighInputLatency', 'N/A')}")
                
                # Enhanced USB detection
                device_name_lower = device_info['name'].lower()
                is_usb = any(keyword in device_name_lower for keyword in [
                    'usb', 'sh-04', 'mv', 'blue', 'wireless', 'bluetooth'
                ])
                
                # Check for hardware device indicators
                if not is_usb:
                    is_usb = any(indicator in device_info['name'] for indicator in [
                        'hw:', 'card', 'device', 'alsa'
                    ])
                
                logger.info(f"   Detected as USB: {is_usb}")
                logger.info(f"   Raw device info: {device_info}")
                
                if device_info['maxInputChannels'] > 0:  # Input device
                    device = AudioDevice(
                        index=i,
                        name=device_info['name'],
                        max_input_channels=device_info['maxInputChannels'],
                        default_sample_rate=device_info['defaultSampleRate'],
                        host_api=device_info['hostApi'],
                        is_usb=is_usb,
                        device_info=device_info
                    )
                    devices.append(device)
                    logger.info(f"   ✅ Added as input device")
                else:
                    logger.info(f"   ⏭️ Skipped (no input channels)")
                    
            except Exception as e:
                logger.error(f"   ❌ Error getting info for device {i}: {e}")
        
        logger.info(f"🎯 Found {len(devices)} input devices total")
        
        # Additional diagnostics for USB devices
        usb_devices = [d for d in devices if d.is_usb]
        logger.info(f"🔌 USB input devices found: {len(usb_devices)}")
        for usb_device in usb_devices:
            logger.info(f"   USB Device: {usb_device.name} (index {usb_device.index})")
        
        return devices
    
    def find_usb_microphone(self) -> Optional[AudioDevice]:
        """Find the first USB microphone device with enhanced detection."""
        logger.info("🎤 Searching for USB microphone...")
        devices = self.list_input_devices()
        
        if not devices:
            logger.error("❌ No input devices found!")
            return None
        
        # First, try to find explicit USB devices
        for device in devices:
            if device.is_usb:
                logger.info(f"✅ Found USB microphone: {device.name} (index {device.index})")
                logger.info(f"   Sample rate: {device.default_sample_rate}")
                logger.info(f"   Channels: {device.max_input_channels}")
                logger.info(f"   Host API: {device.host_api}")
                return device
        
        # If no explicit USB devices, try to find hardware devices
        logger.warning("⚠️ No explicit USB devices found, checking for hardware devices...")
        for device in devices:
            device_name_lower = device.name.lower()
            if any(hw_indicator in device_name_lower for hw_indicator in ['hw:', 'card', 'device']):
                logger.info(f"🔧 Found hardware device: {device.name} (index {device.index})")
                logger.info(f"   Sample rate: {device.default_sample_rate}")
                logger.info(f"   Channels: {device.max_input_channels}")
                logger.info(f"   Host API: {device.host_api}")
                return device
        
        # Last resort: use default device
        logger.warning("⚠️ No USB or hardware devices found, trying default device...")
        default_device = self.get_default_input_device()
        if default_device:
            logger.info(f"🔧 Using default device: {default_device.name} (index {default_device.index})")
            return default_device
        
        logger.error("❌ No suitable microphone device found!")
        return None
    
    def get_default_input_device(self) -> Optional[AudioDevice]:
        """Get the default input device with enhanced error handling."""
        try:
            logger.info("🎯 Getting default input device...")
            default_info = self.pyaudio.get_default_input_device_info()
            default_index = default_info['index']
            
            logger.info(f"   Default device index: {default_index}")
            device_info = self.pyaudio.get_device_info_by_index(default_index)
            
            device_name = device_info['name'].lower()
            original_name = device_info['name']
            
            # Enhanced USB detection for default device
            is_usb = any(keyword in device_name for keyword in [
                'usb', 'hw:', 'card', 'device', 'sh-04', 'mv', 'alsa'
            ])
            
            logger.info(f"   Default device name: {original_name}")
            logger.info(f"   Detected as USB: {is_usb}")
            
            return AudioDevice(
                index=default_index,
                name=device_info['name'],
                max_input_channels=device_info['maxInputChannels'],
                default_sample_rate=device_info['defaultSampleRate'],
                host_api=device_info['hostApi'],
                is_usb=is_usb,
                device_info=device_info
            )
        except Exception as e:
            logger.error(f"❌ Error getting default input device: {e}")
            logger.error(f"   This may indicate no default device is configured")
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
        """Stop recording and close stream with proper device release."""
        if self.current_stream:
            try:
                self.current_stream.stop_stream()
                self.current_stream.close()
                self.current_stream = None
                logger.info("Device released and stream closed")
            except Exception as e:
                logger.error(f"Error closing stream: {e}")
                self.current_stream = None
        else:
            logger.info("No active stream to stop")
    
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