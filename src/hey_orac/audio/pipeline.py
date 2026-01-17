"""
Audio Pipeline for Hey ORAC.

Clean Capture -> Clean -> Distribute architecture for audio processing.

This module provides a unified audio pipeline that:
1. CAPTURE: Uses AudioManager for mic detection and stream creation
2. CLEAN: Placeholder for preprocessing (AGC, compression) - Phase 1/2
3. DISTRIBUTE: Delivers audio to multiple consumers (OWW, ring buffer, etc.)

Architecture:
    USB Mic
        |
        v
    [AudioManager] -- finds mic, creates PyAudio stream
        |
        v
    [AudioPipeline] -- reads audio in thread, applies processing
        |
        v
    [Consumers] -- OWW, RingBuffer, SpeechRecorder, etc.

Key Design Decisions:
- Uses blocking stream.read() in dedicated thread (proven reliable)
- Supports multiple consumers with individual queues
- Preprocessing is optional and controlled by config
- Maintains format compatibility: bytes -> float32 for consumers

Format Notes:
- PyAudio produces: bytes (int16)
- OpenWakeWord needs: UN-normalized float32 (int16 values as float)
- Preprocessing uses: normalized float32 (-1.0 to 1.0)
- Ring buffer stores: int16 for STT pre-roll
"""

import threading
import queue
import logging
import time
from typing import Optional, Dict, Any, Callable
import numpy as np

from .utils import AudioManager, AudioDevice
from .preprocessor import AudioPreprocessor, AudioPreprocessorConfig
from hey_orac import constants

logger = logging.getLogger(__name__)


class AudioPipeline:
    """
    Unified audio pipeline with Capture -> Clean -> Distribute pattern.

    This class encapsulates:
    - Audio stream creation (via AudioManager)
    - Audio reading in a dedicated thread
    - Audio distribution to multiple consumers
    - (Future) Audio preprocessing

    Usage:
        pipeline = AudioPipeline(audio_config)

        # Start the pipeline
        if not pipeline.start():
            raise RuntimeError("Failed to start audio pipeline")

        # Register consumers
        oww_queue = pipeline.register_consumer("openwakeword")
        buffer_queue = pipeline.register_consumer("ring_buffer")

        # Read audio in your loop
        try:
            data = oww_queue.get(timeout=2.0)
            # process data...
        except queue.Empty:
            # handle timeout
            pass

        # Cleanup
        pipeline.stop()
    """

    def __init__(
        self,
        sample_rate: int = constants.SAMPLE_RATE,
        channels: int = constants.CHANNELS_STEREO,
        chunk_size: int = constants.CHUNK_SIZE,
        device_index: Optional[int] = None,
        queue_maxsize: int = constants.AUDIO_READER_QUEUE_MAXSIZE,
        enable_preprocessing: bool = False,  # Phase 1/2 - disabled by default
        preprocessor_config: Optional[AudioPreprocessorConfig] = None,
    ):
        """
        Initialize audio pipeline.

        Args:
            sample_rate: Sample rate in Hz (default 16000)
            channels: Number of audio channels (default 2 for stereo)
            chunk_size: Samples per chunk (default 1280 = 80ms)
            device_index: Specific device index, or None for auto-detect USB mic
            queue_maxsize: Maximum chunks buffered per consumer
            enable_preprocessing: Enable audio preprocessing (AGC, compression, limiting)
            preprocessor_config: Configuration for audio preprocessing
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index
        self.queue_maxsize = queue_maxsize
        self.enable_preprocessing = enable_preprocessing

        # AudioManager for device detection and stream creation
        self._audio_manager: Optional[AudioManager] = None
        self._stream: Optional[Any] = None
        self._device: Optional[AudioDevice] = None

        # Consumer management
        self._consumers: Dict[str, queue.Queue] = {}
        self._consumer_lock = threading.Lock()

        # Reader thread
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False

        # Metrics
        self._last_read_time = time.time()
        self._read_count = 0
        self._error_count = 0
        self._restart_count = 0

        # Preprocessing
        self._preprocessor_config = preprocessor_config or AudioPreprocessorConfig(sample_rate=sample_rate)
        self._preprocessor: Optional[AudioPreprocessor] = None
        if self.enable_preprocessing:
            self._preprocessor = AudioPreprocessor(self._preprocessor_config)
            logger.info(f"AudioPreprocessor enabled: AGC={self._preprocessor_config.enable_agc}, "
                       f"compression={self._preprocessor_config.enable_compression}")

        logger.info(f"AudioPipeline initialized: rate={sample_rate}, channels={channels}, "
                   f"chunk={chunk_size}, preprocessing={enable_preprocessing}")

    def start(self) -> bool:
        """
        Start the audio pipeline.

        This method:
        1. Initializes AudioManager
        2. Finds USB microphone (if device_index not specified)
        3. Opens audio stream
        4. Starts reader thread

        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            logger.warning("AudioPipeline already running")
            return True

        try:
            # Initialize AudioManager
            logger.info("Initializing AudioManager...")
            self._audio_manager = AudioManager()

            # Find device
            if self.device_index is not None:
                # Use specified device index
                logger.info(f"Using specified device index: {self.device_index}")
                # Still get device info for logging
                devices = self._audio_manager.list_input_devices()
                for d in devices:
                    if d.index == self.device_index:
                        self._device = d
                        break
            else:
                # Auto-detect USB microphone
                logger.info("Auto-detecting USB microphone...")
                self._device = self._audio_manager.find_usb_microphone()
                if self._device is None:
                    logger.error("No USB microphone found")
                    return False
                self.device_index = self._device.index

            logger.info(f"Using audio device: {self._device.name if self._device else 'index ' + str(self.device_index)}")

            # Open audio stream
            logger.info("Opening audio stream...")
            self._stream = self._audio_manager.start_stream(
                device_index=self.device_index,
                sample_rate=self.sample_rate,
                channels=self.channels,
                chunk_size=self.chunk_size
            )

            if self._stream is None:
                logger.error("Failed to open audio stream")
                return False

            # Test audio stream
            logger.info("Testing audio stream...")
            try:
                test_data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                logger.info(f"Audio stream test successful, read {len(test_data)} bytes")
            except Exception as e:
                logger.error(f"Audio stream test failed: {e}")
                return False

            # Start reader thread
            self._running = True
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()

            logger.info(f"AudioPipeline started (restart #{self._restart_count})")
            return True

        except Exception as e:
            logger.error(f"Failed to start AudioPipeline: {e}")
            self._cleanup()
            return False

    def stop(self) -> None:
        """Stop the audio pipeline and cleanup resources."""
        if not self._running:
            return

        logger.info("Stopping AudioPipeline...")
        self._running = False

        # Clear all consumer queues to unblock pending puts
        with self._consumer_lock:
            for name, q in self._consumers.items():
                try:
                    while not q.empty():
                        q.get_nowait()
                except queue.Empty:
                    pass

        # Wait for reader thread
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
            if self._reader_thread.is_alive():
                logger.warning("Reader thread did not stop gracefully")

        self._cleanup()
        logger.info("AudioPipeline stopped")

    def _cleanup(self) -> None:
        """Clean up audio resources."""
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error closing stream: {e}")
            self._stream = None

        if self._audio_manager:
            try:
                self._audio_manager.close()
            except Exception as e:
                logger.warning(f"Error closing AudioManager: {e}")
            self._audio_manager = None

    def restart(self) -> bool:
        """Restart the audio pipeline."""
        logger.info("Restarting AudioPipeline...")
        self.stop()
        time.sleep(0.1)  # Brief pause
        self._restart_count += 1
        return self.start()

    def register_consumer(self, name: str) -> queue.Queue:
        """
        Register a consumer and get its dedicated queue.

        Each consumer gets its own queue to receive audio data.
        This allows independent processing at different rates.

        Args:
            name: Unique name for the consumer (e.g., "openwakeword", "ring_buffer")

        Returns:
            Queue object for this consumer
        """
        with self._consumer_lock:
            if name in self._consumers:
                logger.warning(f"Consumer '{name}' already registered, returning existing queue")
                return self._consumers[name]

            consumer_queue = queue.Queue(maxsize=self.queue_maxsize)
            self._consumers[name] = consumer_queue
            logger.info(f"Registered consumer '{name}' (total: {len(self._consumers)})")
            return consumer_queue

    def unregister_consumer(self, name: str) -> None:
        """
        Unregister a consumer.

        Args:
            name: Name of the consumer to unregister
        """
        with self._consumer_lock:
            if name not in self._consumers:
                logger.warning(f"Consumer '{name}' not found")
                return

            # Clear queue
            consumer_queue = self._consumers[name]
            try:
                while not consumer_queue.empty():
                    consumer_queue.get_nowait()
            except queue.Empty:
                pass

            del self._consumers[name]
            logger.info(f"Unregistered consumer '{name}' (remaining: {len(self._consumers)})")

    def _reader_loop(self) -> None:
        """Main loop that reads audio in a dedicated thread."""
        logger.debug("Audio reader loop started")

        while self._running:
            try:
                # Read audio from stream (blocking call)
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)

                if data is None or len(data) == 0:
                    logger.warning("Empty audio data from stream")
                    time.sleep(0.01)
                    continue

                # Update metrics
                self._last_read_time = time.time()
                self._read_count += 1

                # === CLEAN PHASE ===
                # Phase 0: Pass-through (no preprocessing)
                # Phase 1/2: Will add preprocessing here
                processed_data = self._process_audio(data)

                # === DISTRIBUTE PHASE ===
                self._distribute_audio(processed_data)

            except Exception as e:
                self._error_count += 1
                logger.error(f"Error in audio reader: {e}")
                if self._error_count > 10:
                    logger.error("Too many errors, stopping reader")
                    self._running = False
                    break
                time.sleep(0.01)

        logger.debug("Audio reader loop ended")

    def _process_audio(self, data: bytes) -> bytes:
        """
        Process audio data through the pipeline.

        When preprocessing is disabled: Pass-through (no changes)
        When preprocessing is enabled: Apply AGC, compression, limiting

        Format conversion flow (when preprocessing enabled):
        1. bytes (stereo int16) -> mono int16 -> normalized float32 (-1.0 to 1.0)
        2. Apply preprocessing (AGC, compression, limiting)
        3. normalized float32 -> mono int16 -> stereo int16 (duplicated) -> bytes

        Note: OpenWakeWord conversion happens in the consumer, not here.
        We maintain stereo int16 bytes as the distribution format for compatibility.

        Args:
            data: Raw audio bytes from stream (stereo int16)

        Returns:
            Processed audio bytes (stereo int16)
        """
        # Pass-through when preprocessing is disabled
        if not self.enable_preprocessing or self._preprocessor is None:
            return data

        try:
            # Step 1: Convert bytes to numpy array (stereo int16)
            audio_stereo = np.frombuffer(data, dtype=np.int16)

            # Step 2: Convert stereo to mono by averaging channels
            # Stereo data is interleaved: [L0, R0, L1, R1, ...]
            if len(audio_stereo) > self.chunk_size:
                stereo_reshaped = audio_stereo.reshape(-1, 2)
                audio_mono = np.mean(stereo_reshaped, axis=1).astype(np.float32)
            else:
                # Already mono
                audio_mono = audio_stereo.astype(np.float32)

            # Step 3: Normalize to -1.0 to 1.0 range (required by AudioPreprocessor)
            audio_normalized = audio_mono / 32768.0

            # Step 4: Apply preprocessing (AGC, compression, limiting)
            audio_processed = self._preprocessor.process(audio_normalized)

            # Step 5: Convert back to int16 range
            audio_int16 = (audio_processed * 32768.0).clip(-32768, 32767).astype(np.int16)

            # Step 6: Convert mono back to stereo (duplicate to both channels)
            # This maintains compatibility with consumers expecting stereo
            audio_stereo_out = np.column_stack((audio_int16, audio_int16)).flatten()

            # Step 7: Convert back to bytes
            return audio_stereo_out.tobytes()

        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            # Fall back to pass-through on error
            return data

    def _distribute_audio(self, data: bytes) -> None:
        """
        Distribute audio to all registered consumers.

        Args:
            data: Audio bytes to distribute
        """
        with self._consumer_lock:
            for name, consumer_queue in self._consumers.items():
                try:
                    consumer_queue.put(data, timeout=0.01)
                except queue.Full:
                    # Drop oldest and retry
                    try:
                        consumer_queue.get_nowait()
                        consumer_queue.put(data, timeout=0.01)
                    except (queue.Empty, queue.Full):
                        logger.debug(f"Failed to queue audio for '{name}'")

    def is_healthy(self, max_age: float = 3.0) -> bool:
        """
        Check if the pipeline is healthy.

        Args:
            max_age: Maximum age of last read in seconds

        Returns:
            True if healthy, False otherwise
        """
        if not self._running:
            return False

        if not self._reader_thread or not self._reader_thread.is_alive():
            return False

        age = time.time() - self._last_read_time
        if age > max_age:
            logger.warning(f"Pipeline unhealthy: last read {age:.1f}s ago")
            return False

        return True

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        stats = {
            'running': self._running,
            'thread_alive': self._reader_thread.is_alive() if self._reader_thread else False,
            'last_read_age': time.time() - self._last_read_time,
            'read_count': self._read_count,
            'error_count': self._error_count,
            'restart_count': self._restart_count,
            'consumer_count': len(self._consumers),
            'consumers': list(self._consumers.keys()),
            'preprocessing_enabled': self.enable_preprocessing,
        }

        # Add preprocessing metrics if enabled
        if self._preprocessor is not None:
            stats['preprocessing_metrics'] = self._preprocessor.get_metrics()

        return stats

    def get_preprocessing_metrics(self) -> Optional[dict]:
        """
        Get audio preprocessing metrics (AGC gain, peak level, etc.).

        Returns:
            Dict of preprocessing metrics, or None if preprocessing disabled
        """
        if self._preprocessor is None:
            return None
        return self._preprocessor.get_metrics()

    def get_device_info(self) -> Optional[AudioDevice]:
        """Get information about the audio device being used."""
        return self._device

    @property
    def stream(self) -> Optional[Any]:
        """Get the underlying PyAudio stream (for legacy compatibility)."""
        return self._stream

    # === AUTO-CALIBRATION ===

    def calibrate_noise_floor(self, duration: float = 3.0, callback: Optional[Callable[[dict], None]] = None) -> dict:
        """
        Measure ambient noise floor and auto-calibrate thresholds.

        Measures RMS over the specified duration and sets:
        - silence_threshold = noise_floor × 1.5
        - noise_gate_threshold = noise_floor × 1.2

        Args:
            duration: Measurement duration in seconds (default 3.0)
            callback: Optional callback with calibration results

        Returns:
            Dict with calibration results:
            {
                'noise_floor_rms': float,
                'recommended_silence_threshold': float,
                'recommended_noise_gate_threshold': float,
                'samples_measured': int,
                'peak_level': float,
                'calibration_time': float
            }
        """
        if not self._running:
            logger.warning("Cannot calibrate - pipeline not running")
            return {'error': 'Pipeline not running'}

        logger.info(f"Starting noise floor calibration ({duration}s)...")

        # Register temporary consumer for calibration
        cal_queue = self.register_consumer("_calibration")

        rms_samples = []
        peak_level = 0.0
        start_time = time.time()
        samples_measured = 0

        try:
            while time.time() - start_time < duration:
                try:
                    data = cal_queue.get(timeout=0.5)
                    if data is None or len(data) == 0:
                        continue

                    # Convert to numpy and calculate RMS
                    audio_array = np.frombuffer(data, dtype=np.int16)

                    # Handle stereo to mono
                    if len(audio_array) > self.chunk_size:
                        stereo_data = audio_array.reshape(-1, 2)
                        audio_mono = np.mean(stereo_data, axis=1).astype(np.float32)
                    else:
                        audio_mono = audio_array.astype(np.float32)

                    # Normalize for RMS calculation
                    audio_normalized = audio_mono / 32768.0

                    # Calculate RMS
                    rms = float(np.sqrt(np.mean(audio_normalized ** 2)))
                    rms_samples.append(rms)

                    # Track peak
                    current_peak = float(np.max(np.abs(audio_normalized)))
                    peak_level = max(peak_level, current_peak)

                    samples_measured += 1

                except queue.Empty:
                    continue

        finally:
            # Unregister calibration consumer
            self.unregister_consumer("_calibration")

        if not rms_samples:
            logger.error("No audio samples collected during calibration")
            return {'error': 'No samples collected'}

        # Calculate noise floor (use median to ignore outliers)
        noise_floor_rms = float(np.median(rms_samples))
        avg_rms = float(np.mean(rms_samples))
        std_rms = float(np.std(rms_samples))

        # Calculate recommended thresholds
        # Silence threshold: 1.5x noise floor (for STT endpointing)
        # Noise gate: 1.2x noise floor (for gating)
        recommended_silence_threshold = noise_floor_rms * 1.5
        recommended_noise_gate_threshold = noise_floor_rms * 1.2

        calibration_time = time.time() - start_time

        result = {
            'noise_floor_rms': noise_floor_rms,
            'avg_rms': avg_rms,
            'std_rms': std_rms,
            'recommended_silence_threshold': recommended_silence_threshold,
            'recommended_noise_gate_threshold': recommended_noise_gate_threshold,
            'samples_measured': samples_measured,
            'peak_level': peak_level,
            'calibration_time': calibration_time,
            'timestamp': time.time()
        }

        logger.info(f"Calibration complete: noise_floor={noise_floor_rms:.4f}, "
                   f"silence_threshold={recommended_silence_threshold:.4f}, "
                   f"peak={peak_level:.4f}")

        # Store calibration results
        self._last_calibration = result

        # Call callback if provided
        if callback:
            callback(result)

        return result

    def get_last_calibration(self) -> Optional[dict]:
        """Get results from the last calibration run."""
        return getattr(self, '_last_calibration', None)
