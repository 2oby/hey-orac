"""
Metrics collector for TFLite performance monitoring on Raspberry Pi.
Provides lightweight metrics collection optimized for resource-constrained environments.
"""

import time
import psutil
import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class TFLiteMetrics:
    """TFLite-specific performance metrics."""
    inference_count: int = 0
    total_inference_time: float = 0.0
    min_inference_time: float = float('inf')
    max_inference_time: float = 0.0
    avg_inference_time: float = 0.0
    model_load_time: float = 0.0
    model_load_count: int = 0
    
    # Recent inference times (sliding window)
    recent_inference_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Model-specific metrics
    model_sizes: Dict[str, int] = field(default_factory=dict)
    model_formats: Dict[str, str] = field(default_factory=dict)
    
    def update_inference_time(self, inference_time: float):
        """Update inference time metrics."""
        self.inference_count += 1
        self.total_inference_time += inference_time
        self.min_inference_time = min(self.min_inference_time, inference_time)
        self.max_inference_time = max(self.max_inference_time, inference_time)
        self.avg_inference_time = self.total_inference_time / self.inference_count
        self.recent_inference_times.append(inference_time)
    
    def get_recent_avg_inference_time(self) -> float:
        """Get average inference time from recent samples."""
        if not self.recent_inference_times:
            return 0.0
        return sum(self.recent_inference_times) / len(self.recent_inference_times)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    temperature: Optional[float] = None
    disk_usage: float = 0.0
    
    # Raspberry Pi specific
    gpu_memory_split: Optional[int] = None
    throttling_state: Optional[str] = None


@dataclass
class AudioMetrics:
    """Audio processing metrics."""
    chunks_processed: int = 0
    audio_buffer_overruns: int = 0
    audio_buffer_underruns: int = 0
    avg_audio_level: float = 0.0
    peak_audio_level: float = 0.0
    sample_rate: int = 16000
    channels: int = 2
    chunk_size: int = 1280


class MetricsCollector:
    """
    Lightweight metrics collector optimized for Raspberry Pi.
    Provides TFLite-specific performance monitoring and system resource tracking.
    """
    
    def __init__(self, collection_interval: float = 1.0):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: Interval between system metric collections (seconds)
        """
        self.collection_interval = collection_interval
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._collection_thread: Optional[threading.Thread] = None
        
        # Metrics storage
        self.tflite_metrics = TFLiteMetrics()
        self.system_metrics = SystemMetrics()
        self.audio_metrics = AudioMetrics()
        
        # Wake word detection metrics
        self.wake_word_detections = defaultdict(int)
        self.detection_confidence_history = deque(maxlen=1000)
        
        # Performance tracking
        self.start_time = time.time()
        self.last_detection_time: Optional[float] = None
        
        logger.info("MetricsCollector initialized")
        self._start_collection()
    
    def _start_collection(self):
        """Start background metrics collection."""
        self._collection_thread = threading.Thread(
            target=self._collect_system_metrics,
            daemon=True
        )
        self._collection_thread.start()
        logger.info("Background metrics collection started")
    
    def _collect_system_metrics(self):
        """Collect system metrics in background thread."""
        while not self._stop_event.wait(self.collection_interval):
            try:
                with self._lock:
                    # CPU usage
                    self.system_metrics.cpu_usage = psutil.cpu_percent(interval=None)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.system_metrics.memory_usage = memory.percent
                    self.system_metrics.memory_available = memory.available / (1024 * 1024)  # MB
                    
                    # Disk usage
                    disk = psutil.disk_usage('/')
                    self.system_metrics.disk_usage = (disk.used / disk.total) * 100
                    
                    # Temperature (Raspberry Pi specific)
                    try:
                        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                            temp_millicelsius = int(f.read().strip())
                            self.system_metrics.temperature = temp_millicelsius / 1000.0
                    except (FileNotFoundError, ValueError):
                        pass
                    
                    # Throttling state (Raspberry Pi specific)
                    try:
                        with open('/sys/devices/platform/soc/soc:firmware/get_throttled', 'r') as f:
                            throttled = f.read().strip()
                            self.system_metrics.throttling_state = throttled
                    except FileNotFoundError:
                        pass
                    
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
    
    def record_inference_time(self, inference_time: float, model_name: str = "unknown"):
        """
        Record TFLite inference time.
        
        Args:
            inference_time: Time taken for inference (seconds)
            model_name: Name of the model used
        """
        with self._lock:
            self.tflite_metrics.update_inference_time(inference_time)
            logger.debug(f"Recorded inference time: {inference_time:.4f}s for {model_name}")
    
    def record_model_load(self, load_time: float, model_path: str, model_size: int, model_format: str):
        """
        Record model loading metrics.
        
        Args:
            load_time: Time taken to load model (seconds)
            model_path: Path to the model file
            model_size: Size of model file in bytes
            model_format: Format of model ('tflite' or 'onnx')
        """
        with self._lock:
            self.tflite_metrics.model_load_time += load_time
            self.tflite_metrics.model_load_count += 1
            self.tflite_metrics.model_sizes[model_path] = model_size
            self.tflite_metrics.model_formats[model_path] = model_format
            
            logger.info(f"Recorded model load: {load_time:.3f}s, {model_size} bytes, {model_format}")
    
    def record_wake_word_detection(self, wake_word: str, confidence: float):
        """
        Record wake word detection.
        
        Args:
            wake_word: Name of detected wake word
            confidence: Detection confidence score
        """
        with self._lock:
            self.wake_word_detections[wake_word] += 1
            self.detection_confidence_history.append({
                'wake_word': wake_word,
                'confidence': confidence,
                'timestamp': time.time()
            })
            self.last_detection_time = time.time()
            
            logger.info(f"Recorded wake word detection: {wake_word} (confidence: {confidence:.3f})")
    
    def record_audio_metrics(self, chunk_size: int, audio_level: float, buffer_overruns: int = 0):
        """
        Record audio processing metrics.
        
        Args:
            chunk_size: Size of audio chunk processed
            audio_level: RMS audio level
            buffer_overruns: Number of buffer overruns
        """
        with self._lock:
            self.audio_metrics.chunks_processed += 1
            self.audio_metrics.avg_audio_level = (
                (self.audio_metrics.avg_audio_level * (self.audio_metrics.chunks_processed - 1) + audio_level) /
                self.audio_metrics.chunks_processed
            )
            self.audio_metrics.peak_audio_level = max(self.audio_metrics.peak_audio_level, audio_level)
            self.audio_metrics.audio_buffer_overruns += buffer_overruns
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dictionary containing all metrics
        """
        with self._lock:
            uptime = time.time() - self.start_time
            
            return {
                'uptime_seconds': uptime,
                'timestamp': time.time(),
                
                # TFLite metrics
                'tflite': {
                    'inference_count': self.tflite_metrics.inference_count,
                    'total_inference_time': self.tflite_metrics.total_inference_time,
                    'avg_inference_time': self.tflite_metrics.avg_inference_time,
                    'min_inference_time': self.tflite_metrics.min_inference_time if self.tflite_metrics.min_inference_time != float('inf') else 0,
                    'max_inference_time': self.tflite_metrics.max_inference_time,
                    'recent_avg_inference_time': self.tflite_metrics.get_recent_avg_inference_time(),
                    'model_load_count': self.tflite_metrics.model_load_count,
                    'model_load_time': self.tflite_metrics.model_load_time,
                    'loaded_models': len(self.tflite_metrics.model_sizes),
                    'model_formats': dict(self.tflite_metrics.model_formats),
                    'total_model_size_mb': sum(self.tflite_metrics.model_sizes.values()) / (1024 * 1024)
                },
                
                # System metrics
                'system': {
                    'cpu_usage_percent': self.system_metrics.cpu_usage,
                    'memory_usage_percent': self.system_metrics.memory_usage,
                    'memory_available_mb': self.system_metrics.memory_available,
                    'disk_usage_percent': self.system_metrics.disk_usage,
                    'temperature_celsius': self.system_metrics.temperature,
                    'throttling_state': self.system_metrics.throttling_state
                },
                
                # Audio metrics
                'audio': {
                    'chunks_processed': self.audio_metrics.chunks_processed,
                    'buffer_overruns': self.audio_metrics.audio_buffer_overruns,
                    'buffer_underruns': self.audio_metrics.audio_buffer_underruns,
                    'avg_audio_level': self.audio_metrics.avg_audio_level,
                    'peak_audio_level': self.audio_metrics.peak_audio_level,
                    'sample_rate': self.audio_metrics.sample_rate,
                    'channels': self.audio_metrics.channels,
                    'chunk_size': self.audio_metrics.chunk_size
                },
                
                # Wake word detection metrics
                'wake_words': {
                    'total_detections': sum(self.wake_word_detections.values()),
                    'detections_by_word': dict(self.wake_word_detections),
                    'last_detection_time': self.last_detection_time,
                    'recent_confidence_avg': self._get_recent_confidence_avg()
                },
                
                # Performance metrics
                'performance': {
                    'inferences_per_second': self.tflite_metrics.inference_count / uptime if uptime > 0 else 0,
                    'detections_per_hour': (sum(self.wake_word_detections.values()) / uptime) * 3600 if uptime > 0 else 0,
                    'audio_chunks_per_second': self.audio_metrics.chunks_processed / uptime if uptime > 0 else 0
                }
            }
    
    def _get_recent_confidence_avg(self) -> float:
        """Get average confidence of recent detections."""
        if not self.detection_confidence_history:
            return 0.0
        
        recent_confidences = [d['confidence'] for d in self.detection_confidence_history[-10:]]
        return sum(recent_confidences) / len(recent_confidences)
    
    def get_prometheus_metrics(self) -> str:
        """
        Get metrics in Prometheus format.
        
        Returns:
            Prometheus-formatted metrics string
        """
        metrics = self.get_metrics_summary()
        
        prometheus_metrics = []
        
        # TFLite metrics
        prometheus_metrics.extend([
            f"tflite_inference_count {metrics['tflite']['inference_count']}",
            f"tflite_inference_time_total {metrics['tflite']['total_inference_time']}",
            f"tflite_inference_time_avg {metrics['tflite']['avg_inference_time']}",
            f"tflite_inference_time_min {metrics['tflite']['min_inference_time']}",
            f"tflite_inference_time_max {metrics['tflite']['max_inference_time']}",
            f"tflite_model_load_count {metrics['tflite']['model_load_count']}",
            f"tflite_model_load_time_total {metrics['tflite']['model_load_time']}",
            f"tflite_loaded_models {metrics['tflite']['loaded_models']}",
            f"tflite_model_size_mb_total {metrics['tflite']['total_model_size_mb']}"
        ])
        
        # System metrics
        prometheus_metrics.extend([
            f"system_cpu_usage_percent {metrics['system']['cpu_usage_percent']}",
            f"system_memory_usage_percent {metrics['system']['memory_usage_percent']}",
            f"system_memory_available_mb {metrics['system']['memory_available_mb']}",
            f"system_disk_usage_percent {metrics['system']['disk_usage_percent']}"
        ])
        
        if metrics['system']['temperature_celsius'] is not None:
            prometheus_metrics.append(f"system_temperature_celsius {metrics['system']['temperature_celsius']}")
        
        # Audio metrics
        prometheus_metrics.extend([
            f"audio_chunks_processed {metrics['audio']['chunks_processed']}",
            f"audio_buffer_overruns {metrics['audio']['buffer_overruns']}",
            f"audio_avg_level {metrics['audio']['avg_audio_level']}",
            f"audio_peak_level {metrics['audio']['peak_audio_level']}"
        ])
        
        # Wake word metrics
        prometheus_metrics.extend([
            f"wake_word_detections_total {metrics['wake_words']['total_detections']}",
            f"wake_word_recent_confidence_avg {metrics['wake_words']['recent_confidence_avg']}"
        ])
        
        # Performance metrics
        prometheus_metrics.extend([
            f"performance_inferences_per_second {metrics['performance']['inferences_per_second']}",
            f"performance_detections_per_hour {metrics['performance']['detections_per_hour']}",
            f"performance_audio_chunks_per_second {metrics['performance']['audio_chunks_per_second']}"
        ])
        
        return '\n'.join(prometheus_metrics)
    
    def reset_metrics(self):
        """Reset all metrics to initial state."""
        with self._lock:
            self.tflite_metrics = TFLiteMetrics()
            self.system_metrics = SystemMetrics()
            self.audio_metrics = AudioMetrics()
            self.wake_word_detections.clear()
            self.detection_confidence_history.clear()
            self.start_time = time.time()
            self.last_detection_time = None
            
            logger.info("Metrics reset")
    
    def stop(self):
        """Stop metrics collection."""
        self._stop_event.set()
        if self._collection_thread:
            self._collection_thread.join()
        logger.info("Metrics collection stopped")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop()