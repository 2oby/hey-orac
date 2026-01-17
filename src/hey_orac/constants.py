"""
Constants for Hey ORAC wake word detection.

This module contains all magic numbers used throughout the application,
extracted into named constants for better maintainability and clarity.
"""

# Audio Configuration
CHUNK_SIZE = 1280  # samples per chunk (80ms at 16kHz)
SAMPLE_RATE = 16000  # Hz - standard for speech processing
CHANNELS_STEREO = 2  # stereo audio input
SAMPLE_WIDTH_BYTES = 2  # 16-bit audio = 2 bytes per sample

# Ring Buffer Configuration
RING_BUFFER_SECONDS = 10.0  # seconds of audio history to keep for pre-roll

# Monitoring and Logging Intervals
AUDIO_LOG_INTERVAL_CHUNKS = 100  # log every N audio chunks processed
MODERATE_CONFIDENCE_LOG_INTERVAL_CHUNKS = 50  # log moderate confidence every N chunks
CONFIG_CHECK_INTERVAL_SECONDS = 1.0  # check for config changes
HEALTH_CHECK_INTERVAL_SECONDS = 30.0  # check STT service health
THREAD_CHECK_INTERVAL_SECONDS = 5.0  # check thread health

# Thread Health Detection
MAX_STUCK_RMS_COUNT = 10  # max identical RMS values before considering thread stuck
STUCK_RMS_THRESHOLD = 0.0001  # threshold for detecting identical RMS values

# Queue Configuration
QUEUE_TIMEOUT_SECONDS = 2.0  # timeout for queue.get() operations
EVENT_QUEUE_MAXSIZE = 100  # maximum size of event queue
AUDIO_READER_QUEUE_MAXSIZE = 25  # maximum size of audio reader queue (2 seconds at 80ms/chunk)

# Network Timeouts
WEBHOOK_TIMEOUT_SECONDS = 5  # timeout for webhook HTTP requests
THREAD_JOIN_TIMEOUT_SECONDS = 5  # timeout for thread.join() operations

# Detection Thresholds
# Note: These are defaults; actual thresholds are typically loaded from config
DETECTION_THRESHOLD_DEFAULT = 0.3  # default wake word detection threshold
MODERATE_CONFIDENCE_THRESHOLD = 0.1  # threshold for logging moderate confidence
WEAK_SIGNAL_THRESHOLD = 0.05  # threshold for logging weak signals
VERY_WEAK_SIGNAL_THRESHOLD = 0.01  # threshold for logging very weak signals

# Test Recording Configuration
TEST_RECORDING_DURATION_SECONDS = 10  # duration for test audio recordings
TEST_RECORDING_COUNTDOWN_SECONDS = 5  # countdown before starting recording
TEST_RECORDING_STATUS_INTERVAL_SECONDS = 2  # status update interval during recording

# Pipeline Testing
PIPELINE_TEST_LOG_INTERVAL_CHUNKS = 25  # log interval for pipeline testing (~2 seconds)
PIPELINE_TEST_DETECTION_THRESHOLD = 0.05  # low threshold for custom model testing
