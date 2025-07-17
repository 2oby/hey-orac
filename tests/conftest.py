"""
Pytest configuration and fixtures for Hey ORAC tests.
"""

import pytest
from pathlib import Path
import tempfile
import json


@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config = {
            "models": {
                "hey_jarvis": {
                    "enabled": True,
                    "threshold": 0.5,
                    "type": "onnx"
                }
            },
            "audio": {
                "device_index": -1,
                "sample_rate": 16000,
                "chunk_size": 1280,
                "ring_buffer_seconds": 10
            },
            "transport": {
                "endpoint": "http://localhost:8080/stream",
                "timeout": 30,
                "retry_count": 3
            }
        }
        json.dump(config, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def mock_audio_data():
    """Generate mock audio data for testing."""
    import numpy as np
    # Generate 1 second of 16kHz audio
    duration = 1.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    # Generate a simple sine wave
    frequency = 440  # A4 note
    t = np.linspace(0, duration, samples, False)
    audio = np.sin(frequency * 2 * np.pi * t) * 0.5
    return audio.astype(np.float32)