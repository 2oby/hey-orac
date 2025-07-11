#!/usr/bin/env python3
"""
Tests for Hey Orac wake-word detection service
"""

import pytest
import yaml
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import load_config


class TestConfiguration:
    """Test configuration loading and validation."""
    
    def test_load_config(self):
        """Test configuration loading from YAML file."""
        config_path = Path(__file__).parent.parent / "src" / "config.yaml"
        config = load_config(str(config_path))
        
        # Check required sections exist
        assert "audio" in config
        assert "wake_word" in config
        assert "buffer" in config
        assert "network" in config
        assert "logging" in config
        
        # Check audio settings
        assert config["audio"]["sample_rate"] == 16000
        assert config["audio"]["channels"] == 1
        
        # Check wake-word settings
        assert config["wake_word"]["keyword"] == "alexa"  # Using OpenWakeWord alexa model
        assert 0.0 <= config["wake_word"]["sensitivity"] <= 1.0  # OpenWakeWord uses sensitivity


class TestAudioBuffer:
    """Test audio buffer functionality."""
    
    def test_buffer_creation(self):
        """Test audio buffer creation."""
        # TODO: Implement audio buffer tests
        pass
    
    def test_preroll_capture(self):
        """Test pre-roll audio capture."""
        # TODO: Implement pre-roll tests
        pass


class TestWakeWordDetection:
    """Test wake-word detection functionality."""
    
    def test_porcupine_initialization(self):
        """Test Porcupine wake-word engine initialization."""
        # TODO: Implement Porcupine tests
        pass
    
    def test_wake_word_detection(self):
        """Test wake-word detection with sample audio."""
        # TODO: Implement detection tests
        pass


if __name__ == "__main__":
    pytest.main([__file__]) 