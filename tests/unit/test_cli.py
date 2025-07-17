"""
Unit tests for the CLI module.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys


def test_cli_help():
    """Test that CLI help is displayed correctly."""
    with patch.object(sys, 'argv', ['hey-orac', '--help']):
        with pytest.raises(SystemExit) as exc_info:
            from hey_orac.cli import main
            main()
        assert exc_info.value.code == 0


def test_cli_run_command():
    """Test the run command."""
    with patch.object(sys, 'argv', ['hey-orac', 'run']):
        with patch('hey_orac.cli.time.sleep') as mock_sleep:
            mock_sleep.side_effect = KeyboardInterrupt()
            from hey_orac.cli import main
            main()  # Should handle KeyboardInterrupt gracefully