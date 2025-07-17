"""
Command-line interface for Hey ORAC wake-word module.
"""

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )


def run_server(args):
    """Run the wake-word detection server."""
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting Hey ORAC wake-word detection server...")
    
    # Import and start the main application
    from hey_orac.app import HeyOracApplication
    app = HeyOracApplication(config_path=args.config)
    
    if not app.start():
        logger.error("Failed to start application")
        sys.exit(1)
    
    logger.info("Server started. Press Ctrl+C to stop.")
    try:
        # Keep running until interrupted
        while True:
            # Log status periodically
            import time
            time.sleep(30)
            status = app.get_status()
            logger.info(f"Application status: {status}")
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        app.stop()


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Hey ORAC Wake-Word Detection Module"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run the wake-word detection server")
    run_parser.add_argument(
        "--config",
        type=Path,
        default=Path("/config/settings.json"),
        help="Path to configuration file"
    )
    run_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    run_parser.set_defaults(func=run_server)
    
    args = parser.parse_args()
    
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()