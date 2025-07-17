#!/usr/bin/env python3
"""
Test script for M1 implementation - baseline wake detection with ring buffer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time

from hey_orac.audio.microphone import AudioCapture
from hey_orac.models.wake_detector import WakeDetector


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def main():
    """Test M1 components."""
    logger.info("=== M1 Baseline Wake Detection Test ===")
    
    # Initialize components
    logger.info("Initializing audio capture...")
    audio_capture = AudioCapture(
        sample_rate=16000,
        chunk_size=1280,
        ring_buffer_seconds=10.0
    )
    
    logger.info("Initializing wake detector...")
    wake_detector = WakeDetector(
        models=['hey_jarvis'],  # Test with hey_jarvis model
        inference_framework='tflite'
    )
    
    # Set detection threshold - lower for testing
    wake_detector.set_threshold('hey_jarvis', 0.1)  # Lower threshold for testing
    
    # Start audio capture
    if not audio_capture.start():
        logger.error("Failed to start audio capture")
        return 1
    
    logger.info("Audio capture started successfully")
    logger.info("Listening for 'Hey Jarvis' wake word...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        chunk_count = 0
        detection_count = 0
        last_status_time = time.time()
        
        while True:
            # Get audio chunk
            audio_chunk = audio_capture.get_audio_chunk()
            
            if audio_chunk is not None:
                chunk_count += 1
                
                # Process for wake word
                detection = wake_detector.process_audio(audio_chunk)
                
                # Log confidence scores every 100 chunks for debugging
                if chunk_count % 100 == 0:
                    # Get the latest raw predictions from the wake detector
                    if hasattr(wake_detector.model, 'prediction_buffer'):
                        latest_scores = {}
                        for model_name, scores in wake_detector.model.prediction_buffer.items():
                            if scores:
                                latest_scores[model_name] = scores[-1]
                        logger.info(f"Latest confidence scores: {latest_scores}")
                
                if detection:
                    detection_count += 1
                    logger.info(
                        f"🎯 WAKE WORD DETECTED! #{detection_count} - "
                        f"Model: {detection.model_name}, "
                        f"Confidence: {detection.confidence:.3f}"
                    )
                    
                    # Test pre-roll retrieval
                    pre_roll = audio_capture.get_pre_roll(1.0)
                    logger.info(f"Retrieved {len(pre_roll)/16000:.2f}s of pre-roll audio")
                
                # Periodic status update
                current_time = time.time()
                if current_time - last_status_time >= 5.0:
                    rms = audio_capture.get_rms()
                    avg_inference = wake_detector.get_average_inference_time()
                    fill_level = audio_capture.ring_buffer.get_fill_level()
                    
                    logger.info(
                        f"Status - Chunks: {chunk_count}, RMS: {rms:.4f}, "
                        f"Buffer fill: {fill_level:.1%}, "
                        f"Inference: {avg_inference:.1f}ms, "
                        f"Detections: {detection_count}"
                    )
                    last_status_time = current_time
            
            else:
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        logger.info("\nStopping test...")
    
    finally:
        audio_capture.stop()
        logger.info(f"Test complete. Total detections: {detection_count}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())