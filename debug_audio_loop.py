#!/usr/bin/env python3
"""
Debug Audio Processing Loop
Test script to identify where the audio processing is getting stuck
"""

import logging
import time
import numpy as np
from audio_utils import AudioManager
from shared_memory_ipc import shared_memory_ipc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_audio_loop():
    """Debug the audio processing loop to find where it's getting stuck."""
    logger.info("🔍 Starting audio loop debug...")
    
    # Initialize audio manager
    audio_manager = AudioManager()
    
    # Find USB microphone
    devices = audio_manager.list_input_devices()
    if not devices:
        logger.error("❌ No audio devices found!")
        return 1
    
    usb_device = None
    for device in devices:
        if device.is_usb:
            usb_device = device
            break
    
    if not usb_device:
        logger.error("❌ No USB microphone found!")
        return 1
    
    logger.info(f"🎤 Using USB microphone: {usb_device.name}")
    
    # Start audio stream with simple parameters
    sample_rate = 16000
    chunk_size = 1024
    channels = 1
    
    logger.info(f"🎵 Starting audio stream: {sample_rate}Hz, {channels} channel, {chunk_size} samples/chunk")
    
    stream = audio_manager.start_stream(
        device_index=usb_device.index,
        sample_rate=sample_rate,
        channels=channels,
        chunk_size=chunk_size
    )
    
    if not stream:
        logger.error("❌ Failed to start audio stream")
        return 1
    
    logger.info("✅ Audio stream started successfully")
    
    # Test shared memory IPC
    logger.info("🔗 Testing shared memory IPC...")
    try:
        shared_memory_ipc.update_audio_state(0.0)
        logger.info("✅ Shared memory IPC test successful")
    except Exception as e:
        logger.error(f"❌ Shared memory IPC test failed: {e}")
        return 1
    
    # Main debug loop
    try:
        chunk_count = 0
        start_time = time.time()
        
        logger.info("🎯 Starting debug audio loop...")
        logger.info("📊 Will log every 10 chunks to track progress")
        
        while True:
            try:
                # Step 1: Read audio chunk
                logger.debug(f"📖 Reading chunk {chunk_count + 1}...")
                audio_chunk = stream.read(chunk_size, exception_on_overflow=False)
                chunk_count += 1
                
                # Step 2: Convert to numpy array
                logger.debug(f"🔄 Converting chunk {chunk_count} to numpy array...")
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # Step 3: Calculate RMS
                logger.debug(f"📊 Calculating RMS for chunk {chunk_count}...")
                rms_level = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                
                # Step 4: Update shared memory
                logger.debug(f"🔗 Updating shared memory for chunk {chunk_count}...")
                shared_memory_ipc.update_audio_state(rms_level)
                
                # Progress logging every 10 chunks
                if chunk_count % 10 == 0:
                    elapsed = time.time() - start_time
                    chunks_per_second = chunk_count / elapsed
                    logger.info(f"📊 Progress: {chunk_count} chunks, {elapsed:.1f}s elapsed, {chunks_per_second:.1f} chunks/s")
                    logger.info(f"   RMS: {rms_level:.4f}, Max: {np.max(np.abs(audio_data))}")
                    
                    # Test shared memory read
                    try:
                        state = shared_memory_ipc.get_system_state()
                        logger.info(f"   Shared Memory: RMS={state['current_rms']:.4f}, Active={state['is_active']}")
                    except Exception as e:
                        logger.warning(f"   Shared memory read failed: {e}")
                
                # Test for 100 chunks then exit
                if chunk_count >= 100:
                    logger.info("✅ Debug test completed successfully!")
                    break
                
            except Exception as e:
                logger.error(f"❌ Error processing chunk {chunk_count}: {e}")
                logger.error(f"   Audio data length: {len(audio_data) if 'audio_data' in locals() else 'N/A'}")
                logger.error(f"   Stream active: {stream.is_active() if stream else False}")
                continue
                
    except KeyboardInterrupt:
        logger.info("🛑 Debug stopped by user")
    except Exception as e:
        logger.error(f"❌ Debug error: {e}")
        return 1
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up...")
        try:
            if stream:
                stream.stop_stream()
                stream.close()
                logger.info("✅ Audio stream closed")
            
            audio_manager.stop_recording()
            logger.info("✅ Audio manager stopped")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    return 0

if __name__ == "__main__":
    debug_audio_loop() 