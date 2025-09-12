#!/usr/bin/env python3
"""
Test script to verify multi-consumer audio distribution.
Tests that multiple consumers receive the same audio data.
"""

import sys
import os
import time
import threading
import queue

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hey_orac.audio.audio_reader_thread import AudioReaderThread


class MockStream:
    """Mock audio stream that generates sequential test data."""
    def __init__(self):
        self.counter = 0
        
    def read(self, chunk_size, exception_on_overflow=False):
        """Generate unique test data for each call."""
        time.sleep(0.01)  # Simulate audio read delay
        self.counter += 1
        # Create unique data pattern for each chunk
        return bytes([self.counter % 256] * chunk_size * 2)


def consumer_thread(name: str, consumer_queue: queue.Queue, results: dict):
    """Consumer thread that reads from its queue and stores results."""
    chunks_received = []
    
    for i in range(10):  # Read 10 chunks
        try:
            data = consumer_queue.get(timeout=1.0)
            if data:
                # Store first byte as identifier
                chunks_received.append(data[0])
                print(f"  {name}: Received chunk {i+1} (id: {data[0]})")
        except queue.Empty:
            print(f"  {name}: Timeout on chunk {i+1}")
            break
    
    results[name] = chunks_received


def test_multi_consumer():
    """Test that multiple consumers receive the same audio data."""
    print("\n=== Testing Multi-Consumer Audio Distribution ===\n")
    
    # Create mock stream and audio reader
    mock_stream = MockStream()
    audio_reader = AudioReaderThread(mock_stream, chunk_size=128, queue_maxsize=5)
    
    print("1. Starting audio reader thread...")
    if not audio_reader.start():
        print("   ❌ Failed to start audio reader")
        return False
    print("   ✅ Audio reader started")
    
    # Register multiple consumers
    print("\n2. Registering consumers...")
    consumer1_queue = audio_reader.register_consumer("consumer1")
    consumer2_queue = audio_reader.register_consumer("consumer2") 
    consumer3_queue = audio_reader.register_consumer("consumer3")
    print("   ✅ Three consumers registered")
    
    # Start consumer threads
    print("\n3. Starting consumer threads...")
    results = {}
    threads = []
    
    for name, queue_obj in [("consumer1", consumer1_queue), 
                             ("consumer2", consumer2_queue),
                             ("consumer3", consumer3_queue)]:
        t = threading.Thread(target=consumer_thread, args=(name, queue_obj, results))
        t.start()
        threads.append(t)
    
    print("   ✅ Consumer threads started")
    
    # Wait for consumers to finish
    print("\n4. Waiting for consumers to read data...")
    for t in threads:
        t.join()
    
    # Verify results
    print("\n5. Verifying results...")
    
    # Check that all consumers received data
    if len(results) != 3:
        print(f"   ❌ Not all consumers returned results: {list(results.keys())}")
        return False
    
    # Check that all consumers received the same chunks
    chunks1 = results.get("consumer1", [])
    chunks2 = results.get("consumer2", [])
    chunks3 = results.get("consumer3", [])
    
    print(f"   Consumer1 received {len(chunks1)} chunks: {chunks1}")
    print(f"   Consumer2 received {len(chunks2)} chunks: {chunks2}")
    print(f"   Consumer3 received {len(chunks3)} chunks: {chunks3}")
    
    if chunks1 != chunks2 or chunks2 != chunks3:
        print("   ❌ Consumers received different data!")
        return False
    
    if len(chunks1) < 5:
        print(f"   ❌ Consumers received too few chunks: {len(chunks1)}")
        return False
    
    print(f"   ✅ All consumers received identical data ({len(chunks1)} chunks)")
    
    # Test unregistration
    print("\n6. Testing consumer unregistration...")
    audio_reader.unregister_consumer("consumer2")
    print("   ✅ Consumer2 unregistered")
    
    # Clean up
    print("\n7. Cleaning up...")
    audio_reader.unregister_consumer("consumer1")
    audio_reader.unregister_consumer("consumer3")
    audio_reader.stop()
    print("   ✅ All consumers unregistered and audio reader stopped")
    
    return True


def test_legacy_mode():
    """Test backward compatibility with legacy mode."""
    print("\n=== Testing Legacy Mode (Backward Compatibility) ===\n")
    
    mock_stream = MockStream()
    audio_reader = AudioReaderThread(mock_stream, chunk_size=128, queue_maxsize=5)
    
    print("1. Starting audio reader in legacy mode...")
    if not audio_reader.start():
        print("   ❌ Failed to start audio reader")
        return False
    print("   ✅ Audio reader started in legacy mode")
    
    # Use legacy get_audio() method
    print("\n2. Reading data using legacy get_audio() method...")
    chunks = []
    for i in range(5):
        data = audio_reader.get_audio(timeout=1.0)
        if data:
            chunks.append(data[0])
            print(f"   Received chunk {i+1} (id: {data[0]})")
        else:
            print(f"   Timeout on chunk {i+1}")
            
    if len(chunks) < 3:
        print(f"   ❌ Too few chunks received in legacy mode: {len(chunks)}")
        return False
    
    print(f"   ✅ Legacy mode working ({len(chunks)} chunks received)")
    
    audio_reader.stop()
    return True


if __name__ == "__main__":
    print("Multi-Consumer Audio Distribution Test")
    print("=" * 40)
    
    # Run tests
    test1_pass = test_multi_consumer()
    test2_pass = test_legacy_mode()
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    print(f"Multi-consumer test: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Legacy mode test: {'✅ PASS' if test2_pass else '❌ FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)