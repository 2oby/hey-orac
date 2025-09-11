# Complete Implementation Guide: Non-Blocking Audio Capture for HeyORAC

## Objective
Replace the blocking `stream.read()` calls that freeze the audio thread with a callback-based approach using PyAudio's non-blocking mode.

## Step 1: Add Callback Stream Support to AudioManager

**File:** `/Users/2oby/pCloud Box/Projects/ORAC/Hey_Orac/src/hey_orac/audio/utils.py`

Add this method after the existing `start_stream` method (around line 340):

```python
def start_callback_stream(self, device_index: int, callback, sample_rate: int = 16000,
                        channels: int = 1, chunk_size: int = 512):
    """Start audio stream with callback for non-blocking audio capture.
    
    The callback function should have the signature:
    callback(in_data, frame_count, time_info, status) -> (out_data, flag)
    
    Returns:
        PyAudio stream object or None on error
    """
    try:
        logger.info(f"🔍 Creating callback-based audio stream:")
        logger.info(f"   Device index: {device_index}")
        logger.info(f"   Sample rate: {sample_rate}")
        logger.info(f"   Channels: {channels}")
        logger.info(f"   Chunk size: {chunk_size}")
        
        # Get device info
        try:
            device_info = self.pyaudio.get_device_info_by_index(device_index)
            logger.info(f"   Device name: {device_info['name']}")
        except Exception as e:
            logger.error(f"❌ Error getting device info: {e}")
        
        # Create stream with callback
        stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=chunk_size,
            stream_callback=callback
        )
        
        logger.info(f"✅ Callback-based audio stream created successfully")
        logger.info(f"   Stream active: {stream.is_active()}")
        
        return stream
        
    except Exception as e:
        logger.error(f"❌ Error starting callback stream: {e}")
        logger.error(f"   Exception type: {type(e).__name__}")
        logger.error(f"   Exception details: {str(e)}")
        return None
```

## Step 2: Create the Callback Processor Class

**File:** `/Users/2oby/pCloud Box/Projects/ORAC/Hey_Orac/src/hey_orac/wake_word_detection.py`

Add this class BEFORE the `main()` function (around line 442, before `def main():`):

```python
import pyaudio  # Add this import at the top if not already present

class CallbackAudioProcessor:
    """Handles audio processing in a callback-based manner to prevent blocking."""
    
    def __init__(self, shared_data, event_queue, model, active_model_configs, 
                 model_name_mapping, stt_client, speech_recorder, ring_buffer,
                 settings_manager, heartbeat_sender, args):
        """Initialize the callback processor with all required components."""
        self.shared_data = shared_data
        self.event_queue = event_queue
        self.model = model
        self.active_model_configs = active_model_configs
        self.model_name_mapping = model_name_mapping
        self.stt_client = stt_client
        self.speech_recorder = speech_recorder
        self.ring_buffer = ring_buffer
        self.settings_manager = settings_manager
        self.heartbeat_sender = heartbeat_sender
        self.args = args  # Need access to args for WAV file handling
        
        # Audio processing state
        self.chunk_count = 0
        self.last_rms = None
        self.stuck_rms_count = 0
        self.max_stuck_count = 10
        self.audio_buffer = bytearray()  # Buffer for accumulating audio data
        self.target_chunk_size = 1280 * 2  # Size in bytes for int16 mono audio (1280 samples * 2 bytes)
        
        # Timing for periodic checks
        self.last_config_check = time.time()
        self.last_health_check = time.time()
        self.CONFIG_CHECK_INTERVAL = 1.0
        self.HEALTH_CHECK_INTERVAL = 30.0
        
        # Processing lock to prevent concurrent callback execution
        self.processing_lock = threading.Lock()
        self.last_callback_time = time.time()
        
        logger.info("✅ CallbackAudioProcessor initialized")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - called when audio data is available.
        
        This runs in PyAudio's internal thread, so we must be careful
        about thread safety and avoid blocking operations.
        """
        try:
            # Add incoming data to buffer
            self.audio_buffer.extend(in_data)
            
            # Process complete chunks of 1280 samples (2560 bytes)
            while len(self.audio_buffer) >= self.target_chunk_size:
                # Extract a complete chunk
                chunk_data = bytes(self.audio_buffer[:self.target_chunk_size])
                self.audio_buffer = self.audio_buffer[self.target_chunk_size:]
                
                # Process in a separate thread to avoid blocking the callback
                if not self.processing_lock.locked():
                    processing_thread = threading.Thread(
                        target=self.process_audio_chunk,
                        args=(chunk_data,),
                        daemon=True
                    )
                    processing_thread.start()
            
            # Always return continue flag to keep stream active
            return (None, pyaudio.paContinue)
            
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            # Return continue even on error to keep stream alive
            return (None, pyaudio.paContinue)
    
    def process_audio_chunk(self, data):
        """Process a chunk of audio data - runs in separate thread."""
        with self.processing_lock:
            try:
                current_time = time.time()
                
                # Check for configuration changes periodically
                if current_time - self.last_config_check >= self.CONFIG_CHECK_INTERVAL:
                    if self.shared_data.get('config_changed', False):
                        logger.info("📢 Configuration change detected in callback processor")
                        # Note: Full model reloading would require coordination with main thread
                        self.shared_data['config_changed'] = False
                    self.last_config_check = current_time
                
                # Check STT health periodically
                if current_time - self.last_health_check >= self.HEALTH_CHECK_INTERVAL:
                    if self.stt_client:
                        from wake_word_detection import check_all_stt_health  # Import the function
                        stt_health_status = check_all_stt_health(self.active_model_configs, self.stt_client)
                        if self.shared_data.get('stt_health') != stt_health_status:
                            self.shared_data['stt_health'] = stt_health_status
                            self.shared_data['status_changed'] = True
                            logger.info(f"🏥 STT health status changed to: {stt_health_status}")
                    self.last_health_check = current_time
                
                # Convert bytes to numpy array
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # Handle stereo to mono conversion if needed
                if len(audio_array) > 1280:  # Stereo data (2560 samples)
                    stereo_data = audio_array.reshape(-1, 2)
                    audio_data = np.mean(stereo_data, axis=1).astype(np.float32)
                else:
                    # Already mono
                    audio_data = audio_array.astype(np.float32)
                
                # Calculate RMS for monitoring
                rms = np.sqrt(np.mean(audio_data**2))
                self.shared_data['rms'] = float(rms)
                
                # Check for stuck RMS values (indicates problem)
                if self.last_rms is not None and abs(rms - self.last_rms) < 0.0001:
                    self.stuck_rms_count += 1
                    if self.stuck_rms_count >= self.max_stuck_count:
                        logger.error(f"RMS stuck at {rms} for {self.stuck_rms_count} iterations")
                        logger.error("Audio may be frozen - but callback is still running")
                        # Don't exit - let watchdog handle if needed
                else:
                    self.stuck_rms_count = 0
                
                self.last_rms = rms
                self.shared_data['is_listening'] = True
                
                # Feed to ring buffer for STT if enabled
                if self.ring_buffer is not None:
                    audio_int16 = audio_data.astype(np.int16)
                    self.ring_buffer.write(audio_int16)
                
                # Log progress periodically
                self.chunk_count += 1
                if self.chunk_count % 100 == 0:
                    audio_volume = np.abs(audio_data).mean()
                    logger.info(f"📊 Processed {self.chunk_count} audio chunks (callback mode)")
                    logger.info(f"   Audio data shape: {audio_data.shape}, volume: {audio_volume:.4f}, RMS: {rms:.4f}")
                
                # Pass audio to wake word model for prediction
                prediction = self.model.predict(audio_data)
                
                # Process wake word predictions
                self.process_predictions(prediction)
                
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
    
    def process_predictions(self, prediction):
        """Process wake word predictions and trigger actions."""
        try:
            # Get system config for multi-trigger setting
            with self.settings_manager.get_config() as config:
                multi_trigger_enabled = config.system.multi_trigger
            
            # Check each prediction against thresholds
            for wakeword, score in prediction.items():
                # Map prediction key to config name
                config_name = None
                if wakeword in self.model_name_mapping:
                    config_name = self.model_name_mapping[wakeword]
                elif wakeword in self.active_model_configs:
                    config_name = wakeword
                
                if config_name and config_name in self.active_model_configs:
                    model_config = self.active_model_configs[config_name]
                    
                    # Check if detection threshold is met
                    if score >= model_config.threshold:
                        logger.info(f"🎯 WAKE WORD DETECTED! {wakeword}: {score:.6f} (threshold: {model_config.threshold:.6f})")
                        
                        # Create detection event
                        detection_event = {
                            'type': 'detection',
                            'model': config_name,
                            'confidence': float(score),
                            'timestamp': datetime.utcnow().isoformat() + 'Z',
                            'threshold': model_config.threshold
                        }
                        
                        # Update shared data
                        self.shared_data['last_detection'] = detection_event
                        
                        # Add to event queue if space available
                        try:
                            self.event_queue.put_nowait(detection_event)
                        except:
                            pass  # Queue full, skip
                        
                        # Record activation in heartbeat sender
                        if self.heartbeat_sender:
                            self.heartbeat_sender.record_activation(config_name)
                        
                        # Trigger webhook if configured
                        if model_config.webhook_url:
                            self.trigger_webhook(wakeword, config_name, score, model_config, prediction)
                        
                        # Trigger STT recording if enabled and not busy
                        if (self.speech_recorder is not None and 
                            model_config.stt_enabled and
                            not self.speech_recorder.is_busy()):
                            
                            # Get STT language from config
                            with self.settings_manager.get_config() as config:
                                stt_language = config.stt.language
                            
                            # Start recording in background thread
                            logger.info(f"🎤 Starting STT recording for '{config_name}'")
                            recording_thread = threading.Thread(
                                target=self.speech_recorder.record_speech,
                                args=(model_config.webhook_url, model_config.topic, wakeword),
                                daemon=True
                            )
                            recording_thread.start()
                        
                        # In single-trigger mode, stop after first detection
                        if not multi_trigger_enabled:
                            break
                            
        except Exception as e:
            logger.error(f"Error processing predictions: {e}")
    
    def trigger_webhook(self, wakeword, config_name, score, model_config, all_predictions):
        """Trigger webhook in background thread."""
        try:
            webhook_data = {
                "wake_word": wakeword,
                "confidence": float(score),
                "threshold": float(model_config.threshold),
                "timestamp": time.time(),
                "model_name": config_name,
                "all_scores": {k: float(v) for k, v in all_predictions.items()}
            }
            
            logger.info(f"📞 Calling webhook: {model_config.webhook_url}")
            response = requests.post(
                model_config.webhook_url,
                json=webhook_data,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info("✅ Webhook call successful")
            else:
                logger.warning(f"⚠️ Webhook returned status: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Webhook call failed: {e}")
```

## Step 3: Modify the Main Function

**File:** `/Users/2oby/pCloud Box/Projects/ORAC/Hey_Orac/src/hey_orac/wake_word_detection.py`

In the `main()` function, after the audio stream initialization (around line 796), modify the code:

### 3.1: Add callback mode flag
Right after the comment `# Start audio stream if using microphone (skip if using WAV file)` (line 796), add:

```python
# Start audio stream if using microphone (skip if using WAV file)
use_callback_mode = True  # Enable non-blocking callback mode to prevent freezing
logger.info(f"🔄 Audio capture mode: {'callback (non-blocking)' if use_callback_mode else 'blocking'}")
```

### 3.2: Modify the main processing section
Replace the section starting with `# Continuously listen to the audio stream and detect wake words` (around line 1147) with:

```python
# Choose between callback mode and blocking mode
if use_callback_mode and not args.input_wav:
    # CALLBACK MODE - Non-blocking audio processing
    logger.info("🎤 Starting wake word detection with callback mode (non-blocking)...")
    
    # Create callback processor with all required components
    callback_processor = CallbackAudioProcessor(
        shared_data=shared_data,
        event_queue=event_queue,
        model=model,
        active_model_configs=active_model_configs,
        model_name_mapping=model_name_mapping,
        stt_client=stt_client,
        speech_recorder=speech_recorder,
        ring_buffer=ring_buffer,
        settings_manager=settings_manager,
        heartbeat_sender=heartbeat_sender,
        args=args
    )
    
    # Start callback-based stream
    stream = audio_manager.start_callback_stream(
        device_index=usb_mic.index if audio_config.device_index is None else audio_config.device_index,
        sample_rate=audio_config.sample_rate,
        channels=audio_config.channels,
        chunk_size=audio_config.chunk_size,
        callback=callback_processor.audio_callback
    )
    
    if not stream:
        logger.error("Failed to start callback-based audio stream")
        raise RuntimeError("Failed to start callback audio stream")
    
    logger.info("✅ Callback-based audio stream started successfully")
    logger.info("🎧 Audio processing running in background thread")
    
    # Main thread now just monitors the stream and handles shutdown
    try:
        while True:
            time.sleep(1)  # Check every second
            
            # Monitor stream health
            if not stream.is_active():
                logger.error("❌ Audio stream stopped unexpectedly")
                break
            
            # Check for model reload requests (from config changes)
            if shared_data.get('config_changed', False):
                logger.info("📢 Configuration change detected - restart required for model reload")
                # In callback mode, model reloading requires restart
                shared_data['config_changed'] = False
                
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        logger.info("✅ Callback stream closed")
        
else:
    # BLOCKING MODE - Original implementation
    logger.info("🎤 Starting wake word detection loop (blocking mode)...")
    sys.stdout.flush()
    chunk_count = 0
    last_config_check = time.time()
    last_health_check = time.time()
    CONFIG_CHECK_INTERVAL = 1.0
    HEALTH_CHECK_INTERVAL = 30.0
    
    # [Rest of the original blocking implementation stays the same...]
    # The existing while True loop and all its contents remain unchanged
```

## Step 4: Required Imports

Make sure these imports are at the top of `wake_word_detection.py`:

```python
import threading
import pyaudio
import requests
from datetime import datetime
```

## Testing Instructions

1. **Test with callback mode disabled first:**
   - Set `use_callback_mode = False`
   - Deploy and verify everything still works

2. **Enable callback mode:**
   - Set `use_callback_mode = True`
   - Deploy and monitor logs

3. **Verify functionality:**
   - Check logs for "Processed X audio chunks (callback mode)"
   - Verify RMS values are changing
   - Test wake word detection
   - Monitor for 72+ hours to ensure no freezing

## Expected Log Output

When working correctly with callback mode, you should see:
```
🔄 Audio capture mode: callback (non-blocking)
🎤 Starting wake word detection with callback mode (non-blocking)...
✅ Callback-based audio stream started successfully
🎧 Audio processing running in background thread
📊 Processed 100 audio chunks (callback mode)
   Audio data shape: (1280,), volume: 5.1234, RMS: 7.8901
```

## Rollback Plan

If issues occur, simply set `use_callback_mode = False` to revert to the original blocking implementation.

## Key Benefits

1. **No blocking:** Audio processing happens in PortAudio's managed thread
2. **Main thread responsive:** Can handle configuration changes and monitoring
3. **Automatic cleanup:** If callback crashes, PortAudio handles cleanup
4. **No frozen threads:** Eliminates the `stream.read()` blocking issue

## CRITICAL: Indentation Requirements

**⚠️ EXTREMELY IMPORTANT: Python indentation must be PERFECT ⚠️**

When implementing these changes:
1. **PRESERVE ALL EXISTING INDENTATION** - Do not change the indentation of any existing code
2. **MATCH SURROUNDING CODE** - New code must have the exact same indentation level as the code around it
3. **USE SPACES, NOT TABS** - The file uses 4 spaces per indentation level
4. **CHECK EVERY LINE** - A single wrong indentation will break the entire file
5. **DO NOT AUTO-FORMAT** - Do not let any tool auto-format or "fix" indentation

### Indentation Rules:
- Top-level functions and classes: No indentation (column 0)
- Class methods: 4 spaces (1 level)
- Code inside methods: 8 spaces (2 levels)
- Code inside if/for/while blocks: Add 4 more spaces per level
- The existing `while True:` loop in blocking mode: Already at correct indentation - DO NOT CHANGE

### How to Verify Indentation:
```bash
python3 -m py_compile src/hey_orac/wake_word_detection.py
```
If this command produces no output, the indentation is correct.

## Important Notes

- The callback runs in a separate thread managed by PortAudio
- All processing must be thread-safe
- The callback must never block - always return immediately
- Heavy processing is offloaded to separate threads
- The main thread remains available for monitoring and control
- **INDENTATION MUST BE PERFECT - Even one space off will cause a SyntaxError**