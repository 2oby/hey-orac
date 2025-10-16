# Sprint 9: Refactor Massive Main Function - Part 2

## Context from Previous Session

You are continuing the Hey Orac code cleanup project. **Sprints 1-8 are complete** (57% of the project done!) and the application is working perfectly on the Raspberry Pi.

**📋 IMPORTANT**: This sprint is part of a 14-sprint cleanup plan documented in **`CLEANUP.md`**. Read that file first for the full context, testing protocol, and complete sprint breakdown. This document provides focused context for Sprint 9 specifically.

### Project Overview
Hey Orac is a wake word detection service running on a Raspberry Pi in a Docker container. It:
- Detects wake words using OpenWakeWord models
- Sends audio to ORAC STT service for speech-to-text processing
- Provides a web interface for monitoring (port 7171)
- Sends heartbeats to register wake word models with ORAC STT

### Current Branch Status
- **Working Branch**: `code-cleanup`
- **Completed Sprints**: 8/14 (ALL HIGH PRIORITY sprints + Sprint 8 Part 1 complete!)
- **Sprint Progress Tracked In**: `CLEANUP.md`
- **Git Commits**: Clean history with descriptive messages

## Sprint 9 Goal: Extract Detection Loop and Model Reload

**Problem**: After Sprint 8, the `main()` function is still ~900 lines long. The remaining complexity is:
1. A 100+ line nested `reload_models()` function inside main()
2. A 400+ line detection loop (while True) inside main()

**Solution**: Extract these two major sections into standalone functions. After this sprint, main() will be ~200 lines of clean orchestration code.

**Sprint 8 Recap**: In Sprint 8, we extracted 5 setup functions:
- `setup_audio_input()` - Audio source initialization
- `setup_wake_word_models()` - Model loading and initialization
- `setup_heartbeat_sender()` - Heartbeat sender setup
- `setup_web_server()` - Web server initialization
- `setup_stt_components()` - STT components setup

Now in Sprint 9, we'll extract the remaining complex sections.

## Current main() Function Structure (After Sprint 8)

The `main()` function now has this structure:

### Lines 438-480: Initialization (KEEP AS-IS)
```python
def main():
    # Log git commit
    # Parse arguments
    # Initialize shared data
    # Initialize SettingsManager
    # Get configurations
```

### Lines 484-548: Test pipeline mode (KEEP AS-IS - special mode)
```python
    if args.test_pipeline:
        # ... test pipeline logic ...
        return
```

### Lines 550-608: Audio manager and recording mode (KEEP AS-IS)
```python
    audio_manager = None
    usb_mic = None

    if not args.input_wav:
        # Initialize AudioManager
        # Find USB microphone

    if args.record_test:
        # ... recording mode logic ...
        return
```

### After Sprint 8 setup calls (KEEP AS-IS - already clean)
```python
    # Setup audio input
    stream = setup_audio_input(args, audio_config, audio_manager, usb_mic)

    # Initialize wake word models
    model, active_model_configs, model_name_mapping, enabled_models = setup_wake_word_models(
        models_config, system_config, settings_manager, shared_data
    )

    # Initialize heartbeat sender
    heartbeat_sender = setup_heartbeat_sender(enabled_models)

    # Initialize web server
    broadcaster = setup_web_server(settings_manager, shared_data, event_queue)

    # Get STT configuration and initialize components
    with settings_manager.get_config() as config:
        stt_config = config.stt

    ring_buffer, speech_recorder, stt_client, stt_health_status = setup_stt_components(
        stt_config, audio_config, active_model_configs, shared_data
    )
```

### Lines ~830-933: reload_models() nested function (EXTRACT THIS)
```python
    # Function to reload models when configuration changes
    def reload_models():
        nonlocal model, active_model_configs, model_name_mapping
        logger.info("🔄 Reloading models due to configuration change...")

        try:
            # Get current enabled models from configuration
            # Build new model paths and configs
            # Create new model instance
            # Test the new model
            # Replace old model and configs
            # Update shared data
            # Update heartbeat sender with new models
            # Perform health checks for reloaded models
            # Clean up old model
            # Update STT health status after reload

            logger.info("✅ Models reloaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Error reloading models: {e}")
            return False
```
**EXTRACT THIS**: This should be a standalone function that takes all necessary parameters.

### Lines ~935-943: Audio reader thread initialization (KEEP IN MAIN)
```python
    # Initialize audio reader thread for non-blocking audio capture
    audio_reader = AudioReaderThread(stream, chunk_size=constants.CHUNK_SIZE, queue_maxsize=constants.AUDIO_READER_QUEUE_MAXSIZE)
    if not audio_reader.start():
        logger.error("Failed to start audio reader thread")
        raise RuntimeError("Failed to start audio reader thread")

    # Register main loop as a consumer
    main_consumer_queue = audio_reader.register_consumer("main_loop")
    logger.info("✅ Main loop registered as audio consumer")
```
**NOTE**: This is closely tied to the detection loop, so we'll pass it as a parameter to the detection loop function.

### Lines ~945-1380: Main detection loop (EXTRACT THIS)
```python
    # Continuously listen to the audio stream and detect wake words
    logger.info("🎤 Starting wake word detection loop with multi-consumer audio distribution...")
    sys.stdout.flush()
    chunk_count = 0
    last_config_check = time.time()
    last_health_check = time.time()
    last_thread_check = time.time()

    # Variables for detecting stuck RMS
    last_rms = None
    stuck_rms_count = 0

    while True:
        try:
            # Check for configuration changes
            # Check STT health periodically
            # Check audio thread health periodically
            # Get audio data from consumer queue
            # Convert bytes to audio data
            # Calculate RMS
            # Check for stuck RMS values
            # Feed audio to ring buffer
            # Pass audio to model for prediction

            # Multi-trigger mode vs single-trigger mode
            if multi_trigger_enabled:
                # ... multi-trigger logic ...
            else:
                # ... single-trigger logic ...

        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            continue
```
**EXTRACT THIS**: This should be a separate function that contains the entire detection loop.

### Lines ~1381-1444: Exception handlers (KEEP IN MAIN)
```python
    except KeyboardInterrupt:
        # Cleanup logic
    except Exception as e:
        # Error handling and cleanup
```
**NOTE**: These outer exception handlers stay in main() since they're the final safety net.

## Functions to Extract in Sprint 9

### 1. reload_models_on_config_change()

**Purpose**: Reload OpenWakeWord models when configuration changes (extracted from nested function).

**Function Signature**:
```python
def reload_models_on_config_change(
    settings_manager: SettingsManager,
    heartbeat_sender: HeartbeatSender,
    stt_client,
    shared_data: dict
) -> tuple:
    """
    Reload OpenWakeWord models when configuration changes.

    This function is called when the web GUI signals a configuration change.
    It creates a new Model instance with the updated configuration and replaces
    the old model. This allows dynamic model loading without restarting the app.

    Args:
        settings_manager: SettingsManager instance to get updated config
        heartbeat_sender: HeartbeatSender to update with new models
        stt_client: STTClient for health checks
        shared_data: Shared data dict to update with new model info

    Returns:
        tuple: (success, model, active_model_configs, model_name_mapping, enabled_models)
            - success: bool - True if reload succeeded
            - model: OpenWakeWord Model instance (new if success, None if failed)
            - active_model_configs: Dict mapping config names to ModelConfig objects
            - model_name_mapping: Dict mapping OpenWakeWord keys to config names
            - enabled_models: List of enabled ModelConfig objects
    """
```

**Extracted Code** (from nested function at lines ~830-933):
```python
def reload_models_on_config_change(settings_manager, heartbeat_sender, stt_client, shared_data):
    """
    Reload OpenWakeWord models when configuration changes.

    This function is called when the web GUI signals a configuration change.
    It creates a new Model instance with the updated configuration and replaces
    the old model. This allows dynamic model loading without restarting the app.

    Args:
        settings_manager: SettingsManager instance to get updated config
        heartbeat_sender: HeartbeatSender to update with new models
        stt_client: STTClient for health checks
        shared_data: Shared data dict to update with new model info

    Returns:
        tuple: (success, model, active_model_configs, model_name_mapping, enabled_models)
            - success: bool - True if reload succeeded
            - model: OpenWakeWord Model instance (new if success, None if failed)
            - active_model_configs: Dict mapping config names to ModelConfig objects
            - model_name_mapping: Dict mapping OpenWakeWord keys to config names
            - enabled_models: List of enabled ModelConfig objects
    """
    logger.info("🔄 Reloading models due to configuration change...")

    try:
        # Get current enabled models from configuration
        with settings_manager.get_config() as current_config:
            models_config = current_config.models
            system_config = current_config.system

        enabled_models = [model for model in models_config if model.enabled]
        if not enabled_models:
            logger.warning("⚠️  No enabled models found after config change")
            return False, None, {}, {}, []

        # Build new model paths and configs
        new_model_paths = []
        new_active_configs = {}
        new_name_mapping = {}

        for model_cfg in enabled_models:
            if os.path.exists(model_cfg.path):
                logger.info(f"✅ Loading model: {model_cfg.name} from {model_cfg.path}")
                new_model_paths.append(model_cfg.path)
                new_active_configs[model_cfg.name] = model_cfg

                # Create mapping for OpenWakeWord prediction key to config name
                base_name = os.path.basename(model_cfg.path).replace('.tflite', '').replace('.onnx', '')
                new_name_mapping[base_name] = model_cfg.name
                logger.debug(f"   Model name mapping: '{base_name}' -> '{model_cfg.name}'")
            else:
                logger.error(f"❌ Model file NOT found: {model_cfg.path}")

        if not new_model_paths:
            logger.error("No valid model files found after config change")
            return False, None, {}, {}, []

        # Create new model instance
        logger.info(f"Creating new OpenWakeWord instance with {len(new_model_paths)} models: {list(new_active_configs.keys())}")
        new_model = openwakeword.Model(
            wakeword_models=new_model_paths,
            vad_threshold=system_config.vad_threshold,
            enable_speex_noise_suppression=False
        )

        # Test the new model
        test_audio = np.zeros(constants.CHUNK_SIZE, dtype=np.float32)
        test_predictions = new_model.predict(test_audio)
        logger.info(f"✅ New model test successful - predictions: {list(test_predictions.keys())}")

        # Update shared data
        shared_data['loaded_models'] = list(new_active_configs.keys())
        shared_data['models_config'] = {name: {
            'enabled': True,
            'threshold': cfg.threshold,
            'webhook_url': cfg.webhook_url
        } for name, cfg in new_active_configs.items()}

        # Update heartbeat sender with new models
        # Clear existing models and re-register
        heartbeat_sender._models.clear()
        for model_cfg in enabled_models:
            heartbeat_sender.register_model(
                name=model_cfg.name,
                topic=model_cfg.topic,
                wake_word=model_cfg.name,
                enabled=model_cfg.enabled
            )
        logger.info(f"✅ Updated heartbeat sender with {len(enabled_models)} models")

        # Perform health checks for reloaded models with webhook URLs
        if stt_client:
            logger.info("🏥 Performing per-model STT health checks for reloaded models...")
            for name, cfg in new_active_configs.items():
                if cfg.webhook_url:
                    logger.debug(f"Checking STT health for model '{name}' at {cfg.webhook_url}")
                    if stt_client.health_check(webhook_url=cfg.webhook_url):
                        logger.info(f"✅ STT healthy for model '{name}'")
                    else:
                        logger.warning(f"⚠️ STT unhealthy for model '{name}' at {cfg.webhook_url}")

        # Update STT health status after reload
        stt_health_status = check_all_stt_health(new_active_configs, stt_client)
        shared_data['stt_health'] = stt_health_status
        shared_data['status_changed'] = True
        logger.info(f"🏥 STT health after reload: {stt_health_status}")

        logger.info("✅ Models reloaded successfully")
        return True, new_model, new_active_configs, new_name_mapping, enabled_models

    except Exception as e:
        logger.error(f"❌ Error reloading models: {e}")
        return False, None, {}, {}, []
```

**In main(), the nested function becomes**:
```python
    # Create wrapper for model reload that updates local variables
    def reload_models():
        nonlocal model, active_model_configs, model_name_mapping, enabled_models
        success, new_model, new_configs, new_mapping, new_enabled = reload_models_on_config_change(
            settings_manager, heartbeat_sender, stt_client, shared_data
        )
        if success:
            # Clean up old model before replacing
            old_model = model
            model = new_model
            active_model_configs = new_configs
            model_name_mapping = new_mapping
            enabled_models = new_enabled
            del old_model
        return success
```

**NOTE**: We keep a small wrapper function in main() because it needs to update the `nonlocal` variables. The heavy lifting is done in the standalone function.

### 2. run_detection_loop()

**Purpose**: Main wake word detection loop that processes audio and triggers actions.

**Function Signature**:
```python
def run_detection_loop(
    model,
    active_model_configs: dict,
    model_name_mapping: dict,
    audio_reader: AudioReaderThread,
    main_consumer_queue: queue.Queue,
    settings_manager: SettingsManager,
    shared_data: dict,
    ring_buffer,
    speech_recorder,
    stt_client,
    heartbeat_sender: HeartbeatSender,
    reload_models_func
) -> int:
    """
    Main wake word detection loop.

    This is the core detection loop that:
    1. Reads audio from the audio reader thread
    2. Converts audio to appropriate format
    3. Feeds audio through OpenWakeWord model
    4. Detects wake words above threshold
    5. Triggers webhooks and STT recording
    6. Monitors system health (config changes, STT health, thread health)

    The loop runs until interrupted or an error occurs that requires restart.

    Args:
        model: OpenWakeWord Model instance
        active_model_configs: Dict of active model configurations
        model_name_mapping: Dict mapping OpenWakeWord keys to config names
        audio_reader: AudioReaderThread instance providing audio data
        main_consumer_queue: Queue to read audio chunks from
        settings_manager: SettingsManager for config checks
        shared_data: Shared data dict for web GUI updates
        ring_buffer: RingBuffer for STT pre-roll audio
        speech_recorder: SpeechRecorder for STT integration
        stt_client: STTClient for health checks
        heartbeat_sender: HeartbeatSender for model activation tracking
        reload_models_func: Function to call when config changes detected

    Returns:
        int: Exit code (0 for normal, 1 for error requiring restart)
    """
```

**Extracted Code** (lines ~945-1380):
```python
def run_detection_loop(
    model,
    active_model_configs,
    model_name_mapping,
    audio_reader,
    main_consumer_queue,
    settings_manager,
    shared_data,
    ring_buffer,
    speech_recorder,
    stt_client,
    heartbeat_sender,
    reload_models_func
):
    """
    Main wake word detection loop.

    This is the core detection loop that:
    1. Reads audio from the audio reader thread
    2. Converts audio to appropriate format
    3. Feeds audio through OpenWakeWord model
    4. Detects wake words above threshold
    5. Triggers webhooks and STT recording
    6. Monitors system health (config changes, STT health, thread health)

    The loop runs until interrupted or an error occurs that requires restart.

    Args:
        model: OpenWakeWord Model instance
        active_model_configs: Dict of active model configurations
        model_name_mapping: Dict mapping OpenWakeWord keys to config names
        audio_reader: AudioReaderThread instance providing audio data
        main_consumer_queue: Queue to read audio chunks from
        settings_manager: SettingsManager for config checks
        shared_data: Shared data dict for web GUI updates
        ring_buffer: RingBuffer for STT pre-roll audio
        speech_recorder: SpeechRecorder for STT integration
        stt_client: STTClient for health checks
        heartbeat_sender: HeartbeatSender for model activation tracking
        reload_models_func: Function to call when config changes detected

    Returns:
        int: Exit code (0 for normal, 1 for error requiring restart)
    """
    # Continuously listen to the audio stream and detect wake words
    logger.info("🎤 Starting wake word detection loop with multi-consumer audio distribution...")
    sys.stdout.flush()
    chunk_count = 0
    last_config_check = time.time()
    last_health_check = time.time()
    last_thread_check = time.time()
    CONFIG_CHECK_INTERVAL = constants.CONFIG_CHECK_INTERVAL_SECONDS
    HEALTH_CHECK_INTERVAL = constants.HEALTH_CHECK_INTERVAL_SECONDS
    THREAD_CHECK_INTERVAL = constants.THREAD_CHECK_INTERVAL_SECONDS

    # Variables for detecting stuck RMS
    last_rms = None
    stuck_rms_count = 0
    max_stuck_count = constants.MAX_STUCK_RMS_COUNT

    while True:
        try:
            # Check for configuration changes
            current_time = time.time()
            if current_time - last_config_check >= CONFIG_CHECK_INTERVAL:
                if shared_data.get('config_changed', False):
                    logger.info("📢 Configuration change detected")
                    if reload_models_func():
                        shared_data['config_changed'] = False
                        logger.info("✅ Configuration change applied")
                    else:
                        logger.error("❌ Failed to apply configuration change")
                last_config_check = current_time

            # Check STT health periodically
            if current_time - last_health_check >= HEALTH_CHECK_INTERVAL:
                if stt_client:
                    stt_health_status = check_all_stt_health(active_model_configs, stt_client)
                    if shared_data.get('stt_health') != stt_health_status:
                        shared_data['stt_health'] = stt_health_status
                        shared_data['status_changed'] = True
                        logger.info(f"🏥 STT health status changed to: {stt_health_status}")
                    else:
                        logger.debug(f"🏥 Periodic STT health check: {stt_health_status}")
                last_health_check = current_time

            # Check audio thread health periodically
            if current_time - last_thread_check >= THREAD_CHECK_INTERVAL:
                if not audio_reader.is_healthy():
                    logger.warning("⚠️ Audio thread unhealthy, attempting restart...")
                    stats = audio_reader.get_stats()
                    logger.info(f"Thread stats before restart: {stats}")

                    if audio_reader.restart():
                        logger.info("✅ Audio thread restarted successfully")
                    else:
                        logger.error("❌ Failed to restart audio thread - exiting")
                        return 1  # Exit code for restart
                last_thread_check = current_time

            # Get audio data from consumer queue with timeout
            try:
                data = main_consumer_queue.get(timeout=constants.QUEUE_TIMEOUT_SECONDS)
            except queue.Empty:
                data = None

            if data is None:
                logger.warning("No audio data received from queue (timeout)")
                # Check if thread is still alive
                if not audio_reader.is_healthy():
                    logger.error("Audio thread appears dead, attempting restart...")
                    if not audio_reader.restart():
                        logger.error("Failed to restart audio thread - exiting")
                        return 1  # Exit code for restart
                continue

            if len(data) == 0:
                logger.warning("Empty audio data from queue")
                continue

            # Convert bytes to audio data using centralized conversion function
            audio_data = convert_to_openwakeword_format(data)

            # Calculate RMS for web GUI display
            rms = np.sqrt(np.mean(audio_data**2))
            shared_data['rms'] = float(rms)

            # Check for stuck RMS values (indicates frozen audio thread)
            if last_rms is not None and abs(rms - last_rms) < constants.STUCK_RMS_THRESHOLD:
                stuck_rms_count += 1
                if stuck_rms_count >= max_stuck_count:
                    logger.error(f"RMS stuck at {rms} for {stuck_rms_count} iterations - audio thread frozen")
                    logger.error("Forcing exit to trigger container restart")
                    return 1  # Exit code for restart
            else:
                stuck_rms_count = 0

            last_rms = rms

            # Update listening state
            shared_data['is_listening'] = True

            # Feed audio to ring buffer if STT is enabled
            if ring_buffer is not None:
                # Convert to int16 for ring buffer storage
                audio_int16 = audio_data.astype(np.int16)
                ring_buffer.write(audio_int16)

            # Log every 100 chunks to show we're processing audio
            chunk_count += 1
            if chunk_count % constants.AUDIO_LOG_INTERVAL_CHUNKS == 0:
                audio_volume = np.abs(audio_data).mean()
                audio_array = np.frombuffer(data, dtype=np.int16)
                logger.info(f"📊 Processed {chunk_count} audio chunks")
                logger.info(f"   Audio data shape: {audio_data.shape}, volume: {audio_volume:.4f}, RMS: {rms:.4f}")
                logger.info(f"   Raw data size: {len(data)} bytes, samples: {len(audio_array)}")
                if len(audio_array) > constants.CHUNK_SIZE:
                    logger.info(f"   ✅ Stereo→Mono conversion active")

            # Pass the audio data to the model for wake word prediction
            prediction = model.predict(audio_data)

            # Log ALL confidence scores after each processed chunk
            if chunk_count % constants.AUDIO_LOG_INTERVAL_CHUNKS == 0:
                all_scores = {word: f"{score:.6f}" for word, score in prediction.items()}
                logger.debug(f"🎯 All confidence scores: {all_scores}")
                # Also log what models are in active_model_configs for debugging
                if chunk_count == 100:
                    logger.debug(f"Active model configs: {list(active_model_configs.keys())}")
                    logger.debug(f"Model paths: {[cfg.path for cfg in active_model_configs.values()]}")
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            continue

        # Get system config for multi-trigger setting
        with settings_manager.get_config() as config:
            multi_trigger_enabled = config.system.multi_trigger

        if multi_trigger_enabled:
            # MULTI-TRIGGER MODE: Check each model independently
            triggered_models = []

            for wakeword, score in prediction.items():
                # Map OpenWakeWord model name to our config name
                config_name = None
                if wakeword in model_name_mapping:
                    config_name = model_name_mapping[wakeword]
                elif wakeword in active_model_configs:
                    config_name = wakeword

                if config_name and config_name in active_model_configs:
                    model_config = active_model_configs[config_name]
                    detection_threshold = model_config.threshold

                    if score >= detection_threshold:
                        triggered_models.append({
                            'wakeword': wakeword,
                            'config_name': config_name,
                            'confidence': score,
                            'threshold': detection_threshold,
                            'model_config': model_config
                        })

            # Process each triggered model
            for trigger_info in triggered_models:
                logger.info(f"🎯 WAKE WORD DETECTED (MULTI-TRIGGER)! Confidence: {trigger_info['confidence']:.6f} (threshold: {trigger_info['threshold']:.6f}) - Source: {trigger_info['wakeword']}")
                logger.info(f"   All model scores: {[f'{k}: {v:.6f}' for k, v in prediction.items()]}")

                # Create detection event for each triggered model
                detection_event = {
                    'type': 'detection',
                    'model': trigger_info['config_name'],
                    'confidence': float(trigger_info['confidence']),
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'threshold': trigger_info['threshold'],
                    'multi_trigger': True
                }

                # Update shared data with latest detection
                shared_data['last_detection'] = detection_event

                # Add to event queue (use try/except for backward compatibility)
                try:
                    event_queue = shared_data.get('_event_queue')
                    if event_queue:
                        try:
                            event_queue.put_nowait(detection_event)
                        except queue.Full:
                            logger.debug("Event queue full, skipping detection event")
                except Exception:
                    pass

                # Record activation in heartbeat sender
                heartbeat_sender.record_activation(trigger_info['config_name'])

                # Call webhook if configured
                if trigger_info['model_config'].webhook_url:
                    try:
                        # Prepare webhook payload
                        webhook_data = {
                            "wake_word": trigger_info['wakeword'],
                            "confidence": trigger_info['confidence'],
                            "threshold": trigger_info['threshold'],
                            "timestamp": time.time(),
                            "model_name": trigger_info['config_name'],
                            "all_scores": prediction,
                            "multi_trigger": True
                        }

                        # Make webhook call
                        logger.info(f"📞 Calling webhook (multi-trigger): {trigger_info['model_config'].webhook_url}")
                        response = requests.post(
                            trigger_info['model_config'].webhook_url,
                            json=webhook_data,
                            timeout=constants.WEBHOOK_TIMEOUT_SECONDS
                        )

                        if response.status_code == 200:
                            logger.info(f"✅ Webhook call successful (multi-trigger)")
                        else:
                            logger.warning(f"⚠️ Webhook returned status code: {response.status_code}")

                    except requests.exceptions.Timeout:
                        logger.error("❌ Webhook call timed out (multi-trigger)")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"❌ Webhook call failed (multi-trigger): {e}")
                    except Exception as e:
                        logger.error(f"❌ Unexpected error during webhook call (multi-trigger): {e}")

                # Trigger STT recording if enabled
                if (speech_recorder is not None and
                    trigger_info['model_config'].stt_enabled and
                    not speech_recorder.is_busy()):
                    # Get STT language from config
                    with settings_manager.get_config() as config:
                        stt_language = config.stt.language

                    # Start recording in background thread
                    speech_recorder.start_recording(
                        audio_stream=audio_reader,
                        wake_word=trigger_info['config_name'],
                        confidence=trigger_info['confidence'],
                        language=stt_language,
                        webhook_url=trigger_info['model_config'].webhook_url,
                        topic=trigger_info['model_config'].topic
                    )

        else:
            # SINGLE-TRIGGER MODE: Original "winner takes all" behavior
            max_confidence = 0.0
            best_model = None

            # Find the highest confidence score
            for wakeword, score in prediction.items():
                if score > max_confidence:
                    max_confidence = score
                    best_model = wakeword

            # Get threshold from the active model configuration
            config_name = None

            # First check the model name mapping
            if best_model in model_name_mapping:
                config_name = model_name_mapping[best_model]
            # Then try direct match
            elif best_model in active_model_configs:
                config_name = best_model
            else:
                # Log unmapped model for debugging
                if best_model is not None:
                    logger.warning(f"Could not map prediction key '{best_model}' to config name")
                    logger.debug(f"Available mappings: {model_name_mapping}")
                    logger.debug(f"Active configs: {list(active_model_configs.keys())}")

            if config_name and config_name in active_model_configs:
                model_config = active_model_configs[config_name]
                detection_threshold = model_config.threshold
            else:
                # Fallback threshold if model not found in config
                detection_threshold = constants.DETECTION_THRESHOLD_DEFAULT
                if best_model is not None:
                    logger.warning(f"Model '{best_model}' not found in active configs, using default threshold")

            if max_confidence >= detection_threshold:
                logger.info(f"🎯 WAKE WORD DETECTED! Confidence: {max_confidence:.6f} (threshold: {detection_threshold:.6f}) - Source: {best_model}")
                logger.info(f"   All model scores: {[f'{k}: {v:.6f}' for k, v in prediction.items()]}")
                logger.debug(f"Detection details: model={best_model}, config_name={config_name}, stt_enabled={active_model_configs[config_name].stt_enabled if config_name and config_name in active_model_configs else False}")

                # Add detection event to queue for web GUI
                detection_event = {
                    'type': 'detection',
                    'model': config_name or best_model,
                    'confidence': float(max_confidence),
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'threshold': detection_threshold,
                    'multi_trigger': False
                }

                # Update shared data
                shared_data['last_detection'] = detection_event

                # Add to event queue if available (backward compatibility)
                try:
                    event_queue = shared_data.get('_event_queue')
                    if event_queue:
                        try:
                            event_queue.put_nowait(detection_event)
                        except queue.Full:
                            logger.debug("Event queue full, skipping detection event")
                except Exception:
                    pass

                # Record activation in heartbeat sender
                heartbeat_sender.record_activation(config_name or best_model)

                # Call webhook if configured
                if config_name and active_model_configs[config_name].webhook_url:
                    try:
                        # Prepare webhook payload
                        webhook_data = {
                            "wake_word": best_model,
                            "confidence": float(max_confidence),
                            "threshold": float(detection_threshold),
                            "timestamp": time.time(),
                            "model_name": config_name,
                            "all_scores": {k: float(v) for k, v in prediction.items()},
                            "multi_trigger": False
                        }

                        # Make webhook call
                        logger.info(f"📞 Calling webhook: {active_model_configs[config_name].webhook_url}")
                        response = requests.post(
                            active_model_configs[config_name].webhook_url,
                            json=webhook_data,
                            timeout=constants.WEBHOOK_TIMEOUT_SECONDS
                        )

                        if response.status_code == 200:
                            logger.info(f"✅ Webhook call successful")
                        else:
                            logger.warning(f"⚠️ Webhook returned status code: {response.status_code}")

                    except requests.exceptions.Timeout:
                        logger.error("❌ Webhook call timed out")
                    except requests.exceptions.RequestException as e:
                        logger.error(f"❌ Webhook call failed: {e}")
                    except Exception as e:
                        logger.error(f"❌ Unexpected error during webhook call: {e}")

                # Trigger STT recording based on webhook URL presence
                model_has_webhook = config_name and active_model_configs[config_name].webhook_url

                if (speech_recorder is not None and
                    model_has_webhook and
                    not speech_recorder.is_busy()):
                    logger.info(f"🎤 Triggering STT recording for wake word '{config_name}' (webhook URL: {active_model_configs[config_name].webhook_url})")
                    logger.debug(f"STT recording conditions met: speech_recorder={speech_recorder is not None}, config_name={config_name}, webhook_url={active_model_configs[config_name].webhook_url}, is_busy={speech_recorder.is_busy() if speech_recorder else 'N/A'}")

                    # Get STT language from config
                    with settings_manager.get_config() as config:
                        stt_language = config.stt.language

                    logger.debug(f"Starting recording with language={stt_language}")
                    # Start recording in background thread
                    speech_recorder.start_recording(
                        audio_stream=audio_reader,
                        wake_word=config_name,
                        confidence=max_confidence,
                        language=stt_language,
                        webhook_url=active_model_configs[config_name].webhook_url,
                        topic=active_model_configs[config_name].topic
                    )
                else:
                    logger.debug(f"STT recording NOT triggered. Conditions: speech_recorder={speech_recorder is not None}, config_name={config_name}, webhook_url={active_model_configs[config_name].webhook_url if config_name and config_name in active_model_configs else None}, is_busy={speech_recorder.is_busy() if speech_recorder else 'N/A'}")
            else:
                # Enhanced debugging - log more frequent confidence updates
                if chunk_count % constants.MODERATE_CONFIDENCE_LOG_INTERVAL_CHUNKS == 0:
                    logger.debug(f"🎯 Best confidence: {max_confidence:.6f} from '{best_model}' (threshold: {detection_threshold:.6f})")
                    logger.debug(f"   All scores: {[f'{k}: {v:.6f}' for k, v in prediction.items()]}")

                # Also check for moderate confidence levels for debugging
                if max_confidence > constants.MODERATE_CONFIDENCE_THRESHOLD:
                    logger.info(f"🔍 Moderate confidence detected: {best_model} = {max_confidence:.6f}")
                elif max_confidence > constants.WEAK_SIGNAL_THRESHOLD:
                    logger.debug(f"🔍 Weak signal: {best_model} = {max_confidence:.6f}")
                elif max_confidence > constants.VERY_WEAK_SIGNAL_THRESHOLD:
                    logger.debug(f"🔍 Very weak signal: {best_model} = {max_confidence:.6f}")

    # If we exit the loop normally (shouldn't happen), return 0
    return 0
```

**In main(), replace the detection loop with**:
```python
    # Initialize audio reader thread for non-blocking audio capture
    audio_reader = AudioReaderThread(stream, chunk_size=constants.CHUNK_SIZE, queue_maxsize=constants.AUDIO_READER_QUEUE_MAXSIZE)
    if not audio_reader.start():
        logger.error("Failed to start audio reader thread")
        raise RuntimeError("Failed to start audio reader thread")

    # Register main loop as a consumer
    main_consumer_queue = audio_reader.register_consumer("main_loop")
    logger.info("✅ Main loop registered as audio consumer")

    # Run the detection loop
    exit_code = run_detection_loop(
        model=model,
        active_model_configs=active_model_configs,
        model_name_mapping=model_name_mapping,
        audio_reader=audio_reader,
        main_consumer_queue=main_consumer_queue,
        settings_manager=settings_manager,
        shared_data=shared_data,
        ring_buffer=ring_buffer,
        speech_recorder=speech_recorder,
        stt_client=stt_client,
        heartbeat_sender=heartbeat_sender,
        reload_models_func=reload_models
    )

    # If detection loop exits with error code, exit with that code
    if exit_code != 0:
        sys.exit(exit_code)
```

**NOTE**: We need to pass event_queue to the detection loop. Add this line before calling run_detection_loop():
```python
    # Store event_queue in shared_data for detection loop access
    shared_data['_event_queue'] = event_queue
```

## Implementation Steps

### Step 1: Create reload_models_on_config_change() Function (15 minutes)

Add the `reload_models_on_config_change()` function **ABOVE** the `main()` function (after the Sprint 8 setup functions, around line 680).

### Step 2: Update reload_models() Nested Function (5 minutes)

Replace the nested `reload_models()` function inside main() with the small wrapper that calls the standalone function and updates nonlocal variables.

### Step 3: Create run_detection_loop() Function (20 minutes)

Add the `run_detection_loop()` function **ABOVE** the `main()` function (after `reload_models_on_config_change()`, around line 780).

### Step 4: Update main() to Call run_detection_loop() (10 minutes)

Replace the entire detection loop in main() with:
1. Audio reader thread initialization
2. Store event_queue in shared_data
3. Call to run_detection_loop()
4. Check exit code and exit if non-zero

### Step 5: Deploy and Test (10 minutes)

**Commit and deploy**:
```bash
./scripts/deploy_and_test.sh "Sprint 9: Extract detection loop and model reload from main() - Part 2"
```

**What to verify**:
1. Container builds and starts successfully
2. Detection loop runs normally
3. Wake word detection still works
4. Configuration reload works (test via web GUI)
5. STT integration works
6. Multi-trigger mode works (if enabled)
7. Health checks work (STT, audio thread)
8. No regression in functionality

**Check logs**:
```bash
# Watch real-time logs
ssh pi "cd ~/hey-orac && docker-compose logs -f hey-orac"

# Check last 50 lines
ssh pi "cd ~/hey-orac && docker-compose logs --tail=50 hey-orac"

# Check for errors
ssh pi "cd ~/hey-orac && docker-compose logs hey-orac | grep -i error"
```

### Step 6: Update Documentation (5 minutes)

**Update CLEANUP.md**:
- Mark Sprint 9 as complete
- Update statistics (9/14 complete = 64%)
- Add completion notes

**Update devlog.md**:
Add entry at the bottom:
```markdown
## 2025-10-16 - Sprint 9: Refactored Main Function - Part 2

- Extracted 2 major functions from main() in wake_word_detection.py:
  - `reload_models_on_config_change()` - Standalone model reload function (~100 lines)
  - `run_detection_loop()` - Main detection loop extracted (~450 lines)
- Converted nested reload_models() to small wrapper calling standalone function
- Main() function reduced from ~900 lines to ~200 lines (78% reduction from original)
- Combined with Sprint 8, reduced main() from ~1200 lines to ~200 lines
- Main() is now clean orchestration code with clear structure
- Each extracted function has single responsibility and clear purpose
- Detection loop function can be tested independently
- Model reload function can be reused if needed
- Deployed and tested on Raspberry Pi - all functionality working
- Status: Sprint 9 complete (9/14 sprints = 64% complete)
- Note: Main function refactoring complete! Remaining sprints are lower priority cleanup.
```

**Commit documentation**:
```bash
git add CLEANUP.md devlog.md
git commit -m "Update CLEANUP.md and devlog.md: Sprint 9 complete (9/14)"
git push
```

## Expected Outcome

After Sprint 9:
- ✅ Extracted reload_models() as standalone function
- ✅ Extracted detection loop as standalone function
- ✅ Reduced main() from ~1200 lines to ~200 lines (83% reduction!)
- ✅ Main() is now clean orchestration code
- ✅ Each function has single responsibility
- ✅ Functions can be tested independently
- ✅ Code is much more maintainable
- ✅ Application still works perfectly on Pi
- ✅ CLEANUP.md updated (9/14 sprints complete = 64%)
- ✅ devlog.md updated with sprint details
- ✅ Major refactoring complete!

## Final main() Structure After Sprint 9

After Sprint 9, main() will look like this:

```python
def main():
    """Main function - orchestrates wake word detection system."""
    # Log git commit info
    # Parse arguments

    # Initialize shared data and event queue
    # Initialize SettingsManager
    # Get configurations

    # Handle special modes (test_pipeline, record_test) - early return

    # Initialize AudioManager and find USB mic (if not using WAV)

    # Setup all components (5 setup functions from Sprint 8)
    stream = setup_audio_input(args, audio_config, audio_manager, usb_mic)
    model, active_model_configs, model_name_mapping, enabled_models = setup_wake_word_models(...)
    heartbeat_sender = setup_heartbeat_sender(enabled_models)
    broadcaster = setup_web_server(settings_manager, shared_data, event_queue)
    ring_buffer, speech_recorder, stt_client, stt_health_status = setup_stt_components(...)

    # Define model reload wrapper
    def reload_models():
        nonlocal model, active_model_configs, model_name_mapping, enabled_models
        # Call standalone function and update locals
        ...

    # Initialize audio reader thread
    audio_reader = AudioReaderThread(...)
    main_consumer_queue = audio_reader.register_consumer("main_loop")

    # Store event_queue for detection loop
    shared_data['_event_queue'] = event_queue

    # Run detection loop
    exit_code = run_detection_loop(...)
    if exit_code != 0:
        sys.exit(exit_code)
```

That's it! From ~1200 lines down to ~200 lines of clear, orchestrated setup and execution.

## Known Good State

The application currently works with these features:
- ✅ Wake word detection (OpenWakeWord)
- ✅ Audio processing (stereo→mono via conversion.py)
- ✅ Multi-consumer audio distribution (AudioReaderThread)
- ✅ STT integration with ORAC STT service
- ✅ Web interface (Flask-SocketIO on port 7171)
- ✅ Heartbeat sender (registers models with ORAC STT)
- ✅ Configuration management (SettingsManager)
- ✅ Health checks (STT, audio thread, models)
- ✅ Constants extracted (Sprint 6)
- ✅ Audio conversion consolidated (Sprint 7)
- ✅ Setup functions extracted (Sprint 8)

**Don't break these!** Test thoroughly after changes.

## Deployment Troubleshooting

**If deployment fails**:
```bash
# Check what went wrong
ssh pi "cd ~/hey-orac && docker-compose logs --tail=100 hey-orac"

# Rollback locally
git reset --hard HEAD^

# Force deploy previous commit
./scripts/deploy_and_test.sh "Rollback: reverting Sprint 9 changes"
```

**If detection loop doesn't start**:
- Check function signature matches call in main()
- Verify all parameters are passed correctly
- Check for missing imports
- Look for syntax errors in extracted function

**If model reload doesn't work**:
- Check nested function wrapper updates nonlocal variables
- Verify standalone function is called correctly
- Check logs for reload errors

## Important Notes

1. **Always use `deploy_and_test.sh`** - Never test locally
2. **Check logs after deployment** - Verify detection loop starts
3. **Test wake word detection** - Ensure it still works
4. **Test config reload** - Change config via web GUI
5. **Keep exception handlers in main()** - They're the outer safety net
6. **Pass event_queue via shared_data** - Detection loop needs it
7. **Update both CLEANUP.md and devlog.md** - After completing sprint
8. **Commit descriptively** - Use format: "Sprint 9: Extract detection loop and model reload - Part 2"

## Historical Context

The main() function started simple but grew to 1200+ lines as features were added:
- Wake word detection core
- Web interface for monitoring
- STT integration with ORAC
- Heartbeat sender for model registration
- Health monitoring (STT, audio thread, models)
- Configuration reload without restart
- Multi-trigger mode support
- Extensive logging and debugging

Sprint 8 + 9 break this down into:
- 5 focused setup functions (Sprint 8)
- 1 model reload function (Sprint 9)
- 1 detection loop function (Sprint 9)
- 1 orchestrator main() (~200 lines)

This makes the code vastly more maintainable, testable, and extensible.

## Next Steps After Sprint 9

The remaining sprints (10-14) are **LOW PRIORITY** cleanup tasks:
- Sprint 10: Handle wake_word_detection_preprocessing.py
- Sprint 11: Standardize Configuration
- Sprint 12: Remove TODOs
- Sprint 13: Standardize Naming
- Sprint 14: Emoji Logging Standard

All major refactoring is complete after Sprint 9!

---

## Quick Command Reference

```bash
# Deploy changes
./scripts/deploy_and_test.sh "Sprint 9: Extract detection loop and model reload - Part 2"

# Watch logs in real-time
ssh pi "cd ~/hey-orac && docker-compose logs -f hey-orac"

# Check last 30 log lines
ssh pi "cd ~/hey-orac && docker-compose logs --tail=30 hey-orac"

# Check container status
ssh pi "cd ~/hey-orac && docker-compose ps"

# Rollback if broken
git reset --hard HEAD^
./scripts/deploy_and_test.sh "Rollback: reverting broken changes"
```

---

**Ready to start Sprint 9!** Extract reload_models_on_config_change() and run_detection_loop(), update main(), deploy, and test thoroughly. After this sprint, the major refactoring is complete!
