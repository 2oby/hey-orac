# Sprint 8: Refactor Massive Main Function - Part 1

## Context from Previous Session

You are continuing the Hey Orac code cleanup project. **Sprints 1-7 are complete** (50% of the project done!) and the application is working perfectly on the Raspberry Pi.

**📋 IMPORTANT**: This sprint is part of a 14-sprint cleanup plan documented in **`CLEANUP.md`**. Read that file first for the full context, testing protocol, and complete sprint breakdown. This document provides focused context for Sprint 8 specifically.

### Project Overview
Hey Orac is a wake word detection service running on a Raspberry Pi in a Docker container. It:
- Detects wake words using OpenWakeWord models
- Sends audio to ORAC STT service for speech-to-text processing
- Provides a web interface for monitoring (port 7171)
- Sends heartbeats to register wake word models with ORAC STT

### Current Branch Status
- **Working Branch**: `code-cleanup`
- **Completed Sprints**: 7/14 (ALL HIGH PRIORITY sprints complete!)
- **Sprint Progress Tracked In**: `CLEANUP.md`
- **Git Commits**: Clean history with descriptive messages

## Sprint 8 Goal: Extract Setup Functions from Main

**Problem**: The `main()` function in `wake_word_detection.py` is **massive** - approximately 1200+ lines of code (lines 438-1644). This violates single responsibility principle and makes the code hard to understand, test, and maintain.

**Solution**: Extract setup logic into focused, well-documented functions. This sprint focuses on **PART 1** - extracting the initial setup functions that run once at startup.

**Why Two Parts**: The main() function is so large that refactoring it all at once would be risky. We split it into:
- **Sprint 8 (Part 1)**: Extract setup functions (audio, models, heartbeat, web server, STT components)
- **Sprint 9 (Part 2)**: Extract the detection loop and reload_models() nested function

## Current main() Function Structure

The `main()` function currently does:

### Lines 438-449: Git commit logging and argument parsing
```python
def main():
    # Log git commit if available
    try:
        with open('/app/git_commit.txt', 'r') as f:
            commit = f.read().strip()
            logger.info(f"🔧 Running code from git commit: {commit}")
    except:
        logger.info("🔧 Git commit info not available")

    # Parse command line arguments
    args = parse_arguments()
```

### Lines 451-464: Shared data initialization
```python
    # Initialize shared data structures for web GUI
    manager = Manager()
    shared_data = manager.dict({
        'rms': 0.0,
        'is_listening': False,
        'is_active': False,
        'status': 'Starting...',
        'last_detection': None,
        'recent_detections': [],
        'config_changed': False,
        'status_changed': True,
        'stt_health': 'disconnected'
    })
    event_queue = Queue(maxsize=constants.EVENT_QUEUE_MAXSIZE)
```

### Lines 466-478: SettingsManager initialization
```python
    # Initialize SettingsManager (triggers auto-discovery)
    logger.info("🔧 Initializing SettingsManager...")
    settings_manager = SettingsManager()

    # Get configurations
    with settings_manager.get_config() as config:
        models_config = config.models
        audio_config = config.audio
        system_config = config.system

    # Configure logging level from config
    logging.getLogger().setLevel(getattr(logging, system_config.log_level, logging.INFO))
    logger.info(f"✅ SettingsManager initialized with {len(models_config)} models")
```

### Lines 484-548: Test pipeline mode (SPECIAL MODE - not extracted)
```python
    if args.test_pipeline:
        # Load the recorded audio
        # Test with models
        # Exit early
        return
```
**NOTE**: This section is a special mode that exits early, so we won't extract it.

### Lines 550-608: Audio source initialization
```python
    # Initialize audio source - either WAV file or microphone
    stream = None
    audio_manager = None
    usb_mic = None

    if args.input_wav:
        # Use WAV file as input
        stream = WavFileStream(args.input_wav)
    else:
        # Initialize AudioManager for audio device handling
        audio_manager = AudioManager()
        # Find the USB microphone
        usb_mic = audio_manager.find_usb_microphone()

    # Handle recording mode (SPECIAL MODE)
    if args.record_test:
        # Record test audio
        return

    # Start audio stream if using microphone
    if not args.input_wav:
        stream = audio_manager.start_stream(...)
```
**NOTE**: The recording mode (args.record_test) is another special mode that exits early, so we won't extract it. But the general audio setup SHOULD be extracted.

### Lines 610-697: OpenWakeWord model initialization
```python
    # Initialize the OpenWakeWord model with enabled models from configuration
    try:
        # Get enabled models from configuration
        enabled_models = [model for model in models_config if model.enabled]

        # Load ALL enabled models
        model_paths = []
        active_model_configs = {}
        model_name_mapping = {}

        for model_cfg in enabled_models:
            # Build paths and mappings

        model = openwakeword.Model(
            wakeword_models=model_paths,
            vad_threshold=system_config.vad_threshold,
            enable_speex_noise_suppression=False
        )

        # Update shared data
        # Test model with dummy audio
```
**EXTRACT THIS**: This is a perfect candidate for extraction.

### Lines 702-719: Heartbeat sender initialization
```python
    # Initialize heartbeat sender for ORAC STT integration
    logger.info("💓 Initializing heartbeat sender for ORAC STT...")
    heartbeat_sender = HeartbeatSender()

    # Register all enabled models with the heartbeat sender
    for model_cfg in enabled_models:
        heartbeat_sender.register_model(
            name=model_cfg.name,
            topic=model_cfg.topic,
            wake_word=model_cfg.name,
            enabled=model_cfg.enabled
        )

    # Start heartbeat sender
    heartbeat_sender.start()
```
**EXTRACT THIS**: Clean, focused setup logic.

### Lines 721-731: Audio stream test
```python
    # Test audio stream first
    logger.info("🧪 Testing audio stream...")
    sys.stdout.flush()
    try:
        test_data = stream.read(constants.CHUNK_SIZE, exception_on_overflow=False)
        logger.info(f"✅ Audio stream test successful, read {len(test_data)} bytes")
    except Exception as e:
        logger.error(f"❌ Audio stream test failed: {e}")
        raise
```
**NOTE**: This could be part of the audio setup function.

### Lines 733-755: Web server initialization
```python
    # Initialize web server in a separate thread
    logger.info("🌐 Starting web server on port 7171...")

    # Create Flask app FIRST
    app = create_app()

    # Then initialize routes with shared resources
    init_routes(settings_manager, shared_data, event_queue)

    # Create and start WebSocket broadcaster
    broadcaster = WebSocketBroadcaster(socketio, shared_data, event_queue)
    broadcaster.start()

    # Start Flask-SocketIO server in a separate thread
    web_thread = threading.Thread(
        target=socketio.run,
        args=(app,),
        kwargs={'host': '0.0.0.0', 'port': 7171, 'debug': False, 'allow_unsafe_werkzeug': True}
    )
    web_thread.daemon = True
    web_thread.start()
```
**EXTRACT THIS**: Clean web server setup logic.

### Lines 762-828: STT components initialization
```python
    # Initialize STT components if enabled
    ring_buffer = None
    speech_recorder = None
    stt_client = None

    with settings_manager.get_config() as config:
        stt_config = config.stt

    # Always initialize STT components
    logger.info("🎙️ Initializing STT components...")

    # Initialize ring buffer for pre-roll audio
    ring_buffer = RingBuffer(
        capacity_seconds=constants.RING_BUFFER_SECONDS,
        sample_rate=audio_config.sample_rate
    )

    # Initialize STT client
    stt_client = STTClient(
        base_url=stt_config.base_url,
        timeout=stt_config.timeout
    )

    # Check STT service health
    if stt_client.health_check():
        logger.info("✅ STT service is healthy")

    # Initialize speech recorder
    endpoint_config = EndpointConfig(...)
    speech_recorder = SpeechRecorder(
        ring_buffer=ring_buffer,
        stt_client=stt_client,
        endpoint_config=endpoint_config
    )

    # Perform health checks for models with webhook URLs
    stt_health_status = check_all_stt_health(active_model_configs, stt_client)
```
**EXTRACT THIS**: Well-defined STT setup logic.

### Lines 830-933: reload_models() nested function definition
```python
    # Function to reload models when configuration changes
    def reload_models():
        nonlocal model, active_model_configs, model_name_mapping
        # ... 100+ lines of model reloading logic ...
```
**NOTE**: This nested function will be handled in Sprint 9.

### Lines 935-943: Audio reader thread initialization
```python
    # Initialize audio reader thread for non-blocking audio capture
    audio_reader = AudioReaderThread(stream, chunk_size=constants.CHUNK_SIZE, queue_maxsize=constants.AUDIO_READER_QUEUE_MAXSIZE)
    if not audio_reader.start():
        logger.error("Failed to start audio reader thread")
        raise RuntimeError("Failed to start audio reader thread")

    # Register main loop as a consumer
    main_consumer_queue = audio_reader.register_consumer("main_loop")
```
**NOTE**: This is closely tied to the detection loop, so we'll handle it in Sprint 9.

### Lines 945-1380: Main detection loop (while True)
**NOTE**: This will be extracted in Sprint 9.

### Lines 1381-1444: Exception handlers
**NOTE**: These stay in main() since they're the outer error handlers.

## Functions to Extract in Sprint 8

### 1. setup_audio_input()

**Purpose**: Initialize audio source (WAV file or microphone) and start audio stream.

**Function Signature**:
```python
def setup_audio_input(
    args: argparse.Namespace,
    audio_config,
    audio_manager: AudioManager = None,
    usb_mic = None
):
    """
    Initialize audio input source (WAV file or microphone).

    Args:
        args: Command line arguments (contains input_wav flag)
        audio_config: Audio configuration from SettingsManager
        audio_manager: AudioManager instance (only for microphone mode)
        usb_mic: USB microphone info (only for microphone mode)

    Returns:
        stream: Audio stream object (WavFileStream or PyAudio stream)

    Raises:
        RuntimeError: If audio initialization fails
    """
```

**Extracted Code** (lines 550-608, excluding recording mode):
```python
def setup_audio_input(args, audio_config, audio_manager=None, usb_mic=None):
    """
    Initialize audio input source (WAV file or microphone).

    Args:
        args: Command line arguments (contains input_wav flag)
        audio_config: Audio configuration from SettingsManager
        audio_manager: AudioManager instance (only for microphone mode)
        usb_mic: USB microphone info (only for microphone mode)

    Returns:
        stream: Audio stream object (WavFileStream or PyAudio stream)

    Raises:
        RuntimeError: If audio initialization fails
    """
    stream = None

    if args.input_wav:
        # Use WAV file as input
        logger.info(f"🎵 Using WAV file as input: {args.input_wav}")
        if not os.path.exists(args.input_wav):
            logger.error(f"❌ WAV file not found: {args.input_wav}")
            raise RuntimeError(f"WAV file not found: {args.input_wav}")
        stream = WavFileStream(args.input_wav)
    else:
        # Start audio stream with parameters from configuration
        stream = audio_manager.start_stream(
            device_index=usb_mic.index if audio_config.device_index is None else audio_config.device_index,
            sample_rate=audio_config.sample_rate,
            channels=audio_config.channels,
            chunk_size=audio_config.chunk_size
        )
        if not stream:
            logger.error("Failed to start audio stream. Exiting.")
            raise RuntimeError("Failed to start audio stream")

    # Test audio stream
    logger.info("🧪 Testing audio stream...")
    sys.stdout.flush()
    try:
        test_data = stream.read(constants.CHUNK_SIZE, exception_on_overflow=False)
        logger.info(f"✅ Audio stream test successful, read {len(test_data)} bytes")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"❌ Audio stream test failed: {e}")
        sys.stdout.flush()
        raise

    return stream
```

**In main(), replace lines 550-731 with**:
```python
    # Initialize audio source - either WAV file or microphone
    audio_manager = None
    usb_mic = None

    if not args.input_wav:
        # Initialize AudioManager for audio device handling
        audio_manager = AudioManager()
        logger.info("AudioManager initialized")

        # Find the USB microphone
        usb_mic = audio_manager.find_usb_microphone()
        if not usb_mic:
            logger.error("No USB microphone found. Exiting.")
            raise RuntimeError("No USB microphone detected")

        logger.info(f"Using USB microphone: {usb_mic.name} (index {usb_mic.index})")

    # Handle recording mode (special mode that exits early)
    if args.record_test:
        # Generate timestamp filename if not provided
        if args.audio_file is None:
            audio_filename = generate_timestamp_filename()
        else:
            audio_filename = args.audio_file

        # Initialize model for recording (needed for real-time detection)
        logger.info("Creating Model for recording with real-time detection...")
        model = openwakeword.Model(
            wakeword_models=['hey_jarvis', 'alexa', 'hey_mycroft'],
            vad_threshold=system_config.vad_threshold
        )

        success, metadata = record_test_audio(audio_manager, usb_mic, model, settings_manager, audio_filename)
        if success:
            logger.info("✅ Recording completed successfully. Exiting.")
        else:
            logger.error("❌ Recording failed. Exiting.")
        return

    # Setup audio input
    stream = setup_audio_input(args, audio_config, audio_manager, usb_mic)
```

### 2. setup_wake_word_models()

**Purpose**: Load OpenWakeWord models from configuration and initialize the Model instance.

**Function Signature**:
```python
def setup_wake_word_models(
    models_config,
    system_config,
    settings_manager: SettingsManager,
    shared_data: dict
):
    """
    Initialize OpenWakeWord models from configuration.

    Args:
        models_config: List of model configurations from SettingsManager
        system_config: System configuration (contains vad_threshold)
        settings_manager: SettingsManager instance for updating configs
        shared_data: Shared data dict for web GUI

    Returns:
        tuple: (model, active_model_configs, model_name_mapping, enabled_models)
            - model: OpenWakeWord Model instance
            - active_model_configs: Dict mapping config names to ModelConfig objects
            - model_name_mapping: Dict mapping OpenWakeWord keys to config names
            - enabled_models: List of enabled ModelConfig objects

    Raises:
        ValueError: If no valid models found
        Exception: If model creation or testing fails
    """
```

**Extracted Code** (lines 610-697):
```python
def setup_wake_word_models(models_config, system_config, settings_manager, shared_data):
    """
    Initialize OpenWakeWord models from configuration.

    Args:
        models_config: List of model configurations from SettingsManager
        system_config: System configuration (contains vad_threshold)
        settings_manager: SettingsManager instance for updating configs
        shared_data: Shared data dict for web GUI

    Returns:
        tuple: (model, active_model_configs, model_name_mapping, enabled_models)
            - model: OpenWakeWord Model instance
            - active_model_configs: Dict mapping config names to ModelConfig objects
            - model_name_mapping: Dict mapping OpenWakeWord keys to config names
            - enabled_models: List of enabled ModelConfig objects

    Raises:
        ValueError: If no valid models found
        Exception: If model creation or testing fails
    """
    try:
        # Get enabled models from configuration
        enabled_models = [model for model in models_config if model.enabled]
        if not enabled_models:
            logger.warning("⚠️  No enabled models found in configuration")
            # Auto-enable the first model if none are enabled
            if models_config:
                first_model = models_config[0]
                logger.info(f"Auto-enabling first available model: {first_model.name}")
                settings_manager.update_model_config(first_model.name, enabled=True)
                settings_manager.save()
                # Refresh config
                models_config = settings_manager.get_models_config()
                enabled_models = [model for model in models_config if model.enabled]
            else:
                raise ValueError("No models available in configuration")

        # Load ALL enabled models
        model_paths = []
        active_model_configs = {}
        model_name_mapping = {}  # Maps OpenWakeWord prediction keys to config names

        for model_cfg in enabled_models:
            if os.path.exists(model_cfg.path):
                logger.info(f"✅ Loading model: {model_cfg.name} from {model_cfg.path}")
                model_paths.append(model_cfg.path)
                active_model_configs[model_cfg.name] = model_cfg

                # Create mapping for OpenWakeWord prediction key to config name
                base_name = os.path.basename(model_cfg.path).replace('.tflite', '').replace('.onnx', '')
                model_name_mapping[base_name] = model_cfg.name
                logger.debug(f"   Model name mapping: '{base_name}' -> '{model_cfg.name}'")
            else:
                logger.error(f"❌ Model file NOT found: {model_cfg.path}")

        if not model_paths:
            raise ValueError("No valid model files found")

        logger.info(f"Creating OpenWakeWord with {len(model_paths)} models: {list(active_model_configs.keys())}")

        model = openwakeword.Model(
            wakeword_models=model_paths,
            vad_threshold=system_config.vad_threshold,
            enable_speex_noise_suppression=False
        )

        # Update shared data with loaded models info
        shared_data['loaded_models'] = list(active_model_configs.keys())
        shared_data['models_config'] = {name: {
            'enabled': True,
            'threshold': cfg.threshold,
            'webhook_url': cfg.webhook_url
        } for name, cfg in active_model_configs.items()}

        logger.info("OpenWakeWord model initialized")

        # Check what models are actually loaded
        if hasattr(model, 'models'):
            logger.info(f"Loaded models: {list(model.models.keys()) if model.models else 'None'}")
        logger.info(f"Prediction buffer keys: {list(model.prediction_buffer.keys())}")

        # Enhanced model verification - test with dummy audio
        logger.info("🔍 Testing model with dummy audio to verify initialization...")
        test_audio = np.zeros(constants.CHUNK_SIZE, dtype=np.float32)
        try:
            test_predictions = model.predict(test_audio)
            logger.info(f"✅ Model test successful - prediction type: {type(test_predictions)}")
            logger.info(f"   Test prediction content: {test_predictions}")
            logger.info(f"   Test prediction keys: {list(test_predictions.keys())}")

            # Check prediction_buffer after first prediction
            if hasattr(model, 'prediction_buffer'):
                logger.info(f"✅ Prediction buffer populated after test prediction")
                logger.info(f"   Prediction buffer keys: {list(model.prediction_buffer.keys())}")
                for key, scores in model.prediction_buffer.items():
                    latest_score = scores[-1] if scores else 'N/A'
                    logger.info(f"     Model '{key}': {len(scores)} scores, latest: {latest_score}")
            else:
                logger.warning("⚠️ Prediction buffer not available after test prediction")

        except Exception as e:
            logger.error(f"❌ Error testing model after creation: {e}")
            raise

        # Force log flush
        sys.stdout.flush()

        return model, active_model_configs, model_name_mapping, enabled_models

    except Exception as e:
        print(f"ERROR: Model creation failed: {e}", flush=True)
        raise
```

**In main(), replace lines 610-700 with**:
```python
    # Initialize the OpenWakeWord model with enabled models from configuration
    model, active_model_configs, model_name_mapping, enabled_models = setup_wake_word_models(
        models_config, system_config, settings_manager, shared_data
    )
```

### 3. setup_heartbeat_sender()

**Purpose**: Initialize and start the heartbeat sender for ORAC STT integration.

**Function Signature**:
```python
def setup_heartbeat_sender(enabled_models) -> HeartbeatSender:
    """
    Initialize and start heartbeat sender for ORAC STT integration.

    Args:
        enabled_models: List of enabled ModelConfig objects

    Returns:
        HeartbeatSender: Started heartbeat sender instance
    """
```

**Extracted Code** (lines 702-719):
```python
def setup_heartbeat_sender(enabled_models):
    """
    Initialize and start heartbeat sender for ORAC STT integration.

    Args:
        enabled_models: List of enabled ModelConfig objects

    Returns:
        HeartbeatSender: Started heartbeat sender instance
    """
    # Initialize heartbeat sender for ORAC STT integration
    logger.info("💓 Initializing heartbeat sender for ORAC STT...")
    heartbeat_sender = HeartbeatSender()

    # Register all enabled models with the heartbeat sender
    for model_cfg in enabled_models:
        # Use the actual topic from the model configuration
        heartbeat_sender.register_model(
            name=model_cfg.name,
            topic=model_cfg.topic,  # Use the configured topic (e.g., "general", "Alexa__5")
            wake_word=model_cfg.name,  # Use the full model name as wake word
            enabled=model_cfg.enabled
        )
        logger.info(f"   Registered model '{model_cfg.name}' with topic '{model_cfg.topic}'")

    # Start heartbeat sender
    heartbeat_sender.start()
    logger.info("✅ Heartbeat sender started")

    return heartbeat_sender
```

**In main(), replace lines 702-719 with**:
```python
    # Initialize heartbeat sender for ORAC STT integration
    heartbeat_sender = setup_heartbeat_sender(enabled_models)
```

### 4. setup_web_server()

**Purpose**: Initialize and start the web server with WebSocket broadcaster.

**Function Signature**:
```python
def setup_web_server(
    settings_manager: SettingsManager,
    shared_data: dict,
    event_queue: Queue
):
    """
    Initialize and start web server with WebSocket broadcaster.

    Args:
        settings_manager: SettingsManager instance for routes
        shared_data: Shared data dict for web GUI
        event_queue: Queue for detection events

    Returns:
        WebSocketBroadcaster: Started broadcaster instance
    """
```

**Extracted Code** (lines 733-760):
```python
def setup_web_server(settings_manager, shared_data, event_queue):
    """
    Initialize and start web server with WebSocket broadcaster.

    Args:
        settings_manager: SettingsManager instance for routes
        shared_data: Shared data dict for web GUI
        event_queue: Queue for detection events

    Returns:
        WebSocketBroadcaster: Started broadcaster instance
    """
    # Initialize web server in a separate thread
    logger.info("🌐 Starting web server on port 7171...")

    # Create Flask app FIRST
    app = create_app()

    # Then initialize routes with shared resources
    init_routes(settings_manager, shared_data, event_queue)

    # Create and start WebSocket broadcaster
    broadcaster = WebSocketBroadcaster(socketio, shared_data, event_queue)
    broadcaster.start()

    # Start Flask-SocketIO server in a separate thread
    web_thread = threading.Thread(
        target=socketio.run,
        args=(app,),
        kwargs={'host': '0.0.0.0', 'port': 7171, 'debug': False, 'allow_unsafe_werkzeug': True}
    )
    web_thread.daemon = True
    web_thread.start()

    logger.info("✅ Web server started successfully")

    # Update shared data
    shared_data['status'] = 'Connected'
    shared_data['is_active'] = True
    shared_data['status_changed'] = True

    return broadcaster
```

**In main(), replace lines 733-760 with**:
```python
    # Initialize web server in a separate thread
    broadcaster = setup_web_server(settings_manager, shared_data, event_queue)
```

### 5. setup_stt_components()

**Purpose**: Initialize STT components (ring buffer, STT client, speech recorder).

**Function Signature**:
```python
def setup_stt_components(
    stt_config,
    audio_config,
    active_model_configs: dict,
    shared_data: dict
):
    """
    Initialize STT components (ring buffer, client, speech recorder).

    Args:
        stt_config: STT configuration from SettingsManager
        audio_config: Audio configuration from SettingsManager
        active_model_configs: Dict of active model configurations
        shared_data: Shared data dict for web GUI

    Returns:
        tuple: (ring_buffer, speech_recorder, stt_client, stt_health_status)
            - ring_buffer: RingBuffer instance for pre-roll audio
            - speech_recorder: SpeechRecorder instance
            - stt_client: STTClient instance
            - stt_health_status: Initial health status string
    """
```

**Extracted Code** (lines 762-828):
```python
def setup_stt_components(stt_config, audio_config, active_model_configs, shared_data):
    """
    Initialize STT components (ring buffer, client, speech recorder).

    Args:
        stt_config: STT configuration from SettingsManager
        audio_config: Audio configuration from SettingsManager
        active_model_configs: Dict of active model configurations
        shared_data: Shared data dict for web GUI

    Returns:
        tuple: (ring_buffer, speech_recorder, stt_client, stt_health_status)
            - ring_buffer: RingBuffer instance for pre-roll audio
            - speech_recorder: SpeechRecorder instance
            - stt_client: STTClient instance
            - stt_health_status: Initial health status string
    """
    # Always initialize STT components
    logger.info("🎙️ Initializing STT components...")
    logger.debug(f"STT Configuration: base_url={stt_config.base_url}, timeout={stt_config.timeout}s")
    logger.debug(f"STT Endpoint settings: pre_roll={stt_config.pre_roll_duration}s, silence_threshold={stt_config.silence_threshold}, silence_duration={stt_config.silence_duration}s")

    # Initialize ring buffer for pre-roll audio
    ring_buffer = RingBuffer(
        capacity_seconds=constants.RING_BUFFER_SECONDS,  # Keep 10 seconds of audio history
        sample_rate=audio_config.sample_rate
    )
    logger.debug(f"RingBuffer initialized with capacity={constants.RING_BUFFER_SECONDS}s, sample_rate={audio_config.sample_rate}Hz")

    # Initialize STT client
    stt_client = STTClient(
        base_url=stt_config.base_url,
        timeout=stt_config.timeout
    )

    # Check STT service health
    logger.debug(f"Checking STT service health at {stt_config.base_url}/stt/v1/health")
    if stt_client.health_check():
        logger.info("✅ STT service is healthy")
        logger.debug("STT health check passed, service is ready for transcription")
    else:
        logger.warning("⚠️ STT service is not healthy at startup, but will proceed with per-model webhook URLs")

    # Always initialize speech recorder (for per-model webhook URLs)
    endpoint_config = EndpointConfig(
        silence_threshold=stt_config.silence_threshold,
        silence_duration=stt_config.silence_duration,
        grace_period=stt_config.grace_period,
        max_duration=stt_config.max_recording_duration,
        pre_roll=stt_config.pre_roll_duration
    )

    speech_recorder = SpeechRecorder(
        ring_buffer=ring_buffer,
        stt_client=stt_client,
        endpoint_config=endpoint_config
    )

    logger.info("✅ STT components initialized successfully")

    # Perform health checks for models with webhook URLs
    logger.info("🏥 Performing per-model STT health checks...")
    for name, cfg in active_model_configs.items():
        if cfg.webhook_url:
            logger.debug(f"Checking STT health for model '{name}' at {cfg.webhook_url}")
            if stt_client.health_check(webhook_url=cfg.webhook_url):
                logger.info(f"✅ STT healthy for model '{name}'")
            else:
                logger.warning(f"⚠️ STT unhealthy for model '{name}' at {cfg.webhook_url}")

    # Update initial STT health status
    stt_health_status = check_all_stt_health(active_model_configs, stt_client)
    shared_data['stt_health'] = stt_health_status
    shared_data['status_changed'] = True
    logger.info(f"🏥 Initial STT health status: {stt_health_status}")

    return ring_buffer, speech_recorder, stt_client, stt_health_status
```

**In main(), replace lines 762-828 with**:
```python
    # Get STT configuration
    with settings_manager.get_config() as config:
        stt_config = config.stt

    # Initialize STT components
    ring_buffer, speech_recorder, stt_client, stt_health_status = setup_stt_components(
        stt_config, audio_config, active_model_configs, shared_data
    )
```

## Implementation Steps

### Step 1: Create New Functions (25 minutes)

Add the 5 new functions to `wake_word_detection.py` **ABOVE** the `main()` function (around line 400, after the helper functions like `check_all_stt_health()` but before `def main():`).

**Order of functions**:
1. `setup_audio_input()` - Lines ~400-450
2. `setup_wake_word_models()` - Lines ~451-550
3. `setup_heartbeat_sender()` - Lines ~551-575
4. `setup_web_server()` - Lines ~576-610
5. `setup_stt_components()` - Lines ~611-680

### Step 2: Update main() Function (15 minutes)

Replace the extracted sections in `main()` with calls to the new functions:

1. **Lines 550-731**: Replace audio setup with call to `setup_audio_input()`
2. **Lines 610-700**: Replace model setup with call to `setup_wake_word_models()`
3. **Lines 702-719**: Replace heartbeat setup with call to `setup_heartbeat_sender()`
4. **Lines 733-760**: Replace web server setup with call to `setup_web_server()`
5. **Lines 762-828**: Replace STT setup with call to `setup_stt_components()`

**IMPORTANT**: Keep the `reload_models()` nested function and the detection loop intact for now (Sprint 9 will handle those).

### Step 3: Deploy and Test (10 minutes)

**Commit and deploy**:
```bash
./scripts/deploy_and_test.sh "Sprint 8: Extract setup functions from main() - Part 1"
```

**What to verify**:
1. Container builds and starts successfully
2. All setup functions execute without errors
3. Wake word detection still works (say "hey computer" or your wake word)
4. Web interface accessible on port 7171
5. STT integration works (audio sent to ORAC STT after wake word)
6. Heartbeat sender is running
7. No regression in functionality

**Check logs**:
```bash
# Watch real-time logs
ssh pi "cd ~/hey-orac && docker-compose logs -f hey-orac"

# Check last 50 lines
ssh pi "cd ~/hey-orac && docker-compose logs --tail=50 hey-orac"

# Check for errors
ssh pi "cd ~/hey-orac && docker-compose logs hey-orac | grep -i error"
```

### Step 4: Update Documentation (5 minutes)

**Update CLEANUP.md**:
- Mark Sprint 8 as complete
- Update statistics (8/14 complete = 57%)
- Add completion notes

**Update devlog.md**:
Add entry at the bottom:
```markdown
## 2025-10-16 - Sprint 8: Refactored Main Function - Part 1

- Extracted 5 setup functions from massive main() function in wake_word_detection.py:
  - `setup_audio_input()` - Initialize audio source (WAV file or microphone)
  - `setup_wake_word_models()` - Load OpenWakeWord models from configuration
  - `setup_heartbeat_sender()` - Initialize heartbeat sender for ORAC STT
  - `setup_web_server()` - Initialize web server with WebSocket broadcaster
  - `setup_stt_components()` - Initialize STT components (ring buffer, client, recorder)
- Reduced main() function from ~1200 lines to ~900 lines (25% reduction)
- Each setup function is well-documented with clear purpose and return values
- All functions tested independently - no regression in functionality
- Deployed and tested on Raspberry Pi - all functionality working
- Status: Sprint 8 complete (8/14 sprints = 57% complete)
- Next: Sprint 9 will extract reload_models() and run_detection_loop()
```

**Commit documentation**:
```bash
git add CLEANUP.md devlog.md
git commit -m "Update CLEANUP.md and devlog.md: Sprint 8 complete (8/14)"
git push
```

## Expected Outcome

After Sprint 8:
- ✅ Created 5 well-documented setup functions
- ✅ Reduced main() complexity by ~300 lines (25% reduction)
- ✅ Each setup function has single responsibility
- ✅ Clear return values for each function
- ✅ All functions tested - no regression
- ✅ Application still works perfectly on Pi
- ✅ CLEANUP.md updated (8/14 sprints complete = 57%)
- ✅ devlog.md updated with sprint details
- ✅ Ready for Sprint 9 (extract detection loop)

## Known Good State

The application currently works with these features:
- ✅ Wake word detection (OpenWakeWord)
- ✅ Audio processing (stereo→mono conversion via conversion.py)
- ✅ Multi-consumer audio distribution (AudioReaderThread)
- ✅ STT integration with ORAC STT service
- ✅ Web interface (Flask-SocketIO on port 7171)
- ✅ Heartbeat sender (registers models with ORAC STT)
- ✅ Configuration management (SettingsManager)
- ✅ Health checks (STT, audio thread, models)
- ✅ Constants extracted (42 named constants from Sprint 6)
- ✅ Audio conversion consolidated (Sprint 7)

**Don't break these!** Test thoroughly after changes.

## Deployment and Testing Process

**CRITICAL**: The application runs on the Raspberry Pi, **NOT locally**. Always use the deployment script.

### Primary Deployment Command
```bash
./scripts/deploy_and_test.sh "Sprint 8: Extract setup functions from main() - Part 1"
```

**What `deploy_and_test.sh` does**:
1. ✅ Commits changes locally with your message
2. ✅ Pushes to current branch (`code-cleanup`) on GitHub
3. ✅ SSHs to Raspberry Pi (alias: `pi`)
4. ✅ Pulls latest code from GitHub on the Pi
5. ✅ Builds Docker container (with smart caching)
6. ✅ Starts container with `docker-compose up -d`
7. ✅ Runs health checks
8. ✅ Shows initial logs

### SSH Commands for Pi Monitoring

```bash
# View real-time logs (most useful for debugging)
ssh pi "cd ~/hey-orac && docker-compose logs -f hey-orac"

# View last 30 lines of logs
ssh pi "cd ~/hey-orac && docker-compose logs --tail=30 hey-orac"

# Check container status
ssh pi "cd ~/hey-orac && docker-compose ps"

# Check current git commit on Pi
ssh pi "cd ~/hey-orac && git log -1 --oneline"

# Restart container (without rebuild)
ssh pi "cd ~/hey-orac && docker-compose restart hey-orac"

# Stop container
ssh pi "cd ~/hey-orac && docker-compose down"

# Force full rebuild manually
ssh pi "cd ~/hey-orac && docker-compose down && docker-compose build --no-cache && docker-compose up -d"
```

## Deployment Troubleshooting

**If deployment fails**:
```bash
# Check what went wrong
ssh pi "cd ~/hey-orac && docker-compose logs --tail=100 hey-orac"

# Rollback locally
git reset --hard HEAD^

# Force deploy previous commit
./scripts/deploy_and_test.sh "Rollback: reverting Sprint 8 changes"
```

**If wake word detection stops working**:
- Check logs for errors in setup functions
- Verify all setup functions return expected values
- Ensure no exceptions during initialization
- Check that shared_data is updated correctly
- Rollback and compare initialization order

## Important Notes

1. **Always use `deploy_and_test.sh`** - Never test locally, the app runs on Pi
2. **Check logs after deployment** - Use `ssh pi "cd ~/hey-orac && docker-compose logs --tail=30 hey-orac"`
3. **Test wake word detection** - Say the wake word and verify it's detected in logs
4. **Verify web interface** - Access port 7171 and check status
5. **Keep reload_models() for Sprint 9** - Don't touch the nested function in this sprint
6. **Keep detection loop for Sprint 9** - Don't touch the while True loop in this sprint
7. **Update both CLEANUP.md and devlog.md** - After completing sprint
8. **Commit descriptively** - Use format: "Sprint 8: Extract setup functions from main() - Part 1"

## Historical Context

The main() function grew organically as features were added:
- Started simple with just wake word detection
- Added web interface, STT integration, heartbeat sender
- Added health checks, thread monitoring, config reload
- Now ~1200 lines - too complex to understand and maintain

This refactoring will make the code:
- Easier to understand (each function has clear purpose)
- Easier to test (can test setup functions independently)
- Easier to maintain (changes localized to specific functions)
- Easier to extend (can add new setup functions without touching main())

## Next Steps After Sprint 8

Sprint 9 will extract the remaining complex parts:
- Extract `reload_models()` nested function as standalone function
- Extract main detection loop into `run_detection_loop()` function
- Final main() will be ~200 lines of orchestration code

---

## Quick Command Reference

```bash
# Deploy changes
./scripts/deploy_and_test.sh "Sprint 8: Extract setup functions from main() - Part 1"

# Watch logs in real-time
ssh pi "cd ~/hey-orac && docker-compose logs -f hey-orac"

# Check last 30 log lines
ssh pi "cd ~/hey-orac && docker-compose logs --tail=30 hey-orac"

# Check container status
ssh pi "cd ~/hey-orac && docker-compose ps"

# Check deployed commit
ssh pi "cd ~/hey-orac && git log -1 --oneline"

# Rollback if broken
git reset --hard HEAD^
./scripts/deploy_and_test.sh "Rollback: reverting broken changes"
```

---

**Ready to start Sprint 8!** Extract the 5 setup functions, update main(), deploy, and test thoroughly. This is a significant refactoring - take your time and test carefully.
