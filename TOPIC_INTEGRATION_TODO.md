# Hey ORAC - Topic Integration TODO

**Created**: 2025-08-28  
**Related**: [`/TOPIC_MVP_IMPLEMENTATION.md`](../../TOPIC_MVP_IMPLEMENTATION.md)

## Overview
This document outlines the required changes to Hey ORAC to support the ORAC Topic System MVP.

## Required Changes

### 1. Wake Word Configuration UI Update
**Location**: `src/hey_orac/web/static/index.html` and `src/hey_orac/web/static/js/main.js`

#### Tasks:
- [ ] Add topic selection field to each wake word tile configuration
- [ ] Default topic value: "general"
- [ ] Store topic preference per wake word model
- [ ] Display current topic on each tile

#### Implementation:
```html
<!-- Add to model settings modal -->
<div class="form-group">
    <label for="model-topic">Topic</label>
    <input type="text" id="model-topic" class="form-control" 
           placeholder="e.g., general, home_assistant, music" 
           value="general">
    <small class="form-text text-muted">
        Topic determines AI behavior and configuration in ORAC Core
    </small>
</div>
```

### 2. Webhook URL Modification
**Location**: `src/hey_orac/transport/stt_client.py`

#### Current:
```python
webhook_url = "http://192.168.8.191:7272/stt/v1/stream"
```

#### New:
```python
# Include topic in URL path
topic = model_config.get('topic', 'general')
webhook_url = f"http://192.168.8.191:7272/stt/v1/stream/{topic}"
```

#### Tasks:
- [ ] Modify STT client to include topic in URL path
- [ ] Ensure topic is URL-encoded for safety
- [ ] Handle missing topic with "general" default
- [ ] Update any hardcoded URLs to use dynamic topic

### 3. Model Configuration Storage
**Location**: `config/settings.json`

#### Tasks:
- [ ] Update settings schema to include topic per model
- [ ] Migrate existing configurations to include default topic
- [ ] Save topic selection when model settings are updated

#### Example Configuration:
```json
{
  "models": {
    "hey_computer": {
      "webhook_url": "http://192.168.8.191:7272/stt/v1/stream",
      "topic": "general",
      "sensitivity": 0.5,
      "enabled": true
    },
    "home_assistant_wake": {
      "webhook_url": "http://192.168.8.191:7272/stt/v1/stream",
      "topic": "home_assistant",
      "sensitivity": 0.6,
      "enabled": true
    }
  }
}
```

### 4. JavaScript Updates
**Location**: `src/hey_orac/web/static/js/main.js`

#### Tasks:
- [ ] Load available topics from ORAC Core API on startup
- [ ] Populate topic suggestions/dropdown
- [ ] Save topic selection with model configuration
- [ ] Show current topic on wake word tile

#### Code to Add:
```javascript
// Fetch available topics from ORAC Core
async function loadAvailableTopics() {
    try {
        const response = await fetch('http://192.168.8.191:8000/api/topics');
        const data = await response.json();
        return Object.keys(data.topics);
    } catch (error) {
        console.error('Failed to load topics:', error);
        return ['general']; // Fallback to default
    }
}

// When streaming audio, include topic in URL
function streamAudio(audioData, modelConfig) {
    const topic = modelConfig.topic || 'general';
    const webhookUrl = modelConfig.webhook_url.replace(
        '/stt/v1/stream',
        `/stt/v1/stream/${topic}`
    );
    
    fetch(webhookUrl, {
        method: 'POST',
        body: audioData
    });
}
```

### 5. Python Backend Updates
**Location**: `src/hey_orac/web/routes.py`

#### Tasks:
- [ ] Add endpoint to get/set topic for each model
- [ ] Validate topic names
- [ ] Store topic configuration

### 6. Testing Requirements

#### Manual Testing:
- [ ] Configure different wake words with different topics
- [ ] Verify topic is passed in webhook URL
- [ ] Test with non-existent topic (should auto-create in ORAC Core)
- [ ] Test topic persistence after restart
- [ ] Verify default "general" topic for unconfigured models

#### Integration Testing:
- [ ] Wake word → STT → Core flow with topic
- [ ] Multiple wake words with different topics
- [ ] Topic switching between utterances

## Backward Compatibility

- [ ] Ensure existing configurations work without topics (default to "general")
- [ ] Handle old webhook URLs without breaking
- [ ] Graceful fallback if ORAC Core topics API is unavailable

## Deployment Steps

1. Update Hey ORAC code with topic support
2. Test locally with mock ORAC Core
3. Deploy to Jetson Orin (192.168.8.99)
4. Test with full stack:
   ```bash
   # Test with topic in path
   curl -X POST http://192.168.8.191:7272/stt/v1/stream/home_assistant \
     -F "file=@test_audio.wav"
   ```

## Success Criteria

- [ ] Wake words can be configured with specific topics
- [ ] Topics are correctly passed to ORAC STT
- [ ] UI shows current topic for each wake word
- [ ] Settings persist across restarts
- [ ] No breaking changes to existing deployments

## Timeline

Estimated: 1 day of development
- 2 hours: UI updates
- 2 hours: Backend modifications
- 2 hours: Testing and integration
- 2 hours: Deployment and verification

## Notes

- Topics enable different AI behaviors per wake word
- Auto-discovery means new topics can be created on-the-fly
- Path parameters are cleaner than query parameters
- Default to "general" topic for backward compatibility

---

**Status**: Ready for Implementation  
**Priority**: High (required for Topic System MVP)