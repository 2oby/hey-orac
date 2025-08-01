// WebSocket connection and real-time updates
const API_BASE = '/api';
let socket = null;
let sampleModels = [];
let currentEditingModel = null;
let currentVolume = 0;
let volumeHistory = [];
let reconnectTimer = null;

// Test function for manual RMS update
function testRMSUpdate(testValue = 100) {
    console.log('Testing RMS update with value:', testValue);
    updateVolume(testValue);
}

// Initialize WebSocket connection
function initWebSocket() {
    console.log('Initializing WebSocket connection...');
    
    socket = io({
        transports: ['websocket', 'polling'], // Prefer websocket over polling
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionAttempts: 10,
        timeout: 120000,
        pingTimeout: 120000,
        pingInterval: 25000,
        upgrade: true, // Ensure upgrade from polling to websocket
        rememberUpgrade: true
    });

    socket.on('connect', () => {
        console.log('WebSocket connected successfully!');
        console.log('Socket ID:', socket.id);
        console.log('Transport:', socket.io.engine.transport.name);
        updateConnectionStatus(true);
        
        // Subscribe to real-time updates
        socket.emit('subscribe_updates');
        console.log('Sent subscribe_updates event');
        
        // Send periodic pings to keep connection alive
        if (window.pingInterval) {
            clearInterval(window.pingInterval);
        }
        window.pingInterval = setInterval(() => {
            socket.emit('ping', { timestamp: Date.now() });
        }, 10000); // Every 10 seconds
    });

    socket.on('disconnect', () => {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
        
        // Clear ping interval
        if (window.pingInterval) {
            clearInterval(window.pingInterval);
            window.pingInterval = null;
        }
    });

    socket.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        updateConnectionStatus(false);
    });

    socket.on('subscribed', (data) => {
        console.log('Received subscribed confirmation:', data);
        console.log('Should now receive RMS updates...');
    });
    
    socket.on('pong', (data) => {
        console.log('Received pong response:', data);
    });
    
    socket.on('reconnect', (attemptNumber) => {
        console.log('Reconnected after', attemptNumber, 'attempts');
        // Re-subscribe after reconnection
        socket.emit('subscribe_updates');
    });

    socket.on('rms_update', (data) => {
        // No logging for RMS updates to reduce console noise
        updateVolume(data.rms);
    });

    // Debug: Log all Socket.IO events (except RMS updates)
    socket.onAny((eventName, ...args) => {
        if (eventName !== 'rms_update') {
            console.log('Socket.IO event received:', eventName, args);
        }
    });

    socket.on('detection', (data) => {
        handleDetection(data);
    });

    socket.on('status_update', (data) => {
        updateSystemStatus(data);
    });

    socket.on('config_changed', () => {
        loadConfig();
    });
}

// Update connection status in UI
function updateConnectionStatus(connected) {
    const statusElement = document.querySelector('.connection-status');
    if (statusElement) {
        statusElement.textContent = connected ? 'Connected' : 'Disconnected';
        statusElement.style.color = connected ? '#00ff41' : '#ff0000';
    }
}

// Update system status
function updateSystemStatus(status) {
    const listeningElement = document.querySelector('.listening-status');
    if (listeningElement) {
        listeningElement.textContent = status.listening ? 'Listening' : 'Not Listening';
    }

    const activeElement = document.querySelector('.audio-status');
    if (activeElement) {
        activeElement.textContent = status.active ? `Active (RMS: ${currentVolume.toFixed(0)})` : 'Inactive';
    }
}

// Handle detection events
function handleDetection(detection) {
    console.log('Wake word detected:', detection);
    
    // Find the model card
    const modelCard = document.querySelector(`[data-model="${detection.model}"]`);
    if (modelCard) {
        // Add detection animation
        modelCard.classList.add('detected');
        setTimeout(() => modelCard.classList.remove('detected'), 1000);
    }
    
    // Update last detection
    const lastDetectionElement = document.querySelector('.last-detection');
    if (lastDetectionElement) {
        const time = new Date(detection.timestamp).toLocaleTimeString();
        lastDetectionElement.textContent = `${detection.model} at ${time} (${(detection.confidence * 100).toFixed(1)}%)`;
    }
    
    // No audio notification (device has no speaker)
}

// Update volume display
function updateVolume(rms) {
    // Reduced RMS logging
    currentVolume = rms;
    
    // Update the Current RMS display text
    const currentRmsElement = document.getElementById('currentRms');
    if (currentRmsElement) {
        currentRmsElement.textContent = `Current RMS: ${rms.toFixed(2)}`;
    }
    
    updateVolumeDisplay();
}

// API functions
async function loadConfig() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        const config = await response.json();
        
        console.log('Loaded config:', config);
        console.log('Models found:', config.models?.length || 0);
        
        // Convert config to our model format, handling duplicates by prioritizing .tflite models
        const modelMap = new Map();
        config.models.forEach(model => {
            const key = model.name;
            const convertedModel = {
                name: model.name,
                active: model.enabled,
                threshold: model.threshold,
                apiUrl: model.webhook_url,
                framework: model.framework,
                path: model.path
            };
            
            // If we already have this model name, prefer .tflite over .onnx
            if (modelMap.has(key)) {
                const existing = modelMap.get(key);
                // Replace if current is .tflite and existing is .onnx, or if current is enabled
                if ((model.framework === 'tflite' && existing.framework === 'onnx') || 
                    model.enabled) {
                    modelMap.set(key, convertedModel);
                }
            } else {
                modelMap.set(key, convertedModel);
            }
        });
        
        sampleModels = Array.from(modelMap.values());
        
        console.log('Converted models:', sampleModels);
        
        // Update global settings if available
        if (config.system) {
            const rmsSlider = document.getElementById('rms-filter');
            const cooldownSlider = document.getElementById('cooldown');
            
            if (rmsSlider && config.system.rms_filter !== undefined) {
                rmsSlider.value = rmsToSlider(config.system.rms_filter);
                updateSliderDisplay('rms-filter', config.system.rms_filter);
            }
            
            if (cooldownSlider && config.system.cooldown !== undefined) {
                cooldownSlider.value = config.system.cooldown;
                updateSliderDisplay('cooldown', config.system.cooldown);
            }
            
            // Store VAD threshold for global settings modal
            if (config.system.vad_threshold !== undefined) {
                window.currentVadThreshold = config.system.vad_threshold;
            }
        }
        
        updateModelCards();
    } catch (error) {
        console.error('Error loading config:', error);
    }
}

async function saveGlobalSettings() {
    const rmsValue = sliderToRMS(document.getElementById('rms-filter').value);
    const cooldownValue = parseFloat(document.getElementById('cooldown').value);
    
    try {
        const response = await fetch(`${API_BASE}/config/global`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rms_filter: rmsValue,
                cooldown: cooldownValue
            })
        });
        
        if (response.ok) {
            console.log('Global settings saved');
        }
    } catch (error) {
        console.error('Error saving global settings:', error);
    }
}

async function toggleModel(modelName) {
    const model = sampleModels.find(m => m.name === modelName);
    if (!model) return;
    
    const newState = !model.active;
    const endpoint = newState ? 'activate' : 'deactivate';
    
    try {
        const response = await fetch(`${API_BASE}/custom-models/${modelName}/${endpoint}`, {
            method: 'POST'
        });
        
        if (response.ok) {
            model.active = newState;
            updateModelCards();
        }
    } catch (error) {
        console.error('Error toggling model:', error);
    }
}

async function saveModelSettings() {
    if (!currentEditingModel) return;
    
    const threshold = parseFloat(document.getElementById('model-threshold').value);
    const apiUrl = document.getElementById('model-api-url').value;
    
    console.log('Saving model settings:', {model: currentEditingModel, threshold, apiUrl});
    
    try {
        const response = await fetch(`${API_BASE}/config/models/${currentEditingModel}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                threshold: threshold,
                webhook_url: apiUrl
            })
        });
        
        if (response.ok) {
            console.log('Model settings saved successfully');
            // Update the local model data
            const model = sampleModels.find(m => m.name === currentEditingModel);
            if (model) {
                model.threshold = threshold;
                model.apiUrl = apiUrl;
                console.log('Updated local model:', model);
            }
            closeModelSettings();
        } else {
            console.error('Failed to save model settings:', response.status);
        }
    } catch (error) {
        console.error('Error saving model settings:', error);
    }
}

// UI update functions
function updateModelCards() {
    const grid = document.getElementById('models-grid');
    console.log('updateModelCards called, grid element:', grid);
    console.log('sampleModels to display:', sampleModels);
    
    if (!grid) {
        console.error('models-grid element not found!');
        return;
    }
    
    grid.innerHTML = '';
    
    sampleModels.forEach(model => {
        const card = document.createElement('div');
        card.className = `model-card ${model.active ? 'active' : ''}`;
        card.dataset.model = model.name;
        // Removed onclick - now handled by activate button
        
        card.innerHTML = `
            <div class="model-name-header">
                <h3 class="model-name">${model.name}</h3>
            </div>
            <div class="model-footer">
                <button class="settings-cog" onclick="event.stopPropagation(); openModelSettings('${model.name}')">
                    <svg viewBox="0 0 20 20" fill="currentColor" width="20" height="20">
                        <path fill-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd" />
                    </svg>
                </button>
                <button class="activate-btn ${model.active ? 'active' : ''}" onclick="event.stopPropagation(); toggleModel('${model.name}')">${model.active ? 'ACTIVE' : 'ACTIVATE'}</button>
            </div>
        `;
        
        grid.appendChild(card);
    });
}

function openModelSettings(modelName) {
    currentEditingModel = modelName;
    // Find the model, prioritizing active/enabled models over inactive ones
    let model = sampleModels.find(m => m.name === modelName && m.active);
    if (!model) {
        // If no active model found, take the first match
        model = sampleModels.find(m => m.name === modelName);
    }
    
    if (model) {
        console.log('Opening settings for model:', model);
        document.getElementById('model-name').textContent = model.name;
        
        // Set slider values and verify they were set correctly
        const thresholdSlider = document.getElementById('model-threshold');
        const apiUrlField = document.getElementById('model-api-url');
        
        thresholdSlider.value = model.threshold;
        apiUrlField.value = model.apiUrl || '';
        
        console.log('Set threshold slider to:', model.threshold, 'actual value:', thresholdSlider.value);
        
        updateSliderDisplay('model-threshold', model.threshold);
        
        // Attach event listeners for the model sliders (in case they weren't attached yet)
        console.log('Attaching event listeners to model sliders');
        console.log('Threshold slider element:', thresholdSlider);
        
        // Test if sliders are the right elements
        console.log('Threshold slider ID:', thresholdSlider.id);
        
        thresholdSlider.removeEventListener('input', updateModelThreshold);
        thresholdSlider.addEventListener('input', updateModelThreshold);
        console.log('Event listeners attached');
        
        document.getElementById('model-settings-modal').style.display = 'block';
    }
}

// Event handlers for model sliders

function updateModelThreshold() {
    console.log('updateModelThreshold called with value:', this.value);
    updateSliderDisplay('model-threshold', this.value);
}

function closeModelSettings() {
    document.getElementById('model-settings-modal').style.display = 'none';
    currentEditingModel = null;
}

async function openGlobalSettings() {
    // Get current values from existing elements
    const rmsValue = document.getElementById('rms-filter-value').textContent;
    const cooldownValue = document.getElementById('cooldown-value').textContent;
    
    // Set modal values
    document.getElementById('volume-rms-filter').value = document.getElementById('rms-filter').value;
    document.getElementById('volume-cooldown').value = document.getElementById('cooldown').value;
    document.getElementById('volume-rms-filter-value').textContent = rmsValue;
    document.getElementById('volume-cooldown-value').textContent = cooldownValue;
    
    // Set VAD threshold value
    const vadThreshold = window.currentVadThreshold || 0.5;
    document.getElementById('volume-vad-threshold').value = vadThreshold;
    document.getElementById('volume-vad-threshold-value').textContent = vadThreshold.toFixed(2);
    
    // Fetch and set multi-trigger value
    try {
        const response = await fetch(`${API_BASE}/config`);
        if (response.ok) {
            const config = await response.json();
            const multiTrigger = config.system?.multi_trigger || false;
            document.getElementById('multi-trigger-checkbox').checked = multiTrigger;
        }
    } catch (error) {
        console.error('Error fetching config:', error);
        document.getElementById('multi-trigger-checkbox').checked = false;
    }
    
    // Show modal
    document.getElementById('volume-settings-modal').classList.add('active');
}

async function closeGlobalSettings() {
    // Save the settings before closing
    const rmsValue = sliderToRMS(parseFloat(document.getElementById('volume-rms-filter').value));
    const cooldownValue = parseFloat(document.getElementById('volume-cooldown').value);
    const vadThreshold = parseFloat(document.getElementById('volume-vad-threshold').value);
    const multiTrigger = document.getElementById('multi-trigger-checkbox').checked;
    
    try {
        const response = await fetch(`${API_BASE}/config/global`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rms_filter: rmsValue,
                cooldown: cooldownValue,
                vad_threshold: vadThreshold,
                multi_trigger: multiTrigger
            })
        });
        
        if (response.ok) {
            console.log('Global settings saved');
            // Update the main sliders to match
            document.getElementById('rms-filter').value = rmsToSlider(rmsValue);
            document.getElementById('cooldown').value = cooldownValue;
            updateSliderDisplay('rms-filter', rmsValue);
            updateSliderDisplay('cooldown', cooldownValue);
            // Store VAD threshold for future use
            window.currentVadThreshold = vadThreshold;
        } else {
            console.error('Failed to save global settings');
        }
    } catch (error) {
        console.error('Error saving global settings:', error);
    }
    
    document.getElementById('volume-settings-modal').classList.remove('active');
}

// RMS slider conversion functions
function rmsToSlider(rms) {
    if (rms <= 0) return 0;
    if (rms >= 5000) return 100;
    
    const midpoint = 50;
    const midpointRMS = 50;
    
    if (rms <= midpointRMS) {
        return (rms / midpointRMS) * midpoint;
    } else {
        const logScale = Math.log(rms / midpointRMS) / Math.log(5000 / midpointRMS);
        return midpoint + logScale * midpoint;
    }
}

function sliderToRMS(sliderValue) {
    sliderValue = parseFloat(sliderValue);
    if (sliderValue <= 0) return 0;
    if (sliderValue >= 100) return 5000;
    
    const midpoint = 50;
    const midpointRMS = 50;
    
    if (sliderValue <= midpoint) {
        return (sliderValue / midpoint) * midpointRMS;
    } else {
        const normalizedValue = (sliderValue - midpoint) / midpoint;
        return midpointRMS * Math.pow(5000 / midpointRMS, normalizedValue);
    }
}

function updateSliderDisplay(sliderId, value) {
    // Handle different display element ID patterns
    let displayId;
    if (sliderId === 'model-threshold') {
        displayId = sliderId + '-display';  // model settings use -display suffix
    } else if (sliderId.startsWith('volume-')) {
        displayId = sliderId + '-value';     // volume modal elements use -value suffix
    } else {
        displayId = sliderId + '-value';     // global settings use -value suffix
    }
    
    const display = document.getElementById(displayId);
    if (display) {
        // Convert value to number to ensure .toFixed() works
        const numValue = parseFloat(value);
        
        if (sliderId === 'rms-filter' || sliderId === 'volume-rms-filter') {
            display.textContent = Math.round(numValue);
        } else if (sliderId === 'cooldown' || sliderId === 'volume-cooldown') {
            display.textContent = numValue.toFixed(1) + 's';
        } else {
            display.textContent = numValue.toFixed(2);  // Show 2 decimals for model settings
        }
    } else {
        console.warn('Display element not found:', displayId, 'for slider:', sliderId);
    }
}

// Volume display
function updateVolumeDisplay() {
    // Removed logging to reduce console noise
    const meter = document.getElementById('volume-meter');
    const segments = meter.querySelectorAll('.volume-segment');
    const filterThreshold = parseFloat(document.getElementById('rms-filter-value').textContent);
    
    // Update volume history
    volumeHistory.push(currentVolume);
    if (volumeHistory.length > 50) volumeHistory.shift();
    
    // Calculate normalized volume (0-12 scale for segments) using logarithmic scaling
    // RMS values: 0 to 5000, mapped logarithmically to 12 segments
    const maxRMS = 5000;
    const minRMS = 1; // Minimum RMS value for logarithmic scale
    const numSegments = 12;
    
    let normalizedVolume;
    if (currentVolume <= minRMS) {
        normalizedVolume = 0;
    } else if (currentVolume >= maxRMS) {
        normalizedVolume = numSegments;
    } else {
        // Logarithmic scaling: log(current/min) / log(max/min) * segments
        normalizedVolume = Math.log(currentVolume / minRMS) / Math.log(maxRMS / minRMS) * numSegments;
    }
    // Reduced logging: normalizedVolume calculation
    
    segments.forEach((segment, index) => {
        const shouldBeActive = index < normalizedVolume;
        segment.classList.toggle('active', shouldBeActive);
        // Reduced segment logging
        
        // Update filter indicator using logarithmic scale
        const segmentPosition = index / numSegments; // 0 to 1
        const segmentRMS = minRMS * Math.pow(maxRMS / minRMS, segmentPosition);
        if (Math.abs(segmentRMS - filterThreshold) < filterThreshold * 0.2) { // 20% tolerance
            segment.style.borderColor = '#00ff41';
        } else {
            segment.style.borderColor = '';
        }
    });
    
    // Update audio status
    document.querySelector('.audio-status').textContent = `Active (RMS: ${currentVolume.toFixed(0)})`;
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Load initial configuration
    loadConfig();
    
    // Initialize WebSocket connection
    initWebSocket();
    
    // Setup event listeners for volume settings modal sliders
    document.getElementById('volume-rms-filter').addEventListener('input', function() {
        const rmsValue = sliderToRMS(this.value);
        updateSliderDisplay('volume-rms-filter', rmsValue);
        // Update hidden original slider
        document.getElementById('rms-filter').value = this.value;
        updateSliderDisplay('rms-filter', rmsValue);
        saveGlobalSettings();
    });
    
    document.getElementById('volume-cooldown').addEventListener('input', function() {
        updateSliderDisplay('volume-cooldown', this.value);
        // Update hidden original slider
        document.getElementById('cooldown').value = this.value;
        updateSliderDisplay('cooldown', this.value);
        saveGlobalSettings();
    });
    
    document.getElementById('volume-vad-threshold').addEventListener('input', function() {
        updateSliderDisplay('volume-vad-threshold', this.value);
        // Store for global use
        window.currentVadThreshold = parseFloat(this.value);
    });

    // Setup event listeners for sliders
    document.getElementById('rms-filter').addEventListener('input', function() {
        const rmsValue = sliderToRMS(this.value);
        updateSliderDisplay('rms-filter', rmsValue);
        saveGlobalSettings();
    });
    
    document.getElementById('cooldown').addEventListener('input', function() {
        updateSliderDisplay('cooldown', this.value);
        saveGlobalSettings();
    });
    
    // Model slider event listeners are now attached dynamically when modal opens
    
    // Modal close button
    document.querySelector('.close-btn').addEventListener('click', closeModelSettings);
    
    // Save model settings button
    document.querySelector('.save-btn').addEventListener('click', saveModelSettings);
    
    // Close modal on outside click
    document.getElementById('model-settings-modal').addEventListener('click', function(e) {
        if (e.target === this) {
            closeModelSettings();
        }
    });
});