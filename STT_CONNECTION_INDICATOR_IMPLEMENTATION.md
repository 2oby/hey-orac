# STT Connection Indicator Implementation Guide

## Overview
Add a visual STT connection status indicator to the Hey ORAC web interface footer. This indicator will show the health status of all configured webhook URLs for STT services.

## Implementation Steps

### Step 1: Add HTML Structure
Add a new status item to the footer in `/src/hey_orac/web/static/index.html` after the last detection item. Use the same HTML pattern as existing status items with a circular indicator div and descriptive text span.

### Step 2: Add CSS Styling
Define styles in `/src/hey_orac/web/static/style.css` for the three states (green/orange/red). Add an orange color definition (e.g., #ff9500) for the partial connection state, maintaining the same glow effect pattern as other indicators.

### Step 3: Create Health Check Logic
Add a new function in `/src/hey_orac/wake_word_detection.py` to check webhook URL health for all enabled models. This should iterate through models, test each webhook_url with a health check request, and return an aggregated status (all healthy, some healthy, none healthy).

### Step 4: Add Health Status to WebSocket Updates
Modify the WebSocket broadcast function to include STT health status in the updates. Add a new field like 'stt_health' with values 'connected', 'partial', or 'disconnected' based on the health check results.

### Step 5: Update JavaScript Handler
Add JavaScript code in `/src/hey_orac/web/static/js/main.js` to handle the STT health status updates. Update the indicator color and text based on the received status, following the same pattern as other status indicators.

### Step 6: Implement Periodic Health Checks
Set up a background task that runs health checks every 30 seconds to keep the status current. Use the existing event loop pattern to schedule periodic checks without blocking the main detection loop.

### Step 7: Handle Edge Cases
Ensure the indicator shows red when no webhook URLs are configured, handles network timeouts gracefully, and updates immediately when settings change. Add appropriate logging for debugging health check failures.

### Step 8: Test All States
Test all three indicator states by configuring various webhook URLs (valid, invalid, none). Verify the indicator updates correctly on startup, during runtime, and when settings are modified through the web interface.

## Success Criteria
- Indicator appears in footer with consistent styling
- Shows green when all STT endpoints are healthy
- Shows orange when some endpoints are healthy
- Shows red when no endpoints are healthy or none configured
- Updates in real-time as health status changes
- No impact on wake word detection performance