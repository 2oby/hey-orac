#!/usr/bin/env python3
"""
Hey Orac Web Backend
Provides REST API for web interface and settings management
"""

from flask import Flask, jsonify, request, send_from_directory
from settings_manager import get_settings_manager
from shared_memory_ipc import shared_memory_ipc
from pathlib import Path
import time
import logging
import os
import json

# Configure logging - use the same configuration as main.py
# Note: logging.basicConfig() can only be called once per process
# The main application already configures logging, so we just get the logger
logger = logging.getLogger(__name__)

# Ensure our logger has DEBUG level enabled
logger.setLevel(logging.INFO)  # Set to INFO for normal operation

# Configure Flask logging to reduce verbosity for frequent polling endpoints only
import logging
from werkzeug.serving import WSGIRequestHandler

# Custom request handler that only logs errors and non-polling endpoints
class QuietWSGIRequestHandler(WSGIRequestHandler):
    def log_request(self, *args, **kwargs):
        # Only log requests that are not frequent polling endpoints
        path = self.path
        if not (path.startswith('/api/activation') or path.startswith('/api/audio/rms')):
            logger.info(f"🌐 HTTP {self.command} {self.path}")

# Create Flask app
app = Flask(__name__)

# Get settings manager instance
settings_manager = get_settings_manager()

# Add explicit debug log to verify logging is working
logger.debug("🔧 Web backend initialized with DEBUG logging enabled")

# Helper functions
def discover_available_models():
    """Get available models from settings manager"""
    return settings_manager.get_available_models()

def get_model_info(model_name):
    """Get information about a specific model from settings manager"""
    model_config = settings_manager.get_model_config(model_name)
    if model_config:
        # Determine which file type exists
        file_paths = model_config['file_paths']
        if Path(file_paths['onnx']).exists():
            return {
                'name': model_name,
                'path': file_paths['onnx'],
                'type': 'onnx'
            }
        elif Path(file_paths['tflite']).exists():
            return {
                'name': model_name,
                'path': file_paths['tflite'],
                'type': 'tflite'
            }
    return None

# Routes
@app.route('/')
def index():
    """Serve the main web interface"""
    return send_from_directory('/app/web', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files from the web directory"""
    return send_from_directory('/app/web', filename)

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    logger.debug("🌐 API: GET /api/config called")
    # Get all models with their settings
    models = {}
    for model_name in discover_available_models():
        model_config = settings_manager.get_model_config(model_name)
        if model_config:
            models[model_name] = {
                "sensitivity": model_config['sensitivity'],
                "threshold": model_config['threshold'],
                "api_url": model_config['api_url'],
                "active": model_config['active']
            }
    
    # Get global settings
    global_settings = {
        "rms_filter": settings_manager.get("volume_monitoring.rms_filter"),
        "cooldown_s": settings_manager.get("wake_word.cooldown")
    }
    
    logger.debug(f"🌐 API: Returning config with {len(models)} models")
    return jsonify({
        "models": models,
        "global": global_settings
    })

@app.route('/api/config/global', methods=['GET'])
def get_global_config():
    """Get global settings"""
    logger.debug("🌐 API: GET /api/config/global called")
    return jsonify(settings_manager.get_all())

@app.route('/api/config/global', methods=['POST'])
def set_global_config():
    """Update global settings"""
    logger.debug("🌐 API: POST /api/config/global called")
    try:
        settings = request.json
        
        # Convert flat keys to nested structure for proper settings management
        converted_settings = {}
        for key, value in settings.items():
            if '.' in key:
                # Convert flat key like "volume_monitoring.rms_filter" to nested structure
                keys = key.split('.')
                current = converted_settings
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                # Keep non-flat keys as-is
                converted_settings[key] = value
        
        if settings_manager.update(converted_settings):
            logger.debug("🌐 API: Global settings updated successfully")
            return jsonify({"status": "success", "message": "Settings updated"})
        else:
            logger.error("🌐 API: Failed to update global settings")
            return jsonify({"status": "error", "message": "Failed to update settings"}), 500
    except Exception as e:
        logger.error(f"🌐 API: Error updating global settings: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/config/settings/<path:key>', methods=['GET'])
def get_setting_value(key):
    """Get a specific setting value"""
    logger.debug(f"🌐 API: GET /api/config/settings/{key} called")
    value = settings_manager.get(key)
    return jsonify({"key": key, "value": value})

@app.route('/api/config/settings/<path:key>', methods=['POST'])
def set_setting_value(key):
    """Set a specific setting value"""
    logger.debug(f"🌐 API: POST /api/config/settings/{key} called")
    try:
        value = request.json.get('value')
        if settings_manager.set(key, value):
            logger.debug(f"🌐 API: Setting {key} = {value} successful")
            return jsonify({"status": "success", "key": key, "value": value})
        else:
            logger.error(f"🌐 API: Failed to set setting {key}")
            return jsonify({"status": "error", "message": "Failed to update setting"}), 500
    except Exception as e:
        logger.error(f"🌐 API: Error setting {key}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/config/models', methods=['GET'])
def get_models():
    """Get all models with their settings"""
    logger.debug("🌐 API: GET /api/config/models called")
    # Get all models with their complete configuration
    models = {}
    for model_name in discover_available_models():
        model_config = settings_manager.get_model_config(model_name)
        if model_config:
            models[model_name] = {
                "sensitivity": model_config['sensitivity'],
                "threshold": model_config['threshold'],
                "api_url": model_config['api_url'],
                "active": model_config['active']
            }
    
    logger.debug(f"🌐 API: Returning {len(models)} models")
    return jsonify({
        "models": models,
        "active_models": settings_manager.get_active_models()
    })

@app.route('/api/config/models/<model_name>', methods=['GET'])
def get_model_config(model_name):
    """Get specific model settings (per-model sensitivity, threshold, API URL, and active state)"""
    logger.debug(f"🌐 API: GET /api/config/models/{model_name} called")
    model_config = settings_manager.get_model_config(model_name)
    if model_config:
        return jsonify({
            "sensitivity": model_config['sensitivity'],
            "threshold": model_config['threshold'],
            "api_url": model_config['api_url'],
            "active": model_config['active'],
            "model": model_name
        })
    else:
        logger.warning(f"🌐 API: Model {model_name} not found")
        return jsonify({"error": f"Model {model_name} not found"}), 404

@app.route('/api/config/models/<model_name>', methods=['POST'])
def set_model_config(model_name):
    """Update specific model settings (per-model sensitivity, threshold, API URL, and active state)"""
    logger.debug(f"🌐 API: POST /api/config/models/{model_name} called with data: {request.json}")
    try:
        settings = request.json
        success = True
        
        if "sensitivity" in settings:
            logger.debug(f"🌐 API: Setting sensitivity for {model_name} to {settings['sensitivity']}")
            success &= settings_manager.set_model_sensitivity(model_name, settings["sensitivity"])
        if "threshold" in settings:
            logger.debug(f"🌐 API: Setting threshold for {model_name} to {settings['threshold']}")
            success &= settings_manager.set_model_threshold(model_name, settings["threshold"])
        if "api_url" in settings:
            logger.debug(f"🌐 API: Setting API URL for {model_name} to {settings['api_url']}")
            success &= settings_manager.set_model_api_url(model_name, settings["api_url"])
        if "active" in settings:
            if settings["active"]:
                # Set this model as the active model (deactivates others)
                logger.info(f"🌐 API: Changing active model to '{model_name}' via /api/config/models/{model_name}")
                success &= settings_manager.set_active_model(model_name)
            else:
                # Deactivate this specific model
                logger.info(f"🌐 API: Deactivating model '{model_name}' via /api/config/models/{model_name}")
                success &= settings_manager.set_model_active_state(model_name, False)
        
        if success:
            logger.info(f"🌐 API: Model {model_name} settings updated successfully")
            return jsonify({"status": "success", "message": f"Model {model_name} settings updated"})
        else:
            logger.error(f"🌐 API: Failed to update model {model_name} settings")
            return jsonify({"status": "error", "message": "Failed to update model settings"}), 500
    except Exception as e:
        logger.error(f"🌐 API: Error updating model {model_name}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/config/wake_word', methods=['POST'])
def set_wake_word_config():
    """Update wake word detection settings (sensitivity and threshold) - DEPRECATED: Use per-model settings instead"""
    logger.debug("🌐 API: POST /api/config/wake_word called (DEPRECATED)")
    try:
        settings = request.json
        logger.warning("⚠️ Global wake_word settings are deprecated. Use per-model settings instead.")
        return jsonify({"status": "deprecated", "message": "Use per-model settings instead"})
    except Exception as e:
        logger.error(f"🌐 API: Error in deprecated wake_word config: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/models/discover', methods=['GET'])
def discover_models():
    """Get list of available models"""
    logger.debug("🌐 API: GET /api/models/discover called")
    models = discover_available_models()
    logger.debug(f"🌐 API: Found {len(models)} available models")
    return jsonify({
        "models": models,
        "count": len(models)
    })

@app.route('/api/audio/rms', methods=['GET'])
def get_rms_data():
    """Get current RMS audio levels for volume meter"""
    rms_data = shared_memory_ipc.get_audio_state()
    return jsonify(rms_data)

@app.route('/api/custom-models', methods=['GET'])
def get_custom_models():
    """Get list of available custom models and current selection"""
    logger.debug("🌐 API: GET /api/custom-models called")
    available_models = []
    for model_name in discover_available_models():
        model_info = get_model_info(model_name)
        if model_info:
            # Get model configuration including active state
            model_config = settings_manager.get_model_config(model_name)
            if model_config:
                model_info['active'] = model_config['active']
                model_info['sensitivity'] = model_config['sensitivity']
                model_info['threshold'] = model_config['threshold']
                model_info['api_url'] = model_config['api_url']
            available_models.append(model_info)
    
    # Get currently active models
    active_models = settings_manager.get_active_models()
    
    logger.debug(f"🌐 API: Returning {len(available_models)} available models, {len(active_models)} active")
    return jsonify({
        'available_models': available_models,
        'active_models': active_models
    })

@app.route('/api/custom-models/<model_name>', methods=['POST'])
def set_custom_model(model_name):
    """Set the active custom model"""
    logger.info(f"🌐 API: POST /api/custom-models/{model_name} called")
    
    # Validate the model exists
    model_info = get_model_info(model_name)
    if not model_info:
        logger.warning(f"🌐 API: Model {model_name} not found")
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    # Set this model as the active model (deactivates others)
    logger.info(f"🌐 API: Changing active model to '{model_name}' via /api/custom-models/{model_name}")
    if settings_manager.set_active_model(model_name):
        logger.info(f"🌐 API: Model {model_name} activated successfully")
        return jsonify({
            'status': 'success',
            'message': f'Model {model_name} activated',
            'active_model': model_name,
            'active_models': settings_manager.get_active_models()
        })
    else:
        logger.error(f"🌐 API: Failed to activate model {model_name}")
        return jsonify({'error': 'Failed to update model setting'}), 500

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get recent wake word detections from detection file"""
    logger.debug("🌐 API: GET /api/detections called")
    try:
        # Read detection file
        detection_file = "/tmp/recent_detections.json"
        detections = []
        
        if os.path.exists(detection_file):
            try:
                with open(detection_file, 'r') as f:
                    detections = json.load(f)
                
                # Filter to only recent detections (last 5 seconds)
                current_time = int(time.time() * 1000)  # Convert to milliseconds
                recent_detections = [
                    detection for detection in detections 
                    if current_time - detection.get('timestamp', 0) < 5000
                ]
                
                logger.debug(f"🌐 API: Found {len(detections)} total detections, {len(recent_detections)} recent")
                return jsonify(recent_detections)
                
            except Exception as e:
                logger.error(f"❌ Failed to read detection file: {e}")
                return jsonify([])
        else:
            logger.debug("🌐 API: No detection file found")
            return jsonify([])
        
    except Exception as e:
        logger.error(f"❌ Failed to get detections: {e}")
        return jsonify([])

@app.route('/api/activation', methods=['GET'])
def get_activation():
    """Get current activation state from shared memory with enhanced debugging"""
    try:
        # ENHANCED DEBUGGING: Track API calls
        if not hasattr(get_activation, '_call_count'):
            get_activation._call_count = 0
        get_activation._call_count += 1
        
        # Removed excessive logging for frequent polling
        activation_data = shared_memory_ipc.get_activation_state()
        
        return jsonify(activation_data)
        
    except Exception as e:
        logger.error(f"❌ Failed to get activation data: {e}")
        return jsonify({
            'current_rms': 0.0,
            'is_active': False,
            'is_listening': False,
            'last_update': time.time()
        })

@app.route('/api/settings/backup', methods=['POST'])
def backup_settings():
    """Manually backup settings to permanent storage"""
    logger.debug("🌐 API: POST /api/settings/backup called")
    if settings_manager.backup():
        logger.info("🌐 API: Settings backed up successfully")
        return jsonify({"status": "success", "message": "Settings backed up"})
    else:
        logger.error("🌐 API: Failed to backup settings")
        return jsonify({"status": "error", "message": "Failed to backup settings"}), 500

@app.route('/api/settings/restore', methods=['POST'])
def restore_settings():
    """Restore settings from backup"""
    logger.debug("🌐 API: POST /api/settings/restore called")
    if settings_manager.restore():
        logger.info("🌐 API: Settings restored successfully")
        return jsonify({"status": "success", "message": "Settings restored from backup"})
    else:
        logger.error("🌐 API: Failed to restore settings")
        return jsonify({"status": "error", "message": "Failed to restore settings"}), 500

@app.route('/api/settings/reset', methods=['POST'])
def reset_settings():
    """Reset settings to defaults"""
    logger.debug("🌐 API: POST /api/settings/reset called")
    if settings_manager.reset_to_defaults():
        logger.info("🌐 API: Settings reset successfully")
        return jsonify({"status": "success", "message": "Settings reset to defaults"})
    else:
        logger.error("🌐 API: Failed to reset settings")
        return jsonify({"status": "error", "message": "Failed to reset settings"}), 500

if __name__ == '__main__':
    # Run in production mode for service deployment with reduced logging
    logger.info("🚀 Starting web backend")
    app.run(host='0.0.0.0', port=7171, debug=False, threaded=True, use_reloader=False, 
            request_handler=QuietWSGIRequestHandler) 