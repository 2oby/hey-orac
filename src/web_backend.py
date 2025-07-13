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

# Configure logging
logger = logging.getLogger(__name__)

# Configure Flask logging to reduce verbosity for frequent polling endpoints only
import logging
from werkzeug.serving import WSGIRequestHandler

class QuietWSGIRequestHandler(WSGIRequestHandler):
    def log_request(self, *args, **kwargs):
        # Suppress all request logging
        pass

# Create Flask app
app = Flask(__name__)

# Get settings manager instance
settings_manager = get_settings_manager()

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
        "rms_filter": settings_manager.get("volume_monitoring", {}).get("rms_filter", 10),
        "debounce_ms": int(settings_manager.get("wake_word.debounce") * 1000),
        "cooldown_s": settings_manager.get("wake_word.cooldown")
    }
    
    return jsonify({
        "models": models,
        "global": global_settings
    })

@app.route('/api/config/global', methods=['GET'])
def get_global_config():
    """Get global settings"""
    return jsonify(settings_manager.get_all())

@app.route('/api/config/global', methods=['POST'])
def set_global_config():
    """Update global settings"""
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
            return jsonify({"status": "success", "message": "Settings updated"})
        else:
            return jsonify({"status": "error", "message": "Failed to update settings"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/config/settings/<path:key>', methods=['GET'])
def get_setting_value(key):
    """Get a specific setting value"""
    value = settings_manager.get(key)
    return jsonify({"key": key, "value": value})

@app.route('/api/config/settings/<path:key>', methods=['POST'])
def set_setting_value(key):
    """Set a specific setting value"""
    try:
        value = request.json.get('value')
        if settings_manager.set(key, value):
            return jsonify({"status": "success", "key": key, "value": value})
        else:
            return jsonify({"status": "error", "message": "Failed to update setting"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/config/models', methods=['GET'])
def get_models():
    """Get all models with their settings"""
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
    
    return jsonify({
        "models": models,
        "active_models": settings_manager.get_active_models()
    })

@app.route('/api/config/models/<model_name>', methods=['GET'])
def get_model_config(model_name):
    """Get specific model settings (per-model sensitivity, threshold, API URL, and active state)"""
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
        return jsonify({"error": f"Model {model_name} not found"}), 404

@app.route('/api/config/models/<model_name>', methods=['POST'])
def set_model_config(model_name):
    """Update specific model settings (per-model sensitivity, threshold, API URL, and active state)"""
    try:
        settings = request.json
        success = True
        
        if "sensitivity" in settings:
            success &= settings_manager.set_model_sensitivity(model_name, settings["sensitivity"])
        if "threshold" in settings:
            success &= settings_manager.set_model_threshold(model_name, settings["threshold"])
        if "api_url" in settings:
            success &= settings_manager.set_model_api_url(model_name, settings["api_url"])
        if "active" in settings:
            if settings["active"]:
                # Set this model as the active model (deactivates others)
                logger.debug(f"🌐 API: Changing active model to '{model_name}' via /api/config/models/{model_name}")
                success &= settings_manager.set_active_model(model_name)
            else:
                # Deactivate this specific model
                logger.debug(f"🌐 API: Deactivating model '{model_name}' via /api/config/models/{model_name}")
                success &= settings_manager.set_model_active(model_name, False)
        
        if success:
            return jsonify({"status": "success", "message": f"Model {model_name} settings updated"})
        else:
            return jsonify({"status": "error", "message": "Failed to update model settings"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/config/wake_word', methods=['POST'])
def set_wake_word_config():
    """Update wake word detection settings (sensitivity and threshold) - DEPRECATED: Use per-model settings instead"""
    try:
        settings = request.json
        logger.warning("⚠️ Global wake_word settings are deprecated. Use per-model settings instead.")
        return jsonify({"status": "deprecated", "message": "Use per-model settings instead"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/models/discover', methods=['GET'])
def discover_models():
    """Get list of available models"""
    return jsonify({
        "models": discover_available_models(),
        "count": len(discover_available_models())
    })

@app.route('/api/audio/rms', methods=['GET'])
def get_rms_data():
    """Get current RMS audio levels for volume meter"""
    rms_data = shared_memory_ipc.get_audio_state()
    return jsonify(rms_data)

@app.route('/api/custom-models', methods=['GET'])
def get_custom_models():
    """Get list of available custom models and current selection"""
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
    
    return jsonify({
        'available_models': available_models,
        'active_models': active_models
    })

@app.route('/api/custom-models/<model_name>', methods=['POST'])
def set_custom_model(model_name):
    """Set the active custom model"""
    logger.debug(f"🌐 DEBUG TEST: set_custom_model called with model_name={model_name}")
    
    # Validate the model exists
    model_info = get_model_info(model_name)
    if not model_info:
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    # Set this model as the active model (deactivates others)
    logger.debug(f"🌐 API: Changing active model to '{model_name}' via /api/custom-models/{model_name}")
    if settings_manager.set_active_model(model_name):
        return jsonify({
            'status': 'success',
            'message': f'Model {model_name} activated',
            'active_model': model_name,
            'active_models': settings_manager.get_active_models()
        })
    else:
        return jsonify({'error': 'Failed to update model setting'}), 500

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get recent wake word detections from shared memory (replaces file-based system)"""
    try:
        # Get activation data from shared memory
        activation_data = shared_memory_ipc.get_activation_state()
        
        # Convert to detection format for backward compatibility
        detections = []
        if activation_data.get('is_listening', False):
            # Create a detection entry if currently listening
            detection = {
                'model_name': 'Custom Model',  # Will be updated when we add model name to shared memory
                'confidence': 0.0,  # Will be updated when we add confidence to shared memory
                'timestamp': int(activation_data.get('last_update', time.time()) * 1000),  # Convert to milliseconds
                'is_listening': True
            }
            detections.append(detection)
        
        return jsonify(detections)
        
    except Exception as e:
        logger.error(f"❌ Failed to get detections from shared memory: {e}")
        return jsonify([])

@app.route('/api/activation', methods=['GET'])
def get_activation():
    """Get current activation state from shared memory with enhanced debugging"""
    try:
        # ENHANCED DEBUGGING: Track API calls
        if not hasattr(get_activation, '_call_count'):
            get_activation._call_count = 0
        get_activation._call_count += 1
        
        # Log every 50th call to avoid spam
        if get_activation._call_count % 50 == 0:
            logger.info(f"🌐 ACTIVATION API CALL #{get_activation._call_count}")
        
        activation_data = shared_memory_ipc.get_activation_state()
        
        # ENHANCED DEBUGGING: Log activation state
        if activation_data.get('is_listening', False):
            logger.info(f"🎯 ACTIVATION API: Returning listening state - RMS: {activation_data.get('current_rms', 0):.4f}")
        elif get_activation._call_count % 100 == 0:  # Log non-listening state less frequently
            logger.debug(f"🔇 ACTIVATION API: Returning not listening state - RMS: {activation_data.get('current_rms', 0):.4f}")
        
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
    if settings_manager.backup():
        return jsonify({"status": "success", "message": "Settings backed up"})
    else:
        return jsonify({"status": "error", "message": "Failed to backup settings"}), 500

@app.route('/api/settings/restore', methods=['POST'])
def restore_settings():
    """Restore settings from backup"""
    if settings_manager.restore():
        return jsonify({"status": "success", "message": "Settings restored from backup"})
    else:
        return jsonify({"status": "error", "message": "Failed to restore settings"}), 500

@app.route('/api/settings/reset', methods=['POST'])
def reset_settings():
    """Reset settings to defaults"""
    if settings_manager.reset_to_defaults():
        return jsonify({"status": "success", "message": "Settings reset to defaults"})
    else:
        return jsonify({"status": "error", "message": "Failed to reset settings"}), 500

if __name__ == '__main__':
    # Run in production mode for service deployment with reduced logging
    app.run(host='0.0.0.0', port=7171, debug=False, threaded=True, use_reloader=False) 