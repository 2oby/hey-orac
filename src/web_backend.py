from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import logging
from settings_manager import get_settings_manager, get_setting, set_setting
from rms_monitor import rms_monitor

# Configure Flask logging to reduce verbosity for frequent polling endpoints only
logging.getLogger('werkzeug').setLevel(logging.ERROR)  # Set to ERROR to suppress most requests
logging.getLogger('flask').setLevel(logging.INFO)
logging.getLogger('urllib3').setLevel(logging.ERROR)

# Disable Flask's default request logging completely
app = Flask(__name__)
app.logger.disabled = True
CORS(app)

# Configure WSGI server logging to suppress request logs
import logging
from werkzeug.serving import WSGIRequestHandler

# Disable werkzeug's request logging
class QuietWSGIRequestHandler(WSGIRequestHandler):
    def log_request(self, *args, **kwargs):
        # Suppress all request logging
        pass

# Set the custom request handler
WSGIRequestHandler.log_request = QuietWSGIRequestHandler.log_request

# Initialize settings manager
settings_manager = get_settings_manager()

# Serve static files from web directory
@app.route('/')
def index():
    return send_from_directory('/app/web', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('/app/web', filename)

# API endpoints
@app.route('/api/config', methods=['GET'])
def get_config():
    """Get full configuration for web interface"""
    # Return the structure expected by the web interface
    sensitivities = settings_manager.get("wake_word.sensitivities", {})
    api_urls = settings_manager.get("wake_word.api_urls", {})
    return jsonify({
        "global": {
            "rms_filter": settings_manager.get("detection.rms_filter", 50),
            "debounce_ms": settings_manager.get("wake_word.debounce", 200),
            "cooldown_s": settings_manager.get("wake_word.cooldown", 1.5)
        },
        "models": {
            "Hay--compUta_v_lrg": {
                "sensitivity": sensitivities.get("Hay--compUta_v_lrg", 0.4),
                "api_url": api_urls.get("Hay--compUta_v_lrg", "https://api.example.com/webhook")
            },
            "Hey_computer": {
                "sensitivity": sensitivities.get("Hey_computer", 0.4),
                "api_url": api_urls.get("Hey_computer", "https://api.example.com/webhook")
            },
            "hey-CompUter_lrg": {
                "sensitivity": sensitivities.get("hey-CompUter_lrg", 0.4),
                "api_url": api_urls.get("hey-CompUter_lrg", "https://api.example.com/webhook")
            }
        }
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
        if settings_manager.update(settings):
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
    # For now, return wake word settings
    return jsonify({
        "wake_word": {
            "model": settings_manager.get("wake_word.model"),
            "cooldown": settings_manager.get("wake_word.cooldown"),
            "debounce": settings_manager.get("wake_word.debounce")
        }
    })

@app.route('/api/config/models/<model_name>', methods=['GET'])
def get_model_config(model_name):
    """Get specific model settings (per-model sensitivity and API URL)"""
    sensitivity = settings_manager.get_model_sensitivity(model_name, 0.4)
    api_url = settings_manager.get_model_api_url(model_name, "https://api.example.com/webhook")
    return jsonify({
        "sensitivity": sensitivity,
        "api_url": api_url,
        "model": model_name
    })

@app.route('/api/config/models/<model_name>', methods=['POST'])
def set_model_config(model_name):
    """Update specific model settings (per-model sensitivity and API URL)"""
    try:
        settings = request.json
        if "sensitivity" in settings:
            settings_manager.set_model_sensitivity(model_name, settings["sensitivity"])
        if "api_url" in settings:
            settings_manager.set_model_api_url(model_name, settings["api_url"])
        # Optionally update model selection
        if settings.get("activate", False):
            settings_manager.set("wake_word.model", model_name)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/models/discover', methods=['GET'])
def discover_models():
    """Get list of available models"""
    return jsonify({
        "models": ["wake_word"],
        "count": 1
    })

@app.route('/api/audio/rms', methods=['GET'])
def get_rms_data():
    """Get current RMS audio levels for volume meter"""
    rms_data = rms_monitor.get_rms_data()
    return jsonify(rms_data)

@app.route('/api/custom-models', methods=['GET'])
def get_custom_models():
    """Get list of available custom models and current selection"""
    import os
    import glob
    
    # Discover available custom models
    models_dir = '/app/third_party/openwakeword/custom_models'
    onnx_files = glob.glob(os.path.join(models_dir, '*.onnx'))
    
    available_models = []
    for file_path in onnx_files:
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        available_models.append({
            'name': model_name,
            'file': os.path.basename(file_path),
            'path': file_path
        })
    
    # Get current selection from settings
    current_model = settings_manager.get("wake_word.model", "Hay--compUta_v_lrg")
    
    return jsonify({
        'available_models': available_models,
        'current_model': current_model,
        'current_path': f'/app/third_party/openwakeword/custom_models/{current_model}.onnx'
    })

@app.route('/api/custom-models/<model_name>', methods=['POST'])
def set_custom_model(model_name):
    """Set the active custom model"""
    import os
    
    # Validate the model exists
    model_path = f'/app/third_party/openwakeword/custom_models/{model_name}.onnx'
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    # Update the settings
    if settings_manager.set("wake_word.model", model_name):
        return jsonify({
            'status': 'success',
            'message': f'Model {model_name} activated',
            'current_model': model_name,
            'current_path': model_path
        })
    else:
        return jsonify({'error': 'Failed to update model setting'}), 500

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get recent wake word detections (ephemeral - clears file after reading)"""
    import json
    import time
    import os
    
    # Read detections from file
    detection_file = '/tmp/recent_detections.json'
    detections = []
    
    if os.path.exists(detection_file):
        try:
            with open(detection_file, 'r') as f:
                detections = json.load(f)
            # Clear the file after reading (ephemeral behavior)
            os.remove(detection_file)
        except Exception as e:
            # If file is corrupted, return empty list and try to remove the file
            detections = []
            try:
                if os.path.exists(detection_file):
                    os.remove(detection_file)
            except Exception:
                pass
    
    # Return all detections (no time filtering since file is cleared after reading)
    return jsonify(detections)

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