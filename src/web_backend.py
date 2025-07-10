from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from config_handler import ConfigHandler
from rms_monitor import rms_monitor

app = Flask(__name__)
CORS(app)

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
    """Get full configuration"""
    config_handler = ConfigHandler()
    return jsonify(config_handler.get_config())

@app.route('/api/config/global', methods=['GET'])
def get_global_config():
    """Get global settings"""
    config_handler = ConfigHandler()
    return jsonify(config_handler.get_global())

@app.route('/api/config/global', methods=['POST'])
def set_global_config():
    """Update global settings"""
    config_handler = ConfigHandler()
    settings = request.json
    config_handler.set_global(settings)
    return jsonify({"status": "success"})

@app.route('/api/config/models', methods=['GET'])
def get_models():
    """Get all models with their settings"""
    config_handler = ConfigHandler()
    models = {}
    for model_name in config_handler.get_all_models():
        models[model_name] = config_handler.get_model(model_name)
    return jsonify(models)

@app.route('/api/config/models/<model_name>', methods=['GET'])
def get_model_config(model_name):
    """Get specific model settings"""
    config_handler = ConfigHandler()
    return jsonify(config_handler.get_model(model_name))

@app.route('/api/config/models/<model_name>', methods=['POST'])
def set_model_config(model_name):
    """Update specific model settings"""
    config_handler = ConfigHandler()
    settings = request.json
    config_handler.set_model(model_name, settings)
    return jsonify({"status": "success"})

@app.route('/api/models/discover', methods=['GET'])
def discover_models():
    """Get list of available models"""
    config_handler = ConfigHandler()
    return jsonify({
        "models": config_handler.get_all_models(),
        "count": len(config_handler.get_all_models())
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
    
    # Get current selection from config
    config_handler = ConfigHandler()
    config = config_handler.get_config()
    current_model_path = config.get('wake_word', {}).get('custom_model_path', '')
    
    # Find which model is currently active
    current_model = None
    for model in available_models:
        # Compare both the full path and the relative path
        if (model['path'] == current_model_path or 
            model['path'].replace('/app/', '') == current_model_path or
            f"/app/{current_model_path}" == model['path']):
            current_model = model['name']
            break
    
    return jsonify({
        'available_models': available_models,
        'current_model': current_model,
        'current_path': current_model_path
    })

@app.route('/api/custom-models/<model_name>', methods=['POST'])
def set_custom_model(model_name):
    """Set the active custom model"""
    import os
    
    # Validate the model exists
    model_path = f'/app/third_party/openwakeword/custom_models/{model_name}.onnx'
    if not os.path.exists(model_path):
        return jsonify({'error': f'Model {model_name} not found'}), 404
    
    # Update the config
    config_handler = ConfigHandler()
    config = config_handler.get_config()
    
    if 'wake_word' not in config:
        config['wake_word'] = {}
    
    config['wake_word']['custom_model_path'] = model_path
    config_handler.config = config
    config_handler.save()
    
    return jsonify({
        'status': 'success',
        'message': f'Model {model_name} activated',
        'current_model': model_name,
        'current_path': model_path
    })

@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get recent wake word detections"""
    # This is a simple in-memory store for recent detections
    # In a production system, you might want to use a database
    if not hasattr(app, 'recent_detections'):
        app.recent_detections = []
    
    # Return detections from the last 10 seconds
    import time
    current_time = time.time()
    recent_detections = [
        detection for detection in app.recent_detections 
        if current_time - detection['timestamp'] < 10
    ]
    
    return jsonify(recent_detections)

def add_detection(model_name, confidence):
    """Add a detection to the recent detections list"""
    import time
    if not hasattr(app, 'recent_detections'):
        app.recent_detections = []
    
    detection = {
        'model_name': model_name,
        'confidence': confidence,
        'timestamp': time.time()
    }
    
    app.recent_detections.append(detection)
    
    # Keep only the last 50 detections
    if len(app.recent_detections) > 50:
        app.recent_detections = app.recent_detections[-50:]

if __name__ == '__main__':
    # Run in production mode for service deployment
    app.run(host='0.0.0.0', port=7171, debug=False, threaded=True) 