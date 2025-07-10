from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from config_handler import ConfigHandler

app = Flask(__name__)
CORS(app)

# Serve static files from web directory
@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('web', filename)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7171, debug=True) 