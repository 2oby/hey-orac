"""REST API routes and WebSocket handlers for Hey ORAC web GUI."""

import logging
import json
from datetime import datetime
from dataclasses import asdict
from flask import Blueprint, jsonify, request
from flask_socketio import emit, join_room, leave_room

logger = logging.getLogger(__name__)

# Blueprint for REST API
api_bp = Blueprint('api', __name__)

# These will be set by the main application
settings_manager = None
shared_data = None
event_queue = None


def init_routes(sm, sd, eq):
    """Initialize routes with shared resources."""
    global settings_manager, shared_data, event_queue
    settings_manager = sm
    shared_data = sd
    event_queue = eq


# REST API Routes

@api_bp.route('/config', methods=['GET'])
def get_config():
    """Get the complete configuration."""
    try:
        with settings_manager.get_config() as config:
            return jsonify({
                'models': [asdict(model) for model in config.models],
                'audio': asdict(config.audio),
                'system': asdict(config.system),
                'version': config.version
            })
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/config/global', methods=['POST'])
def update_global_config():
    """Update global system configuration."""
    try:
        data = request.get_json()
        # Update system config fields
        if 'rms_filter' in data:
            settings_manager.update_system_config(rms_filter=data['rms_filter'])
        if 'cooldown' in data:
            settings_manager.update_system_config(cooldown=data['cooldown'])
        
        settings_manager.save()
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error updating global config: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/config/models/<model_name>', methods=['GET'])
def get_model_config(model_name):
    """Get configuration for a specific model."""
    try:
        model = settings_manager.get_model_config(model_name)
        if model:
            return jsonify(asdict(model))
        return jsonify({'error': 'Model not found'}), 404
    except Exception as e:
        logger.error(f"Error getting model config: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/config/models/<model_name>', methods=['POST'])
def update_model_config(model_name):
    """Update configuration for a specific model."""
    try:
        data = request.get_json()
        
        # Update model configuration
        updates = {}
        if 'threshold' in data:
            updates['threshold'] = float(data['threshold'])
        if 'sensitivity' in data:
            updates['sensitivity'] = float(data['sensitivity'])
        if 'webhook_url' in data:
            updates['webhook_url'] = data['webhook_url']
        if 'enabled' in data:
            updates['enabled'] = bool(data['enabled'])
        
        settings_manager.update_model_config(model_name, **updates)
        settings_manager.save()
        
        # Notify about config change
        if shared_data:
            shared_data['config_changed'] = True
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error updating model config: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/models', methods=['GET'])
def get_models():
    """Get list of all available models."""
    try:
        models = settings_manager.get_models_config()
        result = []
        for model in models:
            result.append({
                'name': model.name,
                'enabled': model.enabled,
                'threshold': model.threshold,
                'sensitivity': model.sensitivity,
                'webhook_url': model.webhook_url,
                'path': model.path,
                'framework': model.framework
            })
        return jsonify({'models': result})
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/custom-models', methods=['GET'])
def get_custom_models():
    """Get list of all available models (legacy endpoint)."""
    return get_models()


@api_bp.route('/custom-models/<model_name>/activate', methods=['POST'])
def activate_model(model_name):
    """Activate a specific model."""
    try:
        settings_manager.update_model_config(model_name, enabled=True)
        settings_manager.save()
        
        if shared_data:
            shared_data['config_changed'] = True
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error activating model: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/custom-models/<model_name>/deactivate', methods=['POST'])
def deactivate_model(model_name):
    """Deactivate a specific model."""
    try:
        settings_manager.update_model_config(model_name, enabled=False)
        settings_manager.save()
        
        if shared_data:
            shared_data['config_changed'] = True
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error deactivating model: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/audio/rms', methods=['GET'])
def get_rms():
    """Get current RMS value."""
    try:
        rms = shared_data.get('rms', 0.0) if shared_data else 0.0
        return jsonify({
            'rms': rms,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    except Exception as e:
        logger.error(f"Error getting RMS: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/detections', methods=['GET'])
def get_detections():
    """Get recent detection events."""
    try:
        detections = shared_data.get('recent_detections', []) if shared_data else []
        return jsonify({'detections': detections})
    except Exception as e:
        logger.error(f"Error getting detections: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/activation', methods=['GET'])
def get_activation():
    """Get current activation/listening state."""
    try:
        if shared_data:
            return jsonify({
                'listening': shared_data.get('is_listening', False),
                'active': shared_data.get('is_active', False),
                'status': shared_data.get('status', 'Not connected')
            })
        return jsonify({
            'listening': False,
            'active': False,
            'status': 'Not connected'
        })
    except Exception as e:
        logger.error(f"Error getting activation: {e}")
        return jsonify({'error': str(e)}), 500


# WebSocket handlers

def register_socketio_handlers(socketio):
    """Register SocketIO event handlers."""
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        logger.info(f"Client connected: {request.sid}")
        join_room('updates')
        emit('connected', {'status': 'connected'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        logger.info(f"Client disconnected: {request.sid}")
        leave_room('updates')
    
    @socketio.on('subscribe_updates')
    def handle_subscribe():
        """Client subscribes to real-time updates."""
        logger.info(f"Client {request.sid} subscribing to real-time updates")
        join_room('realtime')
        logger.info(f"Client {request.sid} joined 'realtime' room")
        emit('subscribed', {'status': 'subscribed to updates'})