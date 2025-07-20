"""Flask-SocketIO application for Hey ORAC web GUI."""

import os
import logging
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

logger = logging.getLogger(__name__)

socketio = SocketIO(
    cors_allowed_origins="*", 
    async_mode='threading',  # Changed from eventlet to threading
    ping_timeout=120,
    ping_interval=25,
    logger=False,  # Disabled to reduce log spam
    engineio_logger=False  # Disabled to reduce log spam
)


def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__,
                static_folder='static',
                static_url_path='/')
    
    # Configure Flask
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Update config if provided
    if config:
        app.config.update(config)
    
    # Enable CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Initialize SocketIO with the app
    socketio.init_app(app)
    
    # Register blueprints
    from .routes import api_bp, register_socketio_handlers
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Register SocketIO handlers
    register_socketio_handlers(socketio)
    
    # Serve index.html at root
    @app.route('/')
    def index():
        return app.send_static_file('index.html')
    
    logger.info("Flask-SocketIO app created successfully")
    
    return app


def run_web_server(host='0.0.0.0', port=7171, debug=False):
    """Run the Flask-SocketIO server."""
    app = create_app()
    logger.info(f"Starting web server on {host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug)