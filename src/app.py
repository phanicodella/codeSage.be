# Path: codeSage.be/src/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from src.config import active_config
import os
from src.routes.api import api_bp
from src.routes.visualization_routes import visualization
from src.services.file_crawler import FileCrawler
from src.services.nlp_service import NLPService
from src.services.response_formatter import ResponseFormatter
from src.services.visualization_service import VisualizationService
from src.models.dci_engine import DCIEngine
from src.services.visualization_update_handler import VisualizationUpdateHandler
import asyncio
import logging
import sys
from src.services.websocket_factory import websocket_factory
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('codesage.log')
    ]
)

async def init_websocket(app):
    """Initialize WebSocket server"""
    try:
        await websocket_factory.start_server(
            host=app.config['WS_HOST'],
            port=app.config['WS_PORT']
        )
        logger.info(f"WebSocket server started on {app.config['WS_HOST']}:{app.config['WS_PORT']}")
    except Exception as e:
        logger.error(f"Failed to start WebSocket server: {e}")
        return False

async def create_app(config_object=None):
    """Factory function to create and configure the Flask application"""
    app = Flask(__name__)
    
    # Load configuration
    if config_object is None:
        app.config.from_object(active_config)
    else:
        app.config.from_object(config_object)

    # Ensure instance path exists
    os.makedirs(app.instance_path, exist_ok=True)
    
    # Initialize CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": app.config.get('CORS_ORIGINS', "*"),
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Authorization", "Content-Type"]
        }
    })
    
    # Ensure required directories exist
    data_paths = {
        'MODEL_PATH': Path(app.config.get('MODEL_PATH', Path('./data/models'))),
        'CACHE_DIR': Path(app.config.get('CACHE_DIR', Path('./data/cache'))),
        'LOG_DIR': Path(app.config.get('LOG_DIR', Path('./logs'))),
        'DB_PATH': Path(app.config.get('DB_PATH', Path('./data'))).parent
    }

    for path in data_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Initialize services
    try:
        app.file_crawler = FileCrawler()
        app.nlp_service = NLPService(
            cache_dir=str(data_paths['MODEL_PATH']),
            model_name="Salesforce/codet5-base"
        )
        app.response_formatter = ResponseFormatter()
        
        # Initialize visualization services
        app.vis_service = VisualizationService()
        app.dci_engine = DCIEngine()
        app.vis_update_handler = VisualizationUpdateHandler(app.dci_engine, app.vis_service)
        
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(visualization, url_prefix='/api/visualize')
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'version': app.config.get('VERSION', '1.0.0'),
            'services': {
                'file_crawler': 'active',
                'nlp_service': 'active',
                'response_formatter': 'active',
                'visualization': 'active',
                'websocket': 'active' if websocket_factory.running else 'inactive'
            }
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

    # Request hooks
    @app.before_request
    def before_request():
        request.start_time = asyncio.get_event_loop().time()

    @app.after_request
    def after_request(response):
        if hasattr(request, 'start_time'):
            duration = asyncio.get_event_loop().time() - request.start_time
            logger.info(f"{request.method} {request.path} completed in {duration:.2f}s")

        # Security headers
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        return response

    @app.teardown_appcontext
    def cleanup(exception=None):
        """Cleanup on app shutdown"""
        try:
            app.vis_update_handler.cleanup()
            asyncio.create_task(websocket_factory.stop())
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    # Initialize WebSocket if not in testing mode
    if not app.config.get('TESTING'):
        try:
            loop = asyncio.get_event_loop()
            await init_websocket(app)
        except Exception as e:
            logger.error(f"WebSocket initialization failed: {e}")
    
    logger.info("Application created successfully")
    return app