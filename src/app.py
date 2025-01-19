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
from src.services.websocket_factory import websocket_factory

logger = logging.getLogger(__name__)

async def init_websocket(app):
    """Initialize WebSocket server"""
    try:
        await websocket_factory.start_server(
            host=app.config['WS_HOST'],
            port=app.config['WS_PORT']
        )
    except Exception as e:
        logger.error(f"Failed to start WebSocket server: {str(e)}")
        raise

def create_app(config_object=None):
    """
    Factory function to create and configure the Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    if config_object is None:
        app.config.from_object(active_config)
    else:
        app.config.from_object(config_object)
    
    # Initialize CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Ensure required directories exist
    os.makedirs(app.config['MODEL_PATH'], exist_ok=True)
    
    # Initialize services
    file_crawler = FileCrawler()
    nlp_service = NLPService(app.config['MODEL_PATH'])
    response_formatter = ResponseFormatter()
    
    # Initialize visualization services
    vis_service = VisualizationService()
    dci_engine = DCIEngine()
    vis_update_handler = VisualizationUpdateHandler(dci_engine, vis_service)
    
    # Register services to app context
    app.file_crawler = file_crawler
    app.nlp_service = nlp_service
    app.response_formatter = response_formatter
    app.vis_service = vis_service
    app.dci_engine = dci_engine
    app.vis_update_handler = vis_update_handler
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(visualization, url_prefix='/api/visualize')
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'services': {
                'file_crawler': 'active',
                'nlp_service': 'active',
                'response_formatter': 'active',
                'visualization': 'active',
                'websocket': 'active' if websocket_factory.running else 'inactive'
            }
        })
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500

    @app.teardown_appcontext
    def cleanup(exception=None):
        """Cleanup on app shutdown"""
        app.vis_update_handler.cleanup()
        asyncio.create_task(websocket_factory.stop())
    
    # Start WebSocket server if not in testing mode
    if not app.config.get('TESTING'):
        asyncio.create_task(init_websocket(app))
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(
        host=app.config['API_HOST'],
        port=app.config['API_PORT'],
        debug=app.config['DEBUG_MODE']
    )