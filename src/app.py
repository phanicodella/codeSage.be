# File path: E:\codeSage\codeSage.be\src\app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from config import active_config
import os
from routes.api import api_bp
from services.file_crawler import FileCrawler
from services.nlp_service import NLPService
from services.response_formatter import ResponseFormatter

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
    
    # Register services to app context
    app.file_crawler = file_crawler
    app.nlp_service = nlp_service
    app.response_formatter = response_formatter
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'services': {
                'file_crawler': 'active',
                'nlp_service': 'active',
                'response_formatter': 'active'
            }
        })
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(
        host=app.config['API_HOST'],
        port=app.config['API_PORT'],
        debug=app.config['DEBUG_MODE']
    )