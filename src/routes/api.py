# Path: codeSage.be/src/routes/api.py

from flask import Blueprint, request, jsonify, current_app
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import os
from functools import wraps
from ..services.file_crawler import FileCrawler
from ..services.nlp_service import NLPService
from ..services.response_formatter import ResponseFormatter, ResponseType, AnalysisResult
import jwt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
api = Blueprint('api', __name__)

# Initialize services
response_formatter = ResponseFormatter()
file_crawler = FileCrawler()
nlp_service = None  # Lazy initialization

def init_nlp_service():
    """Lazy initialization of NLP service to save resources"""
    global nlp_service
    if nlp_service is None:
        nlp_service = NLPService()

def require_auth(f):
    """Decorator for endpoints requiring authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'No authorization token provided'}), 401

        try:
            token = token.split('Bearer ')[1]
            payload = jwt.decode(
                token, 
                current_app.config['JWT_SECRET_KEY'],
                algorithms=['HS256']
            )
            request.user = payload
        except jwt.InvalidTokenError as e:
            return jsonify({'error': f'Invalid token: {str(e)}'}), 401
        except Exception as e:
            logger.error(f"Auth error: {str(e)}")
            return jsonify({'error': 'Authentication failed'}), 401

        return f(*args, **kwargs)
    return decorated

@api.route('/health', methods=['GET'])
def health_check() -> tuple[Dict[str, Any], int]:
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': current_app.config.get('VERSION', '1.0.0')
    }, 200

@api.route('/analyze/codebase', methods=['POST'])
@require_auth
def analyze_codebase() -> tuple[Dict[str, Any], int]:
    """
    Analyze an entire codebase directory.
    
    Expected JSON payload:
    {
        "path": "string",  // Path to codebase directory
        "file_types": ["string"],  // Optional list of file extensions to analyze
        "analysis_type": "string"  // Type of analysis to perform
    }
    """
    try:
        data = request.get_json()
        if not data or 'path' not in data:
            return jsonify({'error': 'Missing required path parameter'}), 400

        path = data['path']
        if not os.path.exists(path):
            return jsonify({'error': 'Specified path does not exist'}), 404

        file_types = set(data.get('file_types', ['.py', '.js', '.java', '.cpp', '.h']))
        analysis_type = data.get('analysis_type', 'understand')

        # Initialize NLP service if needed
        init_nlp_service()

        # Get project summary
        summary = file_crawler.get_project_summary(path)

        # Analyze each file
        results = []
        for metadata in file_crawler.crawl_directory(path, file_types):
            if not metadata.is_binary:
                try:
                    content = file_crawler.read_file_content(metadata.path)
                    analysis = nlp_service.analyze_code_block(
                        content,
                        task=analysis_type
                    )
                    
                    results.append(AnalysisResult(
                        type=ResponseType.ANALYSIS,
                        message=analysis['analysis'],
                        confidence=analysis['confidence'],
                        metadata={'file_info': asdict(metadata)}
                    ))
                except Exception as e:
                    logger.error(f"Error analyzing {metadata.path}: {str(e)}")

        response = response_formatter.format_batch_results(results)
        response['project_summary'] = summary
        
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in codebase analysis: {str(e)}")
        return jsonify(response_formatter.format_error_response(e)), 500

@api.route('/analyze/file', methods=['POST'])
@require_auth
def analyze_file() -> tuple[Dict[str, Any], int]:
    """
    Analyze a single file.
    
    Expected JSON payload:
    {
        "file_path": "string",
        "analysis_type": "string"
    }
    """
    try:
        data = request.get_json()
        if not data or 'file_path' not in data:
            return jsonify({'error': 'Missing required file_path parameter'}), 400

        file_path = data['file_path']
        if not os.path.exists(file_path):
            return jsonify({'error': 'Specified file does not exist'}), 404

        analysis_type = data.get('analysis_type', 'understand')

        init_nlp_service()

        metadata = file_crawler.get_file_metadata(file_path)
        if metadata.is_binary:
            return jsonify({'error': 'Cannot analyze binary file'}), 400

        content = file_crawler.read_file_content(file_path)
        analysis = nlp_service.analyze_code_block(content, task=analysis_type)

        result = AnalysisResult(
            type=ResponseType.ANALYSIS,
            message=analysis['analysis'],
            confidence=analysis['confidence'],
            metadata={'file_info': asdict(metadata)}
        )

        return jsonify(response_formatter.format_analysis_result(result)), 200

    except Exception as e:
        logger.error(f"Error in file analysis: {str(e)}")
        return jsonify(response_formatter.format_error_response(e)), 500

@api.route('/analyze/snippet', methods=['POST'])
@require_auth
def analyze_snippet() -> tuple[Dict[str, Any], int]:
    """
    Analyze a code snippet.
    
    Expected JSON payload:
    {
        "code": "string",
        "language": "string",
        "analysis_type": "string"
    }
    """
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({'error': 'Missing required code parameter'}), 400

        init_nlp_service()

        analysis = nlp_service.analyze_code_block(
            data['code'],
            task=data.get('analysis_type', 'understand')
        )

        result = AnalysisResult(
            type=ResponseType.ANALYSIS,
            message=analysis['analysis'],
            confidence=analysis['confidence'],
            metadata={'language': data.get('language', 'unknown')}
        )

        return jsonify(response_formatter.format_analysis_result(result)), 200

    except Exception as e:
        logger.error(f"Error in snippet analysis: {str(e)}")
        return jsonify(response_formatter.format_error_response(e)), 500

@api.route('/license/validate', methods=['POST'])
def validate_license() -> tuple[Dict[str, Any], int]:
    """
    Validate license key and return auth token.
    
    Expected JSON payload:
    {
        "license_key": "string"
    }
    """
    try:
        data = request.get_json()
        if not data or 'license_key' not in data:
            return jsonify({'error': 'Missing required license_key parameter'}), 400

        # TODO: Implement actual license validation
        # For now, just check if key exists
        if not data['license_key']:
            return jsonify({'error': 'Invalid license key'}), 401

        # Generate JWT token
        token = jwt.encode(
            {
                'license_key': data['license_key'],
                'exp': datetime.utcnow() + timedelta(days=30)
            },
            current_app.config['JWT_SECRET_KEY'],
            algorithm='HS256'
        )

        return jsonify({
            'token': token,
            'expires_in': 30 * 24 * 60 * 60  # 30 days in seconds
        }), 200

    except Exception as e:
        logger.error(f"Error in license validation: {str(e)}")
        return jsonify(response_formatter.format_error_response(e)), 500

@api.errorhandler(Exception)
def handle_error(error: Exception) -> tuple[Dict[str, Any], int]:
    """Global error handler for all endpoints"""
    logger.error(f"Unhandled error: {str(error)}", exc_info=True)
    return jsonify(response_formatter.format_error_response(error)), 500