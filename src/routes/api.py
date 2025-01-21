# Path: codeSage.be/src/routes/api.py

from flask import Blueprint, request, jsonify, current_app
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import os
import uuid
import shutil
from functools import wraps
from src.services.file_crawler import FileCrawler
from src.services.nlp_service import NLPService
from src.services.response_formatter import ResponseFormatter, ResponseType, AnalysisResult
import jwt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
api_bp = Blueprint('api', __name__)

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

@api_bp.route('/health', methods=['GET'])
def health_check() -> tuple[Dict[str, Any], int]:
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': current_app.config.get('VERSION', '1.0.0')
    }, 200

@api_bp.route('/analyze/codebase', methods=['POST'])
@require_auth
def analyze_codebase() -> tuple[Dict[str, Any], int]:
    """Analyze an entire codebase directory."""
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
        summary = current_app.file_crawler.get_project_summary(path)

        # Analyze files
        results = []
        for metadata in current_app.file_crawler.crawl_directory(path, file_types):
            if not metadata.is_binary:
                try:
                    content = current_app.file_crawler.read_file_content(metadata.path)
                    analysis = current_app.nlp_service.analyze_code_block(
                        content,
                        task=analysis_type
                    )
                    
                    results.append(AnalysisResult(
                        type=ResponseType.ANALYSIS,
                        message=analysis['analysis'],
                        confidence=analysis['confidence'],
                        metadata={'file_info': metadata.__dict__}
                    ))
                except Exception as e:
                    logger.error(f"Error analyzing {metadata.path}: {str(e)}")

        response = current_app.response_formatter.format_batch_results(results)
        response['project_summary'] = summary
        response['total_files_analyzed'] = len(results)
        
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in codebase analysis: {str(e)}")
        return jsonify(current_app.response_formatter.format_error_response(e)), 500

@api_bp.route('/analyze/file', methods=['POST'])
@require_auth
def analyze_file() -> tuple[Dict[str, Any], int]:
    """Analyze a single file."""
    try:
        data = request.get_json()
        if not data or 'file_path' not in data:
            return jsonify({'error': 'Missing required file_path parameter'}), 400

        file_path = data['file_path']
        if not os.path.exists(file_path):
            return jsonify({'error': 'Specified file does not exist'}), 404

        analysis_type = data.get('analysis_type', 'understand')

        init_nlp_service()

        metadata = current_app.file_crawler.get_file_metadata(file_path)
        if metadata.is_binary:
            return jsonify({'error': 'Cannot analyze binary file'}), 400

        content = current_app.file_crawler.read_file_content(file_path)
        analysis = current_app.nlp_service.analyze_code_block(content, task=analysis_type)

        result = AnalysisResult(
            type=ResponseType.ANALYSIS,
            message=analysis['analysis'],
            confidence=analysis['confidence'],
            metadata={'file_info': metadata.__dict__}
        )

        return jsonify(current_app.response_formatter.format_analysis_result(result)), 200

    except Exception as e:
        logger.error(f"Error in file analysis: {str(e)}")
        return jsonify(current_app.response_formatter.format_error_response(e)), 500

@api_bp.route('/analyze/snippet', methods=['POST'])
@require_auth
def analyze_snippet() -> tuple[Dict[str, Any], int]:
    """Analyze a code snippet."""
    try:
        data = request.get_json()
        if not data or 'code' not in data:
            return jsonify({'error': 'Missing required code parameter'}), 400

        init_nlp_service()

        analysis = current_app.nlp_service.analyze_code_block(
            data['code'],
            task=data.get('analysis_type', 'understand')
        )

        result = AnalysisResult(
            type=ResponseType.ANALYSIS,
            message=analysis['analysis'],
            confidence=analysis['confidence'],
            metadata={'language': data.get('language', 'unknown')}
        )

        return jsonify(current_app.response_formatter.format_analysis_result(result)), 200

    except Exception as e:
        logger.error(f"Error in snippet analysis: {str(e)}")
        return jsonify(current_app.response_formatter.format_error_response(e)), 500

@api_bp.route('/analyze/upload', methods=['POST'])
@require_auth
def analyze_uploaded_files() -> tuple[Dict[str, Any], int]:
    """Handle uploaded files for analysis"""
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('files[]')
        temp_dir = Path(current_app.config['UPLOAD_DIR']) / str(uuid.uuid4())
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            saved_files = []
            for file in files:
                if file.filename:
                    file_path = temp_dir / file.filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file.save(str(file_path))
                    saved_files.append(str(file_path))

            if not saved_files:
                return jsonify({'error': 'No valid files uploaded'}), 400

            init_nlp_service()

            # Get project summary
            summary = current_app.file_crawler.get_project_summary(str(temp_dir))

            # Analyze files
            results = []
            for file_path in saved_files:
                try:
                    metadata = current_app.file_crawler.get_file_metadata(file_path)
                    if not metadata.is_binary:
                        content = current_app.file_crawler.read_file_content(file_path)
                        analysis = current_app.nlp_service.analyze_code_block(
                            content,
                            task='understand'
                        )
                        results.append(AnalysisResult(
                            type=ResponseType.ANALYSIS,
                            message=analysis['analysis'],
                            confidence=analysis['confidence'],
                            metadata={'file_info': metadata.__dict__}
                        ))
                except Exception as e:
                    logger.error(f"Error analyzing uploaded file {file_path}: {str(e)}")

            response = current_app.response_formatter.format_batch_results(results)
            response['project_summary'] = summary
            response['total_files_analyzed'] = len(results)

            return jsonify(response), 200

        finally:
            # Cleanup temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        logger.error(f"Error processing uploaded files: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/license/validate', methods=['POST'])
def validate_license() -> tuple[Dict[str, Any], int]:
    """Validate license key and return auth token"""
    try:
        data = request.get_json()
        if not data or 'license_key' not in data:
            return jsonify({'error': 'Missing required license_key parameter'}), 400

        license_key = data['license_key']
        
        # Generate JWT token
        token = jwt.encode(
            {
                'license_key': license_key,
                'exp': datetime.utcnow() + timedelta(days=30)
            },
            current_app.config['JWT_SECRET_KEY'],
            algorithm='HS256'
        )

        return jsonify({
            'token': token,
            'expires_in': 30 * 24 * 60 * 60,  # 30 days in seconds
            'issued_at': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in license validation: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.errorhandler(Exception)
def handle_error(error: Exception) -> tuple[Dict[str, Any], int]:
    """Global error handler for all endpoints"""
    logger.error(f"Unhandled error: {str(error)}", exc_info=True)
    return jsonify(current_app.response_formatter.format_error_response(error)), 500