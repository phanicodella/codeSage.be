# Path: codeSage.be/src/routes/visualization_routes.py

from flask import Blueprint, request, jsonify, current_app
from typing import Dict, Any
import logging
from ..services.visualization_service import VisualizationService
from ..models.dci_engine import DCIEngine
from functools import wraps
import jwt

logger = logging.getLogger(__name__)
visualization = Blueprint('visualization', __name__)

# Initialize services
vis_service = VisualizationService()
dci_engine = DCIEngine()

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

@visualization.route('/dependencies', methods=['POST'])
@require_auth
def get_dependency_graph() -> tuple[Dict[str, Any], int]:
    """Generate dependency graph visualization"""
    try:
        data = request.get_json()
        if not data or 'codebasePath' not in data:
            return jsonify({'error': 'Missing required codebasePath parameter'}), 400

        config = data.get('config', {})
        dependencies = dci_engine.analyze_codebase(data['codebasePath'])
        graph_data = vis_service.create_dependency_graph(dependencies)
        
        return jsonify(graph_data), 200

    except Exception as e:
        logger.error(f"Error generating dependency graph: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization.route('/dataflow', methods=['POST'])
@require_auth
def get_data_flow() -> tuple[Dict[str, Any], int]:
    """Generate data flow visualization"""
    try:
        data = request.get_json()
        if not data or 'codebasePath' not in data or 'componentId' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        config = data.get('config', {})
        insights = dci_engine.analyze_codebase(data['codebasePath'])
        data_flows = insights.get(data['componentId'], {}).get('data_flow', {})
        flow_data = vis_service.create_data_flow_diagram(data_flows)
        
        return jsonify(flow_data), 200

    except Exception as e:
        logger.error(f"Error generating data flow diagram: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization.route('/bug-impact', methods=['POST'])
@require_auth
def get_bug_impact() -> tuple[Dict[str, Any], int]:
    """Generate bug impact visualization"""
    try:
        data = request.get_json()
        if not data or 'codebasePath' not in data or 'bugInfo' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        config = data.get('config', {})
        impact_data = dci_engine.get_bug_impact(
            data['codebasePath'],
            data['bugInfo'].get('line'),
            data['bugInfo'].get('file')
        )
        
        visualization_data = vis_service.visualize_bug_impact(impact_data)
        return jsonify(visualization_data), 200

    except Exception as e:
        logger.error(f"Error generating bug impact visualization: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization.route('/export', methods=['POST'])
@require_auth
def export_visualization() -> tuple[Dict[str, Any], int]:
    """Export visualization in specified format"""
    try:
        data = request.get_json()
        if not data or 'type' not in data or 'data' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        export_format = data.get('format', 'html')
        visualization_data = data['data']
        
        exported_data = vis_service.export_to_format(
            visualization_data,
            export_format
        )
        
        return jsonify({
            'data': exported_data,
            'format': export_format
        }), 200

    except Exception as e:
        logger.error(f"Error exporting visualization: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization.route('/update-layout', methods=['POST'])
@require_auth
def update_layout() -> tuple[Dict[str, Any], int]:
    """Update visualization layout"""
    try:
        data = request.get_json()
        if not data or 'type' not in data or 'layout' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        updated_data = vis_service.update_layout(
            data['type'],
            data['layout']
        )
        
        return jsonify(updated_data), 200

    except Exception as e:
        logger.error(f"Error updating layout: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization.route('/metrics', methods=['POST'])
@require_auth
def get_metrics() -> tuple[Dict[str, Any], int]:
    """Get visualization metrics"""
    try:
        data = request.get_json()
        if not data or 'type' not in data or 'data' not in data:
            return jsonify({'error': 'Missing required parameters'}), 400

        metrics = vis_service.calculate_metrics(
            data['type'],
            data['data']
        )
        
        return jsonify(metrics), 200

    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@visualization.route('/ws', methods=['GET'])
@require_auth
def websocket_handler():
    """WebSocket connection handler for real-time updates"""
    try:
        from flask_sock import Sock
        sock = Sock(current_app)
        
        @sock.route('/api/visualize/ws')
        def ws(ws):
            while True:
                data = ws.receive()
                # Process real-time visualization updates
                ws.send(data)
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        return jsonify({'error': str(e)}), 500