# Path: codeSage.be/src/services/visualization_update_handler.py

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ..services.websocket_factory import websocket_factory, WSMessage
from ..models.dci_engine import DCIEngine
from ..services.visualization_service import VisualizationService
from ..config import active_config as config

logger = logging.getLogger(__name__)

@dataclass
class VisualizationUpdate:
    """Data class for visualization updates"""
    update_type: str
    data: Dict[str, Any]
    codebase_path: str
    component_id: Optional[str] = None

class VisualizationUpdateHandler:
    """Handles real-time updates for visualizations"""

    def __init__(self, dci_engine: DCIEngine, vis_service: VisualizationService):
        self.dci_engine = dci_engine
        self.vis_service = vis_service
        self.update_tasks = {}
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup WebSocket message handlers"""
        websocket_factory.register_handler("dependency_update", self.handle_dependency_update)
        websocket_factory.register_handler("dataflow_update", self.handle_dataflow_update)
        websocket_factory.register_handler("bug_impact_update", self.handle_bug_impact_update)
        websocket_factory.register_handler("layout_update", self.handle_layout_update)

    async def handle_dependency_update(self, message: WSMessage):
        """Handle dependency graph updates"""
        try:
            dependencies = self.dci_engine.analyze_codebase(message.codebase_path)
            graph_data = self.vis_service.create_dependency_graph(dependencies)
            
            await websocket_factory.broadcast(WSMessage(
                type="dependency_graph",
                payload=graph_data,
                codebase_path=message.codebase_path
            ))

        except Exception as e:
            logger.error(f"Dependency update error: {str(e)}")
            await self._send_error(message.codebase_path, str(e))

    async def handle_dataflow_update(self, message: WSMessage):
        """Handle data flow diagram updates"""
        try:
            component_id = message.payload.get('componentId')
            if not component_id:
                raise ValueError("Missing component ID")

            insights = self.dci_engine.analyze_codebase(message.codebase_path)
            data_flows = insights.get(component_id, {}).get('data_flow', {})
            flow_data = self.vis_service.create_data_flow_diagram(data_flows)

            await websocket_factory.broadcast(WSMessage(
                type="data_flow",
                payload=flow_data,
                codebase_path=message.codebase_path
            ))

        except Exception as e:
            logger.error(f"Data flow update error: {str(e)}")
            await self._send_error(message.codebase_path, str(e))

    async def handle_bug_impact_update(self, message: WSMessage):
        """Handle bug impact visualization updates"""
        try:
            bug_info = message.payload.get('bugInfo')
            if not bug_info:
                raise ValueError("Missing bug information")

            impact_data = self.dci_engine.get_bug_impact(
                message.codebase_path,
                bug_info.get('line'),
                bug_info.get('file')
            )
            
            visualization_data = self.vis_service.visualize_bug_impact(impact_data)

            await websocket_factory.broadcast(WSMessage(
                type="bug_impact",
                payload=visualization_data,
                codebase_path=message.codebase_path
            ))

        except Exception as e:
            logger.error(f"Bug impact update error: {str(e)}")
            await self._send_error(message.codebase_path, str(e))

    async def handle_layout_update(self, message: WSMessage):
        """Handle layout updates"""
        try:
            layout_data = message.payload.get('layout')
            if not layout_data:
                raise ValueError("Missing layout data")

            updated_layout = self.vis_service.update_layout(
                message.payload.get('type'),
                layout_data
            )

            await websocket_factory.broadcast(WSMessage(
                type="layout_update",
                payload=updated_layout,
                codebase_path=message.codebase_path
            ))

        except Exception as e:
            logger.error(f"Layout update error: {str(e)}")
            await self._send_error(message.codebase_path, str(e))

    async def start_update_task(self, codebase_path: str):
        """Start periodic update task for a codebase"""
        if codebase_path in self.update_tasks:
            return

        task = asyncio.create_task(self._update_loop(codebase_path))
        self.update_tasks[codebase_path] = task

    async def stop_update_task(self, codebase_path: str):
        """Stop update task for a codebase"""
        if codebase_path in self.update_tasks:
            self.update_tasks[codebase_path].cancel()
            del self.update_tasks[codebase_path]

    async def _update_loop(self, codebase_path: str):
        """Periodic update loop for visualizations"""
        try:
            while True:
                # Update dependency graph
                dependencies = self.dci_engine.analyze_codebase(codebase_path)
                graph_data = self.vis_service.create_dependency_graph(dependencies)
                
                await websocket_factory.broadcast(WSMessage(
                    type="dependency_graph",
                    payload=graph_data,
                    codebase_path=codebase_path
                ))

                # Wait for next update interval
                await asyncio.sleep(config.VIS_UPDATE_INTERVAL / 1000)  # Convert to seconds

        except asyncio.CancelledError:
            logger.info(f"Update loop cancelled for codebase: {codebase_path}")
        except Exception as e:
            logger.error(f"Update loop error: {str(e)}")
            await self._send_error(codebase_path, str(e))

    async def _send_error(self, codebase_path: str, error_message: str):
        """Send error message to clients"""
        await websocket_factory.broadcast(WSMessage(
            type="error",
            payload={"message": error_message},
            codebase_path=codebase_path
        ))

    def cleanup(self):
        """Cleanup all update tasks"""
        for task in self.update_tasks.values():
            task.cancel()
        self.update_tasks.clear()