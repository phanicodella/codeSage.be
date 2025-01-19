# Path: codeSage.be/src/services/websocket_factory.py

import asyncio
import logging
import json
from typing import Dict, Set, Any, Callable
import websockets
from websockets.server import WebSocketServerProtocol
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class WSMessage:
    """WebSocket message structure"""
    type: str
    payload: Any
    codebase_path: str

class WebSocketFactory:
    """Factory for managing WebSocket connections and message routing"""

    def __init__(self):
        self.connections: Dict[str, Set[WebSocketServerProtocol]] = {}
        self.message_handlers: Dict[str, Set[Callable]] = {}
        self.running = False

    async def start_server(self, host: str = 'localhost', port: int = 8765):
        """Start WebSocket server"""
        try:
            async with websockets.serve(self._handle_connection, host, port):
                self.running = True
                logger.info(f"WebSocket server started on ws://{host}:{port}")
                await asyncio.Future()  # Keep server running
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {str(e)}")
            raise

    async def _handle_connection(self, websocket: WebSocketServerProtocol):
        """Handle new WebSocket connection"""
        try:
            # Wait for initial message with codebase path
            message = await websocket.recv()
            data = json.loads(message)
            codebase_path = data.get('codebasePath')

            if not codebase_path:
                await websocket.close(1008, "Missing codebase path")
                return

            # Register connection
            if codebase_path not in self.connections:
                self.connections[codebase_path] = set()
            self.connections[codebase_path].add(websocket)

            try:
                async for message in websocket:
                    await self._handle_message(websocket, message)
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Connection closed for codebase: {codebase_path}")
            finally:
                # Cleanup connection
                self.connections[codebase_path].remove(websocket)
                if not self.connections[codebase_path]:
                    del self.connections[codebase_path]

        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
            await websocket.close(1011, "Internal server error")

    async def _handle_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            ws_message = WSMessage(**data)

            # Route message to registered handlers
            handlers = self.message_handlers.get(ws_message.type, set())
            for handler in handlers:
                try:
                    await handler(ws_message)
                except Exception as e:
                    logger.error(f"Message handler error: {str(e)}")

        except Exception as e:
            logger.error(f"Message handling error: {str(e)}")
            await websocket.send(json.dumps({
                "type": "error",
                "payload": str(e)
            }))

    async def broadcast(self, message: WSMessage):
        """Broadcast message to all connections for a codebase"""
        if message.codebase_path in self.connections:
            dead_connections = set()
            message_data = json.dumps(asdict(message))

            for websocket in self.connections[message.codebase_path]:
                try:
                    await websocket.send(message_data)
                except websockets.exceptions.ConnectionClosed:
                    dead_connections.add(websocket)
                except Exception as e:
                    logger.error(f"Broadcast error: {str(e)}")
                    dead_connections.add(websocket)

            # Cleanup dead connections
            for websocket in dead_connections:
                self.connections[message.codebase_path].remove(websocket)

    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler for specific message type"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = set()
        self.message_handlers[message_type].add(handler)

    def unregister_handler(self, message_type: str, handler: Callable):
        """Unregister message handler"""
        if message_type in self.message_handlers:
            self.message_handlers[message_type].discard(handler)
            if not self.message_handlers[message_type]:
                del self.message_handlers[message_type]

    async def stop(self):
        """Stop WebSocket server and close all connections"""
        self.running = False
        for connections in self.connections.values():
            for websocket in connections:
                await websocket.close(1001, "Server shutting down")
        self.connections.clear()
        self.message_handlers.clear()

# Create singleton instance
websocket_factory = WebSocketFactory()