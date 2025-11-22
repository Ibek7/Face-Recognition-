"""
WebSocket support for real-time face detection.

Enables live video streaming and detection results.
"""

from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class WSConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        """Initialize manager."""
        self.active: Set[WebSocket] = set()
        self.metadata: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept new connection."""
        await websocket.accept()
        self.active.add(websocket)
        self.metadata[websocket] = {"client_id": client_id or str(id(websocket))}
        logger.info(f"WebSocket connected: {self.metadata[websocket]['client_id']}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection."""
        self.active.discard(websocket)
        client_id = self.metadata.pop(websocket, {}).get("client_id", "unknown")
        logger.info(f"WebSocket disconnected: {client_id}")
    
    async def send_message(self, message: str, websocket: WebSocket):
        """Send to specific client."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Send error: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Send to all clients."""
        disconnected = set()
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.add(ws)
        
        for ws in disconnected:
            self.disconnect(ws)
    
    def count(self) -> int:
        """Get active connection count."""
        return len(self.active)


# Global manager instance
ws_manager = WSConnectionManager()


async def handle_face_detection_ws(websocket: WebSocket):
    """
    Handle real-time face detection WebSocket.
    
    Args:
        websocket: WebSocket connection
    """
    await ws_manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                msg = json.loads(data)
                
                if msg.get("type") == "frame":
                    # Process frame and send results
                    result = {
                        "type": "result",
                        "faces": [],  # Add detection logic
                        "count": 0
                    }
                    await ws_manager.send_message(json.dumps(result), websocket)
                
                elif msg.get("type") == "ping":
                    await ws_manager.send_message(
                        json.dumps({"type": "pong"}),
                        websocket
                    )
            
            except json.JSONDecodeError:
                await ws_manager.send_message(
                    json.dumps({"error": "Invalid JSON"}),
                    websocket
                )
    
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# Usage in api_server.py:
"""
from fastapi import WebSocket
from src.ws_handler import handle_face_detection_ws

@app.websocket("/ws/detect")
async def ws_endpoint(websocket: WebSocket):
    await handle_face_detection_ws(websocket)
"""
