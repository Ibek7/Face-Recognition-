#!/usr/bin/env python3
"""
WebSocket Real-Time Notifications

Provides real-time updates for face detection and recognition events using WebSocket.

Features:
- Real-time face detection events
- Recognition match notifications
- System status updates
- Client subscription management
- Event filtering and routing
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, Optional, Any, List
from enum import Enum
import uuid

from fastapi import WebSocket, WebSocketDisconnect, FastAPI, Depends
from fastapi.websockets import WebSocketState
import redis.asyncio as aioredis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """WebSocket event types"""
    FACE_DETECTED = "face_detected"
    FACE_RECOGNIZED = "face_recognized"
    PERSON_ENROLLED = "person_enrolled"
    PERSON_DELETED = "person_deleted"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class Event:
    """Event message"""
    
    def __init__(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        event_id: Optional[str] = None
    ):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.utcnow()
        self.event_id = event_id or str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict())


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_subscriptions: Dict[str, Set[EventType]] = {}
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.pubsub_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("✓ Redis connection established")
            
            # Start pubsub listener
            self.pubsub_task = asyncio.create_task(self._listen_redis_events())
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    async def shutdown(self):
        """Shutdown manager"""
        if self.pubsub_task:
            self.pubsub_task.cancel()
        
        if self.redis:
            await self.redis.close()
        
        # Close all connections
        for client_id in list(self.active_connections.keys()):
            await self.disconnect(client_id)
    
    async def connect(
        self,
        websocket: WebSocket,
        client_id: Optional[str] = None
    ) -> str:
        """
        Accept WebSocket connection
        
        Args:
            websocket: WebSocket connection
            client_id: Optional client identifier
        
        Returns:
            client_id
        """
        await websocket.accept()
        
        if client_id is None:
            client_id = str(uuid.uuid4())
        
        self.active_connections[client_id] = websocket
        self.client_subscriptions[client_id] = set()
        
        logger.info(f"Client connected: {client_id}")
        
        # Send connection confirmation
        await self.send_to_client(
            client_id,
            Event(
                EventType.SYSTEM_STATUS,
                {
                    "status": "connected",
                    "client_id": client_id,
                    "message": "WebSocket connection established"
                }
            )
        )
        
        return client_id
    
    async def disconnect(self, client_id: str):
        """Disconnect client"""
        websocket = self.active_connections.get(client_id)
        
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
        
        self.active_connections.pop(client_id, None)
        self.client_subscriptions.pop(client_id, None)
        
        logger.info(f"Client disconnected: {client_id}")
    
    async def subscribe(self, client_id: str, event_types: List[EventType]):
        """Subscribe client to event types"""
        if client_id in self.client_subscriptions:
            self.client_subscriptions[client_id].update(event_types)
            logger.info(f"Client {client_id} subscribed to {event_types}")
    
    async def unsubscribe(self, client_id: str, event_types: List[EventType]):
        """Unsubscribe client from event types"""
        if client_id in self.client_subscriptions:
            self.client_subscriptions[client_id].difference_update(event_types)
            logger.info(f"Client {client_id} unsubscribed from {event_types}")
    
    async def send_to_client(self, client_id: str, event: Event):
        """Send event to specific client"""
        websocket = self.active_connections.get(client_id)
        
        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(event.to_json())
            except Exception as e:
                logger.error(f"Error sending to client {client_id}: {e}")
                await self.disconnect(client_id)
    
    async def broadcast(
        self,
        event: Event,
        filter_by_subscription: bool = True
    ):
        """
        Broadcast event to all connected clients
        
        Args:
            event: Event to broadcast
            filter_by_subscription: Only send to subscribed clients
        """
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            # Check subscription filter
            if filter_by_subscription:
                subscriptions = self.client_subscriptions.get(client_id, set())
                if event.event_type not in subscriptions:
                    continue
            
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(event.to_json())
                else:
                    disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
    
    async def publish_event(self, event: Event):
        """Publish event to Redis for distributed systems"""
        if self.redis:
            await self.redis.publish(
                "face_recognition:events",
                event.to_json()
            )
    
    async def _listen_redis_events(self):
        """Listen for events from Redis pub/sub"""
        if not self.redis:
            return
        
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("face_recognition:events")
        
        logger.info("✓ Listening for Redis events")
        
        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        event_data = json.loads(message['data'])
                        event = Event(
                            EventType(event_data['event_type']),
                            event_data['data'],
                            datetime.fromisoformat(event_data['timestamp']),
                            event_data['event_id']
                        )
                        await self.broadcast(event)
                    except Exception as e:
                        logger.error(f"Error processing Redis event: {e}")
        except asyncio.CancelledError:
            await pubsub.unsubscribe("face_recognition:events")
            await pubsub.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "active_connections": sum(
                1 for ws in self.active_connections.values()
                if ws.client_state == WebSocketState.CONNECTED
            ),
            "subscriptions": {
                client_id: list(subs)
                for client_id, subs in self.client_subscriptions.items()
            }
        }


# Global connection manager
manager = ConnectionManager()


class WebSocketServer:
    """WebSocket server for face recognition events"""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.setup_routes()
    
    def setup_routes(self):
        """Setup WebSocket routes"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Main WebSocket endpoint"""
            client_id = None
            
            try:
                client_id = await manager.connect(websocket)
                
                # Keep connection alive
                while True:
                    # Receive messages from client
                    message = await websocket.receive_text()
                    await self._handle_client_message(client_id, message)
                    
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected")
            except Exception as e:
                logger.error(f"WebSocket error for client {client_id}: {e}")
            finally:
                if client_id:
                    await manager.disconnect(client_id)
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_with_id(websocket: WebSocket, client_id: str):
            """WebSocket endpoint with custom client ID"""
            try:
                await manager.connect(websocket, client_id)
                
                while True:
                    message = await websocket.receive_text()
                    await self._handle_client_message(client_id, message)
                    
            except WebSocketDisconnect:
                logger.info(f"Client {client_id} disconnected")
            except Exception as e:
                logger.error(f"WebSocket error for client {client_id}: {e}")
            finally:
                await manager.disconnect(client_id)
        
        @self.app.get("/ws/stats")
        async def websocket_stats():
            """Get WebSocket connection statistics"""
            return manager.get_stats()
    
    async def _handle_client_message(self, client_id: str, message: str):
        """Handle messages from client"""
        try:
            data = json.loads(message)
            action = data.get('action')
            
            if action == 'subscribe':
                event_types = [EventType(et) for et in data.get('event_types', [])]
                await manager.subscribe(client_id, event_types)
                
                await manager.send_to_client(
                    client_id,
                    Event(
                        EventType.SYSTEM_STATUS,
                        {
                            "status": "subscribed",
                            "event_types": data.get('event_types', [])
                        }
                    )
                )
            
            elif action == 'unsubscribe':
                event_types = [EventType(et) for et in data.get('event_types', [])]
                await manager.unsubscribe(client_id, event_types)
                
                await manager.send_to_client(
                    client_id,
                    Event(
                        EventType.SYSTEM_STATUS,
                        {
                            "status": "unsubscribed",
                            "event_types": data.get('event_types', [])
                        }
                    )
                )
            
            elif action == 'ping':
                await manager.send_to_client(
                    client_id,
                    Event(
                        EventType.HEARTBEAT,
                        {"status": "pong"}
                    )
                )
            
            else:
                logger.warning(f"Unknown action from client {client_id}: {action}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from client {client_id}: {message}")
        except Exception as e:
            logger.error(f"Error handling message from client {client_id}: {e}")


# Event publishers (to be used by other parts of the application)

async def publish_face_detected(
    face_id: str,
    confidence: float,
    bbox: List[float],
    metadata: Optional[Dict] = None
):
    """Publish face detection event"""
    event = Event(
        EventType.FACE_DETECTED,
        {
            "face_id": face_id,
            "confidence": confidence,
            "bbox": bbox,
            "metadata": metadata or {}
        }
    )
    
    await manager.broadcast(event)
    await manager.publish_event(event)


async def publish_face_recognized(
    face_id: str,
    person_id: str,
    person_name: str,
    confidence: float,
    metadata: Optional[Dict] = None
):
    """Publish face recognition event"""
    event = Event(
        EventType.FACE_RECOGNIZED,
        {
            "face_id": face_id,
            "person_id": person_id,
            "person_name": person_name,
            "confidence": confidence,
            "metadata": metadata or {}
        }
    )
    
    await manager.broadcast(event)
    await manager.publish_event(event)


async def publish_person_enrolled(person_id: str, person_name: str):
    """Publish person enrollment event"""
    event = Event(
        EventType.PERSON_ENROLLED,
        {
            "person_id": person_id,
            "person_name": person_name
        }
    )
    
    await manager.broadcast(event)
    await manager.publish_event(event)


async def publish_system_status(status: str, message: str):
    """Publish system status event"""
    event = Event(
        EventType.SYSTEM_STATUS,
        {
            "status": status,
            "message": message
        }
    )
    
    await manager.broadcast(event, filter_by_subscription=False)


# Application lifecycle events

async def startup_event():
    """Initialize WebSocket manager on startup"""
    await manager.initialize()
    logger.info("✓ WebSocket server started")


async def shutdown_event():
    """Cleanup on shutdown"""
    await manager.shutdown()
    logger.info("✓ WebSocket server stopped")


# Example usage
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Face Recognition WebSocket Server")
    
    # Setup WebSocket server
    ws_server = WebSocketServer(app)
    
    # Add lifecycle events
    app.add_event_handler("startup", startup_event)
    app.add_event_handler("shutdown", shutdown_event)
    
    # Test endpoint to trigger events
    @app.post("/test/detect")
    async def test_detect():
        """Test endpoint to simulate face detection"""
        await publish_face_detected(
            face_id="test_face_123",
            confidence=0.95,
            bbox=[100, 100, 200, 200],
            metadata={"source": "camera_1"}
        )
        return {"status": "event published"}
    
    @app.post("/test/recognize")
    async def test_recognize():
        """Test endpoint to simulate face recognition"""
        await publish_face_recognized(
            face_id="test_face_123",
            person_id="person_456",
            person_name="John Doe",
            confidence=0.92,
            metadata={"source": "camera_1"}
        )
        return {"status": "event published"}
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
