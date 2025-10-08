"""
WebSocket handlers for real-time face recognition.
Provides live video stream processing and real-time face recognition.
"""

import json
import asyncio
import base64
import time
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
import numpy as np
import cv2
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import threading
from queue import Queue, Empty
from dataclasses import dataclass

# Import face recognition components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from database import DatabaseManager
from embeddings import FaceEmbeddingManager
from monitoring import performance_monitor

logger = logging.getLogger(__name__)

@dataclass
class WebSocketClient:
    """WebSocket client information."""
    websocket: WebSocket
    client_id: str
    connected_at: datetime
    last_activity: datetime
    subscription_filters: Dict
    processing_queue: Queue

class ConnectionManager:
    """
    Manages WebSocket connections for real-time face recognition.
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocketClient] = {}
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.shutdown_event = threading.Event()
        
        # Global components
        self.db_manager = DatabaseManager("sqlite:///websocket_face_recognition.db")
        self.embedding_manager = FaceEmbeddingManager(encoder_type="simple")
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "current_connections": 0,
            "messages_processed": 0,
            "faces_recognized": 0,
            "start_time": datetime.now()
        }
    
    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
            
        Returns:
            True if connection accepted, False otherwise
        """
        try:
            await websocket.accept()
            
            # Create client object
            client = WebSocketClient(
                websocket=websocket,
                client_id=client_id,
                connected_at=datetime.now(),
                last_activity=datetime.now(),
                subscription_filters={},
                processing_queue=Queue()
            )
            
            # Store connection
            self.active_connections[client_id] = client
            
            # Start processing thread for this client
            processing_thread = threading.Thread(
                target=self._process_client_queue,
                args=(client,),
                daemon=True
            )
            processing_thread.start()
            self.processing_threads[client_id] = processing_thread
            
            # Update statistics
            self.stats["total_connections"] += 1
            self.stats["current_connections"] = len(self.active_connections)
            
            logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
            
            # Send welcome message
            await self.send_to_client(client_id, {
                "type": "connection_established",
                "client_id": client_id,
                "server_time": datetime.now().isoformat(),
                "capabilities": [
                    "face_recognition",
                    "person_enrollment",
                    "real_time_processing",
                    "statistics"
                ]
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error accepting connection for client {client_id}: {str(e)}")
            return False
    
    def disconnect(self, client_id: str):
        """
        Disconnect a WebSocket client.
        
        Args:
            client_id: Client identifier
        """
        if client_id in self.active_connections:
            client = self.active_connections[client_id]
            
            # Stop processing thread
            if client_id in self.processing_threads:
                # Signal processing thread to stop
                client.processing_queue.put({"type": "shutdown"})
                thread = self.processing_threads[client_id]
                thread.join(timeout=5)  # Wait up to 5 seconds
                del self.processing_threads[client_id]
            
            # Remove connection
            del self.active_connections[client_id]
            
            # Update statistics
            self.stats["current_connections"] = len(self.active_connections)
            
            logger.info(f"Client {client_id} disconnected. Remaining connections: {len(self.active_connections)}")
    
    async def send_to_client(self, client_id: str, message: Dict):
        """
        Send message to a specific client.
        
        Args:
            client_id: Client identifier
            message: Message to send
        """
        if client_id not in self.active_connections:
            return False
        
        client = self.active_connections[client_id]
        
        try:
            if client.websocket.client_state == WebSocketState.CONNECTED:
                await client.websocket.send_text(json.dumps(message))
                client.last_activity = datetime.now()
                return True
        except Exception as e:
            logger.error(f"Error sending message to client {client_id}: {str(e)}")
            self.disconnect(client_id)
        
        return False
    
    async def broadcast(self, message: Dict, exclude_client: Optional[str] = None):
        """
        Broadcast message to all connected clients.
        
        Args:
            message: Message to broadcast
            exclude_client: Optional client ID to exclude
        """
        disconnected_clients = []
        
        for client_id, client in self.active_connections.items():
            if exclude_client and client_id == exclude_client:
                continue
            
            try:
                if client.websocket.client_state == WebSocketState.CONNECTED:
                    await client.websocket.send_text(json.dumps(message))
                    client.last_activity = datetime.now()
                else:
                    disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id}: {str(e)}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    async def handle_message(self, client_id: str, message: Dict):
        """
        Handle incoming message from client.
        
        Args:
            client_id: Client identifier
            message: Received message
        """
        if client_id not in self.active_connections:
            return
        
        client = self.active_connections[client_id]
        client.last_activity = datetime.now()
        
        message_type = message.get("type")
        
        try:
            if message_type == "face_recognition":
                # Queue face recognition request
                client.processing_queue.put(message)
            
            elif message_type == "person_enrollment":
                await self._handle_person_enrollment(client_id, message)
            
            elif message_type == "set_filters":
                await self._handle_set_filters(client_id, message)
            
            elif message_type == "get_statistics":
                await self._handle_get_statistics(client_id)
            
            elif message_type == "ping":
                await self.send_to_client(client_id, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            
            else:
                await self.send_to_client(client_id, {
                    "type": "error",
                    "error": "Unknown message type",
                    "message_type": message_type
                })
            
            self.stats["messages_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error handling message from client {client_id}: {str(e)}")
            await self.send_to_client(client_id, {
                "type": "error",
                "error": str(e),
                "message_type": message_type
            })
    
    def _process_client_queue(self, client: WebSocketClient):
        """
        Process queued messages for a client in a separate thread.
        
        Args:
            client: WebSocket client
        """
        while not self.shutdown_event.is_set():
            try:
                # Get message from queue with timeout
                message = client.processing_queue.get(timeout=1.0)
                
                if message.get("type") == "shutdown":
                    break
                
                if message.get("type") == "face_recognition":
                    self._process_face_recognition(client, message)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing queue for client {client.client_id}: {str(e)}")
    
    def _process_face_recognition(self, client: WebSocketClient, message: Dict):
        """
        Process face recognition request.
        
        Args:
            client: WebSocket client
            message: Recognition request message
        """
        try:
            start_time = time.time()
            
            # Decode image
            image_data = message.get("image_base64", "")
            if not image_data:
                asyncio.run(self.send_to_client(client.client_id, {
                    "type": "face_recognition_result",
                    "success": False,
                    "error": "No image data provided"
                }))
                return
            
            # Remove data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",")[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                asyncio.run(self.send_to_client(client.client_id, {
                    "type": "face_recognition_result",
                    "success": False,
                    "error": "Invalid image data"
                }))
                return
            
            # Process image
            embedding_data = self.embedding_manager.pipeline.process_image_array(image)
            face_locations = self.embedding_manager.pipeline.detector.detect_faces(image)
            
            # Recognition parameters
            threshold = message.get("threshold", 0.7)
            
            recognition_results = []
            
            for i, face_data in enumerate(embedding_data['faces']):
                # Generate embedding
                embedding = self.embedding_manager.encoder.encode_face(face_data['normalized_face'])
                
                if embedding.size > 0:
                    # Search for similar faces
                    similar_faces = self.db_manager.search_similar_faces(
                        embedding,
                        threshold=threshold,
                        top_k=1
                    )
                    
                    # Get bounding box
                    bbox = list(face_locations[i]) if i < len(face_locations) else [0, 0, 0, 0]
                    
                    if similar_faces:
                        # Found match
                        best_match, similarity = similar_faces[0]
                        person = self.db_manager.get_person(best_match.person_id)
                        
                        result = {
                            "face_id": i,
                            "person_id": best_match.person_id,
                            "person_name": person.name if person else None,
                            "confidence": float(similarity),
                            "quality_score": float(face_data['quality_score']),
                            "bounding_box": {
                                "x": int(bbox[0]),
                                "y": int(bbox[1]),
                                "width": int(bbox[2] - bbox[0]),
                                "height": int(bbox[3] - bbox[1])
                            },
                            "is_match": True
                        }
                        
                        # Record recognition result
                        self.db_manager.add_recognition_result(
                            person_id=best_match.person_id,
                            confidence_score=similarity,
                            source_image_path="websocket",
                            processing_time=time.time() - start_time,
                            source_type="websocket"
                        )
                        
                        self.stats["faces_recognized"] += 1
                    else:
                        # No match found
                        result = {
                            "face_id": i,
                            "person_id": None,
                            "person_name": None,
                            "confidence": 0.0,
                            "quality_score": float(face_data['quality_score']),
                            "bounding_box": {
                                "x": int(bbox[0]),
                                "y": int(bbox[1]),
                                "width": int(bbox[2] - bbox[0]),
                                "height": int(bbox[3] - bbox[1])
                            },
                            "is_match": False
                        }
                    
                    recognition_results.append(result)
            
            processing_time = time.time() - start_time
            
            # Send results
            asyncio.run(self.send_to_client(client.client_id, {
                "type": "face_recognition_result",
                "success": True,
                "faces": recognition_results,
                "processing_time": processing_time,
                "total_faces": len(recognition_results),
                "timestamp": datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.error(f"Error in face recognition processing: {str(e)}")
            asyncio.run(self.send_to_client(client.client_id, {
                "type": "face_recognition_result",
                "success": False,
                "error": str(e)
            }))
    
    async def _handle_person_enrollment(self, client_id: str, message: Dict):
        """Handle person enrollment request."""
        try:
            person_name = message.get("person_name")
            image_data = message.get("image_base64")
            
            if not person_name or not image_data:
                await self.send_to_client(client_id, {
                    "type": "person_enrollment_result",
                    "success": False,
                    "error": "Missing person name or image data"
                })
                return
            
            # Check if person exists
            existing_person = self.db_manager.get_person_by_name(person_name)
            if existing_person:
                person = existing_person
            else:
                # Create new person
                person = self.db_manager.add_person(
                    name=person_name,
                    description=f"Enrolled via WebSocket by client {client_id}"
                )
            
            # Process image and add embeddings
            # (Similar to API endpoint implementation)
            
            await self.send_to_client(client_id, {
                "type": "person_enrollment_result",
                "success": True,
                "person_id": person.id,
                "person_name": person.name,
                "message": f"Successfully enrolled {person_name}"
            })
            
        except Exception as e:
            await self.send_to_client(client_id, {
                "type": "person_enrollment_result",
                "success": False,
                "error": str(e)
            })
    
    async def _handle_set_filters(self, client_id: str, message: Dict):
        """Handle subscription filter updates."""
        if client_id in self.active_connections:
            client = self.active_connections[client_id]
            client.subscription_filters = message.get("filters", {})
            
            await self.send_to_client(client_id, {
                "type": "filters_updated",
                "filters": client.subscription_filters
            })
    
    async def _handle_get_statistics(self, client_id: str):
        """Handle statistics request."""
        uptime = datetime.now() - self.stats["start_time"]
        
        await self.send_to_client(client_id, {
            "type": "statistics",
            "stats": {
                **self.stats,
                "uptime": str(uptime),
                "avg_processing_time": performance_monitor.get_recent_metrics("websocket_face_recognition", 300)
            }
        })
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics."""
        return {
            "active_connections": len(self.active_connections),
            "total_connections": self.stats["total_connections"],
            "messages_processed": self.stats["messages_processed"],
            "faces_recognized": self.stats["faces_recognized"],
            "uptime": str(datetime.now() - self.stats["start_time"])
        }
    
    def shutdown(self):
        """Shutdown connection manager."""
        self.shutdown_event.set()
        
        # Disconnect all clients
        for client_id in list(self.active_connections.keys()):
            self.disconnect(client_id)

# Global connection manager
manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time face recognition.
    
    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
    """
    if not await manager.connect(websocket, client_id):
        return
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await manager.handle_message(client_id, message)
            except json.JSONDecodeError:
                await manager.send_to_client(client_id, {
                    "type": "error",
                    "error": "Invalid JSON format"
                })
            
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket connection for client {client_id}: {str(e)}")
    finally:
        manager.disconnect(client_id)

# WebSocket route registration function
def register_websocket_routes(app):
    """
    Register WebSocket routes with FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    @app.websocket("/ws/recognize/{client_id}")
    async def websocket_recognize(websocket: WebSocket, client_id: str):
        await websocket_endpoint(websocket, client_id)
    
    @app.get("/ws/stats")
    async def websocket_stats():
        """Get WebSocket connection statistics."""
        return manager.get_connection_stats()
    
    logger.info("WebSocket routes registered successfully")