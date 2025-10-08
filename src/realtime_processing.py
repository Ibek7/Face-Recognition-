# Advanced Real-Time Face Recognition with WebRTC and Streaming

import asyncio
import websockets
import json
import cv2
import numpy as np
import base64
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import uuid
from abc import ABC, abstractmethod
import socketio
from aiohttp import web, WSMsgType
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRelay
import fractions
import torch
import multiprocessing as mp
from multiprocessing import Process, Manager
import redis
import pickle

@dataclass
class StreamConfig:
    """Configuration for real-time streaming."""
    stream_type: str  # 'webcam', 'rtsp', 'webrtc', 'file'
    input_source: str  # camera index, RTSP URL, file path
    output_format: str  # 'websocket', 'rtmp', 'webrtc', 'file'
    resolution: tuple = (640, 480)
    fps: int = 30
    quality: int = 80  # JPEG quality for streaming
    buffer_size: int = 5
    processing_interval: float = 0.1  # Process every N seconds
    enable_gpu: bool = True

@dataclass
class RecognitionResult:
    """Real-time recognition result."""
    timestamp: float
    frame_id: str
    faces: List[Dict[str, Any]]
    processing_time: float
    confidence_threshold: float
    metadata: Dict[str, Any]

class VideoStreamTrackProcessor(VideoStreamTrack):
    """Custom video track for WebRTC with face recognition."""
    
    def __init__(self, track, face_processor):
        super().__init__()
        self.track = track
        self.face_processor = face_processor
        self.last_frame_time = time.time()
        
    async def recv(self):
        frame = await self.track.recv()
        
        # Process face recognition
        if time.time() - self.last_frame_time > 0.1:  # Process every 100ms
            img = frame.to_ndarray(format="bgr24")
            
            # Run face recognition
            results = await self.face_processor.process_frame_async(img)
            
            # Draw results on frame
            if results and results.faces:
                img = self._draw_faces(img, results.faces)
            
            # Convert back to frame
            frame = frame.from_ndarray(img, format="bgr24")
            self.last_frame_time = time.time()
        
        return frame
    
    def _draw_faces(self, img, faces):
        """Draw face detection results on image."""
        for face in faces:
            if 'bbox' in face:
                x, y, w, h = face['bbox']
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                if 'identity' in face and face['identity']:
                    label = f"{face['identity']} ({face.get('confidence', 0):.2f})"
                    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img

class RealtimeProcessor:
    """High-performance real-time face recognition processor."""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.is_running = False
        self.frame_queue = Queue(maxsize=config.buffer_size)
        self.result_queue = Queue(maxsize=100)
        self.clients = set()
        self.logger = logging.getLogger(__name__)
        
        # Initialize face recognition components
        self.face_detector = None
        self.face_encoder = None
        self.face_database = {}
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.processing_times = []
        
        # Initialize GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.enable_gpu else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=mp.cpu_count())
        
        # Redis for distributed processing (optional)
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
            self.redis_client.ping()
            self.logger.info("Redis connected for distributed processing")
        except:
            self.logger.info("Redis not available, using local processing only")
    
    async def initialize_models(self):
        """Initialize face recognition models."""
        try:
            # Import local modules (assuming they exist)
            from face_detector import FaceDetector
            from face_encoder import FaceEncoder
            
            self.face_detector = FaceDetector()
            self.face_encoder = FaceEncoder()
            
            # Load existing face database
            await self.load_face_database()
            
            self.logger.info("Face recognition models initialized")
            
        except ImportError:
            self.logger.warning("Face recognition modules not found, using mock processor")
            self.face_detector = MockFaceDetector()
            self.face_encoder = MockFaceEncoder()
    
    async def start_stream(self):
        """Start the real-time processing stream."""
        if self.is_running:
            return
        
        self.is_running = True
        await self.initialize_models()
        
        # Start processing threads
        processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        processing_thread.start()
        
        # Start capture based on stream type
        if self.config.stream_type == 'webcam':
            await self._start_webcam_capture()
        elif self.config.stream_type == 'rtsp':
            await self._start_rtsp_capture()
        elif self.config.stream_type == 'file':
            await self._start_file_capture()
        
        self.logger.info(f"Started {self.config.stream_type} stream processing")
    
    async def stop_stream(self):
        """Stop the real-time processing stream."""
        self.is_running = False
        self.logger.info("Stopped stream processing")
    
    async def _start_webcam_capture(self):
        """Start webcam capture loop."""
        cap = cv2.VideoCapture(int(self.config.input_source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue
                
                # Add frame to processing queue
                try:
                    frame_id = str(uuid.uuid4())
                    timestamp = time.time()
                    
                    if not self.frame_queue.full():
                        self.frame_queue.put((frame_id, timestamp, frame))
                    
                    # Control frame rate
                    await asyncio.sleep(1.0 / self.config.fps)
                    
                except Exception as e:
                    self.logger.error(f"Frame capture error: {e}")
        
        finally:
            cap.release()
    
    async def _start_rtsp_capture(self):
        """Start RTSP stream capture."""
        cap = cv2.VideoCapture(self.config.input_source)
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    # Try to reconnect
                    cap.release()
                    await asyncio.sleep(1)
                    cap = cv2.VideoCapture(self.config.input_source)
                    continue
                
                frame_id = str(uuid.uuid4())
                timestamp = time.time()
                
                if not self.frame_queue.full():
                    self.frame_queue.put((frame_id, timestamp, frame))
                
                await asyncio.sleep(1.0 / self.config.fps)
        
        finally:
            cap.release()
    
    async def _start_file_capture(self):
        """Start file-based capture."""
        cap = cv2.VideoCapture(self.config.input_source)
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    # Loop the video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                frame_id = str(uuid.uuid4())
                timestamp = time.time()
                
                if not self.frame_queue.full():
                    self.frame_queue.put((frame_id, timestamp, frame))
                
                await asyncio.sleep(1.0 / self.config.fps)
        
        finally:
            cap.release()
    
    def _processing_loop(self):
        """Main processing loop running in separate thread."""
        while self.is_running:
            try:
                # Get frame from queue
                frame_id, timestamp, frame = self.frame_queue.get(timeout=1.0)
                
                # Process frame
                start_time = time.time()
                result = self._process_frame(frame_id, timestamp, frame)
                processing_time = time.time() - start_time
                
                # Track performance
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                
                # Add result to queue
                if not self.result_queue.full():
                    self.result_queue.put(result)
                
                # Update FPS counter
                self.fps_counter += 1
                if time.time() - self.last_fps_time >= 1.0:
                    current_fps = self.fps_counter / (time.time() - self.last_fps_time)
                    self.logger.debug(f"Processing FPS: {current_fps:.1f}, Avg processing time: {np.mean(self.processing_times):.3f}s")
                    self.fps_counter = 0
                    self.last_fps_time = time.time()
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
    
    def _process_frame(self, frame_id: str, timestamp: float, frame: np.ndarray) -> RecognitionResult:
        """Process a single frame for face recognition."""
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            processed_faces = []
            for face in faces:
                # Extract face embedding
                face_embedding = self.face_encoder.encode_face(frame, face['bbox'])
                
                # Search for matches in database
                identity, confidence = self._search_face_database(face_embedding)
                
                face_result = {
                    'bbox': face['bbox'],
                    'confidence': face.get('confidence', 1.0),
                    'identity': identity,
                    'match_confidence': confidence,
                    'landmarks': face.get('landmarks', []),
                    'embedding': face_embedding.tolist() if isinstance(face_embedding, np.ndarray) else face_embedding
                }
                processed_faces.append(face_result)
            
            return RecognitionResult(
                timestamp=timestamp,
                frame_id=frame_id,
                faces=processed_faces,
                processing_time=time.time() - timestamp,
                confidence_threshold=0.7,
                metadata={'frame_shape': frame.shape}
            )
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return RecognitionResult(
                timestamp=timestamp,
                frame_id=frame_id,
                faces=[],
                processing_time=time.time() - timestamp,
                confidence_threshold=0.7,
                metadata={'error': str(e)}
            )
    
    async def process_frame_async(self, frame: np.ndarray) -> RecognitionResult:
        """Async wrapper for frame processing."""
        frame_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Run processing in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, self._process_frame, frame_id, timestamp, frame
        )
        
        return result
    
    def _search_face_database(self, face_embedding: np.ndarray) -> tuple:
        """Search for face match in database."""
        if not self.face_database:
            return None, 0.0
        
        best_match = None
        best_confidence = 0.0
        threshold = 0.7
        
        for identity, stored_embeddings in self.face_database.items():
            for stored_embedding in stored_embeddings:
                # Calculate cosine similarity
                similarity = np.dot(face_embedding, stored_embedding) / (
                    np.linalg.norm(face_embedding) * np.linalg.norm(stored_embedding)
                )
                
                if similarity > threshold and similarity > best_confidence:
                    best_match = identity
                    best_confidence = similarity
        
        return best_match, best_confidence
    
    async def add_face_to_database(self, identity: str, face_embedding: np.ndarray):
        """Add a face to the recognition database."""
        if identity not in self.face_database:
            self.face_database[identity] = []
        
        self.face_database[identity].append(face_embedding)
        
        # Save to Redis if available
        if self.redis_client:
            try:
                self.redis_client.hset(
                    'face_database', 
                    identity, 
                    pickle.dumps(self.face_database[identity])
                )
            except Exception as e:
                self.logger.warning(f"Failed to save to Redis: {e}")
        
        self.logger.info(f"Added face for identity: {identity}")
    
    async def load_face_database(self):
        """Load face database from storage."""
        try:
            # Try loading from Redis first
            if self.redis_client:
                keys = self.redis_client.hkeys('face_database')
                for key in keys:
                    data = self.redis_client.hget('face_database', key)
                    self.face_database[key.decode()] = pickle.loads(data)
                
                self.logger.info(f"Loaded {len(self.face_database)} identities from Redis")
                return
        except Exception as e:
            self.logger.warning(f"Failed to load from Redis: {e}")
        
        # Fallback to local file
        db_file = Path("face_database.json")
        if db_file.exists():
            with open(db_file, 'r') as f:
                data = json.load(f)
                self.face_database = {k: [np.array(emb) for emb in v] for k, v in data.items()}
            
            self.logger.info(f"Loaded {len(self.face_database)} identities from local file")
    
    async def get_latest_results(self, count: int = 10) -> List[RecognitionResult]:
        """Get latest recognition results."""
        results = []
        
        try:
            for _ in range(min(count, self.result_queue.qsize())):
                result = self.result_queue.get_nowait()
                results.append(result)
        except Empty:
            pass
        
        return results
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'fps': self.fps_counter / max((time.time() - self.last_fps_time), 1),
            'queue_size': self.frame_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'device': str(self.device),
            'connected_clients': len(self.clients)
        }

class WebSocketServer:
    """WebSocket server for real-time face recognition streaming."""
    
    def __init__(self, processor: RealtimeProcessor, host: str = 'localhost', port: int = 8765):
        self.processor = processor
        self.host = host
        self.port = port
        self.clients = set()
        self.logger = logging.getLogger(__name__)
    
    async def register_client(self, websocket):
        """Register a new client."""
        self.clients.add(websocket)
        self.processor.clients.add(websocket)
        self.logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial status
        await websocket.send(json.dumps({
            'type': 'status',
            'message': 'Connected to face recognition stream',
            'timestamp': time.time()
        }))
    
    async def unregister_client(self, websocket):
        """Unregister a client."""
        self.clients.discard(websocket)
        self.processor.clients.discard(websocket)
        self.logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def handle_client(self, websocket, path):
        """Handle client connection and messages."""
        await self.register_client(websocket)
        
        try:
            # Start result broadcasting
            broadcast_task = asyncio.create_task(self.broadcast_results(websocket))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON'
                    }))
                except Exception as e:
                    self.logger.error(f"Message handling error: {e}")
            
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
            broadcast_task.cancel()
    
    async def handle_message(self, websocket, data):
        """Handle incoming messages from clients."""
        message_type = data.get('type')
        
        if message_type == 'get_stats':
            stats = await self.processor.get_performance_stats()
            await websocket.send(json.dumps({
                'type': 'stats',
                'data': stats,
                'timestamp': time.time()
            }))
        
        elif message_type == 'add_face':
            # Handle face addition
            identity = data.get('identity')
            face_data = data.get('face_data')  # Base64 encoded image
            
            if identity and face_data:
                try:
                    # Decode image
                    image_data = base64.b64decode(face_data)
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Process and add to database
                    result = await self.processor.process_frame_async(frame)
                    if result.faces:
                        face_embedding = np.array(result.faces[0]['embedding'])
                        await self.processor.add_face_to_database(identity, face_embedding)
                        
                        await websocket.send(json.dumps({
                            'type': 'face_added',
                            'identity': identity,
                            'timestamp': time.time()
                        }))
                    else:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': 'No face detected in image'
                        }))
                
                except Exception as e:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': f'Failed to add face: {str(e)}'
                    }))
        
        elif message_type == 'configure':
            # Handle configuration updates
            config_updates = data.get('config', {})
            # Apply configuration updates to processor
            for key, value in config_updates.items():
                if hasattr(self.processor.config, key):
                    setattr(self.processor.config, key, value)
            
            await websocket.send(json.dumps({
                'type': 'config_updated',
                'timestamp': time.time()
            }))
    
    async def broadcast_results(self, websocket):
        """Broadcast recognition results to client."""
        while True:
            try:
                results = await self.processor.get_latest_results(5)
                
                if results:
                    for result in results:
                        message = {
                            'type': 'recognition_result',
                            'data': asdict(result),
                            'timestamp': time.time()
                        }
                        
                        await websocket.send(json.dumps(message, default=str))
                
                await asyncio.sleep(0.1)  # Broadcast every 100ms
                
            except websockets.exceptions.ConnectionClosed:
                break
            except Exception as e:
                self.logger.error(f"Broadcast error: {e}")
                await asyncio.sleep(1)
    
    async def start_server(self):
        """Start the WebSocket server."""
        self.logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(
            self.handle_client, 
            self.host, 
            self.port,
            max_size=1048576,  # 1MB max message size
            max_queue=32
        ):
            await asyncio.Future()  # Run forever

class WebRTCServer:
    """WebRTC server for real-time video streaming with face recognition."""
    
    def __init__(self, processor: RealtimeProcessor, port: int = 8080):
        self.processor = processor
        self.port = port
        self.app = web.Application()
        self.pcs = set()
        self.relay = MediaRelay()
        
        # Setup routes
        self._setup_routes()
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    def _setup_routes(self):
        """Setup HTTP routes for WebRTC signaling."""
        self.app.router.add_get('/', self.index)
        self.app.router.add_post('/offer', self.offer)
        self.app.router.add_get('/ws', self.websocket_handler)
    
    async def index(self, request):
        """Serve the main page."""
        content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Face Recognition WebRTC Stream</title>
        </head>
        <body>
            <h1>Real-time Face Recognition</h1>
            <video id="localVideo" autoplay muted></video>
            <video id="remoteVideo" autoplay></video>
            <div id="results"></div>
            
            <script>
                // WebRTC client implementation would go here
                console.log("Face Recognition WebRTC Client");
            </script>
        </body>
        </html>
        """
        return web.Response(text=content, content_type='text/html')
    
    async def offer(self, request):
        """Handle WebRTC offer."""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        
        pc = RTCPeerConnection()
        self.pcs.add(pc)
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)
        
        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                # Process video track with face recognition
                processed_track = VideoStreamTrackProcessor(
                    self.relay.subscribe(track), 
                    self.processor
                )
                pc.addTrack(processed_track)
        
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections for signaling."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    # Handle signaling messages
                    await ws.send_str(json.dumps({"type": "response", "data": "received"}))
                except json.JSONDecodeError:
                    await ws.send_str(json.dumps({"error": "Invalid JSON"}))
            elif msg.type == WSMsgType.ERROR:
                print(f'WebSocket error: {ws.exception()}')
        
        return ws
    
    async def start_server(self):
        """Start the WebRTC server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        
        print(f"WebRTC server started on http://localhost:{self.port}")

# Mock classes for testing when models aren't available
class MockFaceDetector:
    """Mock face detector for testing."""
    
    def detect_faces(self, frame):
        # Return mock detection
        h, w = frame.shape[:2]
        return [{
            'bbox': [w//4, h//4, w//2, h//2],
            'confidence': 0.95,
            'landmarks': []
        }]

class MockFaceEncoder:
    """Mock face encoder for testing."""
    
    def encode_face(self, frame, bbox):
        # Return mock embedding
        return np.random.randn(512)

# Example usage and server startup
async def main():
    """Main function to start the real-time face recognition system."""
    
    # Configuration
    config = StreamConfig(
        stream_type='webcam',
        input_source='0',
        output_format='websocket',
        resolution=(640, 480),
        fps=30,
        quality=80,
        buffer_size=5,
        processing_interval=0.1,
        enable_gpu=True
    )
    
    # Initialize processor
    processor = RealtimeProcessor(config)
    
    # Start stream processing
    await processor.start_stream()
    
    # Start WebSocket server
    ws_server = WebSocketServer(processor, host='localhost', port=8765)
    
    # Start WebRTC server
    webrtc_server = WebRTCServer(processor, port=8080)
    
    # Run servers concurrently
    await asyncio.gather(
        ws_server.start_server(),
        webrtc_server.start_server(),
        processor.start_stream()
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())