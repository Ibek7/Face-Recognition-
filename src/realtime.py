"""
Real-time face recognition using webcam.
Provides live face detection, recognition, and visualization.
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

from src.database import DatabaseManager
from src.embeddings import FaceEmbeddingManager
from src.pipeline import FaceProcessingPipeline
from src.similarity import FaceSimilarity, DistanceMetric

logger = logging.getLogger(__name__)

class RealTimeFaceRecognizer:
    """Real-time face recognition system using webcam."""
    
    def __init__(self, 
                 db_path: str = "face_recognition.db",
                 encoder_type: str = "simple",
                 confidence_threshold: float = 0.7,
                 recognition_interval: float = 0.5):
        """
        Initialize real-time face recognizer.
        
        Args:
            db_path: Path to face database
            encoder_type: Type of face encoder
            confidence_threshold: Minimum confidence for recognition
            recognition_interval: Seconds between recognition attempts
        """
        self.db_manager = DatabaseManager(f"sqlite:///{db_path}")
        self.embedding_manager = FaceEmbeddingManager(encoder_type=encoder_type)
        self.confidence_threshold = confidence_threshold
        self.recognition_interval = recognition_interval
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.stop_event = Event()
        
        # Recognition state
        self.last_recognition_time = 0
        self.current_recognitions = {}
        
        # Display settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.text_thickness = 2
        
        # Colors (BGR format)
        self.colors = {
            'unknown': (0, 0, 255),      # Red
            'known': (0, 255, 0),        # Green
            'processing': (0, 255, 255), # Yellow
            'box': (255, 255, 255),      # White
            'text_bg': (0, 0, 0)         # Black
        }
        
        logger.info("Real-time face recognizer initialized")
    
    def start_webcam(self, camera_id: int = 0, display: bool = True) -> None:
        """
        Start webcam-based face recognition.
        
        Args:
            camera_id: Camera device ID
            display: Whether to show live display
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Start recognition thread
        recognition_thread = Thread(target=self._recognition_worker, daemon=True)
        recognition_thread.start()
        
        logger.info("Starting webcam face recognition. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Add frame to queue for processing (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass  # Skip frame if queue is full
                
                # Get latest recognition results
                while not self.result_queue.empty():
                    try:
                        self.current_recognitions = self.result_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Draw recognition results on frame
                display_frame = self._draw_recognitions(frame)
                
                if display:
                    cv2.imshow('Real-time Face Recognition', display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current frame
                        timestamp = int(time.time())
                        filename = f"capture_{timestamp}.jpg"
                        cv2.imwrite(filename, display_frame)
                        logger.info(f"Saved frame: {filename}")
        
        finally:
            # Cleanup
            self.stop_event.set()
            cap.release()
            if display:
                cv2.destroyAllWindows()
            logger.info("Webcam recognition stopped")
    
    def _recognition_worker(self) -> None:
        """Background thread for face recognition processing."""
        while not self.stop_event.is_set():
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)
                
                # Check if enough time has passed since last recognition
                current_time = time.time()
                if current_time - self.last_recognition_time < self.recognition_interval:
                    continue
                
                # Process frame
                results = self._process_frame(frame)
                
                # Send results back to main thread
                self.result_queue.put(results)
                self.last_recognition_time = current_time
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in recognition worker: {e}")
    
    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process frame for face recognition.
        
        Args:
            frame: Input frame
            
        Returns:
            Recognition results
        """
        try:
            # Generate embeddings for faces in frame
            embedding_data = self.embedding_manager.pipeline.process_image_array(frame)
            
            recognitions = {}
            
            # Process each detected face
            for face_data in embedding_data['faces']:
                face_id = face_data['face_id']
                
                # Generate embedding
                embedding = self.embedding_manager.encoder.encode_face(face_data['normalized_face'])
                
                if embedding.size > 0:
                    # Search for similar faces in database
                    similar_faces = self.db_manager.search_similar_faces(
                        embedding,
                        threshold=self.confidence_threshold,
                        top_k=1
                    )
                    
                    if similar_faces:
                        # Get best match
                        best_match, similarity = similar_faces[0]
                        person = self.db_manager.get_person(best_match.person_id)
                        
                        recognition_result = {
                            'person_name': person.name if person else "Unknown",
                            'person_id': best_match.person_id,
                            'confidence': similarity,
                            'quality_score': face_data['quality_score'],
                            'status': 'known'
                        }
                    else:
                        recognition_result = {
                            'person_name': "Unknown",
                            'person_id': None,
                            'confidence': 0.0,
                            'quality_score': face_data['quality_score'],
                            'status': 'unknown'
                        }
                    
                    recognitions[face_id] = recognition_result
            
            return recognitions
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {}
    
    def _draw_recognitions(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw recognition results on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with recognition annotations
        """
        display_frame = frame.copy()
        
        # Detect faces for bounding boxes
        face_locations = self.embedding_manager.pipeline.detector.detect_faces(frame)
        
        for i, (x, y, w, h) in enumerate(face_locations):
            # Get recognition result for this face
            recognition = self.current_recognitions.get(i, {})
            
            # Determine color based on recognition status
            status = recognition.get('status', 'processing')
            color = self.colors.get(status, self.colors['unknown'])
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare text
            person_name = recognition.get('person_name', 'Processing...')
            confidence = recognition.get('confidence', 0.0)
            quality = recognition.get('quality_score', 0.0)
            
            if confidence > 0:
                text = f"{person_name} ({confidence:.2f})"
            else:
                text = person_name
            
            # Draw text background
            text_size = cv2.getTextSize(text, self.font, self.font_scale, self.text_thickness)[0]
            text_bg_pt1 = (x, y - text_size[1] - 10)
            text_bg_pt2 = (x + text_size[0], y)
            cv2.rectangle(display_frame, text_bg_pt1, text_bg_pt2, self.colors['text_bg'], -1)
            
            # Draw text
            cv2.putText(display_frame, text, (x, y - 5), 
                       self.font, self.font_scale, color, self.text_thickness)
            
            # Draw quality indicator
            quality_text = f"Q: {quality:.1f}"
            cv2.putText(display_frame, quality_text, (x, y + h + 15),
                       self.font, 0.5, color, 1)
        
        # Draw system info
        self._draw_system_info(display_frame)
        
        return display_frame
    
    def _draw_system_info(self, frame: np.ndarray) -> None:
        """Draw system information on frame."""
        h, w = frame.shape[:2]
        
        # Get database stats
        stats = self.db_manager.get_recognition_stats()
        
        # System info text
        info_lines = [
            f"Persons: {stats.get('total_persons', 0)}",
            f"Embeddings: {stats.get('total_embeddings', 0)}",
            f"Recognitions: {stats.get('total_recognitions', 0)}",
            f"Threshold: {self.confidence_threshold:.2f}"
        ]
        
        # Draw background
        info_height = len(info_lines) * 25 + 10
        cv2.rectangle(frame, (10, 10), (300, info_height), self.colors['text_bg'], -1)
        
        # Draw text
        for i, line in enumerate(info_lines):
            y_pos = 30 + i * 25
            cv2.putText(frame, line, (15, y_pos), 
                       self.font, 0.6, self.colors['box'], 1)
    
    def add_person_from_webcam(self, 
                              person_name: str,
                              camera_id: int = 0,
                              num_samples: int = 5) -> bool:
        """
        Add a new person by capturing samples from webcam.
        
        Args:
            person_name: Name of the person to add
            camera_id: Camera device ID
            num_samples: Number of face samples to capture
            
        Returns:
            True if successful, False otherwise
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return False
        
        # Add person to database
        person = self.db_manager.add_person(person_name)
        samples_captured = 0
        
        logger.info(f"Adding person '{person_name}'. Press SPACE to capture, 'q' to quit.")
        
        try:
            while samples_captured < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame to detect faces
                face_locations = self.embedding_manager.pipeline.detector.detect_faces(frame)
                
                # Draw instructions and face boxes
                display_frame = frame.copy()
                
                for x, y, w, h in face_locations:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw instructions
                instruction = f"Capturing {person_name} ({samples_captured}/{num_samples})"
                cv2.putText(display_frame, instruction, (10, 30),
                           self.font, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "Press SPACE to capture", (10, 60),
                           self.font, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Add Person', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and face_locations:
                    # Capture sample
                    try:
                        embedding_data = self.embedding_manager.pipeline.process_image_array(frame)
                        
                        for face_data in embedding_data['faces']:
                            embedding = self.embedding_manager.encoder.encode_face(face_data['normalized_face'])
                            
                            if embedding.size > 0:
                                self.db_manager.add_face_embedding(
                                    embedding=embedding,
                                    person_id=person.id,
                                    source_image_path=f"webcam_sample_{samples_captured}",
                                    quality_score=face_data['quality_score'],
                                    encoder_type=self.embedding_manager.encoder.__class__.__name__,
                                    source_type="webcam"
                                )
                                
                                samples_captured += 1
                                logger.info(f"Captured sample {samples_captured}/{num_samples}")
                                break
                    
                    except Exception as e:
                        logger.error(f"Error capturing sample: {e}")
                
                elif key == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        success = samples_captured > 0
        if success:
            logger.info(f"Successfully added {samples_captured} samples for '{person_name}'")
        else:
            logger.warning(f"No samples captured for '{person_name}'")
        
        return success