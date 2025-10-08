"""
Face detection utilities using OpenCV and dlib.
Provides preprocessing pipeline for face recognition.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    """Face detection using OpenCV cascades and optional dlib."""
    
    def __init__(self, cascade_path: Optional[str] = None):
        """Initialize face detector with cascade classifier."""
        self.cascade_path = cascade_path or cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError(f"Could not load cascade classifier from {self.cascade_path}")
            
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image.
        
        Returns:
            List of (x, y, width, height) tuples for detected faces
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    
    def extract_faces(self, image: np.ndarray, padding: float = 0.2) -> List[np.ndarray]:
        """
        Extract face regions from image with optional padding.
        
        Args:
            image: Input image
            padding: Padding around face (as fraction of face size)
            
        Returns:
            List of extracted face images
        """
        faces = self.detect_faces(image)
        extracted = []
        
        h, w = image.shape[:2]
        
        for x, y, fw, fh in faces:
            # Add padding
            pad_w = int(fw * padding)
            pad_h = int(fh * padding)
            
            # Calculate padded coordinates
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + fw + pad_w)
            y2 = min(h, y + fh + pad_h)
            
            face_img = image[y1:y2, x1:x2]
            if face_img.size > 0:
                extracted.append(face_img)
                
        return extracted