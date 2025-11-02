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
    
    def __init__(self, cascade_path: Optional[str] = None, min_face_size: int = 30):
        """
        Initialize face detector with cascade classifier.
        
        Args:
            cascade_path: Path to cascade classifier XML file
            min_face_size: Minimum face size in pixels (default: 30)
            
        Raises:
            ValueError: If cascade classifier cannot be loaded
            TypeError: If min_face_size is not a positive integer
        """
        if min_face_size <= 0:
            raise TypeError("min_face_size must be a positive integer")
            
        self.min_face_size = min_face_size
        self.cascade_path = cascade_path or cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError(f"Could not load cascade classifier from {self.cascade_path}")
        
        logger.info(f"Face detector initialized with min_face_size={min_face_size}")
            
    def detect_faces(self, image: np.ndarray, scale_factor: float = 1.1, 
                     min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image.
        
        Args:
            image: Input image as numpy array (BGR format)
            scale_factor: Scale factor for multi-scale detection (default: 1.1)
            min_neighbors: Minimum neighbors for detection (default: 5)
            
        Returns:
            List of (x, y, width, height) tuples for detected faces
            
        Raises:
            ValueError: If image is invalid or empty
            TypeError: If parameters are of incorrect type
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
            
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        if len(image.shape) not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (color)")
            
        if scale_factor <= 1.0:
            raise ValueError("scale_factor must be greater than 1.0")
            
        if min_neighbors < 0:
            raise ValueError("min_neighbors must be non-negative")
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=scale_factor, 
                minNeighbors=min_neighbors, 
                minSize=(self.min_face_size, self.min_face_size)
            )
            
            result = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
            logger.debug(f"Detected {len(result)} faces in image")
            return result
            
        except cv2.error as e:
            logger.error(f"OpenCV error during face detection: {str(e)}")
            raise ValueError(f"Face detection failed: {str(e)}")
    
    def extract_faces(self, image: np.ndarray, padding: float = 0.2, 
                     target_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Extract face regions from image with optional padding and resizing.
        
        Args:
            image: Input image as numpy array
            padding: Padding around face (as fraction of face size, 0.0-1.0)
            target_size: Optional (width, height) to resize extracted faces
            
        Returns:
            List of extracted face images
            
        Raises:
            ValueError: If image is invalid or padding is out of range
            TypeError: If parameters are of incorrect type
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
            
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        if not 0.0 <= padding <= 1.0:
            raise ValueError("padding must be between 0.0 and 1.0")
            
        if target_size is not None:
            if not isinstance(target_size, tuple) or len(target_size) != 2:
                raise TypeError("target_size must be a tuple of (width, height)")
            if target_size[0] <= 0 or target_size[1] <= 0:
                raise ValueError("target_size dimensions must be positive")
        
        try:
            faces = self.detect_faces(image)
            extracted = []
            
            h, w = image.shape[:2]
            
            for i, (x, y, fw, fh) in enumerate(faces):
                # Add padding
                pad_w = int(fw * padding)
                pad_h = int(fh * padding)
                
                # Calculate padded coordinates
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(w, x + fw + pad_w)
                y2 = min(h, y + fh + pad_h)
                
                # Extract face region
                face_img = image[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    # Resize if target size specified
                    if target_size is not None:
                        face_img = cv2.resize(face_img, target_size)
                    
                    extracted.append(face_img)
                    logger.debug(f"Extracted face {i+1}/{len(faces)} with shape {face_img.shape}")
                else:
                    logger.warning(f"Face {i+1}/{len(faces)} extraction resulted in empty image")
            
            logger.info(f"Successfully extracted {len(extracted)}/{len(faces)} faces")
            return extracted
            
        except Exception as e:
            logger.error(f"Error extracting faces: {str(e)}")
            raise