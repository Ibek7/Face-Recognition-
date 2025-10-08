"""
Pipeline for processing images through the complete face detection workflow.
Combines detection, extraction, and preprocessing into a unified interface.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path

from .detection import FaceDetector
from .preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)

class FaceProcessingPipeline:
    """Complete pipeline for face detection and preprocessing."""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 quality_threshold: float = 100.0,
                 cascade_path: Optional[str] = None):
        """
        Initialize face processing pipeline.
        
        Args:
            target_size: Target size for normalized faces
            quality_threshold: Minimum quality score for faces
            cascade_path: Path to cascade classifier
        """
        self.detector = FaceDetector(cascade_path)
        self.preprocessor = ImagePreprocessor(target_size)
        self.quality_threshold = quality_threshold
        
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with processed faces and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        return self.process_image_array(image, image_path)
    
    def process_image_array(self, image: np.ndarray, source: str = "array") -> Dict[str, Any]:
        """
        Process image array through the pipeline.
        
        Args:
            image: Input image array
            source: Source identifier for logging
            
        Returns:
            Dictionary with processed faces and metadata
        """
        results = {
            'source': source,
            'input_shape': image.shape,
            'faces': [],
            'metadata': {
                'total_faces_detected': 0,
                'quality_faces': 0,
                'low_quality_faces': 0
            }
        }
        
        # Extract faces
        face_images = self.detector.extract_faces(image)
        results['metadata']['total_faces_detected'] = len(face_images)
        
        # Process each face
        for i, face_img in enumerate(face_images):
            # Assess quality
            quality_score = self.preprocessor.assess_quality(face_img)
            
            if quality_score >= self.quality_threshold:
                # Normalize face
                normalized_face = self.preprocessor.normalize_face(face_img)
                
                face_data = {
                    'face_id': i,
                    'quality_score': quality_score,
                    'original_face': face_img,
                    'normalized_face': normalized_face,
                    'shape': face_img.shape
                }
                
                results['faces'].append(face_data)
                results['metadata']['quality_faces'] += 1
            else:
                results['metadata']['low_quality_faces'] += 1
                logger.warning(f"Face {i} quality too low: {quality_score:.2f}")
                
        return results