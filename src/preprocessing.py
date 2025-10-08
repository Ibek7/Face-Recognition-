"""
Image preprocessing utilities for face recognition pipeline.
Handles normalization, augmentation, and quality assessment.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Image preprocessing for face recognition."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """Initialize with target image dimensions."""
        self.target_size = target_size
        
    def normalize_face(self, face_img: np.ndarray) -> np.ndarray:
        """
        Normalize face image for consistent processing.
        
        Args:
            face_img: Input face image
            
        Returns:
            Normalized face image
        """
        # Resize to target size
        resized = cv2.resize(face_img, self.target_size)
        
        # Convert to float and normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Histogram equalization for better contrast
        if len(normalized.shape) == 3:
            # Convert to LAB, equalize L channel, convert back
            lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.equalizeHist((lab[:,:,0] * 255).astype(np.uint8)) / 255.0
            normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            normalized = cv2.equalizeHist((normalized * 255).astype(np.uint8)) / 255.0
            
        return normalized
    
    def assess_quality(self, face_img: np.ndarray) -> float:
        """
        Assess face image quality using Laplacian variance.
        
        Returns:
            Quality score (higher is better)
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
        return cv2.Laplacian(gray, cv2.CV_64F).var()