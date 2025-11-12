"""
Face encoding and embedding generation using deep learning models.
Supports multiple backends including face_recognition library and custom models.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class FaceEncoder(ABC):
    """Abstract base class for face encoders."""
    
    @abstractmethod
    def encode_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate face encoding from image.
        
        Args:
            face_image: Preprocessed face image
            
        Returns:
            Face encoding vector
        """
        pass
    
    @abstractmethod
    def encode_faces(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate face encodings for multiple images.
        
        Args:
            face_images: List of preprocessed face images
            
        Returns:
            List of face encoding vectors
        """
        pass

class DlibFaceEncoder(FaceEncoder):
    """Face encoder using dlib's face recognition model."""
    
    def __init__(self):
        """Initialize dlib face encoder."""
        try:
            import face_recognition
            self.face_recognition = face_recognition
        except ImportError:
            raise ImportError("face_recognition library not found. Install with: pip install face-recognition")
    
    def encode_face(self, face_image: np.ndarray) -> np.ndarray:
        """Generate face encoding using dlib."""
        # Convert to RGB if needed
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            # Assume BGR input, convert to RGB
            rgb_image = face_image[:, :, ::-1]
        else:
            rgb_image = face_image
            
        # Ensure proper data type
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
            
        encodings = self.face_recognition.face_encodings(rgb_image)
        
        if len(encodings) == 0:
            logger.warning("No face found in image for encoding")
            return np.array([])
        
        # Return first encoding if multiple faces found
        return encodings[0]
    
    def encode_faces(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """Generate face encodings for multiple images."""
        encodings = []
        for face_img in face_images:
            encoding = self.encode_face(face_img)
            if encoding.size > 0:
                encodings.append(encoding)
        return encodings

class SimpleEmbeddingEncoder(FaceEncoder):
    """Simple CNN-based face encoder for demonstration."""
    
    def __init__(self, embedding_dim: int = 128):
        """Initialize simple encoder."""
        self.embedding_dim = embedding_dim
        logger.info(f"Initialized simple encoder with {embedding_dim}D embeddings")
        
    def encode_face(self, face_image: np.ndarray) -> np.ndarray:
        """Generate simple face encoding using basic features."""
        # Flatten and normalize image
        if len(face_image.shape) == 3:
            gray = np.mean(face_image, axis=2)
        else:
            gray = face_image
            
        # Simple feature extraction (for demo purposes)
        # In practice, you'd use a trained CNN
        flattened = gray.flatten()
        
        # Reduce dimensionality using simple averaging
        step = len(flattened) // self.embedding_dim
        if step < 1:
            step = 1
            
        embedding = []
        for i in range(0, len(flattened), step):
            chunk = flattened[i:i+step]
            embedding.append(np.mean(chunk))
            
        # Pad or trim to exact dimension
        embedding = np.array(embedding[:self.embedding_dim])
        if len(embedding) < self.embedding_dim:
            embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
            
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def encode_faces(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """Generate face encodings for multiple images."""
        return [self.encode_face(face_img) for face_img in face_images]

def FaceEncoderFactory(encoder_type: str = "dlib") -> FaceEncoder:
    """
    Factory function to create face encoders.
    
    Args:
        encoder_type: Type of encoder ('dlib' or 'simple')
        
    Returns:
        An instance of a FaceEncoder
    """
    if encoder_type == "dlib":
        return DlibFaceEncoder()
    elif encoder_type == "simple":
        return SimpleEmbeddingEncoder()
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
