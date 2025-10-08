"""
Face embedding utilities for generating and managing face representations.
Combines face detection, preprocessing, and encoding into unified workflow.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import cv2

from .pipeline import FaceProcessingPipeline
from .encoders import FaceEncoder, DlibFaceEncoder, SimpleEmbeddingEncoder
from .similarity import FaceSimilarity, DistanceMetric

logger = logging.getLogger(__name__)

class FaceEmbeddingManager:
    """Manages face embeddings with integrated processing pipeline."""
    
    def __init__(self, 
                 encoder_type: str = "simple",
                 distance_metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
                 quality_threshold: float = 100.0):
        """
        Initialize face embedding manager.
        
        Args:
            encoder_type: Type of encoder ("dlib" or "simple")
            distance_metric: Metric for similarity calculations
            quality_threshold: Minimum face quality threshold
        """
        self.pipeline = FaceProcessingPipeline(quality_threshold=quality_threshold)
        self.similarity = FaceSimilarity(distance_metric)
        
        # Initialize encoder
        if encoder_type == "dlib":
            try:
                self.encoder = DlibFaceEncoder()
            except ImportError:
                logger.warning("Dlib encoder not available, falling back to simple encoder")
                self.encoder = SimpleEmbeddingEncoder()
        else:
            self.encoder = SimpleEmbeddingEncoder()
            
        logger.info(f"Initialized embedding manager with {encoder_type} encoder")
    
    def generate_embeddings_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Generate face embeddings from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with embeddings and metadata
        """
        # Process image through pipeline
        results = self.pipeline.process_image(image_path)
        
        embeddings_data = {
            'source': image_path,
            'embeddings': [],
            'metadata': results['metadata']
        }
        
        # Generate embeddings for each detected face
        for face_data in results['faces']:
            try:
                embedding = self.encoder.encode_face(face_data['normalized_face'])
                
                if embedding.size > 0:
                    embedding_info = {
                        'face_id': face_data['face_id'],
                        'embedding': embedding,
                        'quality_score': face_data['quality_score'],
                        'embedding_dim': len(embedding)
                    }
                    embeddings_data['embeddings'].append(embedding_info)
                    
            except Exception as e:
                logger.error(f"Failed to generate embedding for face {face_data['face_id']}: {e}")
                
        return embeddings_data
    
    def generate_embeddings_from_directory(self, 
                                         directory_path: str,
                                         file_extensions: List[str] = ['jpg', 'jpeg', 'png']) -> List[Dict[str, Any]]:
        """
        Generate embeddings for all images in directory.
        
        Args:
            directory_path: Path to directory containing images
            file_extensions: List of file extensions to process
            
        Returns:
            List of embedding data dictionaries
        """
        directory = Path(directory_path)
        all_embeddings = []
        
        for ext in file_extensions:
            for image_file in directory.glob(f'*.{ext}'):
                try:
                    embedding_data = self.generate_embeddings_from_image(str(image_file))
                    if embedding_data['embeddings']:
                        all_embeddings.append(embedding_data)
                except Exception as e:
                    logger.error(f"Failed to process {image_file}: {e}")
                    
        return all_embeddings
    
    def compare_embeddings(self, 
                          embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> Dict[str, float]:
        """
        Compare two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Dictionary with distance and similarity scores
        """
        distance = self.similarity.calculate_distance(embedding1, embedding2)
        similarity = self.similarity.calculate_similarity(embedding1, embedding2)
        
        return {
            'distance': distance,
            'similarity': similarity,
            'metric': self.similarity.metric.value
        }
    
    def find_similar_faces(self, 
                          target_embedding: np.ndarray,
                          candidate_embeddings: List[np.ndarray],
                          top_k: int = 5,
                          threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find most similar faces to target embedding.
        
        Args:
            target_embedding: Target face embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top matches to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of match dictionaries sorted by similarity
        """
        matches = []
        
        for i, candidate in enumerate(candidate_embeddings):
            comparison = self.compare_embeddings(target_embedding, candidate)
            
            if comparison['similarity'] >= threshold:
                match_info = {
                    'candidate_index': i,
                    'similarity': comparison['similarity'],
                    'distance': comparison['distance']
                }
                matches.append(match_info)
        
        # Sort by similarity (descending) and return top_k
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:top_k]