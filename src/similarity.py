"""
Face similarity calculation utilities.
Provides various distance metrics for comparing face encodings.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DistanceMetric(Enum):
    """Supported distance metrics for face comparison."""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"

class FaceSimilarity:
    """Face similarity calculation using various distance metrics."""
    
    def __init__(self, metric: DistanceMetric = DistanceMetric.EUCLIDEAN):
        """
        Initialize similarity calculator.
        
        Args:
            metric: Distance metric to use
        """
        self.metric = metric
        
    def calculate_distance(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Calculate distance between two face encodings.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            Distance value (lower means more similar)
        """
        if encoding1.size == 0 or encoding2.size == 0:
            return float('inf')
            
        if self.metric == DistanceMetric.EUCLIDEAN:
            return np.linalg.norm(encoding1 - encoding2)
        
        elif self.metric == DistanceMetric.COSINE:
            # Cosine distance = 1 - cosine similarity
            dot_product = np.dot(encoding1, encoding2)
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)
            
            if norm1 == 0 or norm2 == 0:
                return 1.0
                
            cosine_sim = dot_product / (norm1 * norm2)
            return 1.0 - cosine_sim
        
        elif self.metric == DistanceMetric.MANHATTAN:
            return np.sum(np.abs(encoding1 - encoding2))
        
        elif self.metric == DistanceMetric.CHEBYSHEV:
            return np.max(np.abs(encoding1 - encoding2))
        
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
    
    def calculate_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Calculate similarity between two face encodings.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            Similarity value (higher means more similar)
        """
        distance = self.calculate_distance(encoding1, encoding2)
        
        if distance == float('inf'):
            return 0.0
            
        # Convert distance to similarity score
        if self.metric == DistanceMetric.COSINE:
            return 1.0 - distance  # Cosine similarity
        else:
            # For other metrics, use inverse relationship
            return 1.0 / (1.0 + distance)
    
    def find_best_match(self, 
                       target_encoding: np.ndarray, 
                       candidate_encodings: List[np.ndarray],
                       threshold: Optional[float] = None) -> Tuple[int, float]:
        """
        Find best matching face from candidates.
        
        Args:
            target_encoding: Target face encoding to match
            candidate_encodings: List of candidate encodings
            threshold: Optional similarity threshold
            
        Returns:
            Tuple of (best_match_index, similarity_score)
            Returns (-1, 0.0) if no match above threshold
        """
        if not candidate_encodings:
            return -1, 0.0
            
        best_idx = -1
        best_similarity = 0.0
        
        for i, candidate in enumerate(candidate_encodings):
            similarity = self.calculate_similarity(target_encoding, candidate)
            
            if similarity > best_similarity:
                if threshold is None or similarity >= threshold:
                    best_idx = i
                    best_similarity = similarity
                    
        return best_idx, best_similarity
    
    def calculate_similarity_matrix(self, encodings: List[np.ndarray]) -> np.ndarray:
        """
        Calculate pairwise similarity matrix for face encodings.
        
        Args:
            encodings: List of face encodings
            
        Returns:
            Symmetric similarity matrix
        """
        n = len(encodings)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity = self.calculate_similarity(encodings[i], encodings[j])
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
                    
        return similarity_matrix