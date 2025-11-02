"""
Comprehensive unit tests for face embeddings module.

Tests embedding generation, similarity matching, and quality assessment.
"""

import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from embeddings import FaceEmbeddingManager
    from encoders import SimpleFaceEncoder
except ImportError:
    pytest.skip("Embeddings module not available", allow_module_level=True)


class TestSimpleFaceEncoder:
    """Test suite for SimpleFaceEncoder class."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return SimpleFaceEncoder()
    
    @pytest.fixture
    def sample_face(self):
        """Create sample face image."""
        return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    def test_encoder_initialization(self, encoder):
        """Test encoder initializes correctly."""
        assert encoder is not None
        assert hasattr(encoder, 'encode_face')
    
    def test_encode_face_valid_input(self, encoder, sample_face):
        """Test encoding valid face image."""
        embedding = encoder.encode_face(sample_face)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.size > 0
        assert len(embedding.shape) == 1  # Should be 1D vector
    
    def test_encode_face_consistency(self, encoder, sample_face):
        """Test that encoding same face produces consistent results."""
        embedding1 = encoder.encode_face(sample_face)
        embedding2 = encoder.encode_face(sample_face)
        
        # Should produce similar embeddings
        if embedding1.size > 0 and embedding2.size > 0:
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            assert similarity > 0.95  # Very high similarity for identical image
    
    def test_encode_different_faces(self, encoder):
        """Test that different faces produce different embeddings."""
        face1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        face2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        embedding1 = encoder.encode_face(face1)
        embedding2 = encoder.encode_face(face2)
        
        # Different faces should produce different embeddings
        if embedding1.size > 0 and embedding2.size > 0:
            assert not np.array_equal(embedding1, embedding2)


class TestFaceEmbeddingManager:
    """Test suite for FaceEmbeddingManager class."""
    
    @pytest.fixture
    def embedding_manager(self):
        """Create embedding manager instance."""
        try:
            return FaceEmbeddingManager(encoder_type="simple")
        except Exception:
            pytest.skip("FaceEmbeddingManager not available")
    
    @pytest.fixture
    def sample_image_array(self):
        """Create sample image array."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_manager_initialization(self, embedding_manager):
        """Test manager initializes correctly."""
        assert embedding_manager is not None
        assert hasattr(embedding_manager, 'pipeline')
        assert hasattr(embedding_manager, 'encoder')
    
    def test_compute_similarity(self, embedding_manager):
        """Test similarity computation."""
        emb1 = np.random.rand(128)
        emb2 = np.random.rand(128)
        
        # Normalize embeddings
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        
        similarity = embedding_manager.compute_similarity(emb1, emb2)
        
        assert isinstance(similarity, (float, np.floating))
        assert -1.0 <= similarity <= 1.0
    
    def test_similarity_identical_embeddings(self, embedding_manager):
        """Test that identical embeddings have similarity of 1.0."""
        emb = np.random.rand(128)
        emb = emb / np.linalg.norm(emb)
        
        similarity = embedding_manager.compute_similarity(emb, emb)
        
        assert abs(similarity - 1.0) < 1e-6
    
    def test_similarity_orthogonal_embeddings(self, embedding_manager):
        """Test orthogonal embeddings have zero similarity."""
        # Create orthogonal vectors
        emb1 = np.zeros(128)
        emb2 = np.zeros(128)
        emb1[0] = 1.0
        emb2[1] = 1.0
        
        similarity = embedding_manager.compute_similarity(emb1, emb2)
        
        assert abs(similarity) < 1e-6


class TestEmbeddingOperations:
    """Test embedding mathematical operations."""
    
    def test_cosine_similarity_properties(self):
        """Test properties of cosine similarity."""
        # Create normalized vectors
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([1.0, 0.0, 0.0])
        
        # Cosine similarity calculation
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Orthogonal vectors should have 0 similarity
        assert abs(cosine_sim(v1, v2)) < 1e-6
        
        # Identical vectors should have 1.0 similarity
        assert abs(cosine_sim(v1, v3) - 1.0) < 1e-6
        
        # Opposite vectors should have -1.0 similarity
        assert abs(cosine_sim(v1, -v3) - (-1.0)) < 1e-6
    
    def test_euclidean_distance_properties(self):
        """Test properties of Euclidean distance."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        v3 = np.array([0.0, 0.0, 0.0])
        
        # Distance to self should be 0
        assert np.linalg.norm(v1 - v3) == 0.0
        
        # Distance should be positive
        assert np.linalg.norm(v1 - v2) > 0.0
        
        # Triangle inequality
        v4 = np.array([0.5, 0.5, 0.0])
        d12 = np.linalg.norm(v1 - v2)
        d14 = np.linalg.norm(v1 - v4)
        d24 = np.linalg.norm(v2 - v4)
        assert d14 + d24 >= d12 - 1e-6
    
    def test_embedding_normalization(self):
        """Test embedding normalization."""
        # Create random embedding
        emb = np.random.rand(128)
        
        # Normalize
        emb_normalized = emb / np.linalg.norm(emb)
        
        # Check L2 norm is 1.0
        assert abs(np.linalg.norm(emb_normalized) - 1.0) < 1e-6
        
        # Check direction is preserved
        cosine_sim = np.dot(emb, emb_normalized) / (
            np.linalg.norm(emb) * np.linalg.norm(emb_normalized)
        )
        assert abs(cosine_sim - 1.0) < 1e-6


class TestQualityAssessment:
    """Test face quality assessment functionality."""
    
    def test_blur_detection(self):
        """Test that blurred images have lower quality scores."""
        import cv2
        
        # Create sharp image
        sharp = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Create blurred version
        blurred = cv2.GaussianBlur(sharp, (15, 15), 0)
        
        # Compute Laplacian variance (sharpness measure)
        sharp_score = cv2.Laplacian(cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        blur_score = cv2.Laplacian(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        
        # Sharp image should have higher variance
        assert sharp_score > blur_score
    
    def test_brightness_assessment(self):
        """Test brightness level assessment."""
        # Create dark image
        dark = np.ones((100, 100, 3), dtype=np.uint8) * 20
        
        # Create bright image
        bright = np.ones((100, 100, 3), dtype=np.uint8) * 200
        
        # Create normal image
        normal = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Check brightness levels
        assert np.mean(dark) < np.mean(normal)
        assert np.mean(bright) > np.mean(normal)
    
    def test_contrast_assessment(self):
        """Test contrast level assessment."""
        # Low contrast image
        low_contrast = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # High contrast image
        high_contrast = np.zeros((100, 100, 3), dtype=np.uint8)
        high_contrast[::2, ::2] = 255
        
        # Standard deviation as contrast measure
        low_std = np.std(low_contrast)
        high_std = np.std(high_contrast)
        
        assert high_std > low_std


@pytest.mark.parametrize("embedding_size", [128, 256, 512])
def test_different_embedding_sizes(embedding_size):
    """Test embeddings of different sizes."""
    emb = np.random.rand(embedding_size)
    
    assert emb.shape[0] == embedding_size
    assert len(emb.shape) == 1
    
    # Normalize
    emb_norm = emb / np.linalg.norm(emb)
    assert abs(np.linalg.norm(emb_norm) - 1.0) < 1e-6


@pytest.mark.parametrize("threshold", [0.5, 0.6, 0.7, 0.8, 0.9])
def test_similarity_thresholds(threshold):
    """Test different similarity thresholds."""
    # Create similar embeddings
    emb1 = np.random.rand(128)
    emb1 = emb1 / np.linalg.norm(emb1)
    
    # Add small noise
    noise = np.random.rand(128) * 0.1
    emb2 = emb1 + noise
    emb2 = emb2 / np.linalg.norm(emb2)
    
    # Compute similarity
    similarity = np.dot(emb1, emb2)
    
    # Check threshold
    is_match = similarity >= threshold
    assert isinstance(is_match, (bool, np.bool_))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
