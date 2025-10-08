"""
Unit tests for face detection module.
Tests face detection accuracy and performance.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

# Test requires src modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detection import FaceDetector
from preprocessing import ImagePreprocessor

class TestFaceDetector:
    """Test suite for FaceDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a face detector instance."""
        return FaceDetector()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image with a synthetic face."""
        # Create a simple synthetic face-like image
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Draw face-like features
        cv2.circle(image, (100, 100), 80, (200, 200, 200), -1)  # Face
        cv2.circle(image, (80, 80), 10, (0, 0, 0), -1)          # Left eye
        cv2.circle(image, (120, 80), 10, (0, 0, 0), -1)         # Right eye
        cv2.ellipse(image, (100, 120), (15, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        return image
    
    def test_detector_initialization(self, detector):
        """Test that detector initializes properly."""
        assert detector is not None
        assert detector.face_cascade is not None
        assert not detector.face_cascade.empty()
    
    def test_detect_faces_empty_image(self, detector):
        """Test face detection on empty image."""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = detector.detect_faces(empty_image)
        assert isinstance(faces, list)
        # Empty image should not detect faces
        assert len(faces) == 0
    
    def test_detect_faces_sample_image(self, detector, sample_image):
        """Test face detection on sample image."""
        faces = detector.detect_faces(sample_image)
        assert isinstance(faces, list)
        
        # Each face should be a tuple of 4 integers
        for face in faces:
            assert isinstance(face, tuple)
            assert len(face) == 4
            x, y, w, h = face
            assert all(isinstance(coord, int) for coord in [x, y, w, h])
            assert w > 0 and h > 0
    
    def test_extract_faces(self, detector, sample_image):
        """Test face extraction functionality."""
        face_images = detector.extract_faces(sample_image)
        assert isinstance(face_images, list)
        
        for face_img in face_images:
            assert isinstance(face_img, np.ndarray)
            assert len(face_img.shape) == 3  # Color image
            assert face_img.shape[2] == 3    # RGB channels
    
    def test_extract_faces_with_padding(self, detector, sample_image):
        """Test face extraction with different padding values."""
        # Test with different padding values
        for padding in [0.0, 0.1, 0.5]:
            face_images = detector.extract_faces(sample_image, padding=padding)
            assert isinstance(face_images, list)
    
    def test_invalid_cascade_path(self):
        """Test detector with invalid cascade path."""
        with pytest.raises(ValueError):
            FaceDetector(cascade_path="nonexistent_path.xml")

class TestImagePreprocessor:
    """Test suite for ImagePreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create an image preprocessor instance."""
        return ImagePreprocessor(target_size=(64, 64))
    
    @pytest.fixture
    def test_face_image(self):
        """Create a test face image."""
        # Create a simple test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return image
    
    def test_preprocessor_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.target_size == (64, 64)
    
    def test_normalize_face(self, preprocessor, test_face_image):
        """Test face normalization."""
        normalized = preprocessor.normalize_face(test_face_image)
        
        # Check output properties
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape[:2] == preprocessor.target_size
        assert normalized.dtype == np.float32
        assert 0.0 <= normalized.min()
        assert normalized.max() <= 1.0
    
    def test_normalize_face_grayscale(self, preprocessor):
        """Test face normalization with grayscale image."""
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        normalized = preprocessor.normalize_face(gray_image)
        
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == preprocessor.target_size
    
    def test_assess_quality(self, preprocessor, test_face_image):
        """Test image quality assessment."""
        quality = preprocessor.assess_quality(test_face_image)
        
        assert isinstance(quality, (int, float))
        assert quality >= 0.0
    
    def test_assess_quality_blurred_image(self, preprocessor):
        """Test quality assessment on blurred image."""
        # Create a blurred image (should have lower quality)
        sharp_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        blurred_image = cv2.GaussianBlur(sharp_image, (15, 15), 0)
        
        sharp_quality = preprocessor.assess_quality(sharp_image)
        blurred_quality = preprocessor.assess_quality(blurred_image)
        
        # Blurred image should have lower quality score
        assert blurred_quality < sharp_quality
    
    def test_normalize_different_sizes(self):
        """Test normalization with different target sizes."""
        sizes = [(32, 32), (128, 128), (224, 224)]
        test_image = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        
        for size in sizes:
            preprocessor = ImagePreprocessor(target_size=size)
            normalized = preprocessor.normalize_face(test_image)
            assert normalized.shape[:2] == size

@pytest.fixture
def temp_image_file():
    """Create a temporary image file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        # Create a simple test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        cv2.imwrite(tmp.name, test_image)
        yield tmp.name
    
    # Cleanup
    os.unlink(tmp.name)

class TestIntegration:
    """Integration tests for face detection pipeline."""
    
    def test_detector_preprocessor_integration(self, temp_image_file):
        """Test integration between detector and preprocessor."""
        detector = FaceDetector()
        preprocessor = ImagePreprocessor()
        
        # Load test image
        image = cv2.imread(temp_image_file)
        assert image is not None
        
        # Extract faces
        face_images = detector.extract_faces(image)
        
        # Process each face
        for face_img in face_images:
            normalized = preprocessor.normalize_face(face_img)
            quality = preprocessor.assess_quality(face_img)
            
            assert isinstance(normalized, np.ndarray)
            assert isinstance(quality, (int, float))
    
    def test_performance_benchmark(self, temp_image_file):
        """Basic performance test."""
        import time
        
        detector = FaceDetector()
        preprocessor = ImagePreprocessor()
        
        image = cv2.imread(temp_image_file)
        
        # Time the detection process
        start_time = time.time()
        faces = detector.detect_faces(image)
        detection_time = time.time() - start_time
        
        # Detection should be reasonably fast (less than 1 second for small image)
        assert detection_time < 1.0
        
        # Time face extraction
        start_time = time.time()
        face_images = detector.extract_faces(image)
        extraction_time = time.time() - start_time
        
        assert extraction_time < 1.0

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])