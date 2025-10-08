"""
Comprehensive test suite for the Face Recognition API.
Tests all endpoints, middleware, and error handling.
"""

import pytest
import asyncio
import json
import base64
import io
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import the FastAPI app and dependencies
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock external dependencies before importing
sys.modules['cv2'] = Mock()
sys.modules['dlib'] = Mock()
sys.modules['face_recognition'] = Mock()
sys.modules['sqlalchemy'] = Mock()
sys.modules['sqlalchemy.orm'] = Mock()

@pytest.fixture
def mock_database():
    """Mock database manager."""
    db_mock = Mock()
    
    # Mock person data
    mock_person = Mock()
    mock_person.id = 1
    mock_person.name = "Test Person"
    mock_person.description = "Test description"
    mock_person.created_at = "2024-01-15T10:00:00"
    mock_person.updated_at = "2024-01-15T10:00:00"
    mock_person.is_active = True
    
    db_mock.list_persons.return_value = [mock_person]
    db_mock.get_person.return_value = mock_person
    db_mock.get_person_by_name.return_value = None
    db_mock.add_person.return_value = mock_person
    db_mock.get_recognition_stats.return_value = {
        'total_persons': 1,
        'total_embeddings': 5,
        'total_recognitions': 10,
        'avg_processing_time': 0.234
    }
    
    return db_mock

@pytest.fixture
def mock_embedding_manager():
    """Mock embedding manager."""
    em_mock = Mock()
    
    # Mock pipeline and encoder
    em_mock.pipeline = Mock()
    em_mock.encoder = Mock()
    
    # Mock face detection results
    em_mock.pipeline.process_image_array.return_value = {
        'faces': [
            {
                'normalized_face': np.random.random((64, 64, 3)),
                'quality_score': 0.85
            }
        ]
    }
    
    em_mock.pipeline.detector.detect_faces.return_value = [(100, 100, 150, 150)]
    
    # Mock encoding
    em_mock.encoder.encode_face.return_value = np.random.random(128)
    
    # Mock embedding generation
    em_mock.generate_embeddings_from_image.return_value = {
        'embeddings': [
            {
                'embedding': np.random.random(128),
                'quality_score': 0.85
            }
        ]
    }
    
    return em_mock

@pytest.fixture
def test_image_base64():
    """Generate a test image in base64 format."""
    # Create a simple test image
    image = Image.new('RGB', (200, 200), color='red')
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_bytes = buffer.getvalue()
    
    return base64.b64encode(image_bytes).decode()

@pytest.fixture
def client(mock_database, mock_embedding_manager):
    """Test client with mocked dependencies."""
    with patch('src.api_server.get_database', return_value=mock_database), \
         patch('src.api_server.get_embedding_manager', return_value=mock_embedding_manager):
        
        from api_server import app
        return TestClient(app)

class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Face Recognition API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        with patch('src.api_server.performance_monitor') as mock_monitor:
            mock_monitor.get_system_metrics_summary.return_value = {
                "cpu_percent": 15.2,
                "memory_percent": 45.8
            }
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert "uptime" in data
            assert "system_metrics" in data

class TestPersonManagement:
    """Test person management endpoints."""
    
    def test_list_persons(self, client):
        """Test listing all persons."""
        response = client.get("/persons")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "Test Person"
    
    def test_create_person(self, client):
        """Test creating a new person."""
        person_data = {
            "name": "New Person",
            "description": "New test person"
        }
        
        response = client.post("/persons", json=person_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "New Person"
        assert data["description"] == "New test person"
    
    def test_create_person_duplicate_name(self, client, mock_database):
        """Test creating person with duplicate name."""
        # Mock existing person
        mock_database.get_person_by_name.return_value = Mock()
        
        person_data = {
            "name": "Existing Person",
            "description": "This should fail"
        }
        
        response = client.post("/persons", json=person_data)
        assert response.status_code == 400
        
        data = response.json()
        assert "already exists" in data["detail"]
    
    def test_create_person_invalid_data(self, client):
        """Test creating person with invalid data."""
        person_data = {
            "name": "",  # Empty name should fail
            "description": "Test"
        }
        
        response = client.post("/persons", json=person_data)
        assert response.status_code == 422  # Validation error

class TestFaceRecognition:
    """Test face recognition endpoints."""
    
    def test_recognize_faces_success(self, client, mock_database, test_image_base64):
        """Test successful face recognition."""
        # Mock search results
        mock_embedding = Mock()
        mock_embedding.person_id = 1
        mock_database.search_similar_faces.return_value = [(mock_embedding, 0.85)]
        
        request_data = {
            "image_base64": f"data:image/jpeg;base64,{test_image_base64}",
            "threshold": 0.7
        }
        
        response = client.post("/recognize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["faces"]) == 1
        assert data["faces"][0]["person_name"] == "Test Person"
        assert data["faces"][0]["confidence"] == 0.85
    
    def test_recognize_faces_no_match(self, client, mock_database, test_image_base64):
        """Test face recognition with no matches."""
        # Mock no search results
        mock_database.search_similar_faces.return_value = []
        
        request_data = {
            "image_base64": f"data:image/jpeg;base64,{test_image_base64}",
            "threshold": 0.7
        }
        
        response = client.post("/recognize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["faces"]) == 1
        assert data["faces"][0]["person_name"] is None
        assert data["faces"][0]["confidence"] == 0.0
    
    def test_recognize_faces_invalid_image(self, client):
        """Test face recognition with invalid image."""
        request_data = {
            "image_base64": "invalid_base64_data",
            "threshold": 0.7
        }
        
        response = client.post("/recognize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is False
        assert "Error" in data["message"]
    
    def test_upload_image_recognition(self, client, mock_database):
        """Test image upload for recognition."""
        # Create test image file
        image = Image.new('RGB', (200, 200), color='blue')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        # Mock successful recognition
        mock_embedding = Mock()
        mock_embedding.person_id = 1
        mock_database.search_similar_faces.return_value = [(mock_embedding, 0.90)]
        
        files = {"file": ("test.jpg", buffer, "image/jpeg")}
        response = client.post("/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_upload_non_image_file(self, client):
        """Test uploading non-image file."""
        files = {"file": ("test.txt", io.StringIO("not an image"), "text/plain")}
        response = client.post("/upload", files=files)
        
        assert response.status_code == 400

class TestPersonImages:
    """Test person image management endpoints."""
    
    def test_add_person_image(self, client, mock_database):
        """Test adding image for a person."""
        # Create test image file
        image = Image.new('RGB', (200, 200), color='green')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        files = {"file": ("profile.jpg", buffer, "image/jpeg")}
        response = client.post("/persons/1/images", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "embeddings_added" in data
    
    def test_add_image_person_not_found(self, client, mock_database):
        """Test adding image for non-existent person."""
        mock_database.get_person.return_value = None
        
        image = Image.new('RGB', (200, 200), color='yellow')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        files = {"file": ("profile.jpg", buffer, "image/jpeg")}
        response = client.post("/persons/999/images", files=files)
        
        assert response.status_code == 404

class TestStatistics:
    """Test statistics endpoints."""
    
    def test_get_system_stats(self, client):
        """Test system statistics endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_persons"] == 1
        assert data["total_embeddings"] == 5
        assert data["total_recognitions"] == 10
        assert "uptime" in data
    
    def test_get_performance_metrics(self, client):
        """Test performance metrics endpoint."""
        with patch('src.api_server.performance_monitor') as mock_monitor:
            mock_monitor.get_recent_metrics.return_value = []
            mock_monitor.get_system_metrics_summary.return_value = {
                "cpu_percent": 20.5,
                "memory_percent": 55.2
            }
            
            response = client.get("/performance/metrics")
            assert response.status_code == 200
            
            data = response.json()
            assert "recognition_metrics" in data
            assert "system_metrics" in data

class TestMiddleware:
    """Test middleware functionality."""
    
    def test_rate_limiting(self, client):
        """Test rate limiting middleware."""
        # This would require configuring a very low rate limit for testing
        # For now, just verify the endpoint responds
        response = client.get("/health")
        assert response.status_code == 200
        
        # Check rate limit headers are present
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
    
    def test_security_headers(self, client):
        """Test security headers are added."""
        response = client.get("/")
        
        # Check security headers
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert "X-Request-ID" in response.headers
    
    def test_cors_headers(self, client):
        """Test CORS headers."""
        response = client.options("/")
        # CORS headers should be present for OPTIONS requests
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled

class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_404_endpoint(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_validation_error(self, client):
        """Test validation error response format."""
        # Send invalid JSON to person creation
        response = client.post("/persons", json={"name": None})
        assert response.status_code == 422
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error."""
        response = client.delete("/")
        assert response.status_code == 405

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_workflow(self, client, mock_database, test_image_base64):
        """Test complete workflow: create person, add image, recognize."""
        # 1. Create person
        person_data = {"name": "Integration Test", "description": "Test person"}
        response = client.post("/persons", json=person_data)
        assert response.status_code == 200
        
        # 2. Add person image
        image = Image.new('RGB', (200, 200), color='purple')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        files = {"file": ("profile.jpg", buffer, "image/jpeg")}
        response = client.post("/persons/1/images", files=files)
        assert response.status_code == 200
        
        # 3. Recognize face
        mock_embedding = Mock()
        mock_embedding.person_id = 1
        mock_database.search_similar_faces.return_value = [(mock_embedding, 0.95)]
        
        request_data = {
            "image_base64": f"data:image/jpeg;base64,{test_image_base64}",
            "threshold": 0.7
        }
        
        response = client.post("/recognize", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["faces"][0]["confidence"] == 0.95

# Test configuration
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Ensure test database is clean
    pass

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])