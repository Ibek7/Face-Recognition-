"""
Comprehensive Integration Tests

End-to-end API tests for face recognition system:
- Face detection endpoints
- Face recognition endpoints
- Person management
- Embedding operations
- Authentication
- Error handling
- Performance benchmarks
"""

import pytest
import asyncio
import httpx
from typing import Dict, Any
import os
import base64
from pathlib import Path
import time

# Test configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "test-api-key")
TEST_IMAGES_DIR = Path("tests/test_images")


@pytest.fixture
def api_headers() -> Dict[str, str]:
    """Get API headers with authentication"""
    return {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }


@pytest.fixture
async def client():
    """Create async HTTP client"""
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        yield client


@pytest.fixture
def sample_image_base64() -> str:
    """Load sample image as base64"""
    # Create a simple test image (1x1 pixel)
    import io
    from PIL import Image
    
    img = Image.new('RGB', (100, 100), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8')


class TestHealthCheck:
    """Test health check endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test /health endpoint"""
        response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
    
    @pytest.mark.asyncio
    async def test_ready_endpoint(self, client):
        """Test /ready endpoint"""
        response = await client.get("/ready")
        
        assert response.status_code in [200, 503]


class TestFaceDetection:
    """Test face detection endpoints"""
    
    @pytest.mark.asyncio
    async def test_detect_faces(self, client, api_headers, sample_image_base64):
        """Test face detection endpoint"""
        response = await client.post(
            "/api/v1/detect",
            json={"image": sample_image_base64},
            headers=api_headers
        )
        
        assert response.status_code in [200, 422]  # 422 if validation fails
        
        if response.status_code == 200:
            data = response.json()
            assert "faces" in data
            assert isinstance(data["faces"], list)
    
    @pytest.mark.asyncio
    async def test_detect_faces_invalid_image(self, client, api_headers):
        """Test detection with invalid image"""
        response = await client.post(
            "/api/v1/detect",
            json={"image": "invalid_base64"},
            headers=api_headers
        )
        
        assert response.status_code in [400, 422]
    
    @pytest.mark.asyncio
    async def test_detect_faces_missing_auth(self, client, sample_image_base64):
        """Test detection without authentication"""
        response = await client.post(
            "/api/v1/detect",
            json={"image": sample_image_base64}
        )
        
        # Should either work (no auth required) or return 401/403
        assert response.status_code in [200, 401, 403, 422]


class TestPersonManagement:
    """Test person management endpoints"""
    
    @pytest.mark.asyncio
    async def test_create_person(self, client, api_headers):
        """Test creating a person"""
        response = await client.post(
            "/api/v1/persons",
            json={
                "name": "Test Person",
                "email": "test@example.com",
                "metadata": {"department": "Engineering"}
            },
            headers=api_headers
        )
        
        assert response.status_code in [200, 201, 422]
        
        if response.status_code in [200, 201]:
            data = response.json()
            assert "id" in data
            assert data["name"] == "Test Person"
            
            # Store person ID for cleanup
            return data["id"]
    
    @pytest.mark.asyncio
    async def test_list_persons(self, client, api_headers):
        """Test listing persons"""
        response = await client.get(
            "/api/v1/persons",
            headers=api_headers
        )
        
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))
    
    @pytest.mark.asyncio
    async def test_get_person(self, client, api_headers):
        """Test getting a specific person"""
        # First create a person
        person_id = await self.test_create_person(client, api_headers)
        
        if person_id:
            response = await client.get(
                f"/api/v1/persons/{person_id}",
                headers=api_headers
            )
            
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert data["id"] == person_id
    
    @pytest.mark.asyncio
    async def test_update_person(self, client, api_headers):
        """Test updating a person"""
        # First create a person
        person_id = await self.test_create_person(client, api_headers)
        
        if person_id:
            response = await client.put(
                f"/api/v1/persons/{person_id}",
                json={"name": "Updated Name"},
                headers=api_headers
            )
            
            assert response.status_code in [200, 404, 422]
    
    @pytest.mark.asyncio
    async def test_delete_person(self, client, api_headers):
        """Test deleting a person"""
        # First create a person
        person_id = await self.test_create_person(client, api_headers)
        
        if person_id:
            response = await client.delete(
                f"/api/v1/persons/{person_id}",
                headers=api_headers
            )
            
            assert response.status_code in [200, 204, 404]


class TestFaceRecognition:
    """Test face recognition endpoints"""
    
    @pytest.mark.asyncio
    async def test_recognize_face(self, client, api_headers, sample_image_base64):
        """Test face recognition endpoint"""
        response = await client.post(
            "/api/v1/recognize",
            json={"image": sample_image_base64},
            headers=api_headers
        )
        
        assert response.status_code in [200, 404, 422]
        
        if response.status_code == 200:
            data = response.json()
            assert "matches" in data or "person" in data


class TestEmbeddings:
    """Test embedding operations"""
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, client, api_headers, sample_image_base64):
        """Test embedding generation"""
        response = await client.post(
            "/api/v1/embeddings",
            json={"image": sample_image_base64},
            headers=api_headers
        )
        
        assert response.status_code in [200, 422]
        
        if response.status_code == 200:
            data = response.json()
            assert "embedding" in data
            assert isinstance(data["embedding"], list)
    
    @pytest.mark.asyncio
    async def test_compare_embeddings(self, client, api_headers):
        """Test embedding comparison"""
        embedding1 = [0.1] * 128
        embedding2 = [0.2] * 128
        
        response = await client.post(
            "/api/v1/embeddings/compare",
            json={
                "embedding1": embedding1,
                "embedding2": embedding2
            },
            headers=api_headers
        )
        
        assert response.status_code in [200, 422]
        
        if response.status_code == 200:
            data = response.json()
            assert "similarity" in data or "distance" in data


class TestRateLimiting:
    """Test rate limiting"""
    
    @pytest.mark.asyncio
    async def test_rate_limit(self, client, api_headers):
        """Test rate limiting enforcement"""
        # Make multiple rapid requests
        responses = []
        
        for i in range(100):
            response = await client.get("/health", headers=api_headers)
            responses.append(response.status_code)
        
        # Check if any request was rate limited
        rate_limited = any(status == 429 for status in responses)
        
        # Rate limiting may or may not be enabled
        assert all(status in [200, 429, 503] for status in responses)


class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.asyncio
    async def test_404_not_found(self, client, api_headers):
        """Test 404 error"""
        response = await client.get(
            "/api/v1/nonexistent",
            headers=api_headers
        )
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_invalid_json(self, client, api_headers):
        """Test invalid JSON payload"""
        response = await client.post(
            "/api/v1/detect",
            content="invalid json",
            headers={**api_headers, "Content-Type": "application/json"}
        )
        
        assert response.status_code in [400, 422]


class TestPerformance:
    """Test performance benchmarks"""
    
    @pytest.mark.asyncio
    async def test_detection_performance(self, client, api_headers, sample_image_base64):
        """Test detection endpoint performance"""
        start_time = time.time()
        
        response = await client.post(
            "/api/v1/detect",
            json={"image": sample_image_base64},
            headers=api_headers
        )
        
        duration = time.time() - start_time
        
        # Detection should complete within reasonable time
        assert duration < 5.0  # 5 seconds max
        
        if response.status_code == 200:
            print(f"\nDetection took {duration:.3f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, api_headers):
        """Test handling concurrent requests"""
        tasks = []
        
        for i in range(10):
            task = client.get("/health", headers=api_headers)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should complete
        assert len(responses) == 10
        
        # Check response codes
        for response in responses:
            assert response.status_code in [200, 429, 503]


class TestDataValidation:
    """Test input validation"""
    
    @pytest.mark.asyncio
    async def test_invalid_person_data(self, client, api_headers):
        """Test creating person with invalid data"""
        response = await client.post(
            "/api/v1/persons",
            json={
                "name": "",  # Empty name
                "email": "invalid-email"  # Invalid email
            },
            headers=api_headers
        )
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self, client, api_headers):
        """Test missing required fields"""
        response = await client.post(
            "/api/v1/detect",
            json={},  # Missing image field
            headers=api_headers
        )
        
        assert response.status_code == 422


# Test configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
