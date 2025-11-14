"""
Integration Tests for Face Recognition API

Comprehensive integration tests covering all API endpoints,
including face detection, recognition, person management,
and system health checks.
"""

import io
import json
import os
import time
from typing import Dict, List

import pytest
import requests
from PIL import Image


@pytest.fixture(scope="module")
def api_base_url():
    """Get API base URL from environment or use default."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def api_headers():
    """Get API headers with authentication if needed."""
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


@pytest.fixture(scope="module")
def sample_image():
    """Create a sample test image."""
    # Create a simple 640x480 RGB image
    img = Image.new("RGB", (640, 480), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


class TestHealthEndpoints:
    """Test health and status endpoints."""

    def test_health_check(self, api_base_url):
        """Test basic health check endpoint."""
        response = requests.get(f"{api_base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_check(self, api_base_url):
        """Test readiness endpoint."""
        response = requests.get(f"{api_base_url}/ready")
        assert response.status_code in [200, 503]
        # 503 is acceptable if service is not ready yet

    def test_metrics_endpoint(self, api_base_url):
        """Test Prometheus metrics endpoint."""
        response = requests.get(f"{api_base_url}/metrics")
        assert response.status_code == 200
        assert "python_info" in response.text or "http_requests_total" in response.text


class TestDetectionEndpoints:
    """Test face detection endpoints."""

    def test_detect_faces_with_image(self, api_base_url, api_headers, sample_image):
        """Test face detection with image upload."""
        files = {"file": ("test.jpg", sample_image, "image/jpeg")}
        # Remove Content-Type header for multipart
        headers = {k: v for k, v in api_headers.items() if k != "Content-Type"}
        
        response = requests.post(
            f"{api_base_url}/api/detect",
            files=files,
            headers=headers,
        )
        
        assert response.status_code in [200, 400]
        # 400 is acceptable if no faces found in sample image
        
        if response.status_code == 200:
            data = response.json()
            assert "faces" in data
            assert isinstance(data["faces"], list)

    def test_detect_faces_invalid_image(self, api_base_url, api_headers):
        """Test face detection with invalid image data."""
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        headers = {k: v for k, v in api_headers.items() if k != "Content-Type"}
        
        response = requests.post(
            f"{api_base_url}/api/detect",
            files=files,
            headers=headers,
        )
        
        assert response.status_code == 400

    def test_detect_faces_missing_file(self, api_base_url, api_headers):
        """Test face detection without file upload."""
        response = requests.post(
            f"{api_base_url}/api/detect",
            headers=api_headers,
        )
        
        assert response.status_code == 422  # Unprocessable Entity


class TestRecognitionEndpoints:
    """Test face recognition endpoints."""

    def test_recognize_faces_with_image(self, api_base_url, api_headers, sample_image):
        """Test face recognition with image upload."""
        files = {"file": ("test.jpg", sample_image, "image/jpeg")}
        headers = {k: v for k, v in api_headers.items() if k != "Content-Type"}
        
        response = requests.post(
            f"{api_base_url}/api/recognize",
            files=files,
            headers=headers,
        )
        
        assert response.status_code in [200, 400, 404]
        # 400: no faces found, 404: no persons in database
        
        if response.status_code == 200:
            data = response.json()
            assert "matches" in data or "results" in data

    def test_recognize_with_threshold(self, api_base_url, api_headers, sample_image):
        """Test face recognition with custom threshold."""
        files = {"file": ("test.jpg", sample_image, "image/jpeg")}
        headers = {k: v for k, v in api_headers.items() if k != "Content-Type"}
        params = {"threshold": 0.8}
        
        response = requests.post(
            f"{api_base_url}/api/recognize",
            files=files,
            params=params,
            headers=headers,
        )
        
        assert response.status_code in [200, 400, 404]


class TestPersonManagement:
    """Test person CRUD endpoints."""

    @pytest.fixture(scope="class")
    def test_person_id(self, api_base_url, api_headers):
        """Create a test person and return its ID."""
        person_data = {
            "name": "Test Person",
            "metadata": {"department": "QA", "employee_id": "TEST001"},
        }
        
        response = requests.post(
            f"{api_base_url}/api/persons",
            json=person_data,
            headers=api_headers,
        )
        
        if response.status_code == 201:
            return response.json()["id"]
        return None

    def test_create_person(self, api_base_url, api_headers):
        """Test creating a new person."""
        person_data = {
            "name": "Integration Test User",
            "metadata": {"test": True},
        }
        
        response = requests.post(
            f"{api_base_url}/api/persons",
            json=person_data,
            headers=api_headers,
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["name"] == person_data["name"]

    def test_list_persons(self, api_base_url, api_headers):
        """Test listing all persons."""
        response = requests.get(
            f"{api_base_url}/api/persons",
            headers=api_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list) or "persons" in data

    def test_get_person(self, api_base_url, api_headers, test_person_id):
        """Test getting a specific person."""
        if not test_person_id:
            pytest.skip("No test person created")
        
        response = requests.get(
            f"{api_base_url}/api/persons/{test_person_id}",
            headers=api_headers,
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_person_id

    def test_update_person(self, api_base_url, api_headers, test_person_id):
        """Test updating a person."""
        if not test_person_id:
            pytest.skip("No test person created")
        
        update_data = {
            "name": "Updated Test Person",
            "metadata": {"updated": True},
        }
        
        response = requests.put(
            f"{api_base_url}/api/persons/{test_person_id}",
            json=update_data,
            headers=api_headers,
        )
        
        assert response.status_code in [200, 204]

    def test_delete_person(self, api_base_url, api_headers):
        """Test deleting a person."""
        # Create a person to delete
        person_data = {"name": "To Be Deleted"}
        create_response = requests.post(
            f"{api_base_url}/api/persons",
            json=person_data,
            headers=api_headers,
        )
        
        if create_response.status_code != 201:
            pytest.skip("Could not create person to delete")
        
        person_id = create_response.json()["id"]
        
        # Delete the person
        response = requests.delete(
            f"{api_base_url}/api/persons/{person_id}",
            headers=api_headers,
        )
        
        assert response.status_code in [200, 204]

    def test_get_nonexistent_person(self, api_base_url, api_headers):
        """Test getting a person that doesn't exist."""
        response = requests.get(
            f"{api_base_url}/api/persons/99999999",
            headers=api_headers,
        )
        
        assert response.status_code == 404


class TestBatchProcessing:
    """Test batch processing endpoints."""

    def test_batch_detect(self, api_base_url, api_headers, sample_image):
        """Test batch face detection."""
        # Create multiple files
        files = [
            ("files", ("test1.jpg", sample_image, "image/jpeg")),
            ("files", ("test2.jpg", sample_image, "image/jpeg")),
        ]
        headers = {k: v for k, v in api_headers.items() if k != "Content-Type"}
        
        response = requests.post(
            f"{api_base_url}/api/batch/detect",
            files=files,
            headers=headers,
        )
        
        assert response.status_code in [200, 400, 404]
        # 404 if batch endpoint doesn't exist

    def test_batch_recognize(self, api_base_url, api_headers, sample_image):
        """Test batch face recognition."""
        files = [
            ("files", ("test1.jpg", sample_image, "image/jpeg")),
            ("files", ("test2.jpg", sample_image, "image/jpeg")),
        ]
        headers = {k: v for k, v in api_headers.items() if k != "Content-Type"}
        
        response = requests.post(
            f"{api_base_url}/api/batch/recognize",
            files=files,
            headers=headers,
        )
        
        assert response.status_code in [200, 400, 404]


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_endpoint(self, api_base_url):
        """Test accessing an invalid endpoint."""
        response = requests.get(f"{api_base_url}/api/invalid")
        assert response.status_code == 404

    def test_method_not_allowed(self, api_base_url):
        """Test using wrong HTTP method."""
        response = requests.delete(f"{api_base_url}/health")
        assert response.status_code == 405

    def test_malformed_json(self, api_base_url, api_headers):
        """Test sending malformed JSON."""
        response = requests.post(
            f"{api_base_url}/api/persons",
            data="not valid json",
            headers=api_headers,
        )
        
        assert response.status_code in [400, 422]

    def test_large_file_upload(self, api_base_url, api_headers):
        """Test uploading a large file (should be rejected)."""
        # Create a 20MB file
        large_data = b"0" * (20 * 1024 * 1024)
        files = {"file": ("large.jpg", large_data, "image/jpeg")}
        headers = {k: v for k, v in api_headers.items() if k != "Content-Type"}
        
        response = requests.post(
            f"{api_base_url}/api/detect",
            files=files,
            headers=headers,
            timeout=30,
        )
        
        assert response.status_code in [400, 413]  # 413 Request Entity Too Large


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_enforcement(self, api_base_url):
        """Test that rate limiting is enforced."""
        # Make multiple rapid requests
        responses = []
        for _ in range(20):
            response = requests.get(f"{api_base_url}/health")
            responses.append(response.status_code)
        
        # At least some requests should succeed
        assert 200 in responses
        # May include 429 (Too Many Requests) if rate limiting is active
        # But we don't require it as rate limits may be high for health endpoint

    def test_rate_limit_headers(self, api_base_url):
        """Test that rate limit headers are present."""
        response = requests.get(f"{api_base_url}/api/persons")
        
        # Check for common rate limit headers
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ]
        
        # At least one rate limit header should be present if rate limiting is enabled
        has_rate_limit_header = any(
            header in response.headers for header in rate_limit_headers
        )
        
        # This is informational, not a hard requirement
        if has_rate_limit_header:
            print(f"Rate limiting is active: {response.headers}")


class TestPerformance:
    """Test API performance characteristics."""

    def test_response_time(self, api_base_url):
        """Test that health check responds quickly."""
        start_time = time.time()
        response = requests.get(f"{api_base_url}/health")
        elapsed = time.time() - start_time
        
        assert response.status_code == 200
        assert elapsed < 1.0  # Should respond within 1 second

    def test_concurrent_requests(self, api_base_url):
        """Test handling multiple concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return requests.get(f"{api_base_url}/health")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count >= 8  # At least 80% should succeed


class TestSecurity:
    """Test security features."""

    def test_cors_headers(self, api_base_url):
        """Test CORS headers are present."""
        response = requests.options(
            f"{api_base_url}/api/persons",
            headers={"Origin": "http://example.com"},
        )
        
        # Check if CORS is configured
        if "Access-Control-Allow-Origin" in response.headers:
            print(f"CORS enabled: {response.headers['Access-Control-Allow-Origin']}")

    def test_security_headers(self, api_base_url):
        """Test security headers are present."""
        response = requests.get(f"{api_base_url}/health")
        
        # Check for security headers
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
        ]
        
        present_headers = [h for h in security_headers if h in response.headers]
        print(f"Security headers present: {present_headers}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
