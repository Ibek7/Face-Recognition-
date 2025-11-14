"""
Smoke tests for the Face Recognition API.
These tests verify that the API server can start and respond to basic requests.
"""
import pytest
from fastapi.testclient import TestClient


def test_smoke_api_import():
    """Test that the API module can be imported without errors."""
    try:
        from src.api_server import app
        assert app is not None
    except Exception as e:
        pytest.fail(f"Failed to import API server: {e}")


def test_smoke_health_endpoint():
    """Test that the health endpoint responds successfully."""
    from src.api_server import app
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_smoke_docs_available():
    """Test that the API documentation is accessible."""
    from src.api_server import app
    
    client = TestClient(app)
    response = client.get("/docs")
    
    assert response.status_code == 200


def test_smoke_openapi_schema():
    """Test that the OpenAPI schema is available."""
    from src.api_server import app
    
    client = TestClient(app)
    response = client.get("/openapi.json")
    
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
