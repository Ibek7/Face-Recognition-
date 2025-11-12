"""
Unit tests for the Face Recognition API server.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api_server import app
from src.database import DatabaseManager

@pytest.fixture
def client():
    """Test client for the API."""
    return TestClient(app)

@pytest.fixture
def mock_db_manager():
    """Mock DatabaseManager."""
    with patch('src.api_server.get_database') as mock_get_db:
        mock_db = Mock(spec=DatabaseManager)
        mock_get_db.return_value = mock_db
        yield mock_db

class TestAPI:
    def test_health_check(self, client, mock_db_manager):
        """Test the health check endpoint."""
        mock_db_manager.get_session.return_value.query.return_value.first.return_value = (1,)
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["database_status"] == "connected"

    def test_list_persons_paginated(self, client, mock_db_manager):
        """Test the paginated persons endpoint."""
        mock_persons = [Mock(id=i, name=f"Person {i}", description="", created_at="2024-01-01", updated_at="2024-01-01", is_active=True) for i in range(15)]
        mock_db_manager.list_persons_paginated.return_value = (mock_persons[:10], 15)

        response = client.get("/api/v1/persons?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["items"]) == 10
        assert data["total"] == 15
        assert data["page"] == 1
        assert data["total_pages"] == 2

    def test_list_persons_paginated_invalid_page(self, client):
        """Test paginated persons endpoint with invalid page."""
        response = client.get("/api/v1/persons?page=0")
        assert response.status_code == 400

    def test_list_persons_paginated_invalid_page_size(self, client):
        """Test paginated persons endpoint with invalid page size."""
        response = client.get("/api/v1/persons?page_size=200")
        assert response.status_code == 400
