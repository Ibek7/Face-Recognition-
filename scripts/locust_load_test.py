#!/usr/bin/env python3
"""
Load Testing Script using Locust

Simulates realistic user load patterns for the face recognition API.
Tests various endpoints under different load conditions.
"""

from io import BytesIO
from locust import HttpUser, task, between, events
from PIL import Image
import random
import os


class FaceRecognitionUser(HttpUser):
    """Simulates a user interacting with the Face Recognition API."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    def on_start(self):
        """Called when a user starts - initialize test data."""
        self.test_image = self._create_test_image()
        self.person_ids = []

    def _create_test_image(self, size=(640, 480)) -> bytes:
        """Create a test image for requests."""
        img = Image.new("RGB", size, color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return buffer.read()

    @task(10)
    def health_check(self):
        """Test health check endpoint - high frequency."""
        self.client.get("/health")

    @task(5)
    def detect_faces(self):
        """Test face detection endpoint - medium frequency."""
        files = {"file": ("test.jpg", BytesIO(self.test_image), "image/jpeg")}
        with self.client.post(
            "/api/detect",
            files=files,
            catch_response=True,
            name="/api/detect"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 400:
                # No faces found is acceptable
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(3)
    def recognize_faces(self):
        """Test face recognition endpoint - medium frequency."""
        files = {"file": ("test.jpg", BytesIO(self.test_image), "image/jpeg")}
        with self.client.post(
            "/api/recognize",
            files=files,
            catch_response=True,
            name="/api/recognize"
        ) as response:
            if response.status_code in [200, 400, 404]:
                # All these are acceptable
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(2)
    def list_persons(self):
        """Test list persons endpoint - low frequency."""
        self.client.get("/api/persons")

    @task(1)
    def create_person(self):
        """Test create person endpoint - low frequency."""
        person_data = {
            "name": f"Test User {random.randint(1000, 9999)}",
            "metadata": {"test": True, "load_test": True}
        }
        
        with self.client.post(
            "/api/persons",
            json=person_data,
            catch_response=True,
            name="/api/persons [POST]"
        ) as response:
            if response.status_code == 201:
                try:
                    data = response.json()
                    if "id" in data:
                        self.person_ids.append(data["id"])
                    response.success()
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Status: {response.status_code}")

    @task(1)
    def get_person(self):
        """Test get person endpoint - low frequency."""
        if self.person_ids:
            person_id = random.choice(self.person_ids)
            with self.client.get(
                f"/api/persons/{person_id}",
                catch_response=True,
                name="/api/persons/{id} [GET]"
            ) as response:
                if response.status_code in [200, 404]:
                    response.success()
                else:
                    response.failure(f"Status: {response.status_code}")

    @task(1)
    def metrics(self):
        """Test metrics endpoint - low frequency."""
        self.client.get("/metrics")


class HighLoadUser(HttpUser):
    """Simulates high-load scenarios - detection and recognition only."""

    wait_time = between(0.1, 0.5)  # Very short wait time

    def on_start(self):
        """Initialize test data."""
        self.test_image = self._create_test_image()

    def _create_test_image(self, size=(640, 480)) -> bytes:
        """Create a test image."""
        img = Image.new("RGB", size, color=(100, 150, 200))
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return buffer.read()

    @task(7)
    def detect_faces(self):
        """Face detection under high load."""
        files = {"file": ("test.jpg", BytesIO(self.test_image), "image/jpeg")}
        self.client.post("/api/detect", files=files, name="/api/detect [high-load]")

    @task(3)
    def recognize_faces(self):
        """Face recognition under high load."""
        files = {"file": ("test.jpg", BytesIO(self.test_image), "image/jpeg")}
        self.client.post("/api/recognize", files=files, name="/api/recognize [high-load]")


# Event handlers for custom logging
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("=" * 60)
    print("LOAD TEST STARTING")
    print("=" * 60)
    print(f"Target host: {environment.host}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print("\n" + "=" * 60)
    print("LOAD TEST COMPLETED")
    print("=" * 60)
    
    stats = environment.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Requests per second: {stats.total.total_rps:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    print("""
Face Recognition API Load Testing
==================================

Usage: locust -f scripts/locust_load_test.py --host=http://localhost:8000

Examples:
  locust -f scripts/locust_load_test.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 5m --headless
""")
