"""
Face Recognition API Client

Python client library for interacting with the Face Recognition API.
Provides a simple, Pythonic interface for all API operations.
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
from PIL import Image


class FaceRecognitionClient:
    """Client for Face Recognition API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the Face Recognition API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("FACE_RECOGNITION_API_KEY")
        self.timeout = timeout
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update({"X-API-Key": self.api_key})

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """Make HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                files=files,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            raise FaceRecognitionAPIError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise FaceRecognitionAPIError(f"Request failed: {str(e)}") from e

    def _prepare_image(
        self, image: Union[str, Path, bytes, Image.Image]
    ) -> tuple:
        """Prepare image for upload."""
        if isinstance(image, (str, Path)):
            # File path
            with open(image, "rb") as f:
                return ("image.jpg", f.read(), "image/jpeg")
        elif isinstance(image, bytes):
            # Raw bytes
            return ("image.jpg", image, "image/jpeg")
        elif isinstance(image, Image.Image):
            # PIL Image
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            return ("image.jpg", buffer.read(), "image/jpeg")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    # Health Check Methods

    def health_check(self) -> Dict:
        """Check API health status."""
        return self._make_request("GET", "/health")

    def ready_check(self) -> Dict:
        """Check if API is ready to accept requests."""
        return self._make_request("GET", "/ready")

    # Face Detection Methods

    def detect_faces(
        self,
        image: Union[str, Path, bytes, Image.Image],
        min_confidence: float = 0.8,
        return_landmarks: bool = False,
    ) -> Dict:
        """
        Detect faces in an image.

        Args:
            image: Image file path, bytes, or PIL Image
            min_confidence: Minimum detection confidence (0.0 to 1.0)
            return_landmarks: Whether to return facial landmarks

        Returns:
            Dict containing detected faces with bounding boxes
        """
        image_data = self._prepare_image(image)
        files = {"file": image_data}
        params = {
            "min_confidence": min_confidence,
            "return_landmarks": return_landmarks,
        }

        return self._make_request("POST", "/api/detect", files=files, params=params)

    def batch_detect_faces(
        self,
        images: List[Union[str, Path, bytes, Image.Image]],
        min_confidence: float = 0.8,
    ) -> Dict:
        """
        Detect faces in multiple images.

        Args:
            images: List of images (file paths, bytes, or PIL Images)
            min_confidence: Minimum detection confidence

        Returns:
            Dict containing detection results for all images
        """
        files = [("files", self._prepare_image(img)) for img in images]
        params = {"min_confidence": min_confidence}

        return self._make_request(
            "POST", "/api/batch/detect", files=files, params=params
        )

    # Face Recognition Methods

    def recognize_faces(
        self,
        image: Union[str, Path, bytes, Image.Image],
        threshold: float = 0.6,
        top_k: int = 5,
    ) -> Dict:
        """
        Recognize faces in an image.

        Args:
            image: Image file path, bytes, or PIL Image
            threshold: Similarity threshold for matching (0.0 to 1.0)
            top_k: Number of top matches to return

        Returns:
            Dict containing recognized faces with person IDs
        """
        image_data = self._prepare_image(image)
        files = {"file": image_data}
        params = {"threshold": threshold, "top_k": top_k}

        return self._make_request("POST", "/api/recognize", files=files, params=params)

    def batch_recognize_faces(
        self,
        images: List[Union[str, Path, bytes, Image.Image]],
        threshold: float = 0.6,
    ) -> Dict:
        """
        Recognize faces in multiple images.

        Args:
            images: List of images (file paths, bytes, or PIL Images)
            threshold: Similarity threshold for matching

        Returns:
            Dict containing recognition results for all images
        """
        files = [("files", self._prepare_image(img)) for img in images]
        params = {"threshold": threshold}

        return self._make_request(
            "POST", "/api/batch/recognize", files=files, params=params
        )

    # Person Management Methods

    def create_person(
        self,
        name: str,
        image: Optional[Union[str, Path, bytes, Image.Image]] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Create a new person in the database.

        Args:
            name: Person's name
            image: Optional image for face enrollment
            metadata: Optional metadata dictionary

        Returns:
            Dict containing created person information
        """
        data = {"name": name}
        if metadata:
            data["metadata"] = metadata

        if image:
            image_data = self._prepare_image(image)
            files = {"file": image_data}
            return self._make_request("POST", "/api/persons", data=data, files=files)
        else:
            return self._make_request("POST", "/api/persons", data=data)

    def get_person(self, person_id: int) -> Dict:
        """Get person information by ID."""
        return self._make_request("GET", f"/api/persons/{person_id}")

    def list_persons(
        self, skip: int = 0, limit: int = 100, search: Optional[str] = None
    ) -> List[Dict]:
        """
        List all persons.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            search: Optional search query

        Returns:
            List of person dictionaries
        """
        params = {"skip": skip, "limit": limit}
        if search:
            params["search"] = search

        return self._make_request("GET", "/api/persons", params=params)

    def update_person(
        self,
        person_id: int,
        name: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Update person information.

        Args:
            person_id: Person ID
            name: Updated name
            metadata: Updated metadata

        Returns:
            Dict containing updated person information
        """
        data = {}
        if name is not None:
            data["name"] = name
        if metadata is not None:
            data["metadata"] = metadata

        return self._make_request("PUT", f"/api/persons/{person_id}", data=data)

    def delete_person(self, person_id: int) -> Dict:
        """Delete a person from the database."""
        return self._make_request("DELETE", f"/api/persons/{person_id}")

    def add_person_face(
        self, person_id: int, image: Union[str, Path, bytes, Image.Image]
    ) -> Dict:
        """
        Add a face image to an existing person.

        Args:
            person_id: Person ID
            image: Face image to add

        Returns:
            Dict containing update status
        """
        image_data = self._prepare_image(image)
        files = {"file": image_data}

        return self._make_request(
            "POST", f"/api/persons/{person_id}/faces", files=files
        )

    # Embeddings Methods

    def get_embedding(
        self, image: Union[str, Path, bytes, Image.Image]
    ) -> Dict:
        """
        Get face embedding vector from an image.

        Args:
            image: Image file path, bytes, or PIL Image

        Returns:
            Dict containing face embedding vector
        """
        image_data = self._prepare_image(image)
        files = {"file": image_data}

        return self._make_request("POST", "/api/embeddings", files=files)

    def compare_faces(
        self,
        image1: Union[str, Path, bytes, Image.Image],
        image2: Union[str, Path, bytes, Image.Image],
    ) -> Dict:
        """
        Compare two face images.

        Args:
            image1: First face image
            image2: Second face image

        Returns:
            Dict containing similarity score
        """
        image1_data = self._prepare_image(image1)
        image2_data = self._prepare_image(image2)
        files = [("file1", image1_data), ("file2", image2_data)]

        return self._make_request("POST", "/api/compare", files=files)

    # Model Information Methods

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return self._make_request("GET", "/api/model/info")

    def get_metrics(self) -> str:
        """Get Prometheus metrics."""
        url = f"{self.base_url}/metrics"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    # Context Manager Support

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()

    def close(self):
        """Close the session."""
        self.session.close()


class FaceRecognitionAPIError(Exception):
    """Exception raised for Face Recognition API errors."""

    pass


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = FaceRecognitionClient(base_url="http://localhost:8000")

    # Check health
    health = client.health_check()
    print(f"API Health: {health}")

    # Detect faces
    try:
        result = client.detect_faces("path/to/image.jpg", min_confidence=0.8)
        print(f"Detected {len(result.get('faces', []))} faces")
    except FaceRecognitionAPIError as e:
        print(f"Error: {e}")

    # Create a person
    try:
        person = client.create_person(
            name="John Doe",
            image="path/to/face.jpg",
            metadata={"department": "Engineering"},
        )
        print(f"Created person: {person}")
    except FaceRecognitionAPIError as e:
        print(f"Error: {e}")

    # List persons
    persons = client.list_persons(limit=10)
    print(f"Total persons: {len(persons)}")

    # Recognize faces
    try:
        matches = client.recognize_faces("path/to/unknown.jpg", threshold=0.6)
        print(f"Recognition results: {matches}")
    except FaceRecognitionAPIError as e:
        print(f"Error: {e}")

    # Close client
    client.close()
