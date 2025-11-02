"""
Python examples for interacting with the Face Recognition API.

This module demonstrates how to use the API endpoints for:
- Person management
- Face recognition
- Image uploads
- System monitoring
"""

import requests
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional


class FaceRecognitionClient:
    """Client for interacting with the Face Recognition API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """
        Check API health status.
        
        Returns:
            Health check response
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_persons(self) -> List[Dict]:
        """
        List all persons in the database.
        
        Returns:
            List of person objects
        """
        response = self.session.get(f"{self.base_url}/persons")
        response.raise_for_status()
        return response.json()
    
    def create_person(self, name: str, description: str = "") -> Dict:
        """
        Create a new person.
        
        Args:
            name: Person's name
            description: Optional description
            
        Returns:
            Created person object
        """
        data = {
            "name": name,
            "description": description
        }
        response = self.session.post(
            f"{self.base_url}/persons",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def add_person_image(self, person_id: int, image_path: str) -> Dict:
        """
        Add a face image for a person.
        
        Args:
            person_id: ID of the person
            image_path: Path to the image file
            
        Returns:
            Response with embeddings count
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = self.session.post(
                f"{self.base_url}/persons/{person_id}/images",
                files=files
            )
        response.raise_for_status()
        return response.json()
    
    def recognize_from_file(
        self,
        image_path: str,
        threshold: float = 0.7
    ) -> Dict:
        """
        Recognize faces in an image file.
        
        Args:
            image_path: Path to the image file
            threshold: Recognition threshold (0.0-1.0)
            
        Returns:
            Recognition results
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {'threshold': threshold}
            response = self.session.post(
                f"{self.base_url}/upload",
                files=files,
                params=params
            )
        response.raise_for_status()
        return response.json()
    
    def recognize_from_base64(
        self,
        image_base64: str,
        threshold: float = 0.7,
        top_k: int = 5
    ) -> Dict:
        """
        Recognize faces from base64 encoded image.
        
        Args:
            image_base64: Base64 encoded image string
            threshold: Recognition threshold
            top_k: Maximum number of results
            
        Returns:
            Recognition results
        """
        data = {
            "image_base64": image_base64,
            "threshold": threshold,
            "top_k": top_k
        }
        response = self.session.post(
            f"{self.base_url}/recognize",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict:
        """
        Get system statistics.
        
        Returns:
            System statistics
        """
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics.
        
        Returns:
            Performance metrics
        """
        response = self.session.get(f"{self.base_url}/performance/metrics")
        response.raise_for_status()
        return response.json()


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


# Example Usage
def main():
    """Demonstrate API usage with examples."""
    
    # Initialize client
    client = FaceRecognitionClient("http://localhost:8000")
    
    print("=" * 60)
    print("Face Recognition API - Python Examples")
    print("=" * 60)
    
    # 1. Health Check
    print("\n1. Checking API health...")
    try:
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Uptime: {health.get('uptime', 'N/A')}")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # 2. Create a person
    print("\n2. Creating a new person...")
    try:
        person = client.create_person(
            name="John Doe",
            description="Example person for testing"
        )
        person_id = person['id']
        print(f"   Created person: {person['name']} (ID: {person_id})")
    except Exception as e:
        print(f"   Error: {e}")
        person_id = None
    
    # 3. List all persons
    print("\n3. Listing all persons...")
    try:
        persons = client.list_persons()
        print(f"   Found {len(persons)} person(s)")
        for p in persons[:5]:  # Show first 5
            print(f"   - {p['name']} (ID: {p['id']})")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. Add image to person (if image exists)
    print("\n4. Adding face image to person...")
    test_image = "data/images/test_face.jpg"
    if person_id and Path(test_image).exists():
        try:
            result = client.add_person_image(person_id, test_image)
            print(f"   Added {result.get('embeddings_count', 0)} embedding(s)")
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print(f"   Skipped (no person ID or test image not found)")
    
    # 5. Recognize faces from file
    print("\n5. Recognizing faces from file...")
    if Path(test_image).exists():
        try:
            result = client.recognize_from_file(test_image, threshold=0.7)
            print(f"   Success: {result['success']}")
            print(f"   Faces detected: {len(result['faces'])}")
            for i, face in enumerate(result['faces']):
                if face['person_name']:
                    print(f"   - Face {i+1}: {face['person_name']} "
                          f"(confidence: {face['confidence']:.2f})")
                else:
                    print(f"   - Face {i+1}: Unknown")
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print(f"   Skipped (test image not found: {test_image})")
    
    # 6. Recognize faces from base64
    print("\n6. Recognizing faces from base64...")
    if Path(test_image).exists():
        try:
            image_b64 = encode_image_to_base64(test_image)
            result = client.recognize_from_base64(image_b64, threshold=0.7)
            print(f"   Processing time: {result['processing_time']:.3f}s")
            print(f"   Faces detected: {len(result['faces'])}")
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print(f"   Skipped (test image not found)")
    
    # 7. Get system statistics
    print("\n7. Getting system statistics...")
    try:
        stats = client.get_stats()
        print(f"   Total persons: {stats['total_persons']}")
        print(f"   Total embeddings: {stats['total_embeddings']}")
        print(f"   Total recognitions: {stats['total_recognitions']}")
        print(f"   Avg processing time: {stats['avg_processing_time']:.3f}s")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 8. Get performance metrics
    print("\n8. Getting performance metrics...")
    try:
        metrics = client.get_performance_metrics()
        print(f"   Recognition metrics available: "
              f"{metrics['recognition_metrics']['recent_count']}")
        system_metrics = metrics.get('system_metrics', {})
        if system_metrics:
            print(f"   CPU usage: {system_metrics.get('cpu_percent', 'N/A')}%")
            print(f"   Memory usage: {system_metrics.get('memory_percent', 'N/A')}%")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
