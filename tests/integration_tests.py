# Comprehensive Integration Testing System

import asyncio
import pytest
import requests
import websockets
import tempfile
import numpy as np
from pathlib import Path
import json
import time
import subprocess
import threading
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from unittest.mock import Mock, patch
import cv2
from PIL import Image
import io

# Test configuration
@dataclass
class TestConfig:
    """Test configuration settings."""
    api_base_url: str = "http://localhost:8000"
    dashboard_url: str = "http://localhost:8080"
    websocket_url: str = "ws://localhost:8000"
    test_data_dir: str = "tests/data"
    temp_dir: str = "tests/temp"
    timeout: int = 30
    max_concurrent_requests: int = 10

class TestDataGenerator:
    """Generate test data for face recognition system."""
    
    def __init__(self, test_config: TestConfig):
        self.config = test_config
        self.test_data_dir = Path(test_config.test_data_dir)
        self.temp_dir = Path(test_config.temp_dir)
        
        # Create directories
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_test_image(self, width: int = 640, height: int = 480, 
                          face_count: int = 1) -> bytes:
        """Generate a synthetic test image with faces."""
        # Create a random image
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add simple face-like rectangles (for testing purposes)
        for i in range(face_count):
            x = np.random.randint(50, width - 150)
            y = np.random.randint(50, height - 150)
            
            # Draw a simple face-like shape
            cv2.rectangle(image, (x, y), (x + 100, y + 120), (255, 200, 180), -1)
            cv2.rectangle(image, (x + 20, y + 30), (x + 30, y + 40), (0, 0, 0), -1)  # Eye
            cv2.rectangle(image, (x + 70, y + 30), (x + 80, y + 40), (0, 0, 0), -1)  # Eye
            cv2.rectangle(image, (x + 40, y + 70), (x + 60, y + 85), (0, 0, 0), -1)  # Mouth
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()
    
    def generate_test_dataset(self, num_persons: int = 10, 
                            images_per_person: int = 5) -> Dict[str, List[bytes]]:
        """Generate a test dataset with multiple persons."""
        dataset = {}
        
        for person_id in range(num_persons):
            person_name = f"test_person_{person_id:03d}"
            person_images = []
            
            for image_id in range(images_per_person):
                image_data = self.generate_test_image(face_count=1)
                person_images.append(image_data)
            
            dataset[person_name] = person_images
        
        return dataset
    
    def create_test_video(self, duration_seconds: int = 10, fps: int = 30) -> bytes:
        """Create a test video with faces."""
        # Create temporary video file
        temp_video = self.temp_dir / f"test_video_{int(time.time())}.mp4"
        
        # Video properties
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
        
        total_frames = duration_seconds * fps
        
        for frame_num in range(total_frames):
            # Generate frame with moving face
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add moving face
            x = int((frame_num / total_frames) * (width - 100))
            y = height // 2 - 60
            
            cv2.rectangle(frame, (x, y), (x + 100, y + 120), (255, 200, 180), -1)
            cv2.rectangle(frame, (x + 20, y + 30), (x + 30, y + 40), (0, 0, 0), -1)
            cv2.rectangle(frame, (x + 70, y + 30), (x + 80, y + 40), (0, 0, 0), -1)
            cv2.rectangle(frame, (x + 40, y + 70), (x + 60, y + 85), (0, 0, 0), -1)
            
            out.write(frame)
        
        out.release()
        
        # Read video as bytes
        with open(temp_video, 'rb') as f:
            video_data = f.read()
        
        # Cleanup
        temp_video.unlink()
        
        return video_data


class APITestSuite:
    """Comprehensive API testing suite."""
    
    def __init__(self, test_config: TestConfig):
        self.config = test_config
        self.session = requests.Session()
        self.test_data_generator = TestDataGenerator(test_config)
        self.test_results = []
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests."""
        results = {
            "test_suite": "API Integration Tests",
            "start_time": time.time(),
            "tests": {},
            "summary": {}
        }
        
        test_methods = [
            self.test_health_endpoint,
            self.test_person_management,
            self.test_face_detection,
            self.test_face_recognition,
            self.test_batch_processing,
            self.test_real_time_recognition,
            self.test_model_management,
            self.test_performance_metrics,
            self.test_security_features,
            self.test_error_handling,
            self.test_websocket_functionality,
            self.test_concurrent_requests
        ]
        
        for test_method in test_methods:
            test_name = test_method.__name__
            try:
                print(f"Running {test_name}...")
                test_result = await test_method()
                results["tests"][test_name] = {
                    "status": "passed",
                    "result": test_result,
                    "execution_time": test_result.get("execution_time", 0)
                }
                print(f"✓ {test_name} passed")
                
            except Exception as e:
                results["tests"][test_name] = {
                    "status": "failed",
                    "error": str(e),
                    "execution_time": 0
                }
                print(f"✗ {test_name} failed: {e}")
        
        # Generate summary
        total_tests = len(test_methods)
        passed_tests = sum(1 for test in results["tests"].values() if test["status"] == "passed")
        failed_tests = total_tests - passed_tests
        
        results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": passed_tests / total_tests,
            "total_execution_time": time.time() - results["start_time"]
        }
        
        return results
    
    async def test_health_endpoint(self) -> Dict[str, Any]:
        """Test health check endpoint."""
        start_time = time.time()
        
        response = self.session.get(f"{self.config.api_base_url}/health")
        
        assert response.status_code == 200
        health_data = response.json()
        assert "status" in health_data
        
        return {
            "status_code": response.status_code,
            "response_data": health_data,
            "execution_time": time.time() - start_time
        }
    
    async def test_person_management(self) -> Dict[str, Any]:
        """Test person management endpoints."""
        start_time = time.time()
        
        # Create person
        person_data = {
            "name": "Test Person",
            "email": "test@example.com",
            "description": "Integration test person"
        }
        
        response = self.session.post(
            f"{self.config.api_base_url}/persons",
            json=person_data
        )
        assert response.status_code == 201
        person = response.json()
        person_id = person["id"]
        
        # Get person
        response = self.session.get(f"{self.config.api_base_url}/persons/{person_id}")
        assert response.status_code == 200
        retrieved_person = response.json()
        assert retrieved_person["name"] == person_data["name"]
        
        # Update person
        update_data = {"name": "Updated Test Person"}
        response = self.session.put(
            f"{self.config.api_base_url}/persons/{person_id}",
            json=update_data
        )
        assert response.status_code == 200
        
        # List persons
        response = self.session.get(f"{self.config.api_base_url}/persons")
        assert response.status_code == 200
        persons_list = response.json()
        assert len(persons_list) >= 1
        
        # Delete person
        response = self.session.delete(f"{self.config.api_base_url}/persons/{person_id}")
        assert response.status_code == 204
        
        return {
            "person_created": person,
            "operations_tested": ["create", "read", "update", "list", "delete"],
            "execution_time": time.time() - start_time
        }
    
    async def test_face_detection(self) -> Dict[str, Any]:
        """Test face detection functionality."""
        start_time = time.time()
        
        # Generate test image
        test_image = self.test_data_generator.generate_test_image(face_count=2)
        
        # Test face detection
        files = {"image": ("test.jpg", test_image, "image/jpeg")}
        response = self.session.post(
            f"{self.config.api_base_url}/detect-faces",
            files=files
        )
        
        assert response.status_code == 200
        detection_result = response.json()
        assert "faces" in detection_result
        assert "face_count" in detection_result
        
        return {
            "detection_result": detection_result,
            "image_size": len(test_image),
            "execution_time": time.time() - start_time
        }
    
    async def test_face_recognition(self) -> Dict[str, Any]:
        """Test face recognition functionality."""
        start_time = time.time()
        
        # First, create a person and add face embeddings
        person_data = {"name": "Recognition Test Person"}
        response = self.session.post(
            f"{self.config.api_base_url}/persons",
            json=person_data
        )
        person_id = response.json()["id"]
        
        # Add face embedding
        test_image = self.test_data_generator.generate_test_image(face_count=1)
        files = {"image": ("face.jpg", test_image, "image/jpeg")}
        data = {"person_id": person_id}
        
        response = self.session.post(
            f"{self.config.api_base_url}/persons/{person_id}/faces",
            files=files,
            data=data
        )
        assert response.status_code == 201
        
        # Test recognition
        recognition_image = self.test_data_generator.generate_test_image(face_count=1)
        files = {"image": ("recognize.jpg", recognition_image, "image/jpeg")}
        
        response = self.session.post(
            f"{self.config.api_base_url}/recognize",
            files=files
        )
        
        assert response.status_code == 200
        recognition_result = response.json()
        assert "results" in recognition_result
        
        # Cleanup
        self.session.delete(f"{self.config.api_base_url}/persons/{person_id}")
        
        return {
            "recognition_result": recognition_result,
            "person_id": person_id,
            "execution_time": time.time() - start_time
        }
    
    async def test_batch_processing(self) -> Dict[str, Any]:
        """Test batch processing functionality."""
        start_time = time.time()
        
        # Generate multiple test images
        batch_images = []
        for i in range(5):
            image_data = self.test_data_generator.generate_test_image()
            batch_images.append(("images", ("test_{}.jpg".format(i), image_data, "image/jpeg")))
        
        # Test batch face detection
        response = self.session.post(
            f"{self.config.api_base_url}/batch/detect-faces",
            files=batch_images
        )
        
        assert response.status_code == 200
        batch_result = response.json()
        assert "results" in batch_result
        assert len(batch_result["results"]) == 5
        
        return {
            "batch_size": 5,
            "batch_result": batch_result,
            "execution_time": time.time() - start_time
        }
    
    async def test_real_time_recognition(self) -> Dict[str, Any]:
        """Test real-time recognition functionality."""
        start_time = time.time()
        
        # Test video processing
        test_video = self.test_data_generator.create_test_video(duration_seconds=2)
        
        files = {"video": ("test.mp4", test_video, "video/mp4")}
        response = self.session.post(
            f"{self.config.api_base_url}/process-video",
            files=files
        )
        
        assert response.status_code == 200
        video_result = response.json()
        assert "frames_processed" in video_result
        
        return {
            "video_size": len(test_video),
            "video_result": video_result,
            "execution_time": time.time() - start_time
        }
    
    async def test_model_management(self) -> Dict[str, Any]:
        """Test model management endpoints."""
        start_time = time.time()
        
        # List available models
        response = self.session.get(f"{self.config.api_base_url}/models")
        assert response.status_code == 200
        models_list = response.json()
        
        # Get model info
        if models_list:
            model_name = list(models_list.keys())[0]
            response = self.session.get(f"{self.config.api_base_url}/models/{model_name}")
            assert response.status_code == 200
            model_info = response.json()
            assert "metadata" in model_info
        
        return {
            "available_models": models_list,
            "execution_time": time.time() - start_time
        }
    
    async def test_performance_metrics(self) -> Dict[str, Any]:
        """Test performance metrics endpoints."""
        start_time = time.time()
        
        # Get system metrics
        response = self.session.get(f"{self.config.api_base_url}/metrics/system")
        assert response.status_code == 200
        system_metrics = response.json()
        
        # Get application metrics
        response = self.session.get(f"{self.config.api_base_url}/metrics/app")
        assert response.status_code == 200
        app_metrics = response.json()
        
        return {
            "system_metrics": system_metrics,
            "app_metrics": app_metrics,
            "execution_time": time.time() - start_time
        }
    
    async def test_security_features(self) -> Dict[str, Any]:
        """Test security features."""
        start_time = time.time()
        
        # Test rate limiting
        responses = []
        for i in range(10):
            response = self.session.get(f"{self.config.api_base_url}/health")
            responses.append(response.status_code)
        
        # Test input validation
        invalid_data = {"name": "<script>alert('xss')</script>"}
        response = self.session.post(
            f"{self.config.api_base_url}/persons",
            json=invalid_data
        )
        # Should handle XSS attempt gracefully
        
        # Test file upload validation
        malicious_file = b"malicious content"
        files = {"image": ("malicious.exe", malicious_file, "application/octet-stream")}
        response = self.session.post(
            f"{self.config.api_base_url}/detect-faces",
            files=files
        )
        assert response.status_code == 400  # Should reject invalid file type
        
        return {
            "rate_limiting_responses": responses,
            "input_validation_tested": True,
            "file_validation_tested": True,
            "execution_time": time.time() - start_time
        }
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling scenarios."""
        start_time = time.time()
        
        error_scenarios = []
        
        # Test 404 - Non-existent person
        response = self.session.get(f"{self.config.api_base_url}/persons/99999")
        error_scenarios.append({
            "scenario": "non_existent_person",
            "status_code": response.status_code,
            "expected": 404
        })
        
        # Test 400 - Invalid data
        response = self.session.post(
            f"{self.config.api_base_url}/persons",
            json={"invalid": "data"}
        )
        error_scenarios.append({
            "scenario": "invalid_person_data",
            "status_code": response.status_code,
            "expected": 400
        })
        
        # Test 413 - File too large (simulate)
        large_data = b"x" * (11 * 1024 * 1024)  # 11MB
        files = {"image": ("large.jpg", large_data, "image/jpeg")}
        response = self.session.post(
            f"{self.config.api_base_url}/detect-faces",
            files=files
        )
        error_scenarios.append({
            "scenario": "file_too_large",
            "status_code": response.status_code,
            "expected": 413
        })
        
        return {
            "error_scenarios": error_scenarios,
            "execution_time": time.time() - start_time
        }
    
    async def test_websocket_functionality(self) -> Dict[str, Any]:
        """Test WebSocket functionality."""
        start_time = time.time()
        
        try:
            # Test WebSocket connection
            uri = f"ws://localhost:8000/ws/recognition"
            
            async with websockets.connect(uri) as websocket:
                # Send test message
                test_message = {"type": "ping", "data": "test"}
                await websocket.send(json.dumps(test_message))
                
                # Receive response
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                response_data = json.loads(response)
                
                websocket_result = {
                    "connection_successful": True,
                    "message_sent": test_message,
                    "response_received": response_data
                }
        
        except Exception as e:
            websocket_result = {
                "connection_successful": False,
                "error": str(e)
            }
        
        return {
            "websocket_test": websocket_result,
            "execution_time": time.time() - start_time
        }
    
    async def test_concurrent_requests(self) -> Dict[str, Any]:
        """Test concurrent request handling."""
        start_time = time.time()
        
        async def make_request(session_id: int):
            try:
                response = self.session.get(f"{self.config.api_base_url}/health")
                return {
                    "session_id": session_id,
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                return {
                    "session_id": session_id,
                    "error": str(e)
                }
        
        # Create concurrent requests
        concurrent_requests = 20
        tasks = [make_request(i) for i in range(concurrent_requests)]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        successful_requests = [r for r in results if isinstance(r, dict) and "status_code" in r]
        failed_requests = [r for r in results if not isinstance(r, dict) or "error" in r]
        
        avg_response_time = np.mean([r["response_time"] for r in successful_requests]) if successful_requests else 0
        
        return {
            "concurrent_requests": concurrent_requests,
            "successful": len(successful_requests),
            "failed": len(failed_requests),
            "success_rate": len(successful_requests) / concurrent_requests,
            "average_response_time": avg_response_time,
            "execution_time": time.time() - start_time
        }


class PerformanceTestSuite:
    """Performance and load testing suite."""
    
    def __init__(self, test_config: TestConfig):
        self.config = test_config
        self.test_data_generator = TestDataGenerator(test_config)
    
    async def run_load_test(self, duration_seconds: int = 60, 
                          requests_per_second: int = 10) -> Dict[str, Any]:
        """Run load test for specified duration."""
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        results = {
            "test_type": "load_test",
            "duration_seconds": duration_seconds,
            "target_rps": requests_per_second,
            "requests": [],
            "metrics": {}
        }
        
        async def make_request():
            try:
                request_start = time.time()
                
                # Generate test image
                test_image = self.test_data_generator.generate_test_image()
                
                # Make request
                files = {"image": ("test.jpg", test_image, "image/jpeg")}
                response = requests.post(
                    f"{self.config.api_base_url}/detect-faces",
                    files=files,
                    timeout=self.config.timeout
                )
                
                request_end = time.time()
                
                return {
                    "timestamp": request_start,
                    "status_code": response.status_code,
                    "response_time": request_end - request_start,
                    "success": response.status_code == 200
                }
                
            except Exception as e:
                return {
                    "timestamp": time.time(),
                    "error": str(e),
                    "success": False
                }
        
        # Generate requests at specified rate
        while time.time() < end_time:
            # Create batch of requests
            batch_size = min(requests_per_second, int((end_time - time.time()) * requests_per_second))
            
            if batch_size <= 0:
                break
            
            # Execute batch
            batch_start = time.time()
            tasks = [make_request() for _ in range(batch_size)]
            batch_results = await asyncio.gather(*tasks)
            
            results["requests"].extend(batch_results)
            
            # Wait for next batch
            batch_duration = time.time() - batch_start
            sleep_time = max(0, 1.0 - batch_duration)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        # Calculate metrics
        successful_requests = [r for r in results["requests"] if r.get("success", False)]
        failed_requests = [r for r in results["requests"] if not r.get("success", False)]
        
        if successful_requests:
            response_times = [r["response_time"] for r in successful_requests]
            results["metrics"] = {
                "total_requests": len(results["requests"]),
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(successful_requests) / len(results["requests"]),
                "actual_rps": len(results["requests"]) / duration_seconds,
                "avg_response_time": np.mean(response_times),
                "min_response_time": np.min(response_times),
                "max_response_time": np.max(response_times),
                "p95_response_time": np.percentile(response_times, 95),
                "p99_response_time": np.percentile(response_times, 99)
            }
        
        return results
    
    async def run_stress_test(self, max_concurrent_users: int = 100) -> Dict[str, Any]:
        """Run stress test with increasing load."""
        
        results = {
            "test_type": "stress_test",
            "max_concurrent_users": max_concurrent_users,
            "load_points": []
        }
        
        # Test with increasing concurrent users
        for concurrent_users in [1, 5, 10, 20, 50, max_concurrent_users]:
            print(f"Testing with {concurrent_users} concurrent users...")
            
            async def concurrent_user_simulation():
                """Simulate a single user making requests."""
                user_requests = []
                
                for _ in range(5):  # Each user makes 5 requests
                    try:
                        start_time = time.time()
                        
                        test_image = self.test_data_generator.generate_test_image()
                        files = {"image": ("test.jpg", test_image, "image/jpeg")}
                        
                        response = requests.post(
                            f"{self.config.api_base_url}/detect-faces",
                            files=files,
                            timeout=self.config.timeout
                        )
                        
                        user_requests.append({
                            "status_code": response.status_code,
                            "response_time": time.time() - start_time,
                            "success": response.status_code == 200
                        })
                        
                        # Small delay between requests
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        user_requests.append({
                            "error": str(e),
                            "success": False
                        })
                
                return user_requests
            
            # Run concurrent users
            load_start = time.time()
            tasks = [concurrent_user_simulation() for _ in range(concurrent_users)]
            user_results = await asyncio.gather(*tasks)
            load_duration = time.time() - load_start
            
            # Aggregate results
            all_requests = [req for user in user_results for req in user]
            successful_requests = [r for r in all_requests if r.get("success", False)]
            
            load_point = {
                "concurrent_users": concurrent_users,
                "total_requests": len(all_requests),
                "successful_requests": len(successful_requests),
                "success_rate": len(successful_requests) / len(all_requests) if all_requests else 0,
                "load_duration": load_duration,
                "throughput": len(all_requests) / load_duration
            }
            
            if successful_requests:
                response_times = [r["response_time"] for r in successful_requests]
                load_point.update({
                    "avg_response_time": np.mean(response_times),
                    "p95_response_time": np.percentile(response_times, 95)
                })
            
            results["load_points"].append(load_point)
            
            # Brief pause between load levels
            await asyncio.sleep(2)
        
        return results


class IntegrationTestRunner:
    """Main integration test runner."""
    
    def __init__(self, test_config: TestConfig = None):
        self.config = test_config or TestConfig()
        self.api_tests = APITestSuite(self.config)
        self.performance_tests = PerformanceTestSuite(self.config)
    
    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete integration test suite."""
        
        print("Starting comprehensive integration test suite...")
        
        full_results = {
            "test_suite_start": time.time(),
            "configuration": {
                "api_base_url": self.config.api_base_url,
                "timeout": self.config.timeout,
                "max_concurrent_requests": self.config.max_concurrent_requests
            },
            "api_tests": {},
            "performance_tests": {},
            "summary": {}
        }
        
        try:
            # Run API tests
            print("\n=== Running API Integration Tests ===")
            full_results["api_tests"] = await self.api_tests.run_all_tests()
            
            # Run performance tests
            print("\n=== Running Performance Tests ===")
            print("Running load test...")
            load_test_results = await self.performance_tests.run_load_test(
                duration_seconds=30, requests_per_second=5
            )
            
            print("Running stress test...")
            stress_test_results = await self.performance_tests.run_stress_test(
                max_concurrent_users=20
            )
            
            full_results["performance_tests"] = {
                "load_test": load_test_results,
                "stress_test": stress_test_results
            }
            
        except Exception as e:
            full_results["error"] = str(e)
            print(f"Test suite error: {e}")
        
        # Generate final summary
        total_execution_time = time.time() - full_results["test_suite_start"]
        
        api_summary = full_results.get("api_tests", {}).get("summary", {})
        
        full_results["summary"] = {
            "total_execution_time": total_execution_time,
            "api_tests_passed": api_summary.get("passed", 0),
            "api_tests_failed": api_summary.get("failed", 0),
            "api_success_rate": api_summary.get("success_rate", 0),
            "performance_tests_completed": len(full_results.get("performance_tests", {})),
            "overall_status": "PASSED" if api_summary.get("failed", 1) == 0 else "FAILED"
        }
        
        return full_results
    
    def generate_test_report(self, results: Dict[str, Any], output_file: str = None):
        """Generate comprehensive test report."""
        
        if output_file is None:
            output_file = f"test_report_{int(time.time())}.json"
        
        # Save detailed results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary report
        summary_file = output_file.replace('.json', '_summary.txt')
        
        with open(summary_file, 'w') as f:
            f.write("FACE RECOGNITION SYSTEM - INTEGRATION TEST REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            summary = results.get("summary", {})
            f.write(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}\n")
            f.write(f"Total Execution Time: {summary.get('total_execution_time', 0):.2f} seconds\n\n")
            
            f.write("API TESTS SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Tests Passed: {summary.get('api_tests_passed', 0)}\n")
            f.write(f"Tests Failed: {summary.get('api_tests_failed', 0)}\n")
            f.write(f"Success Rate: {summary.get('api_success_rate', 0):.1%}\n\n")
            
            # API test details
            api_tests = results.get("api_tests", {}).get("tests", {})
            for test_name, test_result in api_tests.items():
                status = "PASS" if test_result["status"] == "passed" else "FAIL"
                f.write(f"  {test_name}: {status}\n")
                if test_result["status"] == "failed":
                    f.write(f"    Error: {test_result.get('error', 'Unknown')}\n")
            
            f.write("\nPERFORMANCE TESTS SUMMARY:\n")
            f.write("-" * 25 + "\n")
            
            # Load test summary
            load_test = results.get("performance_tests", {}).get("load_test", {})
            load_metrics = load_test.get("metrics", {})
            if load_metrics:
                f.write(f"Load Test - Success Rate: {load_metrics.get('success_rate', 0):.1%}\n")
                f.write(f"Load Test - Avg Response Time: {load_metrics.get('avg_response_time', 0):.3f}s\n")
                f.write(f"Load Test - P95 Response Time: {load_metrics.get('p95_response_time', 0):.3f}s\n")
            
            # Stress test summary
            stress_test = results.get("performance_tests", {}).get("stress_test", {})
            load_points = stress_test.get("load_points", [])
            if load_points:
                max_load = load_points[-1]
                f.write(f"Stress Test - Max Concurrent Users: {max_load.get('concurrent_users', 0)}\n")
                f.write(f"Stress Test - Max Load Success Rate: {max_load.get('success_rate', 0):.1%}\n")
        
        print(f"Test results saved to: {output_file}")
        print(f"Summary report saved to: {summary_file}")


# Main execution
async def main():
    """Run integration tests."""
    test_runner = IntegrationTestRunner()
    
    results = await test_runner.run_full_test_suite()
    
    # Generate report
    test_runner.generate_test_report(results)
    
    # Print summary
    summary = results.get("summary", {})
    print(f"\n{'='*60}")
    print("INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
    print(f"API Tests: {summary.get('api_tests_passed', 0)} passed, {summary.get('api_tests_failed', 0)} failed")
    print(f"Success Rate: {summary.get('api_success_rate', 0):.1%}")
    print(f"Total Time: {summary.get('total_execution_time', 0):.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())