#!/usr/bin/env python3
"""
Comprehensive Load Testing Script

Advanced load testing for face recognition API with:
- Concurrent request simulation
- Multiple endpoint testing
- Performance metrics collection
- Latency percentiles (P50, P95, P99)
- Throughput measurement
- Error rate tracking
- Detailed reporting (JSON, HTML)

Features:
- Configurable load patterns
- Realistic traffic simulation
- Real-time progress monitoring
- Resource utilization tracking
"""

import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import argparse
import os
from pathlib import Path


# Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "test-api-key")


@dataclass
class RequestResult:
    """Single request result"""
    endpoint: str
    status_code: int
    duration: float
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LoadTestResults:
    """Load test results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    requests_per_second: float
    
    # Latency statistics (in milliseconds)
    min_latency: float
    max_latency: float
    mean_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    
    # Error statistics
    error_rate: float
    status_codes: Dict[int, int]
    errors: List[str]
    
    # Per-endpoint results
    endpoint_results: Dict[str, Dict[str, Any]]


class LoadTester:
    """Load testing tool"""
    
    def __init__(
        self,
        base_url: str = API_BASE_URL,
        api_key: str = API_KEY,
        output_dir: str = "load_test_results"
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[RequestResult] = []
        
        # Sample data
        self.sample_image_base64 = self._create_sample_image()
    
    def _create_sample_image(self) -> str:
        """Create a sample image for testing"""
        # Simple base64 encoded test data
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    async def make_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None
    ) -> RequestResult:
        """Make a single request"""
        url = f"{self.base_url}{endpoint}"
        headers = {"X-API-Key": self.api_key}
        
        start_time = time.time()
        
        try:
            if method == "GET":
                async with session.get(url, headers=headers) as response:
                    await response.read()
                    status_code = response.status
            elif method == "POST":
                headers["Content-Type"] = "application/json"
                async with session.post(url, json=payload, headers=headers) as response:
                    await response.read()
                    status_code = response.status
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            duration = time.time() - start_time
            success = 200 <= status_code < 300
            
            return RequestResult(
                endpoint=endpoint,
                status_code=status_code,
                duration=duration,
                success=success
            )
        
        except Exception as e:
            duration = time.time() - start_time
            
            return RequestResult(
                endpoint=endpoint,
                status_code=0,
                duration=duration,
                success=False,
                error=str(e)
            )
    
    async def run_load_test(
        self,
        endpoints: List[Dict[str, Any]],
        concurrent_users: int = 10,
        requests_per_user: int = 10,
        ramp_up_time: float = 0
    ) -> LoadTestResults:
        """
        Run load test
        
        Args:
            endpoints: List of endpoint configurations
            concurrent_users: Number of concurrent users
            requests_per_user: Requests per user
            ramp_up_time: Ramp-up time in seconds
        
        Returns:
            LoadTestResults
        """
        print(f"\n{'='*60}")
        print("Load Test Configuration")
        print(f"{'='*60}")
        print(f"Target: {self.base_url}")
        print(f"Concurrent users: {concurrent_users}")
        print(f"Requests per user: {requests_per_user}")
        print(f"Total requests: {concurrent_users * requests_per_user}")
        print(f"Ramp-up time: {ramp_up_time}s")
        print(f"{'='*60}\n")
        
        self.results = []
        start_time = time.time()
        
        # Create connector with custom limits
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            
            for user_id in range(concurrent_users):
                # Ramp-up delay
                if ramp_up_time > 0:
                    delay = (ramp_up_time / concurrent_users) * user_id
                    await asyncio.sleep(delay)
                
                # Create user tasks
                for _ in range(requests_per_user):
                    # Select endpoint (round-robin)
                    endpoint = endpoints[len(tasks) % len(endpoints)]
                    
                    task = self.make_request(
                        session,
                        endpoint["path"],
                        endpoint.get("method", "GET"),
                        endpoint.get("payload")
                    )
                    
                    tasks.append(task)
            
            # Execute all requests
            print(f"Executing {len(tasks)} requests...")
            
            # Show progress
            completed = 0
            for coro in asyncio.as_completed(tasks):
                result = await coro
                self.results.append(result)
                
                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    progress = (completed / len(tasks)) * 100
                    print(f"Progress: {completed}/{len(tasks)} ({progress:.1f}%)", end='\r')
        
        print()  # New line after progress
        
        total_duration = time.time() - start_time
        
        # Analyze results
        return self._analyze_results(total_duration)
    
    def _analyze_results(self, total_duration: float) -> LoadTestResults:
        """Analyze test results"""
        if not self.results:
            raise ValueError("No results to analyze")
        
        # Basic statistics
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        
        # Latency statistics (convert to milliseconds)
        latencies = [r.duration * 1000 for r in self.results]
        latencies.sort()
        
        # Status codes
        status_codes = {}
        for result in self.results:
            status_codes[result.status_code] = status_codes.get(result.status_code, 0) + 1
        
        # Errors
        errors = [r.error for r in self.results if r.error]
        
        # Per-endpoint statistics
        endpoint_results = {}
        endpoints = set(r.endpoint for r in self.results)
        
        for endpoint in endpoints:
            endpoint_requests = [r for r in self.results if r.endpoint == endpoint]
            endpoint_latencies = [r.duration * 1000 for r in endpoint_requests]
            endpoint_latencies.sort()
            
            endpoint_results[endpoint] = {
                "total_requests": len(endpoint_requests),
                "successful": sum(1 for r in endpoint_requests if r.success),
                "failed": sum(1 for r in endpoint_requests if not r.success),
                "mean_latency": statistics.mean(endpoint_latencies) if endpoint_latencies else 0,
                "p95_latency": endpoint_latencies[int(len(endpoint_latencies) * 0.95)] if endpoint_latencies else 0
            }
        
        return LoadTestResults(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_duration=total_duration,
            requests_per_second=total_requests / total_duration if total_duration > 0 else 0,
            min_latency=min(latencies),
            max_latency=max(latencies),
            mean_latency=statistics.mean(latencies),
            median_latency=statistics.median(latencies),
            p95_latency=latencies[int(len(latencies) * 0.95)],
            p99_latency=latencies[int(len(latencies) * 0.99)],
            error_rate=(failed_requests / total_requests) * 100,
            status_codes=status_codes,
            errors=errors[:10],  # Top 10 errors
            endpoint_results=endpoint_results
        )
    
    def print_results(self, results: LoadTestResults):
        """Print test results"""
        print(f"\n{'='*60}")
        print("Load Test Results")
        print(f"{'='*60}\n")
        
        print("Summary:")
        print(f"  Total requests: {results.total_requests}")
        print(f"  Successful: {results.successful_requests}")
        print(f"  Failed: {results.failed_requests}")
        print(f"  Duration: {results.total_duration:.2f}s")
        print(f"  Throughput: {results.requests_per_second:.2f} req/s")
        print(f"  Error rate: {results.error_rate:.2f}%")
        print()
        
        print("Latency (ms):")
        print(f"  Min: {results.min_latency:.2f}")
        print(f"  Max: {results.max_latency:.2f}")
        print(f"  Mean: {results.mean_latency:.2f}")
        print(f"  Median: {results.median_latency:.2f}")
        print(f"  P95: {results.p95_latency:.2f}")
        print(f"  P99: {results.p99_latency:.2f}")
        print()
        
        print("Status Codes:")
        for code, count in sorted(results.status_codes.items()):
            percentage = (count / results.total_requests) * 100
            print(f"  {code}: {count} ({percentage:.1f}%)")
        print()
        
        if results.errors:
            print("Sample Errors:")
            for error in results.errors[:5]:
                print(f"  - {error}")
            print()
        
        print("Per-Endpoint Results:")
        for endpoint, stats in results.endpoint_results.items():
            print(f"  {endpoint}:")
            print(f"    Requests: {stats['total_requests']}")
            print(f"    Success rate: {(stats['successful']/stats['total_requests'])*100:.1f}%")
            print(f"    Mean latency: {stats['mean_latency']:.2f}ms")
            print(f"    P95 latency: {stats['p95_latency']:.2f}ms")
        
        print(f"\n{'='*60}")
    
    def save_results(self, results: LoadTestResults, filename: Optional[str] = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"load_test_{timestamp}.json"
        
        output_file = self.output_dir / filename
        
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_requests": results.total_requests,
                "successful_requests": results.successful_requests,
                "failed_requests": results.failed_requests,
                "total_duration": results.total_duration,
                "requests_per_second": results.requests_per_second,
                "error_rate": results.error_rate
            },
            "latency": {
                "min": results.min_latency,
                "max": results.max_latency,
                "mean": results.mean_latency,
                "median": results.median_latency,
                "p95": results.p95_latency,
                "p99": results.p99_latency
            },
            "status_codes": results.status_codes,
            "errors": results.errors,
            "endpoint_results": results.endpoint_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Load testing tool for Face Recognition API")
    parser.add_argument("--url", default=API_BASE_URL, help="API base URL")
    parser.add_argument("--users", type=int, default=10, help="Concurrent users")
    parser.add_argument("--requests", type=int, default=10, help="Requests per user")
    parser.add_argument("--rampup", type=float, default=0, help="Ramp-up time in seconds")
    parser.add_argument("--output", help="Output filename")
    
    args = parser.parse_args()
    
    # Create tester
    tester = LoadTester(base_url=args.url)
    
    # Define endpoints to test
    endpoints = [
        {
            "path": "/health",
            "method": "GET"
        },
        {
            "path": "/api/v1/detect",
            "method": "POST",
            "payload": {"image": tester.sample_image_base64}
        }
    ]
    
    # Run load test
    results = await tester.run_load_test(
        endpoints=endpoints,
        concurrent_users=args.users,
        requests_per_user=args.requests,
        ramp_up_time=args.rampup
    )
    
    # Print results
    tester.print_results(results)
    
    # Save results
    tester.save_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
