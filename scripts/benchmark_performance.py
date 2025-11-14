#!/usr/bin/env python3
"""
Performance Benchmark Script

Comprehensive performance testing for the face recognition system.
Tests detection, recognition, throughput, latency, and resource usage.
"""

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import psutil
import requests
from PIL import Image


@dataclass
class BenchmarkResult:
    """Store benchmark results."""

    operation: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    min_latency_ms: float
    max_latency_ms: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    total_duration_s: float
    cpu_usage_percent: float
    memory_usage_mb: float
    errors: List[str]


class PerformanceBenchmark:
    """Performance benchmark suite."""

    def __init__(self, api_url: str, api_key: str = None):
        """Initialize benchmark."""
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key
        self.results = []

    def _create_test_image(self, size: tuple = (640, 480)) -> bytes:
        """Create a test image."""
        img = Image.new("RGB", size, color=(73, 109, 137))
        import io

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        return buffer.read()

    def _measure_request(
        self, endpoint: str, method: str = "POST", files: dict = None, data: dict = None
    ) -> tuple:
        """Measure a single request."""
        url = f"{self.api_url}{endpoint}"

        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024

        try:
            if method == "POST":
                response = requests.post(url, files=files, json=data, headers=self.headers, timeout=30)
            else:
                response = requests.get(url, headers=self.headers, timeout=30)

            latency = (time.time() - start_time) * 1000  # Convert to ms
            end_cpu = psutil.cpu_percent(interval=None)
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            success = response.status_code == 200
            error = None if success else f"HTTP {response.status_code}"

            return success, latency, (start_cpu + end_cpu) / 2, (start_memory + end_memory) / 2, error

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return False, latency, 0, 0, str(e)

    def benchmark_health_check(self, num_requests: int = 100) -> BenchmarkResult:
        """Benchmark health check endpoint."""
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: Health Check ({num_requests} requests)")
        print(f"{'=' * 60}")

        latencies = []
        cpu_usage = []
        memory_usage = []
        errors = []
        successful = 0
        failed = 0

        start_time = time.time()

        for i in range(num_requests):
            success, latency, cpu, mem, error = self._measure_request("/health", "GET")

            latencies.append(latency)
            cpu_usage.append(cpu)
            memory_usage.append(mem)

            if success:
                successful += 1
            else:
                failed += 1
                if error:
                    errors.append(error)

            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{num_requests}")

        duration = time.time() - start_time

        result = BenchmarkResult(
            operation="health_check",
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            mean_latency_ms=statistics.mean(latencies),
            median_latency_ms=statistics.median(latencies),
            p95_latency_ms=self._percentile(latencies, 95),
            p99_latency_ms=self._percentile(latencies, 99),
            requests_per_second=num_requests / duration,
            total_duration_s=duration,
            cpu_usage_percent=statistics.mean(cpu_usage),
            memory_usage_mb=statistics.mean(memory_usage),
            errors=list(set(errors)),
        )

        self._print_result(result)
        self.results.append(result)
        return result

    def benchmark_face_detection(self, num_requests: int = 50) -> BenchmarkResult:
        """Benchmark face detection endpoint."""
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: Face Detection ({num_requests} requests)")
        print(f"{'=' * 60}")

        test_image = self._create_test_image()
        latencies = []
        cpu_usage = []
        memory_usage = []
        errors = []
        successful = 0
        failed = 0

        start_time = time.time()

        for i in range(num_requests):
            files = {"file": ("test.jpg", test_image, "image/jpeg")}
            success, latency, cpu, mem, error = self._measure_request(
                "/api/detect", "POST", files=files
            )

            latencies.append(latency)
            cpu_usage.append(cpu)
            memory_usage.append(mem)

            if success:
                successful += 1
            else:
                failed += 1
                if error:
                    errors.append(error)

            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{num_requests}")

        duration = time.time() - start_time

        result = BenchmarkResult(
            operation="face_detection",
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            mean_latency_ms=statistics.mean(latencies),
            median_latency_ms=statistics.median(latencies),
            p95_latency_ms=self._percentile(latencies, 95),
            p99_latency_ms=self._percentile(latencies, 99),
            requests_per_second=num_requests / duration,
            total_duration_s=duration,
            cpu_usage_percent=statistics.mean(cpu_usage),
            memory_usage_mb=statistics.mean(memory_usage),
            errors=list(set(errors)),
        )

        self._print_result(result)
        self.results.append(result)
        return result

    def benchmark_face_recognition(self, num_requests: int = 50) -> BenchmarkResult:
        """Benchmark face recognition endpoint."""
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: Face Recognition ({num_requests} requests)")
        print(f"{'=' * 60}")

        test_image = self._create_test_image()
        latencies = []
        cpu_usage = []
        memory_usage = []
        errors = []
        successful = 0
        failed = 0

        start_time = time.time()

        for i in range(num_requests):
            files = {"file": ("test.jpg", test_image, "image/jpeg")}
            success, latency, cpu, mem, error = self._measure_request(
                "/api/recognize", "POST", files=files
            )

            latencies.append(latency)
            cpu_usage.append(cpu)
            memory_usage.append(mem)

            if success:
                successful += 1
            else:
                failed += 1
                if error:
                    errors.append(error)

            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{num_requests}")

        duration = time.time() - start_time

        result = BenchmarkResult(
            operation="face_recognition",
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            mean_latency_ms=statistics.mean(latencies) if latencies else 0,
            median_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            requests_per_second=num_requests / duration,
            total_duration_s=duration,
            cpu_usage_percent=statistics.mean(cpu_usage) if cpu_usage else 0,
            memory_usage_mb=statistics.mean(memory_usage) if memory_usage else 0,
            errors=list(set(errors)),
        )

        self._print_result(result)
        self.results.append(result)
        return result

    def benchmark_concurrent_requests(
        self, num_requests: int = 100, concurrency: int = 10
    ) -> BenchmarkResult:
        """Benchmark concurrent requests."""
        print(f"\n{'=' * 60}")
        print(f"Benchmarking: Concurrent Requests")
        print(f"Total: {num_requests}, Concurrency: {concurrency}")
        print(f"{'=' * 60}")

        test_image = self._create_test_image()
        latencies = []
        errors = []
        successful = 0
        failed = 0

        start_time = time.time()

        def make_request():
            files = {"file": ("test.jpg", test_image, "image/jpeg")}
            success, latency, cpu, mem, error = self._measure_request(
                "/api/detect", "POST", files=files
            )
            return success, latency, error

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]

            for i, future in enumerate(as_completed(futures)):
                success, latency, error = future.result()
                latencies.append(latency)

                if success:
                    successful += 1
                else:
                    failed += 1
                    if error:
                        errors.append(error)

                if (i + 1) % 20 == 0:
                    print(f"Progress: {i + 1}/{num_requests}")

        duration = time.time() - start_time

        result = BenchmarkResult(
            operation=f"concurrent_requests_c{concurrency}",
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            mean_latency_ms=statistics.mean(latencies) if latencies else 0,
            median_latency_ms=statistics.median(latencies) if latencies else 0,
            p95_latency_ms=self._percentile(latencies, 95) if latencies else 0,
            p99_latency_ms=self._percentile(latencies, 99) if latencies else 0,
            requests_per_second=num_requests / duration,
            total_duration_s=duration,
            cpu_usage_percent=0,
            memory_usage_mb=0,
            errors=list(set(errors)),
        )

        self._print_result(result)
        self.results.append(result)
        return result

    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _print_result(self, result: BenchmarkResult):
        """Print benchmark result."""
        print(f"\nResults:")
        print(f"  Total Requests:     {result.total_requests}")
        print(f"  Successful:         {result.successful_requests}")
        print(f"  Failed:             {result.failed_requests}")
        print(f"  Duration:           {result.total_duration_s:.2f}s")
        print(f"  Throughput:         {result.requests_per_second:.2f} req/s")
        print(f"\nLatency:")
        print(f"  Min:                {result.min_latency_ms:.2f}ms")
        print(f"  Max:                {result.max_latency_ms:.2f}ms")
        print(f"  Mean:               {result.mean_latency_ms:.2f}ms")
        print(f"  Median:             {result.median_latency_ms:.2f}ms")
        print(f"  P95:                {result.p95_latency_ms:.2f}ms")
        print(f"  P99:                {result.p99_latency_ms:.2f}ms")

        if result.cpu_usage_percent > 0:
            print(f"\nResources:")
            print(f"  CPU Usage:          {result.cpu_usage_percent:.2f}%")
            print(f"  Memory Usage:       {result.memory_usage_mb:.2f}MB")

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors[:5]:
                print(f"  - {error}")

    def run_all_benchmarks(
        self, num_requests: int = 50, concurrency: int = 10
    ):
        """Run all benchmarks."""
        print("=" * 60)
        print("FACE RECOGNITION PERFORMANCE BENCHMARK")
        print("=" * 60)
        print(f"API URL: {self.api_url}")
        print(f"Timestamp: {datetime.now().isoformat()}")

        # Check if API is available
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code != 200:
                print("\n⚠️  WARNING: API health check failed")
        except Exception as e:
            print(f"\n⚠️  ERROR: Cannot connect to API: {e}")
            return

        # Run benchmarks
        self.benchmark_health_check(num_requests * 2)
        self.benchmark_face_detection(num_requests)
        self.benchmark_face_recognition(num_requests)
        self.benchmark_concurrent_requests(num_requests, concurrency)

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'=' * 60}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 60}")

        print(f"\n{'Operation':<30} {'Throughput':<15} {'Mean Latency':<15}")
        print("-" * 60)

        for result in self.results:
            print(
                f"{result.operation:<30} "
                f"{result.requests_per_second:>8.2f} req/s  "
                f"{result.mean_latency_ms:>8.2f} ms"
            )

    def export_results(self, filename: str):
        """Export results to JSON file."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "api_url": self.api_url,
            "results": [asdict(r) for r in self.results],
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults exported to {filename}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Performance benchmark for Face Recognition API")
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="API base URL",
    )
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument(
        "--requests", type=int, default=50, help="Number of requests per test"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Concurrent requests"
    )
    parser.add_argument("--export", help="Export results to JSON file")

    args = parser.parse_args()

    benchmark = PerformanceBenchmark(args.api_url, args.api_key)
    benchmark.run_all_benchmarks(args.requests, args.concurrency)

    if args.export:
        benchmark.export_results(args.export)


if __name__ == "__main__":
    main()
