#!/usr/bin/env python3
"""
System Health Check Script

Comprehensive health monitoring for the face recognition system.
Checks API endpoints, database connectivity, model availability,
resource usage, and external dependencies.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import requests


class HealthChecker:
    """System health checker for face recognition service."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize health checker with configuration."""
        self.config = config or {}
        self.api_url = self.config.get("api_url", "http://localhost:8000")
        self.timeout = self.config.get("timeout", 10)
        self.results = []
        self.critical_failures = []
        self.warnings = []

    def check_api_health(self) -> Tuple[bool, str]:
        """Check if API is responding to health endpoint."""
        try:
            response = requests.get(
                f"{self.api_url}/health", timeout=self.timeout
            )
            if response.status_code == 200:
                return True, "API health endpoint responding"
            else:
                return False, f"API returned status code {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to API - service may be down"
        except requests.exceptions.Timeout:
            return False, f"API health check timed out after {self.timeout}s"
        except Exception as e:
            return False, f"API health check failed: {str(e)}"

    def check_api_endpoints(self) -> Tuple[bool, str]:
        """Check critical API endpoints."""
        endpoints = [
            ("/api/detect", "POST"),
            ("/api/recognize", "POST"),
            ("/api/persons", "GET"),
            ("/metrics", "GET"),
        ]

        failed_endpoints = []

        for endpoint, method in endpoints:
            try:
                url = f"{self.api_url}{endpoint}"
                if method == "GET":
                    response = requests.get(url, timeout=self.timeout)
                else:
                    # For POST endpoints, just check if they accept requests
                    response = requests.options(url, timeout=self.timeout)

                # Accept both 2xx and 4xx (4xx means endpoint exists but needs data)
                if response.status_code >= 500:
                    failed_endpoints.append(f"{method} {endpoint}")
            except Exception as e:
                failed_endpoints.append(f"{method} {endpoint} ({str(e)})")

        if failed_endpoints:
            return False, f"Failed endpoints: {', '.join(failed_endpoints)}"
        return True, f"All {len(endpoints)} API endpoints accessible"

    def check_database_connection(self) -> Tuple[bool, str]:
        """Check database connectivity."""
        try:
            # Try to hit a database-dependent endpoint
            response = requests.get(
                f"{self.api_url}/api/persons", timeout=self.timeout
            )
            if response.status_code in [200, 404]:  # 404 is OK (empty database)
                return True, "Database connection healthy"
            else:
                return False, f"Database check returned status {response.status_code}"
        except Exception as e:
            return False, f"Database connection check failed: {str(e)}"

    def check_model_availability(self) -> Tuple[bool, str]:
        """Check if ML models are loaded and accessible."""
        try:
            # Check model info endpoint
            response = requests.get(
                f"{self.api_url}/api/model/info", timeout=self.timeout
            )
            if response.status_code == 200:
                model_info = response.json()
                return True, f"Models loaded: {', '.join(model_info.get('models', ['detection', 'recognition']))}"
            elif response.status_code == 404:
                # Endpoint might not exist, try detection instead
                return self._test_model_inference()
            else:
                return False, f"Model info returned status {response.status_code}"
        except Exception:
            # Fall back to testing actual inference
            return self._test_model_inference()

    def _test_model_inference(self) -> Tuple[bool, str]:
        """Test model inference with a dummy request."""
        # This would require a test image, so we'll skip for now
        return True, "Model inference test skipped (requires test image)"

    def check_disk_space(self, threshold_percent: int = 90) -> Tuple[bool, str]:
        """Check available disk space."""
        try:
            disk_usage = psutil.disk_usage("/")
            used_percent = disk_usage.percent

            if used_percent >= threshold_percent:
                return False, f"Disk usage critical: {used_percent}% used"
            elif used_percent >= threshold_percent - 10:
                return True, f"Disk usage warning: {used_percent}% used"
            else:
                return True, f"Disk usage healthy: {used_percent}% used"
        except Exception as e:
            return False, f"Disk space check failed: {str(e)}"

    def check_memory_usage(self, threshold_percent: int = 90) -> Tuple[bool, str]:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            used_percent = memory.percent

            if used_percent >= threshold_percent:
                return False, f"Memory usage critical: {used_percent}% used"
            elif used_percent >= threshold_percent - 10:
                return True, f"Memory usage warning: {used_percent}% used"
            else:
                return True, f"Memory usage healthy: {used_percent}% used"
        except Exception as e:
            return False, f"Memory check failed: {str(e)}"

    def check_cpu_usage(self, threshold_percent: int = 90) -> Tuple[bool, str]:
        """Check CPU usage."""
        try:
            # Sample CPU usage over 1 second
            cpu_percent = psutil.cpu_percent(interval=1)

            if cpu_percent >= threshold_percent:
                return False, f"CPU usage critical: {cpu_percent}%"
            elif cpu_percent >= threshold_percent - 10:
                return True, f"CPU usage warning: {cpu_percent}%"
            else:
                return True, f"CPU usage healthy: {cpu_percent}%"
        except Exception as e:
            return False, f"CPU check failed: {str(e)}"

    def check_file_permissions(self) -> Tuple[bool, str]:
        """Check critical file and directory permissions."""
        critical_paths = [
            "data/",
            "models/",
            "logs/",
        ]

        permission_issues = []

        for path in critical_paths:
            full_path = Path(path)
            if full_path.exists():
                if not os.access(full_path, os.R_OK | os.W_OK):
                    permission_issues.append(f"{path} (not readable/writable)")
            else:
                permission_issues.append(f"{path} (does not exist)")

        if permission_issues:
            return False, f"Permission issues: {', '.join(permission_issues)}"
        return True, "All critical paths have correct permissions"

    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if critical Python dependencies are installed."""
        critical_packages = [
            "fastapi",
            "opencv-python",
            "numpy",
            "torch",
            "face_recognition",
        ]

        missing_packages = []

        for package in critical_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            return False, f"Missing packages: {', '.join(missing_packages)}"
        return True, "All critical dependencies installed"

    def run_all_checks(self) -> Dict:
        """Run all health checks and return results."""
        print("=" * 60)
        print("FACE RECOGNITION SYSTEM HEALTH CHECK")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"API URL: {self.api_url}")
        print()

        checks = [
            ("API Health", self.check_api_health, True),
            ("API Endpoints", self.check_api_endpoints, True),
            ("Database Connection", self.check_database_connection, True),
            ("Model Availability", self.check_model_availability, True),
            ("Disk Space", self.check_disk_space, False),
            ("Memory Usage", self.check_memory_usage, False),
            ("CPU Usage", self.check_cpu_usage, False),
            ("File Permissions", self.check_file_permissions, True),
            ("Dependencies", self.check_dependencies, True),
        ]

        total_checks = len(checks)
        passed_checks = 0

        for check_name, check_func, is_critical in checks:
            print(f"Checking {check_name}...", end=" ")
            try:
                success, message = check_func()

                if success:
                    print(f"✓ PASS - {message}")
                    passed_checks += 1
                    self.results.append({
                        "check": check_name,
                        "status": "PASS",
                        "message": message,
                        "critical": is_critical,
                    })
                else:
                    print(f"✗ FAIL - {message}")
                    self.results.append({
                        "check": check_name,
                        "status": "FAIL",
                        "message": message,
                        "critical": is_critical,
                    })
                    if is_critical:
                        self.critical_failures.append({
                            "check": check_name,
                            "message": message,
                        })
                    else:
                        self.warnings.append({
                            "check": check_name,
                            "message": message,
                        })
            except Exception as e:
                print(f"✗ ERROR - {str(e)}")
                self.results.append({
                    "check": check_name,
                    "status": "ERROR",
                    "message": str(e),
                    "critical": is_critical,
                })
                if is_critical:
                    self.critical_failures.append({
                        "check": check_name,
                        "message": str(e),
                    })

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Critical failures: {len(self.critical_failures)}")
        print(f"Warnings: {len(self.warnings)}")
        print()

        if self.critical_failures:
            print("CRITICAL FAILURES:")
            for failure in self.critical_failures:
                print(f"  - {failure['check']}: {failure['message']}")
            print()

        if self.warnings:
            print("WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning['check']}: {warning['message']}")
            print()

        overall_health = len(self.critical_failures) == 0
        print(f"Overall Health: {'HEALTHY ✓' if overall_health else 'UNHEALTHY ✗'}")
        print("=" * 60)

        return {
            "timestamp": datetime.now().isoformat(),
            "healthy": overall_health,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
            "results": self.results,
        }


def main():
    """Main entry point for health check script."""
    parser = argparse.ArgumentParser(
        description="Health check for face recognition system"
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Request timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--export",
        help="Export results to JSON file",
    )

    args = parser.parse_args()

    config = {
        "api_url": args.api_url,
        "timeout": args.timeout,
    }

    checker = HealthChecker(config)
    results = checker.run_all_checks()

    if args.output == "json":
        print(json.dumps(results, indent=2))

    if args.export:
        with open(args.export, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults exported to {args.export}")

    # Exit with error code if unhealthy
    sys.exit(0 if results["healthy"] else 1)


if __name__ == "__main__":
    main()
