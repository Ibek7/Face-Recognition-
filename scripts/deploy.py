#!/usr/bin/env python3
"""
Deployment Automation Script

Automates the deployment process for the face recognition system.
Handles pre-deployment checks, database migrations, service deployment,
health verification, and rollback capabilities.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class DeploymentManager:
    """Manages automated deployment process."""

    def __init__(self, environment: str, config_path: Optional[str] = None):
        """Initialize deployment manager."""
        self.environment = environment
        self.config = self._load_config(config_path)
        self.deployment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"logs/deployment_{self.deployment_id}.log"
        self.rollback_point = None

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load deployment configuration."""
        default_config = {
            "development": {
                "compose_file": "docker-compose.dev.yml",
                "health_check_retries": 5,
                "health_check_delay": 10,
            },
            "staging": {
                "compose_file": "docker-compose.yml",
                "health_check_retries": 10,
                "health_check_delay": 15,
            },
            "production": {
                "compose_file": "docker-compose.yml",
                "health_check_retries": 15,
                "health_check_delay": 20,
                "require_backup": True,
                "require_smoke_tests": True,
            },
        }

        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                custom_config = json.load(f)
                return custom_config.get(self.environment, default_config.get(self.environment, {}))

        return default_config.get(self.environment, {})

    def log(self, message: str, level: str = "INFO"):
        """Log deployment message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)

        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)

        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")

    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command and log output."""
        self.log(f"Running: {command}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=check,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                self.log(f"Output: {result.stdout.strip()}")
            if result.stderr:
                self.log(f"Error: {result.stderr.strip()}", "WARN")
            return result
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {e}", "ERROR")
            if e.stdout:
                self.log(f"stdout: {e.stdout}", "ERROR")
            if e.stderr:
                self.log(f"stderr: {e.stderr}", "ERROR")
            raise

    def pre_deployment_checks(self) -> bool:
        """Run pre-deployment validation checks."""
        self.log("=" * 60)
        self.log("RUNNING PRE-DEPLOYMENT CHECKS")
        self.log("=" * 60)

        checks_passed = True

        # Check 1: Verify environment file exists
        env_file = f".env.{self.environment}" if self.environment != "development" else ".env"
        if not os.path.exists(env_file):
            self.log(f"Environment file {env_file} not found", "ERROR")
            checks_passed = False
        else:
            self.log(f"✓ Environment file {env_file} exists")

        # Check 2: Verify Docker is running
        try:
            self.run_command("docker info > /dev/null 2>&1")
            self.log("✓ Docker is running")
        except subprocess.CalledProcessError:
            self.log("Docker is not running", "ERROR")
            checks_passed = False

        # Check 3: Verify required files exist
        required_files = ["Dockerfile", "requirements.txt"]
        compose_file = self.config.get("compose_file", "docker-compose.yml")
        if os.path.exists(compose_file):
            required_files.append(compose_file)

        for file in required_files:
            if not os.path.exists(file):
                self.log(f"Required file {file} not found", "ERROR")
                checks_passed = False
            else:
                self.log(f"✓ Required file {file} exists")

        # Check 4: Verify disk space
        try:
            result = self.run_command("df -h / | tail -1 | awk '{print $5}' | sed 's/%//'")
            disk_usage = int(result.stdout.strip())
            if disk_usage > 90:
                self.log(f"Disk usage too high: {disk_usage}%", "ERROR")
                checks_passed = False
            else:
                self.log(f"✓ Disk usage OK: {disk_usage}%")
        except Exception as e:
            self.log(f"Could not check disk usage: {e}", "WARN")

        return checks_passed

    def backup_database(self) -> bool:
        """Create database backup before deployment."""
        if not self.config.get("require_backup", False):
            self.log("Database backup not required for this environment")
            return True

        self.log("Creating database backup...")

        backup_script = "scripts/backup_database.sh"
        if not os.path.exists(backup_script):
            self.log("Backup script not found, skipping backup", "WARN")
            return True

        try:
            self.run_command(f"bash {backup_script}")
            self.log("✓ Database backup completed")
            return True
        except subprocess.CalledProcessError:
            self.log("Database backup failed", "ERROR")
            return False

    def stop_services(self):
        """Stop running services."""
        self.log("Stopping services...")

        compose_file = self.config.get("compose_file")
        if compose_file and os.path.exists(compose_file):
            try:
                self.run_command(f"docker-compose -f {compose_file} down")
                self.log("✓ Services stopped")
            except subprocess.CalledProcessError:
                self.log("Failed to stop services", "WARN")
        else:
            self.log("No compose file found, skipping service stop")

    def build_images(self):
        """Build Docker images."""
        self.log("Building Docker images...")

        compose_file = self.config.get("compose_file")
        if compose_file and os.path.exists(compose_file):
            self.run_command(f"docker-compose -f {compose_file} build --no-cache")
            self.log("✓ Images built successfully")
        else:
            # Fall back to building Dockerfile directly
            self.run_command("docker build -t face-recognition:latest .")
            self.log("✓ Image built successfully")

    def run_migrations(self):
        """Run database migrations."""
        self.log("Running database migrations...")

        # Check if migration script exists
        migration_script = "scripts/migrate_database.py"
        if os.path.exists(migration_script):
            try:
                self.run_command(f"python {migration_script}")
                self.log("✓ Migrations completed")
            except subprocess.CalledProcessError:
                self.log("Migrations failed", "ERROR")
                raise
        else:
            self.log("No migration script found, skipping migrations")

    def start_services(self):
        """Start services."""
        self.log("Starting services...")

        compose_file = self.config.get("compose_file")
        if compose_file and os.path.exists(compose_file):
            self.run_command(f"docker-compose -f {compose_file} up -d")
            self.log("✓ Services started")
        else:
            # Fall back to running container directly
            self.run_command("docker run -d -p 8000:8000 --name face-recognition face-recognition:latest")
            self.log("✓ Container started")

    def wait_for_services(self):
        """Wait for services to be ready."""
        self.log("Waiting for services to be ready...")

        retries = self.config.get("health_check_retries", 10)
        delay = self.config.get("health_check_delay", 10)

        for attempt in range(1, retries + 1):
            self.log(f"Health check attempt {attempt}/{retries}")

            try:
                result = self.run_command(
                    "python scripts/system_health_check.py --output json",
                    check=False
                )

                if result.returncode == 0:
                    self.log("✓ Services are healthy")
                    return True

                self.log(f"Services not ready yet, waiting {delay}s...")
                time.sleep(delay)

            except Exception as e:
                self.log(f"Health check error: {e}", "WARN")
                time.sleep(delay)

        self.log("Services failed to become healthy", "ERROR")
        return False

    def run_smoke_tests(self) -> bool:
        """Run smoke tests to verify deployment."""
        if not self.config.get("require_smoke_tests", False):
            self.log("Smoke tests not required for this environment")
            return True

        self.log("Running smoke tests...")

        try:
            self.run_command("pytest tests/test_smoke.py -v")
            self.log("✓ Smoke tests passed")
            return True
        except subprocess.CalledProcessError:
            self.log("Smoke tests failed", "ERROR")
            return False

    def tag_deployment(self):
        """Tag deployment in version control."""
        self.log("Tagging deployment...")

        tag = f"deploy-{self.environment}-{self.deployment_id}"

        try:
            self.run_command(f"git tag -a {tag} -m 'Deployment to {self.environment} at {self.deployment_id}'")
            self.log(f"✓ Created tag {tag}")
        except subprocess.CalledProcessError:
            self.log("Failed to create git tag", "WARN")

    def rollback(self):
        """Rollback to previous version."""
        self.log("=" * 60)
        self.log("INITIATING ROLLBACK")
        self.log("=" * 60)

        if self.rollback_point:
            self.log(f"Rolling back to {self.rollback_point}")
            try:
                self.run_command(f"git checkout {self.rollback_point}")
                self.stop_services()
                self.build_images()
                self.start_services()
                self.log("✓ Rollback completed")
            except Exception as e:
                self.log(f"Rollback failed: {e}", "ERROR")
        else:
            self.log("No rollback point set", "WARN")

    def deploy(self) -> bool:
        """Execute full deployment process."""
        self.log("=" * 60)
        self.log(f"STARTING DEPLOYMENT TO {self.environment.upper()}")
        self.log(f"Deployment ID: {self.deployment_id}")
        self.log("=" * 60)

        try:
            # Save rollback point
            result = self.run_command("git rev-parse HEAD")
            self.rollback_point = result.stdout.strip()
            self.log(f"Rollback point: {self.rollback_point}")

            # Pre-deployment checks
            if not self.pre_deployment_checks():
                self.log("Pre-deployment checks failed", "ERROR")
                return False

            # Backup database
            if not self.backup_database():
                self.log("Database backup failed", "ERROR")
                return False

            # Stop services
            self.stop_services()

            # Build images
            self.build_images()

            # Run migrations
            self.run_migrations()

            # Start services
            self.start_services()

            # Wait for services to be ready
            if not self.wait_for_services():
                self.log("Services failed health check", "ERROR")
                self.rollback()
                return False

            # Run smoke tests
            if not self.run_smoke_tests():
                self.log("Smoke tests failed", "ERROR")
                self.rollback()
                return False

            # Tag deployment
            self.tag_deployment()

            self.log("=" * 60)
            self.log("DEPLOYMENT COMPLETED SUCCESSFULLY")
            self.log("=" * 60)
            return True

        except Exception as e:
            self.log(f"Deployment failed: {e}", "ERROR")
            self.rollback()
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automated deployment for face recognition system"
    )
    parser.add_argument(
        "environment",
        choices=["development", "staging", "production"],
        help="Target environment",
    )
    parser.add_argument(
        "--config",
        help="Path to deployment configuration file",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip pre-deployment checks (not recommended)",
    )

    args = parser.parse_args()

    manager = DeploymentManager(args.environment, args.config)

    if args.skip_checks:
        manager.log("WARNING: Pre-deployment checks disabled", "WARN")

    success = manager.deploy()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
