#!/usr/bin/env python3
"""
Quick setup script for Face Recognition System.
Automates the initial setup process.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_header(message):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {message}")
    print("=" * 60 + "\n")


def print_success(message):
    """Print a success message."""
    print(f"✓ {message}")


def print_error(message):
    """Print an error message."""
    print(f"✗ {message}", file=sys.stderr)


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_error(f"Python 3.8+ required. Found Python {version.major}.{version.minor}")
        return False
    print_success(f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def create_virtual_environment():
    """Create a virtual environment."""
    print_header("Creating Virtual Environment")
    venv_path = Path(".venv")
    
    if venv_path.exists():
        print("Virtual environment already exists. Skipping...")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True)
        print_success("Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False


def get_pip_path():
    """Get the pip executable path."""
    if os.name == 'nt':  # Windows
        return Path(".venv/Scripts/pip.exe")
    else:  # Unix-like
        return Path(".venv/bin/pip")


def install_dependencies():
    """Install Python dependencies."""
    print_header("Installing Dependencies")
    pip_path = get_pip_path()
    
    if not pip_path.exists():
        print_error("Virtual environment pip not found")
        return False
    
    try:
        # Upgrade pip
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        print_success("Pip upgraded")
        
        # Install requirements
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        print_success("Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print_header("Creating Directories")
    directories = [
        "data/images",
        "data/videos",
        "data/uploads",
        "data/output",
        "logs",
        "models/weights",
        ".cache/models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_success(f"Created {directory}")
    
    return True


def create_env_file():
    """Create .env file from template."""
    print_header("Creating Environment File")
    
    if Path(".env").exists():
        print(".env file already exists. Skipping...")
        return True
    
    if not Path(".env.example").exists():
        print_error(".env.example not found")
        return False
    
    try:
        shutil.copy(".env.example", ".env")
        print_success("Created .env file from template")
        print("\n⚠️  Please edit .env file with your configuration")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False


def initialize_database():
    """Initialize the database."""
    print_header("Initializing Database")
    python_path = Path(".venv/bin/python") if os.name != 'nt' else Path(".venv/Scripts/python.exe")
    
    if not python_path.exists():
        print_error("Virtual environment Python not found")
        return False
    
    try:
        code = "from src.database import DatabaseManager; db = DatabaseManager(); print('Database initialized successfully')"
        result = subprocess.run(
            [str(python_path), "-c", code],
            capture_output=True,
            text=True,
            check=True
        )
        print_success("Database initialized")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to initialize database: {e}")
        print(e.stderr)
        return False


def print_next_steps():
    """Print next steps for the user."""
    print_header("Setup Complete!")
    
    activation_cmd = "source .venv/bin/activate" if os.name != 'nt' else ".venv\\Scripts\\activate"
    
    print("Next steps:")
    print(f"\n1. Activate the virtual environment:")
    print(f"   {activation_cmd}")
    print(f"\n2. Edit the .env file with your configuration:")
    print(f"   nano .env")
    print(f"\n3. Start the API server:")
    print(f"   python src/api_server.py")
    print(f"   # Or use uvicorn directly:")
    print(f"   uvicorn src.api_server:app --reload")
    print(f"\n4. Visit the API documentation:")
    print(f"   http://localhost:8000/docs")
    print(f"\n5. Run tests:")
    print(f"   pytest tests/")
    print("\n")


def main():
    """Main setup function."""
    print_header("Face Recognition System - Quick Setup")
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating virtual environment", create_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directories),
        ("Creating environment file", create_env_file),
        ("Initializing database", initialize_database),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print_error(f"\nSetup failed at: {step_name}")
            sys.exit(1)
    
    print_next_steps()


if __name__ == "__main__":
    main()
