# Face Recognition System

A comprehensive face recognition application with real-time detection, embedding generation, and REST API capabilities. Built with OpenCV, FastAPI, and modern machine learning techniques.

## ✨ Features

- **Face Detection**: High-accuracy face detection using OpenCV Haar Cascades
- **Face Recognition**: Advanced embedding-based face recognition with similarity matching
- **REST API**: Full-featured FastAPI server with endpoints for person management and recognition
- **Real-time Processing**: Live face recognition from video streams
- **Batch Processing**: Process multiple images efficiently
- **Database Management**: SQLite/PostgreSQL support for storing persons and embeddings
- **Performance Monitoring**: Built-in metrics and monitoring capabilities
- **Multi-Model Support**: Flexible encoder architecture supporting various embedding models
- **Web Dashboard**: Frontend interface for visualization and management

## 🏗️ Project Structure

```
Face Recognition/
├── src/                    # Core source code
│   ├── detection.py        # Face detection utilities
│   ├── embeddings.py       # Embedding generation and management
│   ├── api_server.py       # FastAPI REST API server
│   ├── database.py         # Database management
│   ├── realtime.py         # Real-time video processing
│   ├── monitoring.py       # Performance monitoring
│   └── ...                 # Additional modules
├── scripts/                # Utility scripts
│   ├── face_cli.py         # Command-line interface
│   ├── batch_detect.py     # Batch processing
│   └── benchmark.py        # Performance benchmarking
├── frontend/               # Web dashboard
│   ├── index.html
│   ├── dashboard.js
│   └── styles.css
├── data/                   # Data storage
│   ├── images/             # Training/test images
│   └── videos/             # Video files
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile             # Docker build file
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- (Optional) Docker and Docker Compose

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ibek7/Face-Recognition-.git
   cd Face-Recognition-
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the database**:
   ```bash
   python -c "from src.database import DatabaseManager; db = DatabaseManager(); print('Database initialized')"
   ```

### Docker Installation

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. **Or build and run manually**:
   ```bash
   docker build -t face-recognition:latest .
   docker run -p 8000:8000 -v "$(pwd)/data:/app/data" face-recognition:latest
   ```

## 📖 Usage

### Starting the API Server

```bash
python src/api_server.py
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

### Command-Line Interface

**Add a person**:
```bash
python scripts/face_cli.py add-person "John Doe" --images data/images/john/*.jpg
```

**Recognize faces in an image**:
```bash
python scripts/face_cli.py recognize data/images/test.jpg
```

**Batch processing**:
```bash
python scripts/batch_detect.py --input-dir data/images --output-dir results
```

### Using the REST API

**Register a new person**:
```bash
curl -X POST http://localhost:8000/persons \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "description": "Employee"}'
```

**Upload face images**:
```bash
curl -X POST http://localhost:8000/persons/1/images \
  -F "file=@path/to/image.jpg"
```

**Recognize faces**:
```bash
curl -X POST http://localhost:8000/recognize \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "BASE64_ENCODED_IMAGE", "threshold": 0.7}'
```

### Real-time Recognition

```python
from src.realtime import RealTimeFaceRecognizer

recognizer = RealTimeFaceRecognizer()
recognizer.start_recognition(source=0)  # 0 for webcam
```

## 🔧 Configuration

Create a `.env` file in the project root:

```env
DATABASE_URL=sqlite:///face_recognition.db
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
RECOGNITION_THRESHOLD=0.7
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## 📊 Performance Monitoring

Access performance metrics:
```bash
curl http://localhost:8000/performance/metrics
```

View system stats:
```bash
curl http://localhost:8000/stats
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Additional Documentation

- [API Documentation](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Security Guidelines](docs/SECURITY.md)

## 📧 Contact

For questions or support, please open an issue on GitHub.
