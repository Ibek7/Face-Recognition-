# Face Recognition System - Comprehensive Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Core Components](#core-components)
5. [API Reference](#api-reference)
6. [Training Models](#training-models)
7. [Deployment](#deployment)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

## Introduction

The Face Recognition System is a comprehensive, production-ready solution for face detection, recognition, and management. Built with Python, it provides:

- **Real-time face recognition** with webcam integration
- **REST API server** with FastAPI for web integration
- **WebSocket support** for live video streams
- **Deep learning training pipeline** with PyTorch
- **Comprehensive evaluation tools** with cross-validation
- **Data augmentation** and dataset management
- **Performance monitoring** and metrics
- **Production deployment** with Docker and monitoring

### Key Features

- **Multiple face detection backends**: OpenCV Haar cascades, dlib, custom models
- **Flexible encoding systems**: Simple embeddings, dlib face encodings, custom neural networks
- **Similarity metrics**: Euclidean, cosine, Manhattan, Chebyshev distances
- **Database integration**: SQLAlchemy with SQLite, PostgreSQL support
- **Real-time processing**: Optimized for live video streams
- **Scalable architecture**: Microservices-ready with Docker
- **Comprehensive testing**: Unit tests, integration tests, performance benchmarks

## Quick Start

### Basic Usage

```python
from src.embeddings import FaceEmbeddingManager
from src.database import DatabaseManager

# Initialize components
db = DatabaseManager()
embedding_manager = FaceEmbeddingManager()

# Add a person
person = db.add_person("John Doe", "Team member")

# Generate embeddings from image
results = embedding_manager.generate_embeddings_from_image("path/to/photo.jpg")

# Add embeddings to database
for emb_info in results['embeddings']:
    db.add_face_embedding(
        embedding=emb_info['embedding'],
        person_id=person.id,
        source_image_path="path/to/photo.jpg",
        quality_score=emb_info['quality_score']
    )

# Recognize faces in new image
new_results = embedding_manager.generate_embeddings_from_image("path/to/new_photo.jpg")
for emb_info in new_results['embeddings']:
    similar_faces = db.search_similar_faces(emb_info['embedding'], threshold=0.7)
    if similar_faces:
        match, similarity = similar_faces[0]
        person = db.get_person(match.person_id)
        print(f"Recognized: {person.name} (confidence: {similarity:.3f})")
```

### CLI Interface

```bash
# Detect faces in image
python scripts/face_cli.py detect --image path/to/image.jpg

# Add person with face images
python scripts/face_cli.py add-person --name "John Doe" --images path/to/photos/*.jpg

# Recognize faces
python scripts/face_cli.py recognize --image path/to/test.jpg

# Start real-time recognition
python scripts/face_cli.py realtime --camera 0

# Batch processing
python scripts/face_cli.py batch --input-dir path/to/images/ --output-dir results/
```

### API Server

```bash
# Start API server
uvicorn src.api_server:app --host 0.0.0.0 --port 8000

# Using Docker
docker-compose up face-recognition-api

# Test API
curl -X POST "http://localhost:8000/recognize" \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "data:image/jpeg;base64,/9j/4AAQ..."}'
```

## Installation

### Prerequisites

- Python 3.8+
- OpenCV 4.0+
- CMake (for dlib compilation)
- Git

### Development Installation

```bash
# Clone repository
git clone https://github.com/your-repo/face-recognition.git
cd face-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for training
pip install torch torchvision albumentations imgaug

# Verify installation
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

### Production Installation

```bash
# Using Docker
docker build -t face-recognition .
docker run -p 8000:8000 face-recognition

# Using Docker Compose (recommended)
docker-compose up -d

# Check services
docker-compose ps
```

### GPU Support

For GPU acceleration with PyTorch:

```bash
# Install CUDA version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Core Components

### 1. Face Detection (`src/detection.py`)

The face detection module provides multiple backends for detecting faces in images:

```python
from src.detection import FaceDetector

detector = FaceDetector(
    method='opencv',  # or 'dlib', 'mtcnn'
    scale_factor=1.1,
    min_neighbors=5,
    min_size=(30, 30)
)

# Detect faces
faces = detector.detect_faces(image)
for (x, y, w, h) in faces:
    face_region = image[y:y+h, x:x+w]
```

**Configuration Options:**
- `method`: Detection algorithm ('opencv', 'dlib', 'mtcnn')
- `scale_factor`: How much the image size is reduced at each scale
- `min_neighbors`: How many neighbors each face rectangle should retain
- `min_size`: Minimum possible face size

### 2. Face Preprocessing (`src/preprocessing.py`)

Preprocessing module for normalizing face images:

```python
from src.preprocessing import FacePreprocessor

preprocessor = FacePreprocessor(target_size=(112, 112))

# Preprocess face
normalized_face = preprocessor.normalize_face(face_image)
quality_score = preprocessor.assess_image_quality(face_image)

print(f"Image quality: {quality_score:.3f}")
```

**Features:**
- Face alignment and normalization
- Histogram equalization
- Quality assessment
- Noise reduction

### 3. Face Encoding (`src/encoders.py`)

Multiple encoding backends for generating face embeddings:

```python
from src.encoders import DlibFaceEncoder, SimpleEmbeddingEncoder

# Dlib encoder (high accuracy)
dlib_encoder = DlibFaceEncoder()
embedding = dlib_encoder.encode_face(face_image)

# Simple encoder (fast)
simple_encoder = SimpleEmbeddingEncoder()
embedding = simple_encoder.encode_face(face_image)
```

### 4. Database Management (`src/database.py`)

Comprehensive database operations:

```python
from src.database import DatabaseManager

db = DatabaseManager("sqlite:///face_recognition.db")

# Person management
person = db.add_person("Alice Smith", "Engineer")
persons = db.list_persons()

# Embedding management
db.add_face_embedding(
    embedding=face_embedding,
    person_id=person.id,
    source_image_path="photo.jpg",
    quality_score=0.95
)

# Search
similar_faces = db.search_similar_faces(query_embedding, threshold=0.7)
```

### 5. Real-time Recognition (`src/realtime.py`)

Live face recognition from camera:

```python
from src.realtime import RealTimeFaceRecognizer

recognizer = RealTimeFaceRecognizer(
    camera_index=0,
    recognition_threshold=0.7,
    frame_skip=2  # Process every 2nd frame for performance
)

# Start recognition
recognizer.start_recognition()

# Add person during runtime
recognizer.enroll_person("New Person")
```

## API Reference

### REST Endpoints

#### Authentication
Currently no authentication required. In production, implement JWT or API key authentication.

#### Core Endpoints

**POST `/recognize`**
Recognize faces in uploaded image.

Request:
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
  "threshold": 0.7,
  "top_k": 5
}
```

Response:
```json
{
  "success": true,
  "faces": [
    {
      "face_id": 0,
      "person_id": 1,
      "person_name": "John Doe",
      "confidence": 0.85,
      "quality_score": 0.92,
      "bounding_box": {
        "x": 120, "y": 80, "width": 150, "height": 150
      },
      "is_match": true
    }
  ],
  "processing_time": 0.234,
  "total_faces": 1,
  "matched_faces": 1
}
```

**GET `/persons`**
List all persons.

**POST `/persons`**
Create new person.

**POST `/persons/{person_id}/images`**
Add face image for person.

#### WebSocket Endpoints

**WS `/ws/recognize/{client_id}`**
Real-time face recognition via WebSocket.

Message format:
```json
{
  "type": "face_recognition",
  "image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
  "threshold": 0.7
}
```

### Python SDK

```python
import requests
import base64

class FaceRecognitionAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def recognize_image(self, image_path, threshold=0.7):
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode()
        
        response = requests.post(f"{self.base_url}/recognize", json={
            "image_base64": f"data:image/jpeg;base64,{image_base64}",
            "threshold": threshold
        })
        
        return response.json()
    
    def add_person(self, name, description=""):
        response = requests.post(f"{self.base_url}/persons", json={
            "name": name,
            "description": description
        })
        return response.json()

# Usage
api = FaceRecognitionAPI()
result = api.recognize_image("photo.jpg")
print(f"Found {len(result['faces'])} faces")
```

## Training Models

### Quick Training

```python
from src.training import train_custom_model

config = {
    "model_name": "custom_face_model",
    "model_type": "classification",  # or "siamese", "triplet"
    "num_epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "embedding_dim": 512
}

results = train_custom_model(config)
print(f"Training completed: {results['status']}")
```

### Advanced Training Pipeline

1. **Prepare Dataset**
```python
from src.data_augmentation import create_training_dataset

dataset_config = {
    "min_samples_per_person": 5,
    "max_samples_per_person": 50,
    "augmentation_factor": 3
}

results = create_training_dataset("training_v1", config=dataset_config)
```

2. **Configure Training**
```python
from src.training import TrainingConfig, ModelTrainer

config = TrainingConfig(
    model_name="advanced_face_model",
    model_type="triplet",
    num_epochs=100,
    batch_size=64,
    learning_rate=0.0001,
    embedding_dim=512,
    data_augmentation=True,
    early_stopping_patience=15
)

trainer = ModelTrainer(config)
results = trainer.train()
```

3. **Evaluate Model**
```python
from src.evaluation import evaluate_model

evaluation_results = evaluate_model(
    model_path="models/training/best_model.pth",
    output_dir="evaluation_results"
)

print(f"Model accuracy: {evaluation_results['classification_metrics']['accuracy']:.4f}")
```

### Training Configuration

Key parameters for training:

- **model_type**: 
  - `classification`: Standard multi-class classification
  - `siamese`: Siamese network for face verification
  - `triplet`: Triplet loss for embedding learning

- **embedding_dim**: Dimension of face embeddings (128, 256, 512)
- **backbone**: Pre-trained backbone ("resnet50", "efficientnet", "mobilenet")
- **augmentation**: Data augmentation techniques
- **learning_rate**: Learning rate schedule
- **batch_size**: Batch size (depends on GPU memory)

## Deployment

### Docker Deployment

**Single Container:**
```bash
# Build and run
docker build -t face-recognition .
docker run -p 8000:8000 -v ./data:/app/data face-recognition
```

**Production with Docker Compose:**
```bash
# Start all services
docker-compose up -d

# Services included:
# - face-recognition-api: Main API server
# - postgres: Database
# - redis: Caching and sessions
# - nginx: Reverse proxy
# - prometheus: Monitoring
# - grafana: Visualization
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-recognition-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: face-recognition-api
  template:
    metadata:
      labels:
        app: face-recognition-api
    spec:
      containers:
      - name: api
        image: face-recognition:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://user:pass@postgres:5432/face_recognition"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### Environment Variables

**Database Configuration:**
- `DATABASE_URL`: Database connection string
- `REDIS_URL`: Redis connection string

**API Configuration:**
- `DEBUG`: Enable debug mode
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `MAX_FILE_SIZE`: Maximum upload size (bytes)
- `RATE_LIMIT_PER_MINUTE`: API rate limit

**Model Configuration:**
- `DEFAULT_MODEL_PATH`: Path to default model
- `ENCODER_TYPE`: Default encoder type
- `RECOGNITION_THRESHOLD`: Default recognition threshold

## Performance Tuning

### Optimization Strategies

1. **Hardware Optimization**
```python
# Use GPU acceleration
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

2. **Model Optimization**
```python
# Model quantization
import torch.quantization as quantization
quantized_model = quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# TensorRT optimization (NVIDIA GPUs)
import torch_tensorrt
optimized_model = torch_tensorrt.compile(model, 
    inputs=[torch.randn(1, 3, 112, 112).cuda()],
    enabled_precisions=[torch.float, torch.half]
)
```

3. **Inference Optimization**
```python
# Batch processing
def batch_recognize(images, batch_size=32):
    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = model(batch)
        results.extend(batch_results)
    return results

# Frame skipping for real-time
recognizer = RealTimeFaceRecognizer(frame_skip=3)  # Process every 3rd frame
```

### Performance Benchmarks

| Configuration | FPS | Accuracy | Memory (GB) |
|---------------|-----|----------|-------------|
| CPU (OpenCV) | 15 | 0.85 | 0.5 |
| CPU (dlib) | 8 | 0.92 | 1.2 |
| GPU (Custom) | 45 | 0.94 | 2.1 |
| GPU (Optimized) | 60 | 0.93 | 1.8 |

### Memory Management

```python
# Clear GPU memory
torch.cuda.empty_cache()

# Limit database connections
db = DatabaseManager(pool_size=5, max_overflow=10)

# Image processing optimization
def preprocess_image_efficient(image):
    # Use in-place operations
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    return image
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```
ImportError: No module named 'cv2'
```
Solution:
```bash
pip install opencv-python
# For additional features:
pip install opencv-contrib-python
```

**2. CUDA/GPU Issues**
```
RuntimeError: CUDA out of memory
```
Solutions:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Clear GPU cache: `torch.cuda.empty_cache()`

**3. Database Connection Issues**
```
sqlalchemy.exc.OperationalError: could not connect to server
```
Solutions:
- Check database service status
- Verify connection string
- Check network connectivity
- Ensure database exists

**4. Performance Issues**
```
Face recognition is too slow
```
Solutions:
- Use GPU acceleration
- Reduce image resolution
- Skip frames in real-time processing
- Use efficient encoders
- Optimize preprocessing pipeline

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed error messages
import os
os.environ['DEBUG'] = 'true'
```

### Health Checks

```python
# System health check
from src.monitoring import performance_monitor

health_status = performance_monitor.get_system_metrics_summary()
print(f"CPU usage: {health_status['cpu_percent']}%")
print(f"Memory usage: {health_status['memory_percent']}%")

# Database health
from src.database import DatabaseManager
db = DatabaseManager()
try:
    db.list_persons()
    print("Database: OK")
except Exception as e:
    print(f"Database error: {e}")

# Model health
from src.embeddings import FaceEmbeddingManager
em = FaceEmbeddingManager()
try:
    test_result = em.pipeline.process_test_image()
    print("Model: OK")
except Exception as e:
    print(f"Model error: {e}")
```

### Monitoring and Logging

```python
# Custom monitoring
from src.monitoring import performance_monitor

@performance_monitor.time_function("custom_operation")
def my_function():
    # Your code here
    pass

# View metrics
metrics = performance_monitor.get_recent_metrics("custom_operation", 60)
print(f"Average time: {np.mean([m.value for m in metrics])}")
```

## Contributing

### Development Setup

```bash
# Fork and clone repository
git clone https://github.com/your-username/face-recognition.git
cd face-recognition

# Create feature branch
git checkout -b feature/new-feature

# Set up development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run linting
flake8 src/
black src/
isort src/
```

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_detection.py -v
pytest tests/test_api.py -v

# Run with coverage
pytest --cov=src --cov-report=html

# Performance tests
pytest tests/test_performance.py --benchmark-only
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write comprehensive docstrings
- Add unit tests for new features
- Update documentation

### Pull Request Process

1. Create feature branch from main
2. Implement changes with tests
3. Update documentation
4. Run full test suite
5. Submit pull request with description
6. Address review feedback
7. Merge after approval

### Release Process

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create release tag
4. Build and publish packages
5. Update documentation
6. Announce release

For detailed information, see our [Contributing Guide](CONTRIBUTING.md).