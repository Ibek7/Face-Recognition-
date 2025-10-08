# Face Recognition API Documentation

## Overview

The Face Recognition API provides comprehensive REST endpoints for face detection, recognition, and management. Built with FastAPI, it offers real-time processing capabilities with comprehensive monitoring and validation.

## Features

- **Person Management**: Create, update, and manage persons in the database
- **Face Recognition**: Upload images and recognize faces with confidence scores
- **Real-time Processing**: WebSocket support for live face recognition
- **Batch Processing**: Process multiple images asynchronously
- **Performance Monitoring**: Detailed metrics and system health monitoring
- **Comprehensive Validation**: Input validation with detailed error messages

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, implement appropriate authentication mechanisms.

## API Endpoints

### Health & Status

#### GET `/`
Returns API information and available endpoints.

**Response:**
```json
{
  "message": "Face Recognition API",
  "version": "1.0.0",
  "endpoints": {
    "persons": "/persons",
    "recognize": "/recognize",
    "upload": "/upload",
    "stats": "/stats",
    "health": "/health"
  }
}
```

#### GET `/health`
Health check endpoint with system metrics.

**Response:**
```json
{
  "status": "healthy",
  "uptime": "2:30:45",
  "database_connected": true,
  "system_load": {
    "cpu_percent": 15.2,
    "memory_percent": 45.8
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Person Management

#### GET `/persons`
List all persons in the database.

**Response:**
```json
[
  {
    "id": 1,
    "name": "John Doe",
    "description": "Team member",
    "created_at": "2024-01-15T09:00:00Z",
    "updated_at": "2024-01-15T09:00:00Z",
    "is_active": true,
    "embeddings_count": 5
  }
]
```

#### POST `/persons`
Create a new person.

**Request Body:**
```json
{
  "name": "Jane Smith",
  "description": "Project manager"
}
```

**Response:**
```json
{
  "id": 2,
  "name": "Jane Smith",
  "description": "Project manager",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "is_active": true,
  "embeddings_count": 0
}
```

#### PUT `/persons/{person_id}`
Update an existing person.

**Path Parameters:**
- `person_id`: Integer ID of the person

**Request Body:**
```json
{
  "name": "Jane Doe",
  "description": "Senior project manager",
  "is_active": true
}
```

#### DELETE `/persons/{person_id}`
Delete a person and all associated face embeddings.

**Path Parameters:**
- `person_id`: Integer ID of the person

### Face Recognition

#### POST `/recognize`
Recognize faces in a base64 encoded image.

**Request Body:**
```json
{
  "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "threshold": 0.7,
  "top_k": 5,
  "include_unknown": true
}
```

**Response:**
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
        "x": 120,
        "y": 80,
        "width": 150,
        "height": 150
      },
      "is_match": true
    }
  ],
  "processing_time": 0.234,
  "total_faces": 1,
  "matched_faces": 1
}
```

#### POST `/upload`
Upload an image file for face recognition.

**Request:**
- Form data with `file` field containing image
- Optional query parameter `threshold` (default: 0.7)

**Response:** Same as `/recognize` endpoint

### Face Embedding Management

#### POST `/persons/{person_id}/images`
Add a face image for a specific person.

**Path Parameters:**
- `person_id`: Integer ID of the person

**Request:**
- Form data with `file` field containing image

**Response:**
```json
{
  "success": true,
  "filename": "profile.jpg",
  "file_size": 256000,
  "content_type": "image/jpeg",
  "embeddings_added": 1,
  "processing_time": 0.156,
  "message": "Added 1 face embeddings for John Doe"
}
```

#### GET `/persons/{person_id}/embeddings`
List all face embeddings for a person.

**Response:**
```json
[
  {
    "id": 1,
    "person_id": 1,
    "quality_score": 0.92,
    "encoder_type": "simple",
    "source_type": "upload",
    "source_image_path": "profile.jpg",
    "created_at": "2024-01-15T09:15:00Z",
    "dimensions": 128
  }
]
```

### Batch Processing

#### POST `/batch/jobs`
Create a new batch processing job.

**Request Body:**
```json
{
  "name": "Security footage analysis",
  "description": "Process security camera images",
  "image_paths": [
    "/path/to/image1.jpg",
    "/path/to/image2.jpg"
  ],
  "recognition_threshold": 0.8
}
```

#### GET `/batch/jobs`
List all batch processing jobs.

#### GET `/batch/jobs/{job_id}`
Get details of a specific batch job.

#### GET `/batch/jobs/{job_id}/results`
Get results from a completed batch job.

### Statistics & Analytics

#### GET `/stats`
Get system-wide statistics.

**Response:**
```json
{
  "total_persons": 25,
  "total_embeddings": 150,
  "total_recognitions": 1250,
  "total_batch_jobs": 5,
  "avg_processing_time": 0.234,
  "avg_recognition_confidence": 0.78,
  "uptime": "5 days, 12:30:45",
  "version": "1.0.0"
}
```

#### GET `/stats/persons`
Get per-person statistics.

**Response:**
```json
[
  {
    "person_id": 1,
    "person_name": "John Doe",
    "embeddings_count": 5,
    "recognitions_count": 45,
    "avg_confidence": 0.85,
    "last_recognition": "2024-01-15T10:15:00Z"
  }
]
```

#### GET `/performance/metrics`
Get detailed performance metrics.

**Response:**
```json
{
  "recognition_metrics": {
    "recent_count": 150,
    "avg_time": 0.234,
    "min_time": 0.120,
    "max_time": 0.456
  },
  "system_metrics": {
    "cpu_percent": 15.2,
    "memory_percent": 45.8,
    "disk_usage": 67.3
  },
  "recent_activity": [
    {
      "timestamp": "2024-01-15T10:25:00Z",
      "operation": "face_recognition",
      "duration": 0.234,
      "success": true
    }
  ]
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information.

### Error Response Format

```json
{
  "success": false,
  "error_code": "PERSON_NOT_FOUND",
  "message": "Person with ID 123 not found",
  "details": {
    "person_id": 123,
    "requested_at": "2024-01-15T10:30:00Z"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

- `VALIDATION_ERROR`: Input validation failed
- `PERSON_NOT_FOUND`: Requested person does not exist
- `PERSON_ALREADY_EXISTS`: Person with name already exists
- `INVALID_IMAGE`: Image format not supported or corrupted
- `PROCESSING_ERROR`: Error during face processing
- `DATABASE_ERROR`: Database operation failed
- `RATE_LIMIT_EXCEEDED`: Too many requests

## Rate Limiting

The API implements rate limiting to prevent abuse:
- Default: 100 requests per minute per IP
- Batch operations: 10 requests per minute per IP
- File uploads: 20 requests per minute per IP

## File Upload Constraints

- Maximum file size: 10MB
- Supported formats: JPEG, PNG, BMP
- Maximum resolution: 4096x4096 pixels
- Minimum resolution: 100x100 pixels

## WebSocket Endpoints (Future)

### `/ws/recognize`
Real-time face recognition via WebSocket connection.

**Message Format:**
```json
{
  "type": "recognize",
  "image_base64": "...",
  "threshold": 0.7
}
```

## SDK Examples

### Python SDK Usage

```python
import requests
import base64

# Initialize client
api_url = "http://localhost:8000"

# Create person
person_data = {
    "name": "John Doe",
    "description": "Team member"
}
response = requests.post(f"{api_url}/persons", json=person_data)
person = response.json()

# Upload face image
with open("face.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        f"{api_url}/persons/{person['id']}/images",
        files=files
    )

# Recognize faces
with open("test_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

recognition_data = {
    "image_base64": f"data:image/jpeg;base64,{image_base64}",
    "threshold": 0.7
}
response = requests.post(f"{api_url}/recognize", json=recognition_data)
results = response.json()
```

### cURL Examples

```bash
# Create person
curl -X POST "http://localhost:8000/persons" \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "description": "Team member"}'

# Upload face image
curl -X POST "http://localhost:8000/persons/1/images" \
  -F "file=@face.jpg"

# Get system stats
curl "http://localhost:8000/stats"

# Health check
curl "http://localhost:8000/health"
```

## Configuration

### Environment Variables

- `DATABASE_URL`: Database connection string
- `DEBUG`: Enable debug mode (default: false)
- `MAX_FILE_SIZE`: Maximum upload size in bytes
- `RATE_LIMIT_PER_MINUTE`: API rate limit
- `CORS_ORIGINS`: Allowed CORS origins (comma-separated)

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t face-recognition-api .
docker run -p 8000:8000 face-recognition-api
```

## Development

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn src.api_server:app --reload --host 0.0.0.0 --port 8000

# API documentation available at:
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

### Testing

```bash
# Run API tests
pytest tests/test_api.py -v

# Run with coverage
pytest tests/test_api.py --cov=src --cov-report=html
```

## Support

For issues and questions:
- Check the interactive API documentation at `/docs`
- Review error messages and status codes
- Monitor system health at `/health`
- Check performance metrics at `/performance/metrics`