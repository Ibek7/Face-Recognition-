# Face Recognition API Examples

This directory contains examples and integration code for using the Face Recognition API.

## 📁 Contents

- `python_client.py` - Python client library with usage examples
- `curl_examples.sh` - Shell script with cURL command examples
- `javascript_client.html` - Browser-based JavaScript client
- `integration_examples.md` - Integration patterns and best practices

## 🚀 Quick Start

### Python Client

```bash
# Install requests library
pip install requests

# Run the examples
python examples/python_client.py
```

### cURL Examples

```bash
# Make script executable
chmod +x examples/curl_examples.sh

# Run all examples
./examples/curl_examples.sh
```

## 📖 Python Client Usage

### Basic Usage

```python
from examples.python_client import FaceRecognitionClient

# Initialize client
client = FaceRecognitionClient("http://localhost:8000")

# Check API health
health = client.health_check()
print(health)

# Create a person
person = client.create_person("John Doe", "Employee")
person_id = person['id']

# Add face images
result = client.add_person_image(person_id, "path/to/image.jpg")

# Recognize faces
recognition = client.recognize_from_file("path/to/test.jpg")
for face in recognition['faces']:
    print(f"Person: {face['person_name']}, Confidence: {face['confidence']}")
```

### Advanced Usage

```python
# Using base64 encoding
from examples.python_client import encode_image_to_base64

image_b64 = encode_image_to_base64("image.jpg")
result = client.recognize_from_base64(image_b64, threshold=0.8, top_k=3)

# Get system statistics
stats = client.get_stats()
print(f"Total persons: {stats['total_persons']}")
print(f"Average processing time: {stats['avg_processing_time']:.3f}s")

# Monitor performance
metrics = client.get_performance_metrics()
print(metrics['system_metrics'])
```

## 🌐 cURL Examples

### Create a Person

```bash
curl -X POST http://localhost:8000/persons \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "description": "Employee"
  }'
```

### Upload Face Image

```bash
curl -X POST http://localhost:8000/persons/1/images \
  -F "file=@path/to/image.jpg"
```

### Recognize Faces

```bash
curl -X POST http://localhost:8000/upload?threshold=0.7 \
  -F "file=@path/to/test_image.jpg"
```

### Get System Stats

```bash
curl -X GET http://localhost:8000/stats
```

## 🔧 Configuration

### Environment Variables

```bash
# Set custom API URL
export API_URL=http://api.example.com:8000

# Run examples
./examples/curl_examples.sh
```

### Python Configuration

```python
# Custom configuration
client = FaceRecognitionClient("http://custom-api:8000")

# With session customization
import requests

client = FaceRecognitionClient()
client.session.headers.update({
    'X-API-Key': 'your-api-key',
    'User-Agent': 'MyApp/1.0'
})
```

## 📊 Example Workflows

### 1. Person Registration Workflow

```python
# 1. Create person
person = client.create_person("Alice Johnson", "New employee")

# 2. Add multiple face images
images = ["alice1.jpg", "alice2.jpg", "alice3.jpg"]
for img in images:
    client.add_person_image(person['id'], img)

# 3. Verify person was added
persons = client.list_persons()
print(f"Total persons: {len(persons)}")
```

### 2. Face Recognition Workflow

```python
# 1. Load and recognize image
result = client.recognize_from_file("visitor.jpg", threshold=0.7)

# 2. Process results
for face in result['faces']:
    if face['person_name']:
        print(f"Recognized: {face['person_name']}")
        print(f"Confidence: {face['confidence']:.2%}")
    else:
        print("Unknown person detected")

# 3. Log recognition
print(f"Processing time: {result['processing_time']:.3f}s")
```

### 3. Batch Processing Workflow

```python
import os
from pathlib import Path

# Process all images in a directory
image_dir = Path("data/images")
results = []

for image_path in image_dir.glob("*.jpg"):
    try:
        result = client.recognize_from_file(str(image_path))
        results.append({
            'file': image_path.name,
            'faces_detected': len(result['faces']),
            'processing_time': result['processing_time']
        })
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Generate report
print(f"\nProcessed {len(results)} images")
avg_time = sum(r['processing_time'] for r in results) / len(results)
print(f"Average processing time: {avg_time:.3f}s")
```

## 🐛 Error Handling

### Python Error Handling

```python
from requests.exceptions import RequestException

try:
    result = client.recognize_from_file("image.jpg")
except RequestException as e:
    print(f"API request failed: {e}")
except FileNotFoundError:
    print("Image file not found")
except ValueError as e:
    print(f"Invalid response: {e}")
```

### cURL Error Handling

```bash
# Check HTTP status code
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  http://localhost:8000/health)

if [ "$HTTP_CODE" -eq 200 ]; then
    echo "API is healthy"
else
    echo "API returned status: $HTTP_CODE"
fi
```

## 📈 Performance Tips

1. **Reuse client instances**: Create one client and reuse it for multiple requests
2. **Use appropriate thresholds**: Lower threshold (0.6-0.7) for more matches, higher (0.8-0.9) for stricter matching
3. **Batch processing**: Process multiple images in parallel for better throughput
4. **Image optimization**: Resize large images before uploading to reduce transfer time
5. **Monitor metrics**: Regularly check `/performance/metrics` for system health

## 🔗 Additional Resources

- [API Documentation](../docs/API.md)
- [Main README](../README.md)
- [Contributing Guidelines](../CONTRIBUTING.md)

## 💬 Support

For issues or questions about these examples:
1. Check the main documentation
2. Review existing GitHub issues
3. Create a new issue with the `question` label
