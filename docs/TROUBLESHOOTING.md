# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Issue: `pip install` fails with compilation errors

**Solution:**
```bash
# Install system dependencies first
sudo apt-get install build-essential cmake libopencv-dev

# Then install Python packages
pip install -r requirements.txt
```

#### Issue: dlib installation fails

**Solution:**
```bash
# On Ubuntu/Debian
sudo apt-get install libdlib-dev

# On macOS
brew install dlib

# Or use conda
conda install -c conda-forge dlib
```

### Runtime Issues

#### Issue: "No module named 'cv2'"

**Solution:**
```bash
pip install opencv-python
# Or for full OpenCV
pip install opencv-contrib-python
```

#### Issue: Database connection errors

**Solution:**
1. Check DATABASE_URL in `.env`
2. Ensure database file has write permissions
3. For PostgreSQL, verify the server is running:
```bash
psql -h localhost -U user -d face_recognition
```

#### Issue: "CUDA out of memory"

**Solution:**
1. Reduce batch size
2. Process images sequentially
3. Use CPU mode: `ENABLE_GPU=false`

### API Issues

#### Issue: 500 Internal Server Error

**Solution:**
1. Check logs: `tail -f logs/app.log`
2. Verify database connectivity
3. Ensure all required environment variables are set

#### Issue: Slow API response times

**Solution:**
1. Enable caching: `ENABLE_CACHING=true`
2. Increase worker count: `API_WORKERS=8`
3. Use Redis for distributed caching

#### Issue: CORS errors in browser

**Solution:**
Update `CORS_ORIGINS` in `.env`:
```bash
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Face Recognition Issues

#### Issue: Low recognition accuracy

**Solution:**
1. Lower the threshold: `RECOGNITION_THRESHOLD=0.5`
2. Use higher quality images
3. Ensure good lighting in source images
4. Add more training images per person (5+ recommended)

#### Issue: Faces not detected

**Solution:**
1. Check image quality and size
2. Adjust detection confidence: `DETECTION_CONFIDENCE=0.7`
3. Ensure faces are clearly visible (not too small/blurry)
4. Try different face detection models

#### Issue: False positives

**Solution:**
1. Increase threshold: `RECOGNITION_THRESHOLD=0.7`
2. Use better quality training images
3. Remove low-quality embeddings from database

### Docker Issues

#### Issue: Docker build fails

**Solution:**
```bash
# Clear build cache
docker system prune -a

# Build without cache
docker build --no-cache -f Dockerfile.prod -t face-recognition:prod .
```

#### Issue: Permission denied in container

**Solution:**
```bash
# Run as current user
docker run --user $(id -u):$(id -g) ...

# Or set permissions on mounted volumes
chmod -R 755 data/
```

### Testing Issues

#### Issue: Tests fail with import errors

**Solution:**
```bash
# Ensure pytest can find src/
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

#### Issue: Mock errors in tests

**Solution:**
Ensure all external dependencies are mocked:
```python
@patch('cv2.imread')
@patch('dlib.get_frontal_face_detector')
def test_detection(mock_detector, mock_imread):
    # Your test code
```

### Performance Issues

#### Issue: High memory usage

**Solution:**
1. Reduce cache size: `CACHE_MAX_SIZE=256`
2. Process images in smaller batches
3. Use generator patterns for large datasets
4. Clean up old embeddings: 
```python
db.cleanup_old_results(days=30)
```

#### Issue: Slow embedding generation

**Solution:**
1. Enable GPU: `ENABLE_GPU=true`
2. Use simpler encoder: `ENCODER_TYPE=simple`
3. Reduce image resolution before processing
4. Use batch processing for multiple images

## Debugging Tips

### Enable Verbose Logging

```bash
LOG_LEVEL=DEBUG
VERBOSE=true
```

### Profile Performance

```bash
ENABLE_PROFILING=true
python -m cProfile -o profile.stats src/api_server.py
```

View results:
```python
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
```

### Check System Resources

```bash
# CPU and memory
htop

# GPU usage
nvidia-smi

# Disk I/O
iotop
```

## Getting Help

If you're still experiencing issues:

1. Check existing GitHub issues
2. Review the documentation in `docs/`
3. Enable debug logging and check logs
4. Open a new issue with:
   - Error messages
   - System information
   - Steps to reproduce
   - Relevant configuration

## Useful Commands

```bash
# Check Python environment
python --version
pip list

# Verify database
sqlite3 face_recognition.db "SELECT COUNT(*) FROM persons;"

# Test API
curl http://localhost:8000/health

# View logs
tail -f logs/app.log

# Clean cache
rm -rf .cache/

# Reset database (WARNING: deletes all data)
rm face_recognition.db
python -c "from src.database import DatabaseManager; DatabaseManager()"
```
