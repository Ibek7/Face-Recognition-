# Model Versioning and Management Guide

This guide explains how to version, manage, and deploy machine learning models in the face recognition system.

## Table of Contents

- [Overview](#overview)
- [Model Versioning Strategy](#model-versioning-strategy)
- [Model Storage](#model-storage)
- [Model Registry](#model-registry)
- [Deployment Process](#deployment-process)
- [Rollback Procedures](#rollback-procedures)
- [Performance Monitoring](#performance-monitoring)
- [Best Practices](#best-practices)

## Overview

The face recognition system uses multiple ML models:
- **Detection Model**: YOLOv8 or MTCNN for face detection
- **Recognition Model**: FaceNet or ArcFace for face embeddings
- **Landmark Model**: 68-point facial landmark detection

Each model is versioned independently and can be updated without affecting the others.

## Model Versioning Strategy

### Semantic Versioning

We use semantic versioning (MAJOR.MINOR.PATCH) for all models:

- **MAJOR**: Incompatible API changes or architecture changes
- **MINOR**: New features, improved accuracy (backward compatible)
- **PATCH**: Bug fixes, performance improvements

Example: `facenet-v2.1.3`

### Version Naming Convention

```
{model_name}-v{major}.{minor}.{patch}[-{metadata}]
```

Examples:
- `detection-yolov8-v1.0.0`
- `recognition-facenet-v2.1.3`
- `landmark-dlib-v1.2.0-gpu`

### Model Metadata

Each model version includes metadata:

```json
{
  "name": "recognition-facenet",
  "version": "2.1.3",
  "created_at": "2025-11-14T10:30:00Z",
  "framework": "pytorch",
  "framework_version": "2.0.1",
  "architecture": "InceptionResNetV1",
  "input_shape": [160, 160, 3],
  "output_shape": [512],
  "training_dataset": "VGGFace2",
  "accuracy_metrics": {
    "lfw_accuracy": 0.9965,
    "validation_loss": 0.032
  },
  "file_size_mb": 89.5,
  "checksum": "sha256:abc123...",
  "author": "ML Team",
  "description": "Improved FaceNet model with better accuracy on diverse faces"
}
```

## Model Storage

### Directory Structure

```
models/
├── detection/
│   ├── yolov8-v1.0.0/
│   │   ├── model.pt
│   │   ├── metadata.json
│   │   └── config.yaml
│   └── yolov8-v1.1.0/
│       ├── model.pt
│       ├── metadata.json
│       └── config.yaml
├── recognition/
│   ├── facenet-v2.0.0/
│   │   ├── model.pt
│   │   ├── metadata.json
│   │   └── config.yaml
│   └── facenet-v2.1.3/
│       ├── model.pt
│       ├── metadata.json
│       └── config.yaml
└── landmarks/
    └── dlib-v1.0.0/
        ├── model.dat
        ├── metadata.json
        └── config.yaml
```

### Cloud Storage

For production, store models in cloud object storage:

```bash
# AWS S3
s3://face-recognition-models/
├── detection/
├── recognition/
└── landmarks/

# Google Cloud Storage
gs://face-recognition-models/

# Azure Blob Storage
https://facerecognition.blob.core.windows.net/models/
```

### Model Artifacts

Each model version should include:
- **Model weights**: The trained model file (.pt, .h5, .onnx)
- **Metadata**: JSON file with model information
- **Config**: YAML file with model configuration
- **Requirements**: Dependencies needed for the model
- **Checksum**: SHA256 hash for integrity verification

## Model Registry

### Registry Structure

Use a model registry to track all model versions:

```python
# Example model registry entry
{
    "model_id": "rec-facenet-v2.1.3",
    "name": "recognition-facenet",
    "version": "2.1.3",
    "status": "production",  # development, staging, production, deprecated
    "created_at": "2025-11-14T10:30:00Z",
    "deployed_at": "2025-11-14T15:00:00Z",
    "storage_path": "s3://models/recognition/facenet-v2.1.3/model.pt",
    "metrics": {
        "accuracy": 0.9965,
        "inference_time_ms": 45,
        "model_size_mb": 89.5
    },
    "tags": ["production", "high-accuracy", "gpu-optimized"],
    "parent_version": "2.1.2",
    "changelog": "Improved accuracy on diverse face types"
}
```

### Registry Operations

```python
# Register a new model
python scripts/model_registry.py register \
    --name recognition-facenet \
    --version 2.1.3 \
    --path models/recognition/facenet-v2.1.3/model.pt \
    --metadata models/recognition/facenet-v2.1.3/metadata.json

# List all models
python scripts/model_registry.py list

# Get model info
python scripts/model_registry.py info --model-id rec-facenet-v2.1.3

# Update model status
python scripts/model_registry.py update-status \
    --model-id rec-facenet-v2.1.3 \
    --status production

# Deprecate a model
python scripts/model_registry.py deprecate \
    --model-id rec-facenet-v2.0.0 \
    --reason "Superseded by v2.1.3"
```

## Deployment Process

### 1. Model Development

```bash
# Train new model
python scripts/train_model.py \
    --model facenet \
    --dataset data/training \
    --epochs 100 \
    --output models/recognition/facenet-v2.2.0

# Validate model
python scripts/validate_model.py \
    --model models/recognition/facenet-v2.2.0/model.pt \
    --test-data data/validation
```

### 2. Model Packaging

```bash
# Package model with metadata
python scripts/package_model.py \
    --model models/recognition/facenet-v2.2.0/model.pt \
    --name recognition-facenet \
    --version 2.2.0 \
    --output models/recognition/facenet-v2.2.0/
```

### 3. Model Testing

```bash
# Run benchmark tests
python scripts/benchmark_model.py \
    --model models/recognition/facenet-v2.2.0/model.pt \
    --test-suite tests/model_tests/

# Compare with current production model
python scripts/compare_models.py \
    --model-a models/recognition/facenet-v2.1.3/model.pt \
    --model-b models/recognition/facenet-v2.2.0/model.pt \
    --metrics accuracy,speed,memory
```

### 4. Staging Deployment

```bash
# Deploy to staging
python scripts/deploy_model.py \
    --model-id rec-facenet-v2.2.0 \
    --environment staging \
    --config config/staging.yaml

# Run integration tests
pytest tests/integration/ --model-version 2.2.0
```

### 5. Production Deployment

```bash
# Deploy to production (with gradual rollout)
python scripts/deploy_model.py \
    --model-id rec-facenet-v2.2.0 \
    --environment production \
    --strategy canary \
    --canary-percent 10

# Monitor metrics
python scripts/monitor_model.py \
    --model-id rec-facenet-v2.2.0 \
    --duration 24h

# Complete rollout
python scripts/deploy_model.py \
    --model-id rec-facenet-v2.2.0 \
    --environment production \
    --strategy complete
```

## Rollback Procedures

### Immediate Rollback

If issues are detected, immediately rollback to the previous version:

```bash
# Rollback to previous version
python scripts/rollback_model.py \
    --model-type recognition \
    --to-version 2.1.3 \
    --reason "High error rate detected"

# Verify rollback
curl http://api.example.com/api/model/info
```

### Automated Rollback

Configure automated rollback triggers:

```yaml
# config/rollback_triggers.yaml
rollback_triggers:
  error_rate_threshold: 0.05  # 5% error rate
  latency_threshold_ms: 500   # 500ms latency
  memory_threshold_mb: 2000   # 2GB memory usage
  accuracy_drop_threshold: 0.02  # 2% accuracy drop
  monitoring_window_minutes: 30
```

## Performance Monitoring

### Key Metrics

Monitor these metrics for each model:

1. **Accuracy Metrics**
   - Precision, Recall, F1-Score
   - False Positive Rate (FPR)
   - True Positive Rate (TPR)

2. **Performance Metrics**
   - Inference time (p50, p95, p99)
   - Throughput (requests/second)
   - Memory usage
   - GPU utilization

3. **Business Metrics**
   - API success rate
   - User satisfaction scores
   - Cost per inference

### Monitoring Dashboard

```python
# Example: Track model performance
from prometheus_client import Counter, Histogram

model_inference_duration = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_name', 'model_version']
)

model_predictions_total = Counter(
    'model_predictions_total',
    'Total predictions made',
    ['model_name', 'model_version', 'status']
)

# Track inference
with model_inference_duration.labels(
    model_name='recognition-facenet',
    model_version='2.1.3'
).time():
    result = model.predict(image)

model_predictions_total.labels(
    model_name='recognition-facenet',
    model_version='2.1.3',
    status='success'
).inc()
```

## Best Practices

### 1. Version Control

- **Always version models**: Never deploy unversioned models
- **Track model lineage**: Document parent models and training changes
- **Use Git for code**: Version training code alongside models

### 2. Testing

- **Unit tests**: Test model input/output shapes and types
- **Integration tests**: Test model with API endpoints
- **Performance tests**: Benchmark inference time and memory
- **Accuracy tests**: Validate on held-out test sets

### 3. Documentation

- **Changelog**: Document changes in each version
- **Training details**: Record hyperparameters, dataset info
- **Known issues**: Document limitations and edge cases

### 4. Security

- **Checksum verification**: Always verify model integrity
- **Access control**: Restrict model storage access
- **Audit logs**: Track who deployed which models

### 5. Deployment

- **Canary deployments**: Gradually roll out new models
- **A/B testing**: Compare new vs old models in production
- **Feature flags**: Use flags to control model usage
- **Monitoring**: Set up alerts for performance degradation

### 6. Model Lifecycle

```
Development → Testing → Staging → Production → Monitoring → Deprecation
     ↑                                                            ↓
     └────────────────── Rollback/Improvements ──────────────────┘
```

### 7. Naming Conventions

- Use descriptive names: `recognition-facenet-v2.1.3` not `model_v1`
- Include metadata in name when relevant: `detection-yolov8-v1.0.0-gpu`
- Be consistent across all models

### 8. Storage Management

- **Retention policy**: Keep last 3 production versions
- **Compression**: Compress archived models
- **Cleanup**: Remove deprecated models after grace period
- **Backup**: Maintain backups of production models

## Example Workflow

```bash
# 1. Train new model
python scripts/train_model.py --config config/train_facenet.yaml

# 2. Validate and package
python scripts/validate_model.py --model output/facenet_latest.pt
python scripts/package_model.py --model output/facenet_latest.pt --version 2.2.0

# 3. Register in model registry
python scripts/model_registry.py register --model models/recognition/facenet-v2.2.0

# 4. Deploy to staging
python scripts/deploy_model.py --model-id rec-facenet-v2.2.0 --env staging

# 5. Run tests
pytest tests/integration/ --model-version 2.2.0

# 6. Canary deployment to production
python scripts/deploy_model.py --model-id rec-facenet-v2.2.0 --env production --strategy canary

# 7. Monitor for 24 hours
python scripts/monitor_model.py --model-id rec-facenet-v2.2.0 --duration 24h

# 8. Complete rollout or rollback
# If successful:
python scripts/deploy_model.py --model-id rec-facenet-v2.2.0 --env production --strategy complete
# If issues found:
python scripts/rollback_model.py --model-type recognition --to-version 2.1.3
```

## References

- [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html)
- [DVC for Model Versioning](https://dvc.org/)
- [Semantic Versioning](https://semver.org/)
- [Model Deployment Best Practices](https://neptune.ai/blog/model-deployment-best-practices)
