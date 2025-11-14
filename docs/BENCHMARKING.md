# Performance Benchmarking Guide

## Overview

This guide explains how to benchmark the Face Recognition System's performance.

## Quick Start

Run the benchmark script:

```bash
python scripts/benchmark.py --test-images data/images --output-dir benchmark_results
```

## Benchmark Metrics

### 1. Detection Speed
- **Metric**: Faces detected per second
- **Target**: > 30 FPS for single face
- **Command**: `python scripts/benchmark.py --mode detection`

### 2. Recognition Accuracy
- **Metric**: True positive rate at various thresholds
- **Target**: > 95% at threshold 0.6
- **Command**: `python scripts/benchmark.py --mode accuracy`

### 3. Embedding Generation
- **Metric**: Time to generate embeddings
- **Target**: < 50ms per face
- **Command**: `python scripts/benchmark.py --mode embedding`

### 4. Database Query Speed
- **Metric**: Query time for similarity search
- **Target**: < 100ms for 10K faces
- **Command**: `python scripts/benchmark.py --mode query`

## Interpreting Results

Results are saved in JSON format:

```json
{
  "timestamp": "2025-11-13T23:00:00",
  "detection_fps": 45.2,
  "recognition_accuracy": 0.96,
  "embedding_time_ms": 42,
  "query_time_ms": 85
}
```

## Performance Optimization Tips

### 1. Use GPU Acceleration
Set `ENABLE_GPU=true` in your `.env` file

### 2. Batch Processing
Process multiple images at once:
```python
results = processor.process_batch(images)
```

### 3. Enable Caching
Set `ENABLE_CACHING=true` to cache embeddings

### 4. Optimize Database
Create indexes on frequently queried columns

## Continuous Benchmarking

Add benchmarks to your CI/CD pipeline:

```yaml
- name: Run benchmarks
  run: |
    python scripts/benchmark.py --quick
    python scripts/compare_benchmarks.py
```

## Regression Testing

Compare current performance against baseline:

```bash
python scripts/benchmark.py --compare baseline.json
```

## Hardware Requirements

### Minimum
- CPU: 4 cores
- RAM: 8GB
- Storage: SSD

### Recommended
- CPU: 8+ cores
- RAM: 16GB+
- GPU: CUDA-capable with 4GB+ VRAM
- Storage: NVMe SSD

## Reporting Issues

If benchmarks show degradation:
1. Run with `--verbose` flag
2. Check system resources
3. Review recent code changes
4. Open an issue with benchmark results
