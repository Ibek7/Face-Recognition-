# Edge Device Optimization and Model Compression

import logging
import numpy as np
import json
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import threading

class DeviceType(Enum):
    """Target device types."""
    MOBILE = "mobile"
    EMBEDDED = "embedded"
    IOT = "iot"
    EDGE_SERVER = "edge_server"
    DESKTOP = "desktop"

class OptimizationLevel(Enum):
    """Optimization intensity levels."""
    MINIMAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    AGGRESSIVE = 5

@dataclass
class DeviceCapabilities:
    """Device capabilities and constraints."""
    device_type: DeviceType
    memory_mb: int
    storage_mb: int
    cpu_cores: int
    has_gpu: bool
    has_npu: bool
    battery_powered: bool
    network_bandwidth_mbps: float
    
    def get_recommended_model_size_mb(self) -> float:
        """Get recommended model size based on device."""
        # Use 20-30% of available memory
        return self.memory_mb * 0.25

@dataclass
class CompressionMetrics:
    """Metrics for model compression."""
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    inference_time_increase: float  # percentage
    accuracy_loss: float  # percentage
    memory_reduction: float  # percentage

class ModelCompressor(ABC):
    """Base class for model compression."""
    
    @abstractmethod
    def compress(self, model: Any, **kwargs) -> Tuple[Any, CompressionMetrics]:
        """Compress model."""
        pass

class QuantizationCompressor(ModelCompressor):
    """INT8 Quantization for model compression."""
    
    def __init__(self, quantization_bits: int = 8):
        self.quantization_bits = quantization_bits
        self.logger = logging.getLogger(__name__)
    
    def compress(self, weights: np.ndarray, 
                preserve_accuracy: bool = True) -> Tuple[np.ndarray, CompressionMetrics]:
        """Quantize model weights to lower precision."""
        
        original_dtype_size = weights.dtype.itemsize
        original_size_mb = weights.nbytes / 1024 / 1024
        
        # Calculate quantization range
        min_val = np.min(weights)
        max_val = np.max(weights)
        
        # Quantize to specified bits
        scale = (max_val - min_val) / (2 ** self.quantization_bits - 1)
        zero_point = int(-min_val / scale)
        
        # Quantize
        quantized = np.clip(
            np.round(weights / scale + zero_point),
            0,
            2 ** self.quantization_bits - 1
        ).astype(f'uint{self.quantization_bits}')
        
        # Dequantize for accuracy measurement
        dequantized = (quantized - zero_point) * scale
        
        # Calculate metrics
        compressed_size_mb = quantized.nbytes / 1024 / 1024
        accuracy_loss = np.mean(np.abs(weights - dequantized)) / np.mean(np.abs(weights)) * 100
        
        metrics = CompressionMetrics(
            original_size_mb=original_size_mb,
            compressed_size_mb=compressed_size_mb,
            compression_ratio=original_size_mb / compressed_size_mb,
            inference_time_increase=0.0,  # Typically faster
            accuracy_loss=accuracy_loss,
            memory_reduction=(original_size_mb - compressed_size_mb) / original_size_mb * 100
        )
        
        self.logger.info(f"Quantization: {original_size_mb:.2f}MB -> {compressed_size_mb:.2f}MB "
                        f"({metrics.compression_ratio:.1f}x compression)")
        
        return quantized, metrics

class PruningCompressor(ModelCompressor):
    """Magnitude-based pruning for model compression."""
    
    def __init__(self, pruning_ratio: float = 0.3):
        self.pruning_ratio = pruning_ratio
        self.logger = logging.getLogger(__name__)
    
    def compress(self, weights: np.ndarray) -> Tuple[np.ndarray, CompressionMetrics]:
        """Prune weights by magnitude."""
        
        original_size_mb = weights.nbytes / 1024 / 1024
        
        # Calculate pruning threshold
        abs_weights = np.abs(weights)
        threshold = np.percentile(abs_weights, self.pruning_ratio * 100)
        
        # Prune weights below threshold
        pruned = np.where(abs_weights >= threshold, weights, 0)
        
        # Calculate sparsity
        zeros = np.sum(pruned == 0)
        total = pruned.size
        sparsity = zeros / total * 100
        
        compressed_size_mb = original_size_mb * (1 - sparsity / 100)
        
        metrics = CompressionMetrics(
            original_size_mb=original_size_mb,
            compressed_size_mb=compressed_size_mb,
            compression_ratio=original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 1.0,
            inference_time_increase=5.0,  # May increase due to sparse operations
            accuracy_loss=2.0,  # Typical accuracy loss from pruning
            memory_reduction=sparsity
        )
        
        self.logger.info(f"Pruning: Achieved {sparsity:.1f}% sparsity")
        
        return pruned, metrics

class DistillationCompressor(ModelCompressor):
    """Knowledge distillation for model compression."""
    
    def __init__(self, compression_factor: float = 0.5):
        self.compression_factor = compression_factor
        self.logger = logging.getLogger(__name__)
    
    def compress(self, weights: np.ndarray) -> Tuple[np.ndarray, CompressionMetrics]:
        """Apply knowledge distillation compression."""
        
        original_size_mb = weights.nbytes / 1024 / 1024
        
        # Reduce layer dimensions
        new_shape = tuple(int(dim * self.compression_factor) 
                         for dim in weights.shape)
        
        # Simplified distillation: average pooling
        distilled = np.zeros(new_shape)
        
        for idx in np.ndindex(new_shape):
            # Calculate source indices
            src_indices = tuple(
                slice(int(i * len(weights.shape[j])) // new_shape[j],
                      int((i + 1) * len(weights.shape[j])) // new_shape[j])
                for i, j in enumerate(idx)
            )
            
            if src_indices[0].stop > src_indices[0].start:
                distilled[idx] = np.mean(weights[src_indices])
        
        compressed_size_mb = distilled.nbytes / 1024 / 1024
        
        metrics = CompressionMetrics(
            original_size_mb=original_size_mb,
            compressed_size_mb=compressed_size_mb,
            compression_ratio=original_size_mb / compressed_size_mb if compressed_size_mb > 0 else 1.0,
            inference_time_increase=-30.0,  # Faster inference
            accuracy_loss=5.0,  # Larger accuracy loss but significant speedup
            memory_reduction=(original_size_mb - compressed_size_mb) / original_size_mb * 100
        )
        
        self.logger.info(f"Distillation: {original_size_mb:.2f}MB -> {compressed_size_mb:.2f}MB")
        
        return distilled, metrics

class EdgeOptimizer:
    """Optimize models for edge devices."""
    
    def __init__(self, device_capabilities: DeviceCapabilities):
        self.device_capabilities = device_capabilities
        self.logger = logging.getLogger(__name__)
        self.compression_history: List[CompressionMetrics] = []
    
    def get_optimization_strategy(self, 
                                 model_size_mb: float,
                                 target_latency_ms: float) -> Dict[str, Any]:
        """Determine optimal optimization strategy."""
        
        available_memory_mb = self.device_capabilities.memory_mb * 0.25
        
        strategy = {
            'device_type': self.device_capabilities.device_type.value,
            'optimizations': [],
            'estimated_final_size_mb': model_size_mb,
            'estimated_latency_ms': target_latency_ms
        }
        
        # Quantization is almost always beneficial
        strategy['optimizations'].append({
            'technique': 'quantization',
            'level': OptimizationLevel.HIGH.name
        })
        
        # Pruning for larger models
        if model_size_mb > available_memory_mb * 0.8:
            strategy['optimizations'].append({
                'technique': 'pruning',
                'level': OptimizationLevel.MEDIUM.name
            })
        
        # Distillation for very large models
        if model_size_mb > available_memory_mb * 1.5:
            strategy['optimizations'].append({
                'technique': 'distillation',
                'level': OptimizationLevel.AGGRESSIVE.name
            })
        
        return strategy
    
    def estimate_compressed_size(self, original_size_mb: float,
                                optimizations: List[str]) -> float:
        """Estimate final model size after optimizations."""
        
        estimated_size = original_size_mb
        
        for opt in optimizations:
            if opt == 'quantization':
                estimated_size *= 0.25  # INT8 is 1/4 of float32
            elif opt == 'pruning':
                estimated_size *= 0.7  # 30% pruning
            elif opt == 'distillation':
                estimated_size *= 0.5  # 50% size reduction
        
        return estimated_size

class EdgeInferenceEngine:
    """Optimized inference engine for edge devices."""
    
    def __init__(self, device_capabilities: DeviceCapabilities,
                 model_path: str):
        self.device_capabilities = device_capabilities
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
        self.inference_cache = {}
        self.lock = threading.RLock()
    
    def optimize_input(self, input_data: np.ndarray) -> np.ndarray:
        """Optimize input for edge inference."""
        
        # Reduce precision
        if input_data.dtype == np.float64:
            input_data = input_data.astype(np.float32)
        
        # Reduce dimensionality
        if input_data.size > 1000000:  # 1M elements
            # Use bilinear downsampling
            scale_factor = np.sqrt(1000000 / input_data.size)
            new_shape = tuple(int(dim * scale_factor) for dim in input_data.shape)
            # Simplified downsampling
            input_data = input_data[:int(new_shape[0]), :int(new_shape[1])]
        
        return input_data
    
    def batch_inference(self, inputs: List[np.ndarray], 
                       batch_size: int = 8) -> List[Dict]:
        """Perform batched inference."""
        
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            optimized_batch = [self.optimize_input(inp) for inp in batch]
            
            # Simulate inference
            batch_results = []
            for inp in optimized_batch:
                cache_key = hash(inp.tobytes())
                
                with self.lock:
                    if cache_key in self.inference_cache:
                        result = self.inference_cache[cache_key]
                    else:
                        # Simulated inference
                        result = {
                            'detections': int(inp.mean() * 10),
                            'confidence': 0.95
                        }
                        self.inference_cache[cache_key] = result
                
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization report."""
        
        return {
            'device_type': self.device_capabilities.device_type.value,
            'memory_mb': self.device_capabilities.memory_mb,
            'has_gpu': self.device_capabilities.has_gpu,
            'has_npu': self.device_capabilities.has_npu,
            'cache_size': len(self.inference_cache),
            'battery_powered': self.device_capabilities.battery_powered
        }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Define mobile device
    mobile_device = DeviceCapabilities(
        device_type=DeviceType.MOBILE,
        memory_mb=2048,
        storage_mb=1024,
        cpu_cores=4,
        has_gpu=True,
        has_npu=False,
        battery_powered=True,
        network_bandwidth_mbps=10.0
    )
    
    # Create optimizer
    optimizer = EdgeOptimizer(mobile_device)
    
    # Get optimization strategy
    strategy = optimizer.get_optimization_strategy(
        model_size_mb=100,
        target_latency_ms=50
    )
    
    print("Optimization Strategy:")
    print(json.dumps(strategy, indent=2))
    
    # Test compression
    test_weights = np.random.randn(1024, 512).astype(np.float32)
    
    quantizer = QuantizationCompressor()
    quantized, q_metrics = quantizer.compress(test_weights)
    
    print("\nQuantization Metrics:")
    print(json.dumps({
        'original_size_mb': q_metrics.original_size_mb,
        'compressed_size_mb': q_metrics.compressed_size_mb,
        'compression_ratio': q_metrics.compression_ratio,
        'accuracy_loss_percent': q_metrics.accuracy_loss
    }, indent=2))
    
    # Create inference engine
    engine = EdgeInferenceEngine(mobile_device, "model.tflite")
    
    # Run optimized inference
    test_inputs = [np.random.randn(224, 224, 3).astype(np.float32) for _ in range(3)]
    results = engine.batch_inference(test_inputs, batch_size=2)
    
    print(f"\nInference Results: {len(results)} samples processed")