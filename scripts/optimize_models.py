#!/usr/bin/env python3
"""
Model Optimization Script

Optimizes ML models for production deployment using:
- Quantization (INT8, FP16)
- Pruning (structured and unstructured)
- Knowledge distillation
- ONNX export for cross-platform deployment
- TensorRT optimization for NVIDIA GPUs
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize_static, get_default_qconfig
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic as onnx_quantize_dynamic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimize ML models for production deployment"""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "models/optimized",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Device: {self.device}")
    
    def load_model(self) -> nn.Module:
        """Load PyTorch model"""
        try:
            model = torch.load(self.model_path, map_location=self.device)
            model.eval()
            logger.info(f"✓ Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def quantize_dynamic(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply dynamic quantization
        
        Benefits:
        - Smaller model size (4x reduction)
        - Faster inference on CPU
        - No calibration data needed
        """
        logger.info("Applying dynamic quantization...")
        
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},  # Layers to quantize
            dtype=dtype
        )
        
        # Save quantized model
        output_path = self.output_dir / f"{self.model_path.stem}_quantized_dynamic.pth"
        torch.save(quantized_model, output_path)
        
        logger.info(f"✓ Dynamic quantization complete: {output_path}")
        return quantized_model
    
    def quantize_static(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply static quantization
        
        Benefits:
        - Better accuracy than dynamic
        - Faster inference
        - Requires calibration data
        """
        logger.info("Applying static quantization...")
        
        # Prepare model for quantization
        model.qconfig = get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model)
        
        # Calibrate with sample data
        logger.info("Calibrating model...")
        with torch.no_grad():
            for data in calibration_data:
                model_prepared(data.to(self.device))
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        # Save quantized model
        output_path = self.output_dir / f"{self.model_path.stem}_quantized_static.pth"
        torch.save(quantized_model, output_path)
        
        logger.info(f"✓ Static quantization complete: {output_path}")
        return quantized_model
    
    def prune_unstructured(
        self,
        model: nn.Module,
        amount: float = 0.3
    ) -> nn.Module:
        """
        Apply unstructured pruning (remove individual weights)
        
        Args:
            amount: Fraction of weights to prune (0.0 to 1.0)
        """
        logger.info(f"Applying unstructured pruning (amount: {amount})...")
        
        import torch.nn.utils.prune as prune
        
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        # Save pruned model
        output_path = self.output_dir / f"{self.model_path.stem}_pruned_unstructured.pth"
        torch.save(model, output_path)
        
        logger.info(f"✓ Unstructured pruning complete: {output_path}")
        return model
    
    def prune_structured(
        self,
        model: nn.Module,
        amount: float = 0.3
    ) -> nn.Module:
        """
        Apply structured pruning (remove entire filters/neurons)
        
        Args:
            amount: Fraction of structures to prune
        """
        logger.info(f"Applying structured pruning (amount: {amount})...")
        
        import torch.nn.utils.prune as prune
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Prune filters
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=amount,
                    n=2,
                    dim=0  # Prune output channels
                )
                prune.remove(module, 'weight')
        
        # Save pruned model
        output_path = self.output_dir / f"{self.model_path.stem}_pruned_structured.pth"
        torch.save(model, output_path)
        
        logger.info(f"✓ Structured pruning complete: {output_path}")
        return model
    
    def export_onnx(
        self,
        model: nn.Module,
        input_shape: tuple = (1, 3, 224, 224),
        opset_version: int = 14
    ) -> str:
        """
        Export model to ONNX format
        
        Benefits:
        - Cross-platform deployment
        - Optimized inference
        - Hardware acceleration
        """
        logger.info("Exporting to ONNX format...")
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        output_path = self.output_dir / f"{self.model_path.stem}.onnx"
        
        # Export
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"✓ ONNX export complete: {output_path}")
        return str(output_path)
    
    def optimize_onnx(
        self,
        onnx_path: str,
        quantize: bool = True
    ) -> str:
        """
        Optimize ONNX model
        
        Args:
            onnx_path: Path to ONNX model
            quantize: Apply INT8 quantization
        """
        logger.info("Optimizing ONNX model...")
        
        onnx_path = Path(onnx_path)
        output_path = self.output_dir / f"{onnx_path.stem}_optimized.onnx"
        
        if quantize:
            # Apply dynamic quantization
            from onnxruntime.quantization import QuantType
            
            onnx_quantize_dynamic(
                str(onnx_path),
                str(output_path),
                weight_type=QuantType.QInt8
            )
            logger.info("✓ ONNX quantization applied")
        else:
            # Basic optimization
            import onnx
            from onnxruntime.transformers import optimizer
            
            model = onnx.load(str(onnx_path))
            optimized_model = optimizer.optimize_model(
                str(onnx_path),
                model_type='bert',  # or appropriate type
                num_heads=0,
                hidden_size=0
            )
            optimized_model.save_model_to_file(str(output_path))
        
        logger.info(f"✓ ONNX optimization complete: {output_path}")
        return str(output_path)
    
    def convert_to_fp16(
        self,
        model: nn.Module
    ) -> nn.Module:
        """
        Convert model to FP16 (half precision)
        
        Benefits:
        - 2x smaller model size
        - 2x faster on supported GPUs
        - Minimal accuracy loss
        """
        logger.info("Converting to FP16...")
        
        model = model.half()
        
        # Save FP16 model
        output_path = self.output_dir / f"{self.model_path.stem}_fp16.pth"
        torch.save(model, output_path)
        
        logger.info(f"✓ FP16 conversion complete: {output_path}")
        return model
    
    def benchmark_model(
        self,
        model: nn.Module,
        input_shape: tuple = (1, 3, 224, 224),
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark model performance
        
        Returns:
            Dictionary with latency and throughput metrics
        """
        logger.info(f"Benchmarking model ({num_iterations} iterations)...")
        
        # Create test input
        test_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = model(test_input)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                latencies.append(end_time - start_time)
        
        # Calculate metrics
        latencies = np.array(latencies) * 1000  # Convert to ms
        
        metrics = {
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "throughput_fps": float(1000.0 / np.mean(latencies))
        }
        
        logger.info("Benchmark results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.2f}")
        
        return metrics
    
    def compare_models(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        test_data: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Compare original and optimized models
        
        Returns:
            Comparison metrics (size, speed, accuracy)
        """
        logger.info("Comparing models...")
        
        # Size comparison
        original_size = self.get_model_size(original_model)
        optimized_size = self.get_model_size(optimized_model)
        size_reduction = (1 - optimized_size / original_size) * 100
        
        # Speed comparison
        original_metrics = self.benchmark_model(original_model)
        optimized_metrics = self.benchmark_model(optimized_model)
        speedup = original_metrics['mean_latency_ms'] / optimized_metrics['mean_latency_ms']
        
        comparison = {
            "size": {
                "original_mb": original_size,
                "optimized_mb": optimized_size,
                "reduction_percent": size_reduction
            },
            "speed": {
                "original_latency_ms": original_metrics['mean_latency_ms'],
                "optimized_latency_ms": optimized_metrics['mean_latency_ms'],
                "speedup": speedup
            }
        }
        
        # Accuracy comparison (if test data provided)
        if test_data is not None:
            accuracy_original = self.evaluate_accuracy(original_model, test_data)
            accuracy_optimized = self.evaluate_accuracy(optimized_model, test_data)
            
            comparison["accuracy"] = {
                "original": accuracy_original,
                "optimized": accuracy_optimized,
                "difference": accuracy_optimized - accuracy_original
            }
        
        logger.info("\nComparison Results:")
        logger.info(f"  Size reduction: {size_reduction:.1f}%")
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        return comparison
    
    def get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        temp_path = self.output_dir / "temp_model.pth"
        torch.save(model.state_dict(), temp_path)
        size_mb = temp_path.stat().st_size / (1024 * 1024)
        temp_path.unlink()
        return size_mb
    
    def evaluate_accuracy(
        self,
        model: nn.Module,
        test_data: List[torch.Tensor]
    ) -> float:
        """Evaluate model accuracy on test data"""
        # Placeholder - implement based on your task
        logger.warning("Accuracy evaluation not implemented")
        return 0.0
    
    def save_optimization_report(
        self,
        comparison: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> None:
        """Save optimization report to JSON"""
        if output_path is None:
            output_path = self.output_dir / "optimization_report.json"
        
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"✓ Optimization report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Optimize ML models for production")
    parser.add_argument("model_path", type=str, help="Path to model file")
    parser.add_argument("--output-dir", type=str, default="models/optimized",
                        help="Output directory for optimized models")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply quantization")
    parser.add_argument("--prune", action="store_true",
                        help="Apply pruning")
    parser.add_argument("--prune-amount", type=float, default=0.3,
                        help="Pruning amount (0.0 to 1.0)")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export to ONNX format")
    parser.add_argument("--fp16", action="store_true",
                        help="Convert to FP16")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark model performance")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = ModelOptimizer(args.model_path, args.output_dir)
    
    # Load original model
    model = optimizer.load_model()
    
    # Apply optimizations
    optimized_model = model
    
    if args.quantize:
        optimized_model = optimizer.quantize_dynamic(optimized_model)
    
    if args.prune:
        optimized_model = optimizer.prune_unstructured(
            optimized_model,
            amount=args.prune_amount
        )
    
    if args.fp16:
        optimized_model = optimizer.convert_to_fp16(optimized_model)
    
    if args.export_onnx:
        onnx_path = optimizer.export_onnx(optimized_model)
        optimizer.optimize_onnx(onnx_path, quantize=args.quantize)
    
    if args.benchmark:
        logger.info("\n=== Benchmarking ===")
        optimizer.benchmark_model(optimized_model)
    
    # Compare models
    comparison = optimizer.compare_models(model, optimized_model)
    optimizer.save_optimization_report(comparison)
    
    logger.info("\n✓ Model optimization complete!")


if __name__ == "__main__":
    main()
