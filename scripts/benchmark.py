#!/usr/bin/env python3
"""
Face recognition benchmarking tool.
Evaluates performance of different encoders and similarity metrics.
"""

import argparse
import time
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from embeddings import FaceEmbeddingManager
from similarity import DistanceMetric
from encoders import DlibFaceEncoder, SimpleEmbeddingEncoder

class FaceRecognitionBenchmark:
    """Benchmark suite for face recognition components."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def benchmark_encoders(self, test_images_dir: str) -> Dict[str, Any]:
        """
        Benchmark different face encoders.
        
        Args:
            test_images_dir: Directory with test images
            
        Returns:
            Benchmark results dictionary
        """
        encoder_results = {}
        test_images = list(Path(test_images_dir).glob('*.jpg')) + \
                     list(Path(test_images_dir).glob('*.png'))
        
        if not test_images:
            print(f"No test images found in {test_images_dir}")
            return {}
            
        print(f"Benchmarking encoders with {len(test_images)} images...")
        
        # Test different encoders
        encoders = {
            'simple': SimpleEmbeddingEncoder(),
            'dlib': None  # Will try to initialize
        }
        
        # Try to initialize dlib encoder
        try:
            encoders['dlib'] = DlibFaceEncoder()
        except ImportError:
            print("Dlib encoder not available, skipping...")
            del encoders['dlib']
        
        for encoder_name, encoder in encoders.items():
            if encoder is None:
                continue
                
            print(f"\nTesting {encoder_name} encoder...")
            
            start_time = time.time()
            total_faces = 0
            successful_encodings = 0
            embedding_times = []
            
            manager = FaceEmbeddingManager(encoder_type=encoder_name)
            
            for img_path in test_images[:10]:  # Limit for benchmark
                try:
                    img_start = time.time()
                    embedding_data = manager.generate_embeddings_from_image(str(img_path))
                    img_time = time.time() - img_start
                    
                    total_faces += embedding_data['metadata']['total_faces_detected']
                    successful_encodings += len(embedding_data['embeddings'])
                    embedding_times.append(img_time)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            total_time = time.time() - start_time
            
            encoder_results[encoder_name] = {
                'total_time': total_time,
                'avg_time_per_image': np.mean(embedding_times) if embedding_times else 0,
                'total_faces_detected': total_faces,
                'successful_encodings': successful_encodings,
                'success_rate': successful_encodings / max(total_faces, 1),
                'images_per_second': len(test_images[:10]) / total_time
            }
            
        return encoder_results
    
    def benchmark_similarity_metrics(self, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        Benchmark different similarity metrics.
        
        Args:
            embeddings: List of face embeddings to compare
            
        Returns:
            Benchmark results for similarity metrics
        """
        if len(embeddings) < 2:
            return {}
            
        print("\nBenchmarking similarity metrics...")
        
        metrics_results = {}
        metrics = [
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.COSINE,
            DistanceMetric.MANHATTAN,
            DistanceMetric.CHEBYSHEV
        ]
        
        # Use subset for timing
        test_embeddings = embeddings[:min(50, len(embeddings))]
        n_comparisons = len(test_embeddings) * (len(test_embeddings) - 1) // 2
        
        for metric in metrics:
            from similarity import FaceSimilarity
            similarity_calc = FaceSimilarity(metric)
            
            start_time = time.time()
            
            # Calculate all pairwise similarities
            similarity_matrix = similarity_calc.calculate_similarity_matrix(test_embeddings)
            
            calculation_time = time.time() - start_time
            
            metrics_results[metric.value] = {
                'calculation_time': calculation_time,
                'comparisons_per_second': n_comparisons / calculation_time if calculation_time > 0 else 0,
                'mean_similarity': np.mean(similarity_matrix),
                'std_similarity': np.std(similarity_matrix)
            }
            
        return metrics_results
    
    def generate_report(self, results: Dict[str, Any]) -> None:
        """Generate benchmark report with visualizations."""
        print("\nGenerating benchmark report...")
        
        # Save raw results
        with open(self.output_dir / 'benchmark_results.json', 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Generate visualizations if matplotlib available
        try:
            self._create_visualizations(results)
        except ImportError:
            print("Matplotlib not available, skipping visualizations")
        
        # Generate text report
        self._generate_text_report(results)
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def _create_visualizations(self, results: Dict[str, Any]) -> None:
        """Create benchmark visualization plots."""
        plt.style.use('seaborn-v0_8')
        
        # Encoder performance comparison
        if 'encoders' in results:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Face Encoder Performance Comparison')
            
            encoder_data = results['encoders']
            encoder_names = list(encoder_data.keys())
            
            if encoder_names:
                # Processing time
                times = [encoder_data[name]['avg_time_per_image'] for name in encoder_names]
                axes[0, 0].bar(encoder_names, times)
                axes[0, 0].set_title('Average Processing Time per Image')
                axes[0, 0].set_ylabel('Time (seconds)')
                
                # Success rates
                success_rates = [encoder_data[name]['success_rate'] for name in encoder_names]
                axes[0, 1].bar(encoder_names, success_rates)
                axes[0, 1].set_title('Encoding Success Rate')
                axes[0, 1].set_ylabel('Success Rate')
                axes[0, 1].set_ylim(0, 1)
                
                # Images per second
                ips = [encoder_data[name]['images_per_second'] for name in encoder_names]
                axes[1, 0].bar(encoder_names, ips)
                axes[1, 0].set_title('Processing Throughput')
                axes[1, 0].set_ylabel('Images per Second')
                
                # Total faces detected
                faces = [encoder_data[name]['total_faces_detected'] for name in encoder_names]
                axes[1, 1].bar(encoder_names, faces)
                axes[1, 1].set_title('Total Faces Detected')
                axes[1, 1].set_ylabel('Number of Faces')
                
            plt.tight_layout()
            plt.savefig(self.output_dir / 'encoder_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Similarity metrics comparison
        if 'similarity_metrics' in results:
            metrics_data = results['similarity_metrics']
            metric_names = list(metrics_data.keys())
            
            if metric_names:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle('Similarity Metrics Performance')
                
                # Calculation speed
                speeds = [metrics_data[name]['comparisons_per_second'] for name in metric_names]
                ax1.bar(metric_names, speeds)
                ax1.set_title('Calculation Speed')
                ax1.set_ylabel('Comparisons per Second')
                ax1.tick_params(axis='x', rotation=45)
                
                # Mean similarity scores
                mean_sims = [metrics_data[name]['mean_similarity'] for name in metric_names]
                ax2.bar(metric_names, mean_sims)
                ax2.set_title('Mean Similarity Scores')
                ax2.set_ylabel('Mean Similarity')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'similarity_metrics.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    def _generate_text_report(self, results: Dict[str, Any]) -> None:
        """Generate detailed text report."""
        report_path = self.output_dir / 'benchmark_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("Face Recognition Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Encoder results
            if 'encoders' in results:
                f.write("ENCODER PERFORMANCE\n")
                f.write("-" * 20 + "\n")
                
                for encoder_name, data in results['encoders'].items():
                    f.write(f"\n{encoder_name.upper()} Encoder:\n")
                    f.write(f"  Average time per image: {data['avg_time_per_image']:.4f}s\n")
                    f.write(f"  Success rate: {data['success_rate']:.2%}\n")
                    f.write(f"  Images per second: {data['images_per_second']:.2f}\n")
                    f.write(f"  Total faces detected: {data['total_faces_detected']}\n")
                    f.write(f"  Successful encodings: {data['successful_encodings']}\n")
            
            # Similarity metrics results
            if 'similarity_metrics' in results:
                f.write("\n\nSIMILARITY METRICS PERFORMANCE\n")
                f.write("-" * 30 + "\n")
                
                for metric_name, data in results['similarity_metrics'].items():
                    f.write(f"\n{metric_name.upper()} Metric:\n")
                    f.write(f"  Comparisons per second: {data['comparisons_per_second']:.2f}\n")
                    f.write(f"  Mean similarity: {data['mean_similarity']:.4f}\n")
                    f.write(f"  Std similarity: {data['std_similarity']:.4f}\n")
        
        print(f"Detailed report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Face Recognition Benchmark Tool')
    parser.add_argument('--test-images', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = FaceRecognitionBenchmark(args.output_dir)
    
    # Run benchmarks
    results = {}
    
    # Benchmark encoders
    encoder_results = benchmark.benchmark_encoders(args.test_images)
    if encoder_results:
        results['encoders'] = encoder_results
    
    # Generate some test embeddings for similarity metric benchmark
    try:
        manager = FaceEmbeddingManager(encoder_type="simple")
        test_images = list(Path(args.test_images).glob('*.jpg'))[:5]
        
        all_embeddings = []
        for img_path in test_images:
            embedding_data = manager.generate_embeddings_from_image(str(img_path))
            for emb_info in embedding_data['embeddings']:
                all_embeddings.append(emb_info['embedding'])
        
        if all_embeddings:
            similarity_results = benchmark.benchmark_similarity_metrics(all_embeddings)
            if similarity_results:
                results['similarity_metrics'] = similarity_results
    
    except Exception as e:
        print(f"Error in similarity benchmark: {e}")
    
    # Generate report
    if results:
        benchmark.generate_report(results)
        print(f"\nBenchmark complete! Results saved to: {args.output_dir}")
    else:
        print("No benchmark results generated.")

if __name__ == '__main__':
    main()