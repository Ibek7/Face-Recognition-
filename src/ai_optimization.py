# Advanced AI/ML Model Optimization System

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import optuna
import time
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    optimization_type: str  # 'hyperparameter', 'architecture', 'quantization', 'pruning'
    target_metric: str  # 'accuracy', 'speed', 'memory', 'balanced'
    optimization_budget: int  # Number of trials/iterations
    target_accuracy_threshold: float  # Minimum acceptable accuracy
    target_speed_improvement: float  # Target speed improvement factor
    memory_constraint_mb: float  # Memory constraint in MB
    hardware_target: str  # 'cpu', 'gpu', 'mobile', 'edge'

@dataclass
class ModelMetrics:
    """Comprehensive model metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_ms: float
    memory_usage_mb: float
    model_size_mb: float
    flops: int
    energy_consumption: float
    latency_p95: float
    throughput_fps: float

class FaceRecognitionDataset(Dataset):
    """Custom dataset for face recognition training."""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, 
                 transform: Optional[Callable] = None):
        self.embeddings = embeddings
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        
        if self.transform:
            embedding = self.transform(embedding)
        
        return torch.FloatTensor(embedding), torch.LongTensor([label])

class AdvancedFaceRecognitionModel(nn.Module):
    """Advanced face recognition model with configurable architecture."""
    
    def __init__(self, input_dim: int = 512, hidden_dims: List[int] = [256, 128], 
                 num_classes: int = 1000, dropout_rate: float = 0.2,
                 activation: str = 'relu', use_batch_norm: bool = True):
        super(AdvancedFaceRecognitionModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build dynamic architecture
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'swish':
                layers.append(nn.SiLU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(x)

class ModelOptimizer:
    """Advanced model optimization system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimization_history = []
        self.best_model = None
        self.best_metrics = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def optimize_hyperparameters(self, train_data: np.ndarray, train_labels: np.ndarray,
                                val_data: np.ndarray, val_labels: np.ndarray) -> Dict[str, Any]:
        """Optimize model hyperparameters using Optuna."""
        
        def objective(trial):
            # Suggest hyperparameters
            hidden_dims = []
            num_layers = trial.suggest_int('num_layers', 1, 4)
            
            for i in range(num_layers):
                dim = trial.suggest_int(f'hidden_dim_{i}', 64, 512, step=64)
                hidden_dims.append(dim)
            
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'swish'])
            optimizer_type = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
            
            # Create model
            model = AdvancedFaceRecognitionModel(
                input_dim=train_data.shape[1],
                hidden_dims=hidden_dims,
                num_classes=len(np.unique(train_labels)),
                dropout_rate=dropout_rate,
                activation=activation
            )
            
            # Train and evaluate
            metrics = self._train_and_evaluate(
                model, train_data, train_labels, val_data, val_labels,
                learning_rate, batch_size, optimizer_type, epochs=20
            )
            
            # Multi-objective optimization
            if self.config.target_metric == 'accuracy':
                return metrics.accuracy
            elif self.config.target_metric == 'speed':
                return -metrics.inference_time_ms  # Minimize inference time
            elif self.config.target_metric == 'memory':
                return -metrics.memory_usage_mb   # Minimize memory usage
            elif self.config.target_metric == 'balanced':
                # Balanced score considering accuracy, speed, and memory
                accuracy_score = metrics.accuracy
                speed_score = 1.0 / (metrics.inference_time_ms + 1e-6)
                memory_score = 1.0 / (metrics.memory_usage_mb + 1e-6)
                
                return 0.5 * accuracy_score + 0.3 * speed_score + 0.2 * memory_score
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.optimization_budget)
        
        # Get best parameters
        best_params = study.best_params
        
        # Train final model with best parameters
        final_model = self._create_model_from_params(
            best_params, train_data.shape[1], len(np.unique(train_labels))
        )
        
        final_metrics = self._train_and_evaluate(
            final_model, train_data, train_labels, val_data, val_labels,
            best_params['learning_rate'], best_params['batch_size'],
            best_params['optimizer'], epochs=50
        )
        
        self.best_model = final_model
        self.best_metrics = final_metrics
        
        return {
            "best_parameters": best_params,
            "best_value": study.best_value,
            "optimization_history": [trial.value for trial in study.trials if trial.value is not None],
            "final_metrics": asdict(final_metrics),
            "study_summary": {
                "n_trials": len(study.trials),
                "best_trial": study.best_trial.number,
                "optimization_time": sum(trial.duration.total_seconds() for trial in study.trials if trial.duration)
            }
        }
    
    def optimize_architecture(self, train_data: np.ndarray, train_labels: np.ndarray,
                            val_data: np.ndarray, val_labels: np.ndarray) -> Dict[str, Any]:
        """Neural Architecture Search (NAS) for optimal model architecture."""
        
        architecture_candidates = []
        
        # Generate architecture candidates
        for num_layers in range(1, 6):
            for base_width in [64, 128, 256]:
                for width_pattern in ['constant', 'decreasing', 'increasing', 'hourglass']:
                    hidden_dims = self._generate_architecture(num_layers, base_width, width_pattern)
                    
                    for activation in ['relu', 'gelu', 'swish']:
                        for use_batch_norm in [True, False]:
                            architecture_candidates.append({
                                'hidden_dims': hidden_dims,
                                'activation': activation,
                                'use_batch_norm': use_batch_norm,
                                'num_layers': num_layers,
                                'base_width': base_width,
                                'width_pattern': width_pattern
                            })
        
        # Limit candidates based on budget
        if len(architecture_candidates) > self.config.optimization_budget:
            # Sample architectures intelligently
            import random
            architecture_candidates = random.sample(architecture_candidates, self.config.optimization_budget)
        
        best_architecture = None
        best_score = -float('inf')
        architecture_results = []
        
        for arch in architecture_candidates:
            try:
                # Create model with this architecture
                model = AdvancedFaceRecognitionModel(
                    input_dim=train_data.shape[1],
                    hidden_dims=arch['hidden_dims'],
                    num_classes=len(np.unique(train_labels)),
                    activation=arch['activation'],
                    use_batch_norm=arch['use_batch_norm']
                )
                
                # Quick evaluation (fewer epochs for speed)
                metrics = self._train_and_evaluate(
                    model, train_data, train_labels, val_data, val_labels,
                    learning_rate=1e-3, batch_size=64, optimizer_type='adam', epochs=10
                )
                
                # Calculate architecture score
                score = self._calculate_architecture_score(metrics, arch)
                
                architecture_results.append({
                    'architecture': arch,
                    'metrics': asdict(metrics),
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_architecture = arch
                
                self.logger.info(f"Architecture {len(architecture_results)}: Score={score:.4f}, Accuracy={metrics.accuracy:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Architecture evaluation failed: {e}")
                continue
        
        # Train best architecture with full training
        if best_architecture:
            final_model = AdvancedFaceRecognitionModel(
                input_dim=train_data.shape[1],
                hidden_dims=best_architecture['hidden_dims'],
                num_classes=len(np.unique(train_labels)),
                activation=best_architecture['activation'],
                use_batch_norm=best_architecture['use_batch_norm']
            )
            
            final_metrics = self._train_and_evaluate(
                final_model, train_data, train_labels, val_data, val_labels,
                learning_rate=1e-3, batch_size=64, optimizer_type='adam', epochs=50
            )
            
            self.best_model = final_model
            self.best_metrics = final_metrics
        
        return {
            "best_architecture": best_architecture,
            "best_score": best_score,
            "architecture_results": architecture_results,
            "final_metrics": asdict(final_metrics) if best_architecture else None
        }
    
    def quantize_model(self, model: nn.Module) -> Tuple[nn.Module, ModelMetrics]:
        """Quantize model for faster inference."""
        
        # Dynamic quantization (post-training)
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        
        # Measure performance improvement
        original_metrics = self._benchmark_model(model)
        quantized_metrics = self._benchmark_model(quantized_model)
        
        self.logger.info(f"Quantization results:")
        self.logger.info(f"Speed improvement: {original_metrics.inference_time_ms / quantized_metrics.inference_time_ms:.2f}x")
        self.logger.info(f"Memory reduction: {original_metrics.memory_usage_mb / quantized_metrics.memory_usage_mb:.2f}x")
        self.logger.info(f"Model size reduction: {original_metrics.model_size_mb / quantized_metrics.model_size_mb:.2f}x")
        
        return quantized_model, quantized_metrics
    
    def prune_model(self, model: nn.Module, sparsity: float = 0.3) -> Tuple[nn.Module, ModelMetrics]:
        """Prune model to reduce parameters and improve efficiency."""
        
        import torch.nn.utils.prune as prune
        
        # Apply structured pruning
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
        
        # Remove pruning masks (make pruning permanent)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                prune.remove(module, 'weight')
        
        pruned_metrics = self._benchmark_model(model)
        
        self.logger.info(f"Pruning results:")
        self.logger.info(f"Sparsity applied: {sparsity:.1%}")
        self.logger.info(f"Model size: {pruned_metrics.model_size_mb:.2f} MB")
        self.logger.info(f"Inference time: {pruned_metrics.inference_time_ms:.2f} ms")
        
        return model, pruned_metrics
    
    def knowledge_distillation(self, teacher_model: nn.Module, train_data: np.ndarray,
                             train_labels: np.ndarray, val_data: np.ndarray,
                             val_labels: np.ndarray, student_architecture: Dict) -> nn.Module:
        """Train a smaller student model using knowledge distillation."""
        
        # Create student model
        student_model = AdvancedFaceRecognitionModel(
            input_dim=train_data.shape[1],
            hidden_dims=student_architecture['hidden_dims'],
            num_classes=len(np.unique(train_labels)),
            activation=student_architecture.get('activation', 'relu'),
            use_batch_norm=student_architecture.get('use_batch_norm', True)
        )
        
        # Distillation training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        teacher_model.to(device)
        student_model.to(device)
        teacher_model.eval()
        
        # Create data loaders
        train_dataset = FaceRecognitionDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(student_model.parameters(), lr=1e-3)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_kd = nn.KLDivLoss(reduction='batchmean')
        
        # Distillation parameters
        temperature = 4.0
        alpha = 0.7  # Weight for distillation loss
        
        student_model.train()
        
        for epoch in range(30):
            total_loss = 0
            
            for batch_embeddings, batch_labels in train_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.squeeze().to(device)
                
                # Student predictions
                student_outputs = student_model(batch_embeddings)
                
                # Teacher predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(batch_embeddings)
                
                # Calculate losses
                ce_loss = criterion_ce(student_outputs, batch_labels)
                
                kd_loss = criterion_kd(
                    torch.log_softmax(student_outputs / temperature, dim=1),
                    torch.softmax(teacher_outputs / temperature, dim=1)
                ) * (temperature ** 2)
                
                total_loss = alpha * kd_loss + (1 - alpha) * ce_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0:
                val_accuracy = self._evaluate_model(student_model, val_data, val_labels)
                self.logger.info(f"Distillation Epoch {epoch}: Val Accuracy = {val_accuracy:.4f}")
        
        student_metrics = self._benchmark_model(student_model)
        teacher_metrics = self._benchmark_model(teacher_model)
        
        self.logger.info(f"Knowledge Distillation Results:")
        self.logger.info(f"Student model size: {student_metrics.model_size_mb:.2f} MB (vs Teacher: {teacher_metrics.model_size_mb:.2f} MB)")
        self.logger.info(f"Student inference time: {student_metrics.inference_time_ms:.2f} ms (vs Teacher: {teacher_metrics.inference_time_ms:.2f} ms)")
        self.logger.info(f"Student accuracy: {student_metrics.accuracy:.4f} (vs Teacher: {teacher_metrics.accuracy:.4f})")
        
        return student_model
    
    def _train_and_evaluate(self, model: nn.Module, train_data: np.ndarray,
                          train_labels: np.ndarray, val_data: np.ndarray,
                          val_labels: np.ndarray, learning_rate: float,
                          batch_size: int, optimizer_type: str, epochs: int) -> ModelMetrics:
        """Train and evaluate model."""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Create data loaders
        train_dataset = FaceRecognitionDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_embeddings, batch_labels in train_loader:
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.squeeze().to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # Evaluate model
        return self._benchmark_model(model, val_data, val_labels)
    
    def _benchmark_model(self, model: nn.Module, test_data: np.ndarray = None,
                        test_labels: np.ndarray = None) -> ModelMetrics:
        """Comprehensive model benchmarking."""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # Generate test data if not provided
        if test_data is None:
            test_data = np.random.randn(100, 512)
            test_labels = np.random.randint(0, 10, 100)
        
        test_tensor = torch.FloatTensor(test_data).to(device)
        
        # Measure inference time
        inference_times = []
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(test_tensor[:1])
            
            # Actual timing
            for i in range(len(test_data)):
                start_time = time.time()
                _ = model(test_tensor[i:i+1])
                inference_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        # Calculate accuracy if labels provided
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        
        if test_labels is not None:
            with torch.no_grad():
                outputs = model(test_tensor)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                
                accuracy = accuracy_score(test_labels, predictions)
                precision, recall, f1_score, _ = precision_recall_fscore_support(
                    test_labels, predictions, average='weighted', zero_division=0
                )
        
        # Calculate model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)  # MB
        
        # Estimate memory usage (rough approximation)
        memory_usage = model_size * 2  # Rough estimate
        
        # Calculate FLOPs (simplified)
        flops = self._calculate_flops(model, test_data.shape[1])
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            inference_time_ms=np.mean(inference_times),
            memory_usage_mb=memory_usage,
            model_size_mb=model_size,
            flops=flops,
            energy_consumption=0.0,  # Would need specialized hardware to measure
            latency_p95=np.percentile(inference_times, 95),
            throughput_fps=1000.0 / np.mean(inference_times)
        )
    
    def _calculate_flops(self, model: nn.Module, input_dim: int) -> int:
        """Calculate FLOPs for the model (simplified)."""
        total_flops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_flops += module.in_features * module.out_features
        
        return total_flops
    
    def _generate_architecture(self, num_layers: int, base_width: int, pattern: str) -> List[int]:
        """Generate architecture based on pattern."""
        
        if pattern == 'constant':
            return [base_width] * num_layers
        elif pattern == 'decreasing':
            return [max(base_width // (2 ** i), 32) for i in range(num_layers)]
        elif pattern == 'increasing':
            return [min(base_width * (2 ** i), 512) for i in range(num_layers)]
        elif pattern == 'hourglass':
            # Decrease then increase
            mid = num_layers // 2
            decreasing = [max(base_width // (2 ** i), 32) for i in range(mid)]
            increasing = [max(base_width // (2 ** (mid - i - 1)), 32) for i in range(num_layers - mid)]
            return decreasing + increasing
        else:
            return [base_width] * num_layers
    
    def _calculate_architecture_score(self, metrics: ModelMetrics, arch: Dict) -> float:
        """Calculate score for architecture based on multiple criteria."""
        
        # Weights for different criteria
        accuracy_weight = 0.4
        speed_weight = 0.3
        memory_weight = 0.2
        complexity_weight = 0.1
        
        # Normalize metrics (assuming reasonable ranges)
        accuracy_score = metrics.accuracy  # Already 0-1
        speed_score = min(1.0, 10.0 / max(metrics.inference_time_ms, 0.1))  # Faster is better
        memory_score = min(1.0, 100.0 / max(metrics.memory_usage_mb, 1.0))  # Less memory is better
        
        # Complexity penalty (prefer simpler architectures)
        num_params = sum(arch['hidden_dims']) + arch['hidden_dims'][-1] * 1000  # Rough estimate
        complexity_score = min(1.0, 100000.0 / max(num_params, 1000))
        
        total_score = (
            accuracy_weight * accuracy_score +
            speed_weight * speed_score +
            memory_weight * memory_score +
            complexity_weight * complexity_score
        )
        
        return total_score
    
    def _create_model_from_params(self, params: Dict, input_dim: int, num_classes: int) -> nn.Module:
        """Create model from hyperparameter dictionary."""
        
        # Extract hidden dimensions
        hidden_dims = []
        for i in range(params['num_layers']):
            hidden_dims.append(params[f'hidden_dim_{i}'])
        
        return AdvancedFaceRecognitionModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout_rate=params['dropout_rate'],
            activation=params['activation']
        )
    
    def _evaluate_model(self, model: nn.Module, val_data: np.ndarray, val_labels: np.ndarray) -> float:
        """Evaluate model accuracy on validation data."""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        val_tensor = torch.FloatTensor(val_data).to(device)
        
        with torch.no_grad():
            outputs = model(val_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            accuracy = accuracy_score(val_labels, predictions)
        
        return accuracy
    
    def generate_optimization_report(self, results: Dict[str, Any], output_dir: str = "optimization_results"):
        """Generate comprehensive optimization report."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save results as JSON
        with open(output_path / "optimization_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate visualizations
        self._create_optimization_visualizations(results, output_path)
        
        # Generate text report
        self._create_text_report(results, output_path)
        
        self.logger.info(f"Optimization report saved to {output_path}")
    
    def _create_optimization_visualizations(self, results: Dict, output_path: Path):
        """Create visualization plots for optimization results."""
        
        plt.style.use('seaborn-v0_8')
        
        # Hyperparameter optimization history
        if 'optimization_history' in results:
            plt.figure(figsize=(10, 6))
            plt.plot(results['optimization_history'])
            plt.title('Hyperparameter Optimization Progress')
            plt.xlabel('Trial')
            plt.ylabel('Objective Value')
            plt.grid(True)
            plt.savefig(output_path / 'optimization_history.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Architecture comparison
        if 'architecture_results' in results:
            arch_results = results['architecture_results']
            scores = [r['score'] for r in arch_results]
            accuracies = [r['metrics']['accuracy'] for r in arch_results]
            inference_times = [r['metrics']['inference_time_ms'] for r in arch_results]
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Score distribution
            ax1.hist(scores, bins=20, alpha=0.7)
            ax1.set_title('Architecture Score Distribution')
            ax1.set_xlabel('Score')
            ax1.set_ylabel('Frequency')
            
            # Accuracy vs Inference Time
            ax2.scatter(inference_times, accuracies, alpha=0.6)
            ax2.set_title('Accuracy vs Inference Time')
            ax2.set_xlabel('Inference Time (ms)')
            ax2.set_ylabel('Accuracy')
            
            # Score vs Accuracy
            ax3.scatter(accuracies, scores, alpha=0.6)
            ax3.set_title('Score vs Accuracy')
            ax3.set_xlabel('Accuracy')
            ax3.set_ylabel('Score')
            
            # Architecture complexity analysis
            complexities = [sum(r['architecture']['hidden_dims']) for r in arch_results]
            ax4.scatter(complexities, scores, alpha=0.6)
            ax4.set_title('Score vs Architecture Complexity')
            ax4.set_xlabel('Total Hidden Units')
            ax4.set_ylabel('Score')
            
            plt.tight_layout()
            plt.savefig(output_path / 'architecture_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_text_report(self, results: Dict, output_path: Path):
        """Create detailed text report."""
        
        with open(output_path / 'optimization_report.txt', 'w') as f:
            f.write("AI/ML MODEL OPTIMIZATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Optimization Type: {self.config.optimization_type}\n")
            f.write(f"Target Metric: {self.config.target_metric}\n")
            f.write(f"Optimization Budget: {self.config.optimization_budget}\n")
            f.write(f"Hardware Target: {self.config.hardware_target}\n\n")
            
            if 'best_parameters' in results:
                f.write("BEST HYPERPARAMETERS:\n")
                f.write("-" * 25 + "\n")
                for param, value in results['best_parameters'].items():
                    f.write(f"{param}: {value}\n")
                f.write(f"\nBest Validation Score: {results['best_value']:.4f}\n\n")
            
            if 'final_metrics' in results:
                f.write("FINAL MODEL METRICS:\n")
                f.write("-" * 20 + "\n")
                metrics = results['final_metrics']
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
                f.write(f"Inference Time: {metrics['inference_time_ms']:.2f} ms\n")
                f.write(f"Memory Usage: {metrics['memory_usage_mb']:.2f} MB\n")
                f.write(f"Model Size: {metrics['model_size_mb']:.2f} MB\n")
                f.write(f"Throughput: {metrics['throughput_fps']:.1f} FPS\n\n")
            
            if 'study_summary' in results:
                f.write("OPTIMIZATION SUMMARY:\n")
                f.write("-" * 22 + "\n")
                summary = results['study_summary']
                f.write(f"Total Trials: {summary['n_trials']}\n")
                f.write(f"Best Trial: #{summary['best_trial']}\n")
                f.write(f"Optimization Time: {summary['optimization_time']:.2f} seconds\n")


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 512
    n_classes = 50
    
    # Create synthetic face embeddings
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Setup optimization configuration
    config = OptimizationConfig(
        optimization_type='hyperparameter',
        target_metric='balanced',
        optimization_budget=20,
        target_accuracy_threshold=0.8,
        target_speed_improvement=2.0,
        memory_constraint_mb=100.0,
        hardware_target='gpu'
    )
    
    # Run optimization
    optimizer = ModelOptimizer(config)
    
    print("Running hyperparameter optimization...")
    hp_results = optimizer.optimize_hyperparameters(X_train, y_train, X_val, y_val)
    
    print("Running architecture search...")
    arch_results = optimizer.optimize_architecture(X_train, y_train, X_val, y_val)
    
    # Generate comprehensive report
    all_results = {
        "hyperparameter_optimization": hp_results,
        "architecture_search": arch_results,
        "configuration": asdict(config)
    }
    
    optimizer.generate_optimization_report(all_results)
    
    print("Optimization complete! Check optimization_results/ directory for detailed reports.")