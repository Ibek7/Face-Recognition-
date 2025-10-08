"""
Model evaluation and validation system for face recognition models.
Provides comprehensive metrics, cross-validation, and performance analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
import time

# ML and evaluation imports
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_curve, auc,
        precision_recall_curve, average_precision_score
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    import torch
    import torch.nn.functional as F
except ImportError:
    # Mock for development
    accuracy_score = precision_score = recall_score = f1_score = lambda *args, **kwargs: 0.0
    confusion_matrix = lambda *args, **kwargs: np.array([[1, 0], [0, 1]])
    classification_report = lambda *args, **kwargs: "Mock report"
    roc_curve = auc = lambda *args, **kwargs: ([0, 1], [0, 1], 0.5)
    StratifiedKFold = None
    torch = None

# Import face recognition components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from database import DatabaseManager
    from embeddings import FaceEmbeddingManager
    from training import TrainingConfig, FaceRecognitionModel, FaceRecognitionDataset
    from monitoring import performance_monitor
except ImportError:
    # Mock for development
    DatabaseManager = object
    FaceEmbeddingManager = object
    TrainingConfig = object
    FaceRecognitionModel = object
    FaceRecognitionDataset = object
    performance_monitor = object

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    average_precision: float
    confusion_matrix: np.ndarray
    classification_report: str
    inference_time_mean: float
    inference_time_std: float
    model_size_mb: float
    memory_usage_mb: float

class ModelEvaluator:
    """
    Comprehensive model evaluation system.
    """
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            config_path: Path to training configuration
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else None
        
        # Load model and config
        self.model = None
        self.config = None
        self.device = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.embedding_manager = FaceEmbeddingManager()
        
        # Evaluation results
        self.evaluation_results = {}
        
        self._load_model()
        
    def _load_model(self):
        """Load trained model and configuration."""
        try:
            if torch and self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Load config
                if 'config' in checkpoint:
                    self.config = TrainingConfig(**checkpoint['config'])
                elif self.config_path and self.config_path.exists():
                    with open(self.config_path, 'r') as f:
                        config_dict = json.load(f)
                    self.config = TrainingConfig(**config_dict)
                else:
                    self.config = TrainingConfig()  # Default config
                
                # Create and load model
                self.model = FaceRecognitionModel(
                    num_classes=10,  # Will be updated based on data
                    embedding_dim=self.config.embedding_dim,
                    backbone=self.config.pretrained_backbone,
                    mode=self.config.model_type
                )
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.warning("Model file not found or PyTorch not available")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
    
    def prepare_test_data(self, test_split: float = 0.2) -> Tuple[List, List]:
        """
        Prepare test dataset from database.
        
        Args:
            test_split: Fraction of data to use for testing
            
        Returns:
            Tuple of (test_data, labels)
        """
        logger.info("Preparing test dataset...")
        
        # Get all persons with sufficient samples
        persons = self.db_manager.list_persons()
        test_data = []
        labels = []
        
        for person in persons:
            embeddings = self.db_manager.get_person_embeddings(person.id)
            
            if len(embeddings) >= 3:  # Minimum for testing
                # Use portion for testing
                test_count = max(1, int(len(embeddings) * test_split))
                test_embeddings = np.random.choice(embeddings, test_count, replace=False)
                
                for embedding in test_embeddings:
                    test_data.append({
                        'person_id': person.id,
                        'person_name': person.name,
                        'embedding_id': embedding.id,
                        'image_path': embedding.source_image_path,
                        'quality_score': embedding.quality_score,
                        'embedding_vector': embedding.embedding
                    })
                    labels.append(person.name)
        
        logger.info(f"Prepared test dataset with {len(test_data)} samples")
        return test_data, labels
    
    def evaluate_classification_accuracy(self, test_data: List, labels: List) -> Dict:
        """
        Evaluate classification accuracy.
        
        Args:
            test_data: Test data samples
            labels: True labels
            
        Returns:
            Evaluation metrics dictionary
        """
        if not self.model or not torch:
            logger.warning("Model not available for evaluation")
            return {"accuracy": 0.0, "error": "Model not available"}
        
        logger.info("Evaluating classification accuracy...")
        
        predicted_labels = []
        predicted_probs = []
        inference_times = []
        
        # Encode labels
        label_encoder = LabelEncoder()
        true_labels_encoded = label_encoder.fit_transform(labels)
        
        with torch.no_grad():
            for sample in test_data:
                start_time = time.time()
                
                # Load and preprocess image
                # For now, use stored embedding
                if isinstance(sample.get('embedding_vector'), np.ndarray):
                    # Use pre-computed embedding
                    embedding = sample['embedding_vector']
                    
                    # Find most similar in database
                    similar_faces = self.db_manager.search_similar_faces(
                        embedding, threshold=0.5, top_k=1
                    )
                    
                    if similar_faces:
                        best_match, similarity = similar_faces[0]
                        person = self.db_manager.get_person(best_match.person_id)
                        predicted_label = person.name if person else "unknown"
                        confidence = similarity
                    else:
                        predicted_label = "unknown"
                        confidence = 0.0
                    
                    predicted_labels.append(predicted_label)
                    predicted_probs.append(confidence)
                    
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
        
        # Calculate metrics
        predicted_labels_encoded = []
        for pred in predicted_labels:
            try:
                encoded = label_encoder.transform([pred])[0]
                predicted_labels_encoded.append(encoded)
            except ValueError:
                # Unknown label
                predicted_labels_encoded.append(-1)
        
        # Filter out unknown predictions for metric calculation
        valid_indices = [i for i, pred in enumerate(predicted_labels_encoded) if pred != -1]
        
        if valid_indices:
            valid_true = [true_labels_encoded[i] for i in valid_indices]
            valid_pred = [predicted_labels_encoded[i] for i in valid_indices]
            
            accuracy = accuracy_score(valid_true, valid_pred)
            precision = precision_score(valid_true, valid_pred, average='weighted', zero_division=0)
            recall = recall_score(valid_true, valid_pred, average='weighted', zero_division=0)
            f1 = f1_score(valid_true, valid_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(valid_true, valid_pred)
            
            # Classification report
            report = classification_report(
                valid_true, valid_pred, 
                target_names=label_encoder.classes_,
                zero_division=0
            )
        else:
            accuracy = precision = recall = f1 = 0.0
            cm = np.array([[0]])
            report = "No valid predictions"
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "inference_time_mean": np.mean(inference_times),
            "inference_time_std": np.std(inference_times),
            "total_samples": len(test_data),
            "valid_predictions": len(valid_indices)
        }
        
        logger.info(f"Classification accuracy: {accuracy:.4f}")
        return metrics
    
    def evaluate_face_verification(self, test_data: List) -> Dict:
        """
        Evaluate face verification performance (1:1 matching).
        
        Args:
            test_data: Test data samples
            
        Returns:
            Verification metrics
        """
        logger.info("Evaluating face verification...")
        
        verification_results = []
        thresholds = np.arange(0.1, 1.0, 0.1)
        
        # Generate positive and negative pairs
        positive_pairs = []
        negative_pairs = []
        
        # Group by person
        person_groups = {}
        for sample in test_data:
            person_name = sample['person_name']
            if person_name not in person_groups:
                person_groups[person_name] = []
            person_groups[person_name].append(sample)
        
        # Generate positive pairs (same person)
        for person_name, samples in person_groups.items():
            if len(samples) > 1:
                for i in range(len(samples)):
                    for j in range(i + 1, len(samples)):
                        positive_pairs.append((samples[i], samples[j], 1))
        
        # Generate negative pairs (different persons)
        person_list = list(person_groups.keys())
        for i, person1 in enumerate(person_list):
            for j, person2 in enumerate(person_list):
                if i != j:
                    sample1 = np.random.choice(person_groups[person1])
                    sample2 = np.random.choice(person_groups[person2])
                    negative_pairs.append((sample1, sample2, 0))
        
        # Limit number of negative pairs
        if len(negative_pairs) > len(positive_pairs) * 2:
            negative_pairs = np.random.choice(
                negative_pairs, len(positive_pairs) * 2, replace=False
            ).tolist()
        
        all_pairs = positive_pairs + negative_pairs
        logger.info(f"Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs")
        
        # Calculate similarities for all pairs
        similarities = []
        true_labels = []
        
        for sample1, sample2, label in all_pairs:
            # Calculate similarity between embeddings
            emb1 = sample1.get('embedding_vector', np.random.random(128))
            emb2 = sample2.get('embedding_vector', np.random.random(128))
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            similarities.append(similarity)
            true_labels.append(label)
        
        similarities = np.array(similarities)
        true_labels = np.array(true_labels)
        
        # Calculate metrics for different thresholds
        threshold_metrics = []
        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)
            
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)
            
            threshold_metrics.append({
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })
        
        # Find best threshold
        best_metric = max(threshold_metrics, key=lambda x: x['f1_score'])
        
        # ROC curve
        if len(np.unique(true_labels)) > 1:
            fpr, tpr, _ = roc_curve(true_labels, similarities)
            roc_auc = auc(fpr, tpr)
        else:
            fpr, tpr, roc_auc = [0, 1], [0, 1], 0.5
        
        verification_metrics = {
            "best_threshold": best_metric,
            "threshold_metrics": threshold_metrics,
            "roc_auc": roc_auc,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "total_pairs": len(all_pairs),
            "positive_pairs": len(positive_pairs),
            "negative_pairs": len(negative_pairs)
        }
        
        logger.info(f"Best verification F1-score: {best_metric['f1_score']:.4f} at threshold {best_metric['threshold']:.2f}")
        return verification_metrics
    
    def evaluate_model_performance(self) -> Dict:
        """
        Evaluate overall model performance and characteristics.
        
        Returns:
            Performance metrics
        """
        logger.info("Evaluating model performance characteristics...")
        
        performance_metrics = {}
        
        if self.model and torch:
            # Model size
            model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
            performance_metrics["model_size_mb"] = model_size
            
            # Parameter count
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            performance_metrics["total_parameters"] = total_params
            performance_metrics["trainable_parameters"] = trainable_params
            
            # Memory usage estimation
            dummy_input = torch.randn(1, 3, 112, 112).to(self.device)
            
            # Measure inference time
            inference_times = []
            with torch.no_grad():
                for _ in range(100):
                    start_time = time.time()
                    _ = self.model(dummy_input)
                    inference_times.append(time.time() - start_time)
            
            performance_metrics["inference_time_mean"] = np.mean(inference_times)
            performance_metrics["inference_time_std"] = np.std(inference_times)
            performance_metrics["fps"] = 1.0 / np.mean(inference_times)
            
        else:
            performance_metrics = {
                "model_size_mb": 0.0,
                "total_parameters": 0,
                "trainable_parameters": 0,
                "inference_time_mean": 0.0,
                "inference_time_std": 0.0,
                "fps": 0.0
            }
        
        return performance_metrics
    
    def cross_validate_model(self, k_folds: int = 5) -> Dict:
        """
        Perform k-fold cross-validation.
        
        Args:
            k_folds: Number of folds for cross-validation
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {k_folds}-fold cross-validation...")
        
        # Prepare full dataset
        test_data, labels = self.prepare_test_data(test_split=1.0)
        
        if len(test_data) < k_folds:
            logger.warning("Not enough data for cross-validation")
            return {"error": "Insufficient data"}
        
        # Group data by person for stratified split
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        cv_results = {
            "fold_accuracies": [],
            "fold_f1_scores": [],
            "fold_precisions": [],
            "fold_recalls": []
        }
        
        if StratifiedKFold:
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(test_data, encoded_labels)):
                logger.info(f"Evaluating fold {fold + 1}/{k_folds}")
                
                # Split data
                fold_test_data = [test_data[i] for i in test_idx]
                fold_test_labels = [labels[i] for i in test_idx]
                
                # Evaluate fold
                fold_metrics = self.evaluate_classification_accuracy(fold_test_data, fold_test_labels)
                
                cv_results["fold_accuracies"].append(fold_metrics["accuracy"])
                cv_results["fold_f1_scores"].append(fold_metrics["f1_score"])
                cv_results["fold_precisions"].append(fold_metrics["precision"])
                cv_results["fold_recalls"].append(fold_metrics["recall"])
        
        # Calculate statistics
        if cv_results["fold_accuracies"]:
            cv_results["mean_accuracy"] = np.mean(cv_results["fold_accuracies"])
            cv_results["std_accuracy"] = np.std(cv_results["fold_accuracies"])
            cv_results["mean_f1"] = np.mean(cv_results["fold_f1_scores"])
            cv_results["std_f1"] = np.std(cv_results["fold_f1_scores"])
        
        logger.info(f"Cross-validation completed. Mean accuracy: {cv_results.get('mean_accuracy', 0):.4f}")
        return cv_results
    
    @performance_monitor.time_function("model_evaluation")
    def evaluate_comprehensive(self, output_dir: str = "evaluation_results") -> Dict:
        """
        Perform comprehensive model evaluation.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting comprehensive model evaluation...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare test data
        test_data, labels = self.prepare_test_data()
        
        if not test_data:
            logger.warning("No test data available")
            return {"error": "No test data available"}
        
        # Perform evaluations
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_path": str(self.model_path),
            "config": self.config.__dict__ if self.config else {},
            "test_data_size": len(test_data)
        }
        
        # Classification accuracy
        results["classification_metrics"] = self.evaluate_classification_accuracy(test_data, labels)
        
        # Face verification
        results["verification_metrics"] = self.evaluate_face_verification(test_data)
        
        # Model performance
        results["performance_metrics"] = self.evaluate_model_performance()
        
        # Cross-validation
        results["cross_validation"] = self.cross_validate_model()
        
        # Save results
        results_path = output_path / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate visualizations
        self.generate_evaluation_plots(results, output_path)
        
        # Generate report
        self.generate_evaluation_report(results, output_path)
        
        logger.info(f"Comprehensive evaluation completed. Results saved to {output_path}")
        return results
    
    def generate_evaluation_plots(self, results: Dict, output_dir: Path):
        """Generate visualization plots for evaluation results."""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
        # Confusion Matrix
        if "classification_metrics" in results and "confusion_matrix" in results["classification_metrics"]:
            cm = np.array(results["classification_metrics"]["confusion_matrix"])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # ROC Curve
        if "verification_metrics" in results and "fpr" in results["verification_metrics"]:
            fpr = results["verification_metrics"]["fpr"]
            tpr = results["verification_metrics"]["tpr"]
            roc_auc = results["verification_metrics"]["roc_auc"]
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'r--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Face Verification')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Cross-validation results
        if "cross_validation" in results and "fold_accuracies" in results["cross_validation"]:
            accuracies = results["cross_validation"]["fold_accuracies"]
            
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.bar(range(1, len(accuracies) + 1), accuracies)
            plt.xlabel('Fold')
            plt.ylabel('Accuracy')
            plt.title('Cross-Validation Accuracies')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.boxplot([accuracies], labels=['Accuracy'])
            plt.ylabel('Score')
            plt.title('Cross-Validation Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "cross_validation.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Evaluation plots generated successfully")
    
    def generate_evaluation_report(self, results: Dict, output_dir: Path):
        """Generate comprehensive evaluation report."""
        report_lines = [
            "# Face Recognition Model Evaluation Report",
            f"\n**Generated on:** {results.get('timestamp', 'Unknown')}",
            f"**Model Path:** {results.get('model_path', 'Unknown')}",
            f"**Test Data Size:** {results.get('test_data_size', 0)} samples\n",
            
            "## Classification Performance",
            "| Metric | Value |",
            "|--------|-------|"
        ]
        
        # Classification metrics
        if "classification_metrics" in results:
            cm = results["classification_metrics"]
            report_lines.extend([
                f"| Accuracy | {cm.get('accuracy', 0):.4f} |",
                f"| Precision | {cm.get('precision', 0):.4f} |",
                f"| Recall | {cm.get('recall', 0):.4f} |",
                f"| F1-Score | {cm.get('f1_score', 0):.4f} |",
                f"| Inference Time (mean) | {cm.get('inference_time_mean', 0):.4f}s |",
                f"| Valid Predictions | {cm.get('valid_predictions', 0)}/{cm.get('total_samples', 0)} |"
            ])
        
        # Verification metrics
        if "verification_metrics" in results:
            vm = results["verification_metrics"]
            best = vm.get("best_threshold", {})
            
            report_lines.extend([
                "\n## Face Verification Performance",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Best Threshold | {best.get('threshold', 0):.2f} |",
                f"| Best F1-Score | {best.get('f1_score', 0):.4f} |",
                f"| ROC AUC | {vm.get('roc_auc', 0):.4f} |",
                f"| Total Pairs | {vm.get('total_pairs', 0)} |"
            ])
        
        # Performance metrics
        if "performance_metrics" in results:
            pm = results["performance_metrics"]
            
            report_lines.extend([
                "\n## Model Performance",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Model Size | {pm.get('model_size_mb', 0):.2f} MB |",
                f"| Total Parameters | {pm.get('total_parameters', 0):,} |",
                f"| Inference FPS | {pm.get('fps', 0):.1f} |"
            ])
        
        # Cross-validation
        if "cross_validation" in results and "mean_accuracy" in results["cross_validation"]:
            cv = results["cross_validation"]
            
            report_lines.extend([
                "\n## Cross-Validation Results",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Mean Accuracy | {cv.get('mean_accuracy', 0):.4f} ± {cv.get('std_accuracy', 0):.4f} |",
                f"| Mean F1-Score | {cv.get('mean_f1', 0):.4f} ± {cv.get('std_f1', 0):.4f} |"
            ])
        
        # Save report
        report_path = output_dir / "evaluation_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Evaluation report saved to {report_path}")

def evaluate_model(model_path: str, config_path: str = None, output_dir: str = "evaluation_results") -> Dict:
    """
    Evaluate a trained face recognition model.
    
    Args:
        model_path: Path to trained model file
        config_path: Path to training configuration
        output_dir: Directory to save evaluation results
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = ModelEvaluator(model_path, config_path)
    results = evaluator.evaluate_comprehensive(output_dir)
    
    return results

if __name__ == "__main__":
    # Example usage
    model_path = "models/training/best_model.pth"
    results = evaluate_model(model_path)
    print(f"Evaluation completed. Accuracy: {results.get('classification_metrics', {}).get('accuracy', 0):.4f}")