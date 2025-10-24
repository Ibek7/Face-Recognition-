# Machine Learning Pipeline Framework

import threading
from typing import List, Dict, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

class ModelStatus(Enum):
    """Model lifecycle status."""
    DRAFT = "DRAFT"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    EVALUATING = "EVALUATING"
    DEPLOYED = "DEPLOYED"
    ARCHIVED = "ARCHIVED"

@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc: float = 0.0
    additional_metrics: Dict[str, float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc': self.auc,
            'additional_metrics': self.additional_metrics or {}
        }

@dataclass
class ModelVersion:
    """Model version information."""
    version_id: str
    model_name: str
    version_number: int
    status: ModelStatus = ModelStatus.DRAFT
    created_at: float = None
    updated_at: float = None
    metrics: ModelMetrics = None
    hyperparameters: Dict = None
    training_data_size: int = 0
    inference_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'version_id': self.version_id,
            'model_name': self.model_name,
            'version_number': self.version_number,
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'metrics': self.metrics.to_dict() if self.metrics else {},
            'hyperparameters': self.hyperparameters or {},
            'training_data_size': self.training_data_size,
            'inference_latency_ms': self.inference_latency_ms
        }

class DataPreprocessor:
    """Data preprocessing for ML."""
    
    def __init__(self):
        self.transformers: List[Callable] = []
    
    def add_transformer(self, transformer: Callable) -> None:
        """Add preprocessing transformer."""
        self.transformers.append(transformer)
    
    def preprocess(self, data: List[Dict]) -> List[Dict]:
        """Apply preprocessing."""
        result = data
        
        for transformer in self.transformers:
            result = [transformer(item) for item in result]
        
        return result
    
    def normalize(self, data: List[float], min_val: float = None,
                 max_val: float = None) -> List[float]:
        """Normalize data."""
        if min_val is None:
            min_val = min(data)
        if max_val is None:
            max_val = max(data)
        
        if max_val == min_val:
            return [0.0] * len(data)
        
        return [(x - min_val) / (max_val - min_val) for x in data]
    
    def standardize(self, data: List[float]) -> List[float]:
        """Standardize data (z-score)."""
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        
        if variance == 0:
            return [0.0] * len(data)
        
        std_dev = variance ** 0.5
        return [(x - mean) / std_dev for x in data]

class FeatureEngineer:
    """Feature engineering."""
    
    @staticmethod
    def create_polynomial_features(features: List[float],
                                  degree: int = 2) -> List[float]:
        """Create polynomial features."""
        poly_features = []
        for i in range(1, degree + 1):
            for x in features:
                poly_features.append(x ** i)
        return poly_features
    
    @staticmethod
    def create_interaction_features(features: Dict[str, float]) -> Dict[str, float]:
        """Create interaction features."""
        interactions = {}
        feature_list = list(features.items())
        
        for i, (k1, v1) in enumerate(feature_list):
            for k2, v2 in feature_list[i + 1:]:
                interactions[f"{k1}_{k2}"] = v1 * v2
        
        return {**features, **interactions}
    
    @staticmethod
    def select_features(features: Dict[str, float],
                       importance_scores: Dict[str, float],
                       threshold: float = 0.1) -> Dict[str, float]:
        """Select features by importance."""
        return {
            k: v for k, v in features.items()
            if importance_scores.get(k, 0) >= threshold
        }

class ModelTrainer:
    """Model training pipeline."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.versions: Dict[str, ModelVersion] = {}
        self.current_version = None
        self.lock = threading.RLock()
    
    def create_version(self, version_number: int,
                      hyperparameters: Dict = None) -> ModelVersion:
        """Create new model version."""
        import time
        import uuid
        
        version_id = str(uuid.uuid4())
        
        version = ModelVersion(
            version_id=version_id,
            model_name=self.model_name,
            version_number=version_number,
            status=ModelStatus.DRAFT,
            created_at=time.time(),
            hyperparameters=hyperparameters or {}
        )
        
        with self.lock:
            self.versions[version_id] = version
        
        return version
    
    def train(self, version_id: str, training_data: List[Any],
             training_func: Callable) -> bool:
        """Train model."""
        import time
        
        with self.lock:
            if version_id not in self.versions:
                return False
            
            version = self.versions[version_id]
            version.status = ModelStatus.TRAINING
        
        try:
            start_time = time.time()
            
            # Call training function
            metrics = training_func(training_data, version.hyperparameters)
            
            elapsed = time.time() - start_time
            
            with self.lock:
                version.metrics = ModelMetrics(**metrics) if metrics else ModelMetrics()
                version.training_data_size = len(training_data)
                version.status = ModelStatus.TRAINED
                version.updated_at = time.time()
            
            return True
        
        except Exception as e:
            print(f"Training error: {e}")
            with self.lock:
                version.status = ModelStatus.DRAFT
            return False
    
    def evaluate(self, version_id: str, test_data: List[Any],
                eval_func: Callable) -> Dict:
        """Evaluate model."""
        import time
        
        with self.lock:
            if version_id not in self.versions:
                return {}
            
            version = self.versions[version_id]
            version.status = ModelStatus.EVALUATING
        
        try:
            metrics = eval_func(test_data)
            
            with self.lock:
                version.metrics = ModelMetrics(**metrics) if metrics else version.metrics
                version.status = ModelStatus.TRAINED
            
            return metrics or {}
        
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {}
    
    def deploy(self, version_id: str) -> bool:
        """Deploy model."""
        import time
        
        with self.lock:
            if version_id not in self.versions:
                return False
            
            version = self.versions[version_id]
            version.status = ModelStatus.DEPLOYED
            version.updated_at = time.time()
            self.current_version = version_id
        
        return True
    
    def get_versions(self) -> List[ModelVersion]:
        """Get all versions."""
        with self.lock:
            return list(self.versions.values())
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get specific version."""
        with self.lock:
            return self.versions.get(version_id)

class ModelServingEngine:
    """Serve trained models."""
    
    def __init__(self):
        self.models: Dict[str, Callable] = {}
        self.inference_cache: Dict[str, Any] = {}
        self.lock = threading.RLock()
    
    def register_model(self, model_name: str, predict_func: Callable) -> None:
        """Register model for serving."""
        with self.lock:
            self.models[model_name] = predict_func
    
    def predict(self, model_name: str, input_data: Any) -> Optional[Any]:
        """Make prediction."""
        with self.lock:
            if model_name not in self.models:
                return None
            
            predict_func = self.models[model_name]
        
        return predict_func(input_data)
    
    def batch_predict(self, model_name: str, inputs: List[Any]) -> List[Any]:
        """Batch prediction."""
        results = []
        
        for input_data in inputs:
            result = self.predict(model_name, input_data)
            if result is not None:
                results.append(result)
        
        return results

class HyperparameterTuner:
    """Hyperparameter optimization."""
    
    @staticmethod
    def grid_search(param_grid: Dict[str, List],
                   train_func: Callable) -> Tuple[Dict, float]:
        """Grid search optimization."""
        best_params = {}
        best_score = 0
        
        import itertools
        
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))
            score = train_func(params)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params, best_score
    
    @staticmethod
    def random_search(param_ranges: Dict[str, Tuple],
                     train_func: Callable,
                     iterations: int = 10) -> Tuple[Dict, float]:
        """Random search optimization."""
        import random
        
        best_params = {}
        best_score = 0
        
        for _ in range(iterations):
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                params[param_name] = random.uniform(min_val, max_val)
            
            score = train_func(params)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params, best_score

class ModelRegistry:
    """Central model registry."""
    
    def __init__(self):
        self.trainers: Dict[str, ModelTrainer] = {}
        self.lock = threading.RLock()
    
    def register_model(self, model_name: str) -> ModelTrainer:
        """Register model."""
        with self.lock:
            if model_name not in self.trainers:
                self.trainers[model_name] = ModelTrainer(model_name)
            return self.trainers[model_name]
    
    def get_model(self, model_name: str) -> Optional[ModelTrainer]:
        """Get model trainer."""
        with self.lock:
            return self.trainers.get(model_name)
    
    def get_deployed_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get deployed version."""
        with self.lock:
            trainer = self.trainers.get(model_name)
            if trainer and trainer.current_version:
                return trainer.versions.get(trainer.current_version)
        return None

# Example usage
if __name__ == "__main__":
    # Create trainer
    registry = ModelRegistry()
    trainer = registry.register_model("face_recognition_model")
    
    # Create version
    version = trainer.create_version(
        version_number=1,
        hyperparameters={'learning_rate': 0.001, 'epochs': 100}
    )
    
    print(f"Created version: {version.version_id}")
    print(f"Status: {version.status.value}")
    
    # Mock training
    def mock_train(data, params):
        return {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.93,
            'f1_score': 0.925,
            'auc': 0.97
        }
    
    # Train
    success = trainer.train(version.version_id, [], mock_train)
    print(f"Training: {success}")
    
    # Get version info
    trained_version = trainer.get_version(version.version_id)
    print(f"Trained version: {json.dumps(trained_version.to_dict(), indent=2, default=str)}")
