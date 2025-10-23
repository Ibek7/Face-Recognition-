# Adaptive Learning & Model Evolution Framework

import threading
import time
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

class AdaptationStrategy(Enum):
    """Model adaptation strategies."""
    INCREMENTAL = "incremental"  # Continuous learning
    DRIFT_DETECTION = "drift_detection"  # React to distribution shift
    ACTIVE_LEARNING = "active_learning"  # Learn from uncertain samples
    ONLINE = "online"  # Real-time updates

class ModelQuality(Enum):
    """Model quality indicators."""
    EXCELLENT = "excellent"  # > 95% accuracy
    GOOD = "good"  # 85-95% accuracy
    ACCEPTABLE = "acceptable"  # 75-85% accuracy
    POOR = "poor"  # < 75% accuracy

@dataclass
class ModelVersion:
    """Version information for model."""
    version_id: str
    timestamp: float = field(default_factory=time.time)
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    samples_trained: int = 0
    training_duration_sec: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'version_id': self.version_id,
            'timestamp': self.timestamp,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'samples_trained': self.samples_trained,
            'training_duration_sec': self.training_duration_sec
        }

@dataclass
class PredictionResult:
    """Prediction with confidence and metadata."""
    prediction_id: str
    prediction: Any
    confidence: float
    timestamp: float = field(default_factory=time.time)
    uncertainty: float = 0.0
    is_uncertain: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'prediction_id': self.prediction_id,
            'prediction': str(self.prediction),
            'confidence': self.confidence,
            'uncertainty': self.uncertainty,
            'is_uncertain': self.is_uncertain,
            'timestamp': self.timestamp
        }

class ConceptDriftDetector:
    """Detect distribution shifts in data."""
    
    def __init__(self, window_size: int = 100, drift_threshold: float = 0.15):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.prediction_window: deque = deque(maxlen=window_size)
        self.performance_history: List[float] = []
        self.lock = threading.RLock()
    
    def update(self, predicted: Any, actual: Any) -> bool:
        """Update detector with new prediction."""
        is_correct = predicted == actual
        
        with self.lock:
            self.prediction_window.append(is_correct)
            
            if len(self.prediction_window) == self.window_size:
                accuracy = sum(self.prediction_window) / self.window_size
                self.performance_history.append(accuracy)
        
        return self._detect_drift()
    
    def _detect_drift(self) -> bool:
        """Detect concept drift."""
        with self.lock:
            if len(self.performance_history) < 2:
                return False
            
            recent_accuracy = self.performance_history[-1]
            previous_accuracy = self.performance_history[-2]
            
            drift = abs(recent_accuracy - previous_accuracy)
            return drift > self.drift_threshold
    
    def get_status(self) -> Dict:
        """Get drift detection status."""
        with self.lock:
            current_accuracy = (
                sum(self.prediction_window) / len(self.prediction_window)
                if self.prediction_window else 0
            )
            
            return {
                'current_accuracy': current_accuracy,
                'drift_detected': self._detect_drift(),
                'history_length': len(self.performance_history),
                'window_size': len(self.prediction_window)
            }

class UncertaintyEstimator:
    """Estimate prediction uncertainty."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_history: Dict[str, List[float]] = {}
        self.lock = threading.RLock()
    
    def estimate_uncertainty(self, prediction: Any, confidence: float,
                            ensemble_predictions: List[Any] = None) -> Tuple[float, bool]:
        """Estimate prediction uncertainty."""
        base_uncertainty = 1.0 - confidence
        
        # Check agreement in ensemble
        if ensemble_predictions:
            agreement = sum(1 for p in ensemble_predictions if p == prediction) / len(ensemble_predictions)
            base_uncertainty *= (1 - agreement)
        
        is_uncertain = base_uncertainty > (1 - self.confidence_threshold)
        
        return base_uncertainty, is_uncertain
    
    def record_uncertainty(self, model_id: str, uncertainty: float) -> None:
        """Record uncertainty for analysis."""
        with self.lock:
            if model_id not in self.uncertainty_history:
                self.uncertainty_history[model_id] = []
            
            self.uncertainty_history[model_id].append(uncertainty)

class AdaptiveModelManager:
    """Manage adaptive model learning."""
    
    def __init__(self, model_id: str, strategy: AdaptationStrategy = AdaptationStrategy.INCREMENTAL):
        self.model_id = model_id
        self.strategy = strategy
        self.current_version: Optional[ModelVersion] = None
        self.version_history: List[ModelVersion] = []
        self.drift_detector = ConceptDriftDetector()
        self.uncertainty_estimator = UncertaintyEstimator()
        
        self.prediction_buffer: List[Tuple[Any, Any]] = []
        self.retraining_threshold = 50  # Retrain after N predictions
        
        self.lock = threading.RLock()
    
    def make_prediction(self, input_data: Any, model_func: Callable,
                       ensemble_funcs: List[Callable] = None) -> PredictionResult:
        """Make adaptive prediction."""
        prediction = model_func(input_data)
        
        # Get confidence
        confidence = self._get_confidence(prediction)
        
        # Estimate uncertainty
        ensemble_preds = None
        if ensemble_funcs:
            ensemble_preds = [f(input_data) for f in ensemble_funcs]
        
        uncertainty, is_uncertain = self.uncertainty_estimator.estimate_uncertainty(
            prediction, confidence, ensemble_preds
        )
        
        result = PredictionResult(
            prediction_id=self._generate_id(),
            prediction=prediction,
            confidence=confidence,
            uncertainty=uncertainty,
            is_uncertain=is_uncertain
        )
        
        # Store for active learning
        if is_uncertain:
            with self.lock:
                self.prediction_buffer.append((input_data, None))  # Queued for labeling
        
        return result
    
    def update_with_ground_truth(self, prediction: Any, actual: Any) -> None:
        """Update model with ground truth."""
        # Check for drift
        drift_detected = self.drift_detector.update(prediction, actual)
        
        with self.lock:
            self.prediction_buffer.append((prediction, actual))
            
            # Trigger retraining if needed
            if drift_detected or len(self.prediction_buffer) >= self.retraining_threshold:
                self._trigger_retraining()
    
    def _trigger_retraining(self) -> None:
        """Trigger model retraining."""
        # Implementation would call actual retraining logic
        new_version = ModelVersion(
            version_id=self._generate_id(),
            samples_trained=len(self.prediction_buffer)
        )
        
        with self.lock:
            self.current_version = new_version
            self.version_history.append(new_version)
            self.prediction_buffer.clear()
    
    def get_model_quality(self) -> ModelQuality:
        """Assess model quality."""
        if not self.current_version:
            return ModelQuality.POOR
        
        accuracy = self.current_version.accuracy
        
        if accuracy > 0.95:
            return ModelQuality.EXCELLENT
        elif accuracy > 0.85:
            return ModelQuality.GOOD
        elif accuracy > 0.75:
            return ModelQuality.ACCEPTABLE
        else:
            return ModelQuality.POOR
    
    def _get_confidence(self, prediction: Any) -> float:
        """Extract confidence from prediction."""
        if isinstance(prediction, tuple) and len(prediction) > 1:
            return float(prediction[1])
        return 0.7  # Default confidence
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        import hashlib
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    def get_status(self) -> Dict:
        """Get adaptive learning status."""
        with self.lock:
            return {
                'model_id': self.model_id,
                'strategy': self.strategy.value,
                'current_version': self.current_version.to_dict() if self.current_version else None,
                'version_count': len(self.version_history),
                'model_quality': self.get_model_quality().value,
                'drift_status': self.drift_detector.get_status(),
                'predictions_buffered': len(self.prediction_buffer)
            }

class ModelEnsembleEvolver:
    """Evolve ensemble of models."""
    
    def __init__(self, max_models: int = 5):
        self.max_models = max_models
        self.models: Dict[str, AdaptiveModelManager] = {}
        self.model_weights: Dict[str, float] = {}
        self.lock = threading.RLock()
    
    def register_model(self, model_id: str, manager: AdaptiveModelManager) -> None:
        """Register model in ensemble."""
        with self.lock:
            if len(self.models) >= self.max_models:
                # Remove worst performing model
                self._remove_worst_model()
            
            self.models[model_id] = manager
            self.model_weights[model_id] = 1.0 / (len(self.models) + 1)
    
    def _remove_worst_model(self) -> None:
        """Remove worst performing model."""
        worst_id = min(
            self.models.keys(),
            key=lambda m_id: self.models[m_id].current_version.accuracy
            if self.models[m_id].current_version else 0
        )
        del self.models[worst_id]
        del self.model_weights[worst_id]
    
    def make_ensemble_prediction(self, input_data: Any) -> Any:
        """Make prediction using ensemble."""
        with self.lock:
            predictions = []
            
            for model_id, manager in self.models.items():
                pred = manager.make_prediction(input_data, lambda x: x)
                weight = self.model_weights[model_id]
                predictions.append((pred, weight))
            
            # Weighted voting
            if predictions:
                best_pred = max(predictions, key=lambda x: x[0].confidence * x[1])
                return best_pred[0].prediction
            
            return None
    
    def get_ensemble_status(self) -> Dict:
        """Get ensemble status."""
        with self.lock:
            return {
                'model_count': len(self.models),
                'models': {
                    m_id: manager.get_status()
                    for m_id, manager in self.models.items()
                },
                'weights': self.model_weights
            }

# Example usage
if __name__ == "__main__":
    # Create adaptive manager
    manager = AdaptiveModelManager("model_v1", AdaptationStrategy.INCREMENTAL)
    
    # Simulate predictions
    def dummy_model(x):
        return ("face_detected", 0.92)
    
    for i in range(10):
        result = manager.make_prediction(f"image_{i}.jpg", dummy_model)
        manager.update_with_ground_truth(result.prediction, ("face_detected", True))
    
    # Get status
    status = manager.get_status()
    print("Adaptive Manager Status:")
    print(json.dumps(status, indent=2, default=str))
