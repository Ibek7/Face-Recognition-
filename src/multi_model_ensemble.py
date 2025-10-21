# Multi-Model Inference and Ensemble System

import logging
import numpy as np
import json
import time
import threading
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict

class EnsembleStrategy(Enum):
    """Ensemble aggregation strategies."""
    VOTING = "voting"
    AVERAGING = "averaging"
    WEIGHTED_AVERAGING = "weighted_averaging"
    MAX_CONFIDENCE = "max_confidence"
    RANKED_VOTING = "ranked_voting"

class ModelType(Enum):
    """Types of face recognition models."""
    DETECTION = "detection"
    ENCODING = "encoding"
    ALIGNMENT = "alignment"
    RECOGNITION = "recognition"

@dataclass
class ModelMetadata:
    """Metadata for a model."""
    model_id: str
    model_name: str
    model_type: ModelType
    version: str
    accuracy: float
    latency_ms: float
    is_active: bool = True
    inference_count: int = 0
    error_count: int = 0
    last_used: float = field(default_factory=time.time)

@dataclass
class InferenceResult:
    """Result from single model inference."""
    model_id: str
    predictions: np.ndarray
    confidence: float
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnsembleResult:
    """Aggregated ensemble result."""
    ensemble_id: str
    predictions: np.ndarray
    confidence: float
    strategy: EnsembleStrategy
    individual_results: List[InferenceResult] = field(default_factory=list)
    latency_ms: float = 0.0
    model_consensus: float = 0.0  # How much models agree

class ModelWrapper:
    """Wrapper for different model implementations."""
    
    def __init__(self, model_id: str, model_name: str, 
                 model_type: ModelType, model_func: Callable):
        self.model_id = model_id
        self.model_name = model_name
        self.model_type = model_type
        self.model_func = model_func
        
        self.metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            version="1.0",
            accuracy=0.95,
            latency_ms=50
        )
        
        self.logger = logging.getLogger(__name__)
    
    def infer(self, input_data: np.ndarray) -> InferenceResult:
        """Run inference with timing."""
        start_time = time.time()
        
        try:
            predictions, confidence = self.model_func(input_data)
            latency_ms = (time.time() - start_time) * 1000
            
            self.metadata.inference_count += 1
            self.metadata.last_used = time.time()
            
            return InferenceResult(
                model_id=self.model_id,
                predictions=predictions,
                confidence=confidence,
                latency_ms=latency_ms
            )
        
        except Exception as e:
            self.metadata.error_count += 1
            self.logger.error(f"Inference error in {self.model_id}: {e}")
            raise
    
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return self.metadata

class EnsembleAggregator:
    """Aggregate predictions from multiple models."""
    
    def __init__(self, strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGING):
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
    
    def aggregate(self, results: List[InferenceResult], 
                  weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Aggregate multiple model predictions."""
        
        if not results:
            raise ValueError("No results to aggregate")
        
        if self.strategy == EnsembleStrategy.VOTING:
            return self._voting_aggregation(results)
        elif self.strategy == EnsembleStrategy.AVERAGING:
            return self._averaging_aggregation(results)
        elif self.strategy == EnsembleStrategy.WEIGHTED_AVERAGING:
            return self._weighted_averaging_aggregation(results, weights)
        elif self.strategy == EnsembleStrategy.MAX_CONFIDENCE:
            return self._max_confidence_aggregation(results)
        elif self.strategy == EnsembleStrategy.RANKED_VOTING:
            return self._ranked_voting_aggregation(results)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _voting_aggregation(self, results: List[InferenceResult]) -> Dict[str, Any]:
        """Simple voting aggregation."""
        predictions_list = [r.predictions.argmax() for r in results]
        consensus = max(predictions_list.count(p) for p in set(predictions_list)) / len(predictions_list)
        
        return {
            'predictions': np.array(predictions_list),
            'consensus': consensus,
            'final_prediction': max(set(predictions_list), 
                                   key=predictions_list.count)
        }
    
    def _averaging_aggregation(self, results: List[InferenceResult]) -> Dict[str, Any]:
        """Average predictions."""
        avg_predictions = np.mean(
            [r.predictions for r in results],
            axis=0
        )
        avg_confidence = np.mean([r.confidence for r in results])
        
        return {
            'predictions': avg_predictions,
            'confidence': avg_confidence,
            'final_prediction': avg_predictions.argmax()
        }
    
    def _weighted_averaging_aggregation(self, results: List[InferenceResult],
                                       weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Weighted averaging based on model accuracy."""
        
        if weights is None:
            # Use model accuracy as weights
            weights = [r.confidence for r in results]
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        weighted_predictions = np.average(
            [r.predictions for r in results],
            axis=0,
            weights=weights
        )
        
        weighted_confidence = np.average(
            [r.confidence for r in results],
            weights=weights
        )
        
        return {
            'predictions': weighted_predictions,
            'confidence': weighted_confidence,
            'final_prediction': weighted_predictions.argmax(),
            'weights': weights.tolist()
        }
    
    def _max_confidence_aggregation(self, results: List[InferenceResult]) -> Dict[str, Any]:
        """Select prediction with highest confidence."""
        best_result = max(results, key=lambda r: r.confidence)
        
        return {
            'predictions': best_result.predictions,
            'confidence': best_result.confidence,
            'final_prediction': best_result.predictions.argmax(),
            'selected_model': best_result.model_id
        }
    
    def _ranked_voting_aggregation(self, results: List[InferenceResult]) -> Dict[str, Any]:
        """Ranking-based voting."""
        rankings = []
        
        for r in results:
            # Get top 3 predictions
            top_indices = np.argsort(r.predictions)[-3:][::-1]
            rankings.append(top_indices)
        
        # Weighted vote based on ranking
        vote_counts = defaultdict(float)
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                vote_counts[idx] += 3 - rank  # First place: 3 points, etc.
        
        final_prediction = max(vote_counts, key=vote_counts.get)
        
        return {
            'predictions': np.array(list(vote_counts.values())),
            'confidence': vote_counts[final_prediction] / (len(rankings) * 3),
            'final_prediction': final_prediction,
            'vote_counts': dict(vote_counts)
        }

class MultiModelInferenceEngine:
    """Engine for multi-model inference with ensemble."""
    
    def __init__(self, ensemble_strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGING):
        self.models: Dict[str, ModelWrapper] = {}
        self.ensemble_strategy = ensemble_strategy
        self.aggregator = EnsembleAggregator(ensemble_strategy)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        self.inference_history: List[EnsembleResult] = []
    
    def register_model(self, wrapper: ModelWrapper):
        """Register a model."""
        with self.lock:
            self.models[wrapper.model_id] = wrapper
            self.logger.info(f"Registered model: {wrapper.model_name}")
    
    def remove_model(self, model_id: str):
        """Remove a model."""
        with self.lock:
            if model_id in self.models:
                del self.models[model_id]
                self.logger.info(f"Removed model: {model_id}")
    
    def infer_single(self, model_id: str, input_data: np.ndarray) -> InferenceResult:
        """Run single model inference."""
        with self.lock:
            if model_id not in self.models:
                raise ValueError(f"Model not found: {model_id}")
            
            model = self.models[model_id]
        
        return model.infer(input_data)
    
    def infer_ensemble(self, input_data: np.ndarray, 
                      model_ids: Optional[List[str]] = None) -> EnsembleResult:
        """Run ensemble inference."""
        
        start_time = time.time()
        
        with self.lock:
            # Use all models if not specified
            if model_ids is None:
                model_ids = list(self.models.keys())
            
            active_models = [self.models[mid] for mid in model_ids 
                           if mid in self.models and self.models[mid].metadata.is_active]
        
        if not active_models:
            raise ValueError("No active models available")
        
        # Run inference with all models
        results = []
        for model in active_models:
            try:
                result = model.infer(input_data)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Model {model.model_id} failed: {e}")
        
        if not results:
            raise RuntimeError("All models failed")
        
        # Aggregate results
        aggregated = self.aggregator.aggregate(results)
        
        # Calculate consensus
        consensus = self._calculate_consensus(results)
        
        latency_ms = (time.time() - start_time) * 1000
        
        ensemble_result = EnsembleResult(
            ensemble_id=f"ensemble_{time.time()}",
            predictions=aggregated['predictions'],
            confidence=aggregated.get('confidence', np.mean([r.confidence for r in results])),
            strategy=self.ensemble_strategy,
            individual_results=results,
            latency_ms=latency_ms,
            model_consensus=consensus
        )
        
        with self.lock:
            self.inference_history.append(ensemble_result)
        
        return ensemble_result
    
    def _calculate_consensus(self, results: List[InferenceResult]) -> float:
        """Calculate how much models agree."""
        if len(results) < 2:
            return 1.0
        
        predictions = [r.predictions.argmax() for r in results]
        consensus_pred = max(set(predictions), key=predictions.count)
        agreement = predictions.count(consensus_pred) / len(predictions)
        
        return agreement
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics for all models."""
        with self.lock:
            stats = {}
            for model_id, model in self.models.items():
                metadata = model.get_metadata()
                error_rate = (metadata.error_count / 
                            (metadata.inference_count + 1)) * 100
                
                stats[model_id] = {
                    'name': metadata.model_name,
                    'type': metadata.model_type.value,
                    'accuracy': metadata.accuracy,
                    'latency_ms': metadata.latency_ms,
                    'inference_count': metadata.inference_count,
                    'error_rate': error_rate,
                    'is_active': metadata.is_active
                }
            
            return stats
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        with self.lock:
            if not self.inference_history:
                return {}
            
            recent = self.inference_history[-100:]
            
            return {
                'total_inferences': len(self.inference_history),
                'avg_latency_ms': np.mean([r.latency_ms for r in recent]),
                'avg_confidence': np.mean([r.confidence for r in recent]),
                'avg_consensus': np.mean([r.model_consensus for r in recent]),
                'strategy': self.ensemble_strategy.value
            }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create simulated models
    def model1_inference(input_data):
        """Model 1: High accuracy, slightly slower."""
        predictions = np.array([0.1, 0.8, 0.1])
        return predictions, 0.95
    
    def model2_inference(input_data):
        """Model 2: Good accuracy, fast."""
        predictions = np.array([0.15, 0.75, 0.1])
        return predictions, 0.92
    
    def model3_inference(input_data):
        """Model 3: Slightly different prediction."""
        predictions = np.array([0.05, 0.85, 0.1])
        return predictions, 0.90
    
    # Create engine
    engine = MultiModelInferenceEngine(
        ensemble_strategy=EnsembleStrategy.WEIGHTED_AVERAGING
    )
    
    # Register models
    engine.register_model(ModelWrapper("model1", "ResNet-50", 
                                     ModelType.ENCODING, model1_inference))
    engine.register_model(ModelWrapper("model2", "VGG-16", 
                                     ModelType.ENCODING, model2_inference))
    engine.register_model(ModelWrapper("model3", "MobileNet", 
                                     ModelType.ENCODING, model3_inference))
    
    # Run ensemble inference
    test_input = np.random.randn(128)
    
    result = engine.infer_ensemble(test_input)
    
    print(f"Ensemble Result:")
    print(f"  Final Prediction: {result.predictions.argmax()}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Model Consensus: {result.model_consensus:.4f}")
    print(f"  Latency: {result.latency_ms:.2f}ms")
    
    # Print statistics
    print(f"\nModel Statistics:")
    stats = engine.get_model_statistics()
    for model_id, stat in stats.items():
        print(f"  {model_id}: {stat['name']} (accuracy: {stat['accuracy']:.2f})")
    
    print(f"\nEnsemble Statistics:")
    ens_stats = engine.get_ensemble_statistics()
    print(json.dumps(ens_stats, indent=2))