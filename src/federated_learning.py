# Federated Learning Framework

import logging
import json
import time
import threading
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
import numpy as np

class ClientStatus(Enum):
    """Status of federated learning client."""
    IDLE = "idle"
    TRAINING = "training"
    UPLOADING = "uploading"
    FAILED = "failed"

class AggregationStrategy(Enum):
    """Model aggregation strategies."""
    FEDAVG = "fedavg"  # Federated Averaging
    FEDPROX = "fedprox"  # Federated Proximal
    SCAFFOLD = "scaffold"  # SCAFFOLD
    MEDIAN = "median"  # Median aggregation

@dataclass
class ClientModelUpdate:
    """Model update from a client."""
    client_id: str
    round_number: int
    model_weights: np.ndarray
    num_samples: int
    accuracy: float
    loss: float
    training_time_seconds: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class FederatedRound:
    """Single round of federated learning."""
    round_number: int
    start_time: float
    end_time: Optional[float] = None
    client_updates: List[ClientModelUpdate] = field(default_factory=list)
    aggregated_weights: Optional[np.ndarray] = None
    global_accuracy: float = 0.0
    global_loss: float = 0.0

class FederatedClient:
    """Federated learning client."""
    
    def __init__(self, client_id: str, local_data_size: int):
        self.client_id = client_id
        self.local_data_size = local_data_size
        self.model_weights: Optional[np.ndarray] = None
        self.status = ClientStatus.IDLE
        self.logger = logging.getLogger(__name__)
        self.training_history: List[Dict[str, float]] = []
    
    def receive_global_model(self, weights: np.ndarray):
        """Receive global model from server."""
        self.model_weights = weights.copy()
        self.status = ClientStatus.IDLE
    
    def train_local_model(self, epochs: int = 1) -> ClientModelUpdate:
        """Train model locally."""
        self.status = ClientStatus.TRAINING
        
        try:
            start_time = time.time()
            
            # Simulate local training
            # In practice, this would be actual model training
            local_accuracy = 0.85 + np.random.normal(0, 0.05)
            local_loss = 0.3 + np.random.normal(0, 0.05)
            
            # Simulate model update
            if self.model_weights is not None:
                # Add small noise to simulate training effect
                updated_weights = self.model_weights + np.random.normal(0, 0.001, 
                                                                        self.model_weights.shape)
            else:
                updated_weights = np.random.randn(1024, 512)
            
            training_time = time.time() - start_time
            
            # Record training history
            self.training_history.append({
                'accuracy': local_accuracy,
                'loss': local_loss,
                'training_time': training_time
            })
            
            update = ClientModelUpdate(
                client_id=self.client_id,
                round_number=len(self.training_history),
                model_weights=updated_weights,
                num_samples=self.local_data_size,
                accuracy=local_accuracy,
                loss=local_loss,
                training_time_seconds=training_time
            )
            
            self.status = ClientStatus.UPLOADING
            return update
        
        except Exception as e:
            self.status = ClientStatus.FAILED
            self.logger.error(f"Client training failed: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status."""
        return {
            'client_id': self.client_id,
            'status': self.status.value,
            'local_data_size': self.local_data_size,
            'training_rounds': len(self.training_history),
            'last_training_time': self.training_history[-1]['training_time'] if self.training_history else 0
        }

class ModelAggregator:
    """Aggregate model updates from clients."""
    
    def __init__(self, strategy: AggregationStrategy = AggregationStrategy.FEDAVG):
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
    
    def aggregate(self, updates: List[ClientModelUpdate]) -> np.ndarray:
        """Aggregate model updates."""
        
        if not updates:
            raise ValueError("No updates to aggregate")
        
        if self.strategy == AggregationStrategy.FEDAVG:
            return self._fedavg(updates)
        elif self.strategy == AggregationStrategy.FEDPROX:
            return self._fedprox(updates)
        elif self.strategy == AggregationStrategy.MEDIAN:
            return self._median(updates)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _fedavg(self, updates: List[ClientModelUpdate]) -> np.ndarray:
        """Federated Averaging aggregation."""
        
        # Weight by number of samples
        total_samples = sum(u.num_samples for u in updates)
        
        aggregated = None
        for update in updates:
            weight = update.num_samples / total_samples
            
            if aggregated is None:
                aggregated = update.model_weights * weight
            else:
                aggregated += update.model_weights * weight
        
        return aggregated
    
    def _fedprox(self, updates: List[ClientModelUpdate]) -> np.ndarray:
        """Federated Proximal aggregation."""
        
        # Similar to FedAvg but with proximal term
        # For now, use simple FedAvg
        return self._fedavg(updates)
    
    def _median(self, updates: List[ClientModelUpdate]) -> np.ndarray:
        """Median aggregation (robust to outliers)."""
        
        weights_list = np.array([u.model_weights for u in updates])
        
        # Compute median element-wise
        aggregated = np.median(weights_list, axis=0)
        
        return aggregated
    
    def get_update_statistics(self, updates: List[ClientModelUpdate]) -> Dict[str, Any]:
        """Get statistics about updates."""
        
        if not updates:
            return {}
        
        accuracies = [u.accuracy for u in updates]
        losses = [u.loss for u in updates]
        
        return {
            'num_updates': len(updates),
            'avg_accuracy': np.mean(accuracies),
            'avg_loss': np.mean(losses),
            'max_accuracy': np.max(accuracies),
            'min_accuracy': np.min(accuracies),
            'total_samples': sum(u.num_samples for u in updates)
        }

class FederatedLearningServer:
    """Central server for federated learning."""
    
    def __init__(self, num_clients: int, aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDAVG):
        self.num_clients = num_clients
        self.aggregator = ModelAggregator(aggregation_strategy)
        self.global_model_weights: Optional[np.ndarray] = None
        self.round_history: List[FederatedRound] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize global model
        self.global_model_weights = np.random.randn(1024, 512)
    
    def start_round(self) -> int:
        """Start new federated learning round."""
        with self.lock:
            round_number = len(self.round_history) + 1
            
            round_info = FederatedRound(
                round_number=round_number,
                start_time=time.time()
            )
            
            self.round_history.append(round_info)
            
            self.logger.info(f"Starting federated learning round {round_number}")
            
            return round_number
    
    def end_round(self, round_number: int, updates: List[ClientModelUpdate]) -> np.ndarray:
        """End federated learning round and aggregate updates."""
        
        with self.lock:
            if not self.round_history or self.round_history[-1].round_number != round_number:
                raise ValueError(f"Invalid round number: {round_number}")
            
            current_round = self.round_history[-1]
            current_round.client_updates = updates
            current_round.end_time = time.time()
            
            # Aggregate updates
            aggregated_weights = self.aggregator.aggregate(updates)
            current_round.aggregated_weights = aggregated_weights
            self.global_model_weights = aggregated_weights
            
            # Calculate global metrics
            stats = self.aggregator.get_update_statistics(updates)
            current_round.global_accuracy = stats.get('avg_accuracy', 0.0)
            current_round.global_loss = stats.get('avg_loss', 0.0)
            
            # Log round completion
            round_time = current_round.end_time - current_round.start_time
            self.logger.info(
                f"Completed round {round_number}: "
                f"accuracy={current_round.global_accuracy:.4f}, "
                f"loss={current_round.global_loss:.4f}, "
                f"time={round_time:.2f}s"
            )
            
            return aggregated_weights
    
    def get_global_model(self) -> np.ndarray:
        """Get current global model."""
        with self.lock:
            return self.global_model_weights.copy() if self.global_model_weights is not None else None
    
    def get_round_history(self) -> List[Dict[str, Any]]:
        """Get federated learning round history."""
        with self.lock:
            return [
                {
                    'round_number': r.round_number,
                    'duration_seconds': (r.end_time - r.start_time) if r.end_time else 0,
                    'num_updates': len(r.client_updates),
                    'global_accuracy': r.global_accuracy,
                    'global_loss': r.global_loss
                }
                for r in self.round_history
            ]
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        with self.lock:
            if not self.round_history:
                return {}
            
            completed_rounds = [r for r in self.round_history if r.end_time]
            
            accuracies = [r.global_accuracy for r in completed_rounds]
            losses = [r.global_loss for r in completed_rounds]
            round_times = [(r.end_time - r.start_time) for r in completed_rounds]
            
            return {
                'total_rounds': len(self.round_history),
                'completed_rounds': len(completed_rounds),
                'avg_accuracy': np.mean(accuracies) if accuracies else 0,
                'max_accuracy': np.max(accuracies) if accuracies else 0,
                'avg_loss': np.mean(losses) if losses else 0,
                'avg_round_time_seconds': np.mean(round_times) if round_times else 0
            }

class PrivacyPreserver:
    """Handle privacy in federated learning."""
    
    @staticmethod
    def add_differential_privacy(weights: np.ndarray, epsilon: float = 1.0,
                                delta: float = 1e-5) -> np.ndarray:
        """Add differential privacy via noise."""
        
        # Compute L2 sensitivity
        l2_sensitivity = np.max(np.abs(weights))
        
        # Calculate noise scale
        noise_scale = l2_sensitivity / epsilon
        
        # Add Laplace noise
        noise = np.random.laplace(0, noise_scale, weights.shape)
        
        return weights + noise
    
    @staticmethod
    def compute_parameter_hash(weights: np.ndarray) -> str:
        """Compute hash of model parameters."""
        
        weights_bytes = weights.tobytes()
        return hashlib.sha256(weights_bytes).hexdigest()
    
    @staticmethod
    def verify_model_integrity(weights: np.ndarray, expected_hash: str) -> bool:
        """Verify model integrity via hash."""
        
        computed_hash = PrivacyPreserver.compute_parameter_hash(weights)
        return computed_hash == expected_hash

class FederatedLearningCoordinator:
    """Coordinate federated learning process."""
    
    def __init__(self, num_clients: int, num_rounds: int = 5):
        self.server = FederatedLearningServer(num_clients)
        self.clients: Dict[str, FederatedClient] = {
            f"client_{i}": FederatedClient(f"client_{i}", 100 * (i + 1))
            for i in range(num_clients)
        }
        self.num_rounds = num_rounds
        self.privacy_preserver = PrivacyPreserver()
        self.logger = logging.getLogger(__name__)
    
    def run_federated_learning(self) -> Dict[str, Any]:
        """Run complete federated learning process."""
        
        results = {
            'rounds': [],
            'final_statistics': {}
        }
        
        for round_num in range(1, self.num_rounds + 1):
            self.logger.info(f"\n=== Federated Learning Round {round_num} ===")
            
            # Start round
            self.server.start_round()
            
            # Distribute global model to clients
            global_model = self.server.get_global_model()
            for client in self.clients.values():
                client.receive_global_model(global_model)
            
            # Clients train locally
            updates = []
            for client in self.clients.values():
                try:
                    update = client.train_local_model()
                    updates.append(update)
                except Exception as e:
                    self.logger.warning(f"Client {client.client_id} training failed: {e}")
            
            # Server aggregates updates
            if updates:
                self.server.end_round(round_num, updates)
            
            # Record round results
            history = self.server.get_round_history()
            if history:
                results['rounds'].append(history[-1])
        
        # Get final statistics
        results['final_statistics'] = self.server.get_training_statistics()
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        
        client_statuses = {
            cid: client.get_status()
            for cid, client in self.clients.items()
        }
        
        return {
            'num_clients': len(self.clients),
            'client_statuses': client_statuses,
            'server_statistics': self.server.get_training_statistics(),
            'round_history': self.server.get_round_history()
        }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create coordinator
    coordinator = FederatedLearningCoordinator(num_clients=3, num_rounds=5)
    
    # Run federated learning
    results = coordinator.run_federated_learning()
    
    # Print results
    print("\n=== Federated Learning Results ===")
    print(f"Completed {len(results['rounds'])} rounds")
    
    print("\nFinal Statistics:")
    import json
    stats = results['final_statistics']
    print(json.dumps({
        'avg_accuracy': f"{stats['avg_accuracy']:.4f}",
        'max_accuracy': f"{stats['max_accuracy']:.4f}",
        'avg_loss': f"{stats['avg_loss']:.4f}",
        'avg_round_time': f"{stats['avg_round_time_seconds']:.2f}s"
    }, indent=2))
    
    # Print system status
    print("\nSystem Status:")
    status = coordinator.get_system_status()
    print(f"Active Clients: {len(status['client_statuses'])}")
    print(f"Total Rounds: {status['server_statistics']['total_rounds']}")