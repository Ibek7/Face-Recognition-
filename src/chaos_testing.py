# Chaos Testing Framework

import threading
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import random

class FailureType(Enum):
    """Chaos failure types."""
    LATENCY = "latency"
    ERROR = "error"
    TIMEOUT = "timeout"
    NETWORK_PARTITION = "network_partition"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CASCADING_FAILURE = "cascading_failure"

@dataclass
class ChaosScenario:
    """Chaos testing scenario."""
    scenario_id: str
    name: str
    failure_type: FailureType
    target_service: str
    failure_rate: float = 0.1  # 0-1
    duration_seconds: int = 60
    intensity: int = 1  # 1-10
    enabled: bool = False
    created_at: float = field(default_factory=time.time)

@dataclass
class ChaosEvent:
    """Chaos event."""
    event_id: str
    scenario_id: str
    failure_type: FailureType
    timestamp: float
    recovered: bool = False
    recovery_time: float = 0.0

class ChaosInjector:
    """Inject chaos into system."""
    
    def __init__(self):
        self.scenarios: Dict[str, ChaosScenario] = {}
        self.active_failures: Dict[str, ChaosEvent] = {}
        self.events: List[ChaosEvent] = []
        self.lock = threading.RLock()
    
    def create_scenario(self, scenario: ChaosScenario) -> str:
        """Create chaos scenario."""
        with self.lock:
            self.scenarios[scenario.scenario_id] = scenario
            return scenario.scenario_id
    
    def enable_scenario(self, scenario_id: str) -> bool:
        """Enable scenario."""
        with self.lock:
            if scenario_id in self.scenarios:
                self.scenarios[scenario_id].enabled = True
                return True
            return False
    
    def disable_scenario(self, scenario_id: str) -> bool:
        """Disable scenario."""
        with self.lock:
            if scenario_id in self.scenarios:
                self.scenarios[scenario_id].enabled = False
                return True
            return False
    
    def inject_failure(self, scenario_id: str) -> Optional[ChaosEvent]:
        """Inject failure for scenario."""
        with self.lock:
            if scenario_id not in self.scenarios:
                return None
            
            scenario = self.scenarios[scenario_id]
            
            if not scenario.enabled:
                return None
            
            # Determine if failure occurs
            if random.random() > scenario.failure_rate:
                return None
            
            # Create failure event
            import uuid
            event = ChaosEvent(
                event_id=str(uuid.uuid4()),
                scenario_id=scenario_id,
                failure_type=scenario.failure_type,
                timestamp=time.time()
            )
            
            self.active_failures[event.event_id] = event
            self.events.append(event)
            
            return event
    
    def recover_failure(self, event_id: str) -> bool:
        """Recover from failure."""
        with self.lock:
            if event_id not in self.active_failures:
                return False
            
            event = self.active_failures[event_id]
            event.recovered = True
            event.recovery_time = time.time() - event.timestamp
            
            del self.active_failures[event_id]
            
            return True
    
    def get_active_failures(self) -> List[ChaosEvent]:
        """Get active failures."""
        with self.lock:
            return list(self.active_failures.values())
    
    def get_events(self) -> List[ChaosEvent]:
        """Get all events."""
        with self.lock:
            return self.events.copy()

class LatencyInjector:
    """Inject latency into requests."""
    
    def __init__(self):
        self.scenarios: Dict[str, Dict] = {}
        self.lock = threading.RLock()
    
    def add_latency_scenario(self, scenario_id: str, min_ms: int,
                           max_ms: int, probability: float = 0.1) -> None:
        """Add latency scenario."""
        with self.lock:
            self.scenarios[scenario_id] = {
                'min_ms': min_ms,
                'max_ms': max_ms,
                'probability': probability
            }
    
    def inject_latency(self, scenario_id: str) -> float:
        """Inject latency."""
        with self.lock:
            if scenario_id not in self.scenarios:
                return 0.0
            
            scenario = self.scenarios[scenario_id]
            
            if random.random() > scenario['probability']:
                return 0.0
            
            delay_ms = random.uniform(scenario['min_ms'], scenario['max_ms'])
            return delay_ms / 1000.0

class ErrorInjector:
    """Inject errors into requests."""
    
    def __init__(self):
        self.scenarios: Dict[str, Dict] = {}
        self.lock = threading.RLock()
    
    def add_error_scenario(self, scenario_id: str, error_code: int,
                          probability: float = 0.1, message: str = None) -> None:
        """Add error scenario."""
        with self.lock:
            self.scenarios[scenario_id] = {
                'error_code': error_code,
                'probability': probability,
                'message': message or f"Chaos error {error_code}"
            }
    
    def inject_error(self, scenario_id: str) -> Optional[Dict]:
        """Inject error."""
        with self.lock:
            if scenario_id not in self.scenarios:
                return None
            
            scenario = self.scenarios[scenario_id]
            
            if random.random() > scenario['probability']:
                return None
            
            return {
                'error_code': scenario['error_code'],
                'message': scenario['message']
            }

class NetworkPartitionSimulator:
    """Simulate network partitions."""
    
    def __init__(self):
        self.partitions: Dict[str, List[str]] = {}
        self.lock = threading.RLock()
    
    def create_partition(self, partition_id: str, isolated_services: List[str]) -> None:
        """Create network partition."""
        with self.lock:
            self.partitions[partition_id] = isolated_services
    
    def remove_partition(self, partition_id: str) -> None:
        """Remove partition."""
        with self.lock:
            if partition_id in self.partitions:
                del self.partitions[partition_id]
    
    def is_partitioned(self, service1: str, service2: str) -> bool:
        """Check if services are partitioned."""
        with self.lock:
            for partition in self.partitions.values():
                if (service1 in partition and service2 not in partition) or \
                   (service2 in partition and service1 not in partition):
                    return True
            
            return False

class ResourceExhaustionSimulator:
    """Simulate resource exhaustion."""
    
    def __init__(self):
        self.exhausted_resources: Dict[str, Dict] = {}
        self.lock = threading.RLock()
    
    def exhaust_resource(self, resource_id: str, resource_type: str,
                        percentage: float = 0.9) -> None:
        """Mark resource as exhausted."""
        with self.lock:
            self.exhausted_resources[resource_id] = {
                'type': resource_type,
                'available': 100 - (percentage * 100),
                'timestamp': time.time()
            }
    
    def release_resource(self, resource_id: str) -> None:
        """Release exhausted resource."""
        with self.lock:
            if resource_id in self.exhausted_resources:
                del self.exhausted_resources[resource_id]
    
    def get_available_capacity(self, resource_id: str) -> float:
        """Get available resource capacity."""
        with self.lock:
            if resource_id in self.exhausted_resources:
                return self.exhausted_resources[resource_id]['available']
            
            return 100.0

class ChaosTestRunner:
    """Run chaos testing experiments."""
    
    def __init__(self):
        self.injector = ChaosInjector()
        self.latency_injector = LatencyInjector()
        self.error_injector = ErrorInjector()
        self.network_sim = NetworkPartitionSimulator()
        self.resource_sim = ResourceExhaustionSimulator()
        self.results: List[Dict] = []
        self.lock = threading.RLock()
    
    def run_experiment(self, scenario: ChaosScenario) -> Dict:
        """Run chaos experiment."""
        scenario_id = self.injector.create_scenario(scenario)
        self.injector.enable_scenario(scenario_id)
        
        start_time = time.time()
        failures = []
        recoveries = []
        
        # Simulate scenario for duration
        while time.time() - start_time < scenario.duration_seconds:
            # Inject failures
            failure_event = self.injector.inject_failure(scenario_id)
            
            if failure_event:
                failures.append({
                    'time': failure_event.timestamp,
                    'type': failure_event.failure_type.value
                })
                
                # Simulate recovery
                time.sleep(0.1 * scenario.intensity)
                
                if self.injector.recover_failure(failure_event.event_id):
                    recovered_event = self.injector.active_failures.get(failure_event.event_id)
                    recoveries.append({
                        'time': time.time(),
                        'recovery_time': 0.1 * scenario.intensity
                    })
            
            time.sleep(0.1)
        
        self.injector.disable_scenario(scenario_id)
        
        result = {
            'scenario_id': scenario_id,
            'duration': time.time() - start_time,
            'total_failures': len(failures),
            'total_recoveries': len(recoveries),
            'recovery_rate': len(recoveries) / len(failures) if failures else 0,
            'failures': failures,
            'recoveries': recoveries
        }
        
        with self.lock:
            self.results.append(result)
        
        return result
    
    def get_results(self) -> List[Dict]:
        """Get experiment results."""
        with self.lock:
            return self.results.copy()
    
    def generate_report(self) -> Dict:
        """Generate chaos testing report."""
        with self.lock:
            if not self.results:
                return {}
            
            total_failures = sum(r['total_failures'] for r in self.results)
            total_recoveries = sum(r['total_recoveries'] for r in self.results)
            avg_recovery_rate = sum(r['recovery_rate'] for r in self.results) / len(self.results)
            
            return {
                'experiments': len(self.results),
                'total_failures': total_failures,
                'total_recoveries': total_recoveries,
                'avg_recovery_rate': avg_recovery_rate,
                'resilience_score': avg_recovery_rate * 100
            }

# Example usage
if __name__ == "__main__":
    runner = ChaosTestRunner()
    
    # Create scenario
    scenario = ChaosScenario(
        scenario_id="chaos-1",
        name="API Timeout Scenario",
        failure_type=FailureType.TIMEOUT,
        target_service="face-api",
        failure_rate=0.2,
        duration_seconds=5,
        intensity=3
    )
    
    # Run experiment
    result = runner.run_experiment(scenario)
    
    print(f"Experiment Results:")
    print(f"  Total Failures: {result['total_failures']}")
    print(f"  Total Recoveries: {result['total_recoveries']}")
    print(f"  Recovery Rate: {result['recovery_rate']:.2%}")
    
    # Generate report
    report = runner.generate_report()
    print(f"\nChaos Testing Report:")
    print(json.dumps(report, indent=2))
