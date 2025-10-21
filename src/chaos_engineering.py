# Chaos Engineering & Resilience Testing System

import random
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime

class ChaosType(Enum):
    """Types of chaos experiments."""
    LATENCY_INJECTION = "latency_injection"
    PACKET_LOSS = "packet_loss"
    ERROR_INJECTION = "error_injection"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    NETWORK_PARTITION = "network_partition"

class ResiliencePattern(Enum):
    """Resilience patterns."""
    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    BULKHEAD = "bulkhead"
    TIMEOUT = "timeout"
    FALLBACK = "fallback"

@dataclass
class ChaosEvent:
    """Chaos experiment event."""
    chaos_type: ChaosType
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    target: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    affected_requests: int = 0
    errors_induced: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'chaos_type': self.chaos_type.value,
            'timestamp': self.timestamp,
            'duration_ms': self.duration_ms,
            'target': self.target,
            'parameters': self.parameters,
            'affected_requests': self.affected_requests,
            'errors_induced': self.errors_induced
        }

class ChaosMonkey(ABC):
    """Base class for chaos monkeys."""
    
    @abstractmethod
    def inject_chaos(self) -> Optional[Exception]:
        """Inject chaos and return exception if applicable."""
        pass

class LatencyMonkey(ChaosMonkey):
    """Inject latency."""
    
    def __init__(self, min_delay_ms: float = 100, max_delay_ms: float = 500,
                injection_rate: float = 0.1):
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.injection_rate = injection_rate
    
    def inject_chaos(self) -> Optional[Exception]:
        """Inject latency."""
        if random.random() < self.injection_rate:
            delay = random.uniform(self.min_delay_ms, self.max_delay_ms)
            time.sleep(delay / 1000)
            return None
        return None

class ErrorMonkey(ChaosMonkey):
    """Inject errors."""
    
    def __init__(self, error_rate: float = 0.05, 
                error_types: List[Exception] = None):
        self.error_rate = error_rate
        self.error_types = error_types or [Exception("Simulated chaos error")]
    
    def inject_chaos(self) -> Optional[Exception]:
        """Inject error."""
        if random.random() < self.error_rate:
            return random.choice(self.error_types)
        return None

class PacketLossMonkey(ChaosMonkey):
    """Simulate packet loss."""
    
    def __init__(self, loss_rate: float = 0.05):
        self.loss_rate = loss_rate
    
    def inject_chaos(self) -> Optional[Exception]:
        """Inject packet loss."""
        if random.random() < self.loss_rate:
            return ConnectionError("Simulated packet loss")
        return None

class ResourceExhaustionMonkey(ChaosMonkey):
    """Simulate resource exhaustion."""
    
    def __init__(self, memory_increase_mb: float = 100):
        self.memory_increase_mb = memory_increase_mb
        self.allocated_memory = []
    
    def inject_chaos(self) -> Optional[Exception]:
        """Simulate resource exhaustion."""
        try:
            # Allocate memory
            data = [0] * (int(self.memory_increase_mb * 1024 * 1024 / 8))
            self.allocated_memory.append(data)
            return None
        except MemoryError:
            return MemoryError("Resource exhaustion simulated")

class ChaosExperiment:
    """Chaos engineering experiment."""
    
    def __init__(self, name: str, target: str, duration_seconds: int = 60):
        self.name = name
        self.target = target
        self.duration_seconds = duration_seconds
        self.monkeys: List[ChaosMonkey] = []
        self.events: List[ChaosEvent] = []
        self.is_running = False
        self.lock = threading.RLock()
    
    def add_monkey(self, monkey: ChaosMonkey):
        """Add chaos monkey."""
        self.monkeys.append(monkey)
    
    def start(self) -> Dict:
        """Start experiment."""
        self.is_running = True
        start_time = time.time()
        
        while time.time() - start_time < self.duration_seconds and self.is_running:
            for monkey in self.monkeys:
                try:
                    exception = monkey.inject_chaos()
                    if exception:
                        raise exception
                except Exception as e:
                    with self.lock:
                        event = ChaosEvent(
                            chaos_type=type(monkey).__name__,
                            target=self.target,
                            duration_ms=(time.time() - start_time) * 1000,
                            errors_induced=1
                        )
                        self.events.append(event)
            
            time.sleep(0.1)
        
        self.is_running = False
        
        return self.get_report()
    
    def stop(self):
        """Stop experiment."""
        self.is_running = False
    
    def get_report(self) -> Dict:
        """Get experiment report."""
        with self.lock:
            total_errors = sum(e.errors_induced for e in self.events)
            
            return {
                'name': self.name,
                'target': self.target,
                'duration_seconds': self.duration_seconds,
                'total_chaos_events': len(self.events),
                'total_errors': total_errors,
                'events': [e.to_dict() for e in self.events]
            }

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, 
                reset_timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker."""
        
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception(f"Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _on_success(self):
        """Handle successful call."""
        self.success_count += 1
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset."""
        return (self.last_failure_time and 
               time.time() - self.last_failure_time > self.reset_timeout_seconds)
    
    def get_status(self) -> Dict:
        """Get circuit breaker status."""
        with self.lock:
            return {
                'state': self.state,
                'failure_count': self.failure_count,
                'success_count': self.success_count
            }

class Bulkhead:
    """Bulkhead pattern implementation."""
    
    def __init__(self, max_concurrent_calls: int = 10):
        self.max_concurrent_calls = max_concurrent_calls
        self.semaphore = threading.Semaphore(max_concurrent_calls)
        self.current_calls = 0
        self.rejected_calls = 0
        self.lock = threading.RLock()
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with bulkhead."""
        
        if not self.semaphore.acquire(blocking=False):
            with self.lock:
                self.rejected_calls += 1
            raise Exception(f"Bulkhead limit reached")
        
        try:
            with self.lock:
                self.current_calls += 1
            
            return func(*args, **kwargs)
        finally:
            with self.lock:
                self.current_calls -= 1
            self.semaphore.release()
    
    def get_status(self) -> Dict:
        """Get bulkhead status."""
        with self.lock:
            return {
                'max_concurrent_calls': self.max_concurrent_calls,
                'current_calls': self.current_calls,
                'rejected_calls': self.rejected_calls
            }

class RetryPolicy:
    """Retry policy."""
    
    def __init__(self, max_retries: int = 3, 
                backoff_factor: float = 1.0,
                jitter: bool = True):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.attempt_count = 0
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retries."""
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.attempt_count += 1
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = (self.backoff_factor ** attempt)
                    if self.jitter:
                        delay *= random.uniform(0.5, 1.5)
                    
                    time.sleep(delay)
        
        raise last_exception

class ResilienceTestSuite:
    """Suite of resilience tests."""
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker()
        self.bulkhead = Bulkhead()
        self.retry_policy = RetryPolicy()
        self.results: Dict[str, Any] = {}
    
    def test_circuit_breaker(self, failing_func: Callable) -> Dict:
        """Test circuit breaker."""
        results = {
            'test_name': 'circuit_breaker',
            'passed': 0,
            'failed': 0
        }
        
        for i in range(10):
            try:
                self.circuit_breaker.call(failing_func)
                results['passed'] += 1
            except Exception:
                results['failed'] += 1
        
        results['status'] = self.circuit_breaker.get_status()
        self.results['circuit_breaker'] = results
        
        return results
    
    def test_bulkhead(self, func: Callable) -> Dict:
        """Test bulkhead."""
        results = {
            'test_name': 'bulkhead',
            'passed': 0,
            'rejected': 0
        }
        
        threads = []
        
        for i in range(20):
            def worker():
                try:
                    self.bulkhead.execute(func)
                    results['passed'] += 1
                except Exception:
                    results['rejected'] += 1
            
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        results['status'] = self.bulkhead.get_status()
        self.results['bulkhead'] = results
        
        return results
    
    def test_retry(self, func: Callable) -> Dict:
        """Test retry policy."""
        results = {
            'test_name': 'retry',
            'attempts': 0,
            'success': False
        }
        
        try:
            self.retry_policy.execute(func)
            results['success'] = True
        except Exception as e:
            results['error'] = str(e)
        
        results['attempts'] = self.retry_policy.attempt_count
        self.results['retry'] = results
        
        return results

# Example usage
if __name__ == "__main__":
    # Create chaos experiment
    experiment = ChaosExperiment("api_resilience_test", "face-recognition-api", duration_seconds=10)
    
    # Add chaos monkeys
    experiment.add_monkey(LatencyMonkey(100, 500, 0.2))
    experiment.add_monkey(ErrorMonkey(0.1))
    experiment.add_monkey(PacketLossMonkey(0.05))
    
    # Run experiment
    print("Starting chaos experiment...")
    report = experiment.start()
    print(f"Chaos Events: {report['total_chaos_events']}")
    print(f"Errors Induced: {report['total_errors']}")
    
    # Test resilience patterns
    print("\nTesting resilience patterns...")
    suite = ResilienceTestSuite()
    
    def failing_function():
        if random.random() < 0.6:
            raise Exception("Random failure")
        return "Success"
    
    cb_result = suite.test_circuit_breaker(failing_function)
    print(f"Circuit Breaker - Passed: {cb_result['passed']}, Failed: {cb_result['failed']}")
