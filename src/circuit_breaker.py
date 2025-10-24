# Circuit Breaker Pattern for Fault Tolerance

import threading
import time
import json
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery

class FailureType(Enum):
    """Types of failures."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"

@dataclass
class FailureEvent:
    """Record of failure event."""
    timestamp: float = field(default_factory=time.time)
    failure_type: FailureType = FailureType.EXCEPTION
    error_message: str = ""
    service_name: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'type': self.failure_type.value,
            'message': self.error_message,
            'service': self.service_name
        }

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Failures to trigger open
    success_threshold: int = 2  # Successes to close from half-open
    timeout_sec: float = 60.0  # Time before attempting recovery
    half_open_max_calls: int = 3  # Max calls in half-open state
    window_size_sec: int = 120  # Rolling window for metrics

class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, service_name: str, config: CircuitBreakerConfig = None):
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state_change_time = time.time()
        
        self.failure_history: deque = deque()
        self.success_history: deque = deque()
        self.half_open_calls = 0
        
        self.lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs) -> Tuple[bool, Any]:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker is OPEN for {self.service_name}"
                    )
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenException(
                        f"Half-open limit reached for {self.service_name}"
                    )
                self.half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return True, result
        
        except Exception as e:
            self._on_failure(FailureType.EXCEPTION, str(e))
            return False, None
    
    def _on_success(self) -> None:
        """Handle successful call."""
        with self.lock:
            self.success_count += 1
            self.success_history.append(time.time())
            
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self._close_circuit()
            
            # Clear failures on success
            if self.state == CircuitState.CLOSED:
                self.failure_count = 0
    
    def _on_failure(self, failure_type: FailureType, error_msg: str) -> None:
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.failure_history.append(self.last_failure_time)
            
            event = FailureEvent(
                timestamp=self.last_failure_time,
                failure_type=failure_type,
                error_message=error_msg,
                service_name=self.service_name
            )
            
            if self.state == CircuitState.HALF_OPEN:
                self._open_circuit()
            elif self.failure_count >= self.config.failure_threshold:
                self._open_circuit()
    
    def _open_circuit(self) -> None:
        """Transition to OPEN state."""
        self.state = CircuitState.OPEN
        self.state_change_time = time.time()
        self.success_count = 0
    
    def _close_circuit(self) -> None:
        """Transition to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.state_change_time = time.time()
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt reset from OPEN."""
        if self.last_failure_time is None:
            return False
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.timeout_sec
    
    def get_state(self) -> Dict:
        """Get circuit breaker state."""
        with self.lock:
            time_in_state = time.time() - self.state_change_time
            
            return {
                'service': self.service_name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'time_in_state_sec': time_in_state,
                'last_failure_time': self.last_failure_time,
                'half_open_calls': self.half_open_calls
            }

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit is open."""
    pass

class CircuitBreakerManager:
    """Manage multiple circuit breakers."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.RLock()
    
    def register_breaker(self, service_name: str, 
                        config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Register circuit breaker for service."""
        with self.lock:
            if service_name not in self.breakers:
                self.breakers[service_name] = CircuitBreaker(service_name, config)
            
            return self.breakers[service_name]
    
    def get_breaker(self, service_name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for service."""
        with self.lock:
            return self.breakers.get(service_name)
    
    def call_with_fallback(self, service_name: str, func: Callable,
                          fallback_func: Callable = None,
                          *args, **kwargs) -> Any:
        """Call function with fallback."""
        breaker = self.get_breaker(service_name)
        
        if not breaker:
            return func(*args, **kwargs)
        
        try:
            success, result = breaker.call(func, *args, **kwargs)
            
            if success:
                return result
            
            if fallback_func:
                return fallback_func(*args, **kwargs)
            
            return None
        
        except CircuitBreakerOpenException:
            if fallback_func:
                return fallback_func(*args, **kwargs)
            
            raise
    
    def get_status(self) -> Dict:
        """Get all breakers status."""
        with self.lock:
            return {
                service_name: breaker.get_state()
                for service_name, breaker in self.breakers.items()
            }
    
    def reset_breaker(self, service_name: str) -> bool:
        """Manually reset circuit breaker."""
        with self.lock:
            breaker = self.breakers.get(service_name)
            
            if breaker:
                breaker._close_circuit()
                return True
            
            return False

class BulkheadPolicy:
    """Isolate resources to prevent cascade failures."""
    
    def __init__(self, max_concurrent_calls: int = 10, queue_size: int = 100):
        self.max_concurrent_calls = max_concurrent_calls
        self.queue_size = queue_size
        self.active_calls = 0
        self.queued_calls = deque(maxlen=queue_size)
        self.lock = threading.RLock()
        self.semaphore = threading.Semaphore(max_concurrent_calls)
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with bulkhead isolation."""
        if not self.semaphore.acquire(blocking=False):
            # Queue request if at capacity
            with self.lock:
                if len(self.queued_calls) < self.queue_size:
                    self.queued_calls.append((func, args, kwargs))
                    raise Exception("Queue full - request rejected")
        
        try:
            with self.lock:
                self.active_calls += 1
            
            return func(*args, **kwargs)
        
        finally:
            with self.lock:
                self.active_calls -= 1
            
            self.semaphore.release()
    
    def get_metrics(self) -> Dict:
        """Get bulkhead metrics."""
        with self.lock:
            return {
                'active_calls': self.active_calls,
                'max_concurrent': self.max_concurrent_calls,
                'queued_calls': len(self.queued_calls),
                'queue_capacity': self.queue_size,
                'utilization': self.active_calls / self.max_concurrent_calls
            }

class RetryPolicy:
    """Retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay_ms: float = 100,
                 max_delay_ms: float = 10000, backoff_multiplier: float = 2.0):
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.backoff_multiplier = backoff_multiplier
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay / 1000.0)  # Convert ms to sec
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.base_delay_ms * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay_ms)

# Example usage
if __name__ == "__main__":
    # Create manager
    manager = CircuitBreakerManager()
    
    # Register breaker for API service
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_sec=5
    )
    breaker = manager.register_breaker('api_service', config)
    
    # Simulate API calls
    def unreliable_api():
        import random
        if random.random() < 0.7:
            raise Exception("API timeout")
        return {'status': 'ok'}
    
    def fallback_api():
        return {'status': 'fallback'}
    
    # Make calls
    for i in range(10):
        try:
            result = manager.call_with_fallback(
                'api_service', unreliable_api, fallback_api
            )
            print(f"Call {i+1}: {result}")
        except CircuitBreakerOpenException as e:
            print(f"Call {i+1}: Circuit open - {e}")
        
        time.sleep(0.1)
    
    # Get status
    status = manager.get_status()
    print(f"\nCircuit Breaker Status:")
    print(json.dumps(status, indent=2))
