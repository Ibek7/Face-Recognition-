# Advanced Error Handling and Recovery System

import logging
import traceback
import time
import json
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque
import numpy as np

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class RecoveryStrategy(Enum):
    """Recovery strategies for errors."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ABORT = "abort"

@dataclass
class ErrorMetric:
    """Record for error tracking."""
    timestamp: float
    error_type: str
    severity: ErrorSeverity
    message: str
    traceback: str
    recovered: bool
    recovery_strategy: Optional[str]
    retry_count: int
    processing_time: float

class ErrorTracker:
    """Track and analyze errors."""
    
    def __init__(self, max_history: int = 10000):
        self.errors: deque = deque(maxlen=max_history)
        self.error_counts: Dict[str, int] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def record_error(self, error: ErrorMetric):
        """Record an error."""
        with self.lock:
            self.errors.append(error)
            error_key = f"{error.error_type}:{error.severity.name}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def get_error_rate(self, window_seconds: int = 300) -> float:
        """Get error rate in the last N seconds."""
        with self.lock:
            cutoff_time = time.time() - window_seconds
            recent_errors = sum(1 for e in self.errors if e.timestamp > cutoff_time)
            total_in_window = len([e for e in self.errors if e.timestamp > cutoff_time])
            
            return (recent_errors / total_in_window * 100) if total_in_window > 0 else 0
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        with self.lock:
            critical_errors = [e for e in self.errors if e.severity == ErrorSeverity.CRITICAL]
            recovered_count = sum(1 for e in self.errors if e.recovered)
            
            return {
                'total_errors': len(self.errors),
                'critical_errors': len(critical_errors),
                'recovered_count': recovered_count,
                'recovery_rate': (recovered_count / len(self.errors) * 100) if len(self.errors) > 0 else 0,
                'error_breakdown': dict(self.error_counts),
                'recent_errors': [asdict(e) for e in list(self.errors)[-10:]]
            }

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
        self.is_half_open = False
        self.lock = threading.RLock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.is_open:
                if self._should_attempt_reset():
                    self.is_half_open = True
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            self.failure_count = 0
            self.is_open = False
            self.is_half_open = False
    
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
            
            self.is_half_open = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'is_open': self.is_open,
            'is_half_open': self.is_half_open,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time
        }

class RobustRetry:
    """Robust retry mechanism with exponential backoff."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 0.1, 
                 max_delay: float = 10.0, exponential_base: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_attempts:
                    delay = self._calculate_delay(attempt)
                    time.sleep(delay)
                else:
                    break
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff."""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        return min(delay, self.max_delay)

class FallbackStrategy:
    """Fallback mechanism for graceful degradation."""
    
    def __init__(self):
        self.fallbacks: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, error_type: str, fallback_func: Callable):
        """Register fallback function for error type."""
        self.fallbacks[error_type] = fallback_func
    
    def execute(self, primary_func: Callable, error_type: str, 
                *args, **kwargs) -> Any:
        """Execute with fallback."""
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            if error_type in self.fallbacks:
                self.logger.warning(f"Fallback triggered for {error_type}: {str(e)}")
                return self.fallbacks[error_type](*args, **kwargs)
            raise

def robust_handler(max_retries: int = 3, 
                   fallback: Optional[Callable] = None,
                   error_tracker: Optional[ErrorTracker] = None):
    """Decorator for robust error handling."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_strategy = RobustRetry(max_attempts=max_retries)
            start_time = time.time()
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    
                    if error_tracker and attempt > 0:
                        error_tracker.record_error(ErrorMetric(
                            timestamp=time.time(),
                            error_type=func.__name__,
                            severity=ErrorSeverity.LOW,
                            message="Recovered after retries",
                            traceback="",
                            recovered=True,
                            recovery_strategy=RecoveryStrategy.RETRY.value,
                            retry_count=attempt,
                            processing_time=time.time() - start_time
                        ))
                    
                    return result
                
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        delay = 0.1 * (2 ** attempt)
                        time.sleep(min(delay, 10))
            
            # Try fallback if available
            if fallback:
                try:
                    if error_tracker:
                        error_tracker.record_error(ErrorMetric(
                            timestamp=time.time(),
                            error_type=func.__name__,
                            severity=ErrorSeverity.MEDIUM,
                            message=str(last_exception),
                            traceback=traceback.format_exc(),
                            recovered=True,
                            recovery_strategy=RecoveryStrategy.FALLBACK.value,
                            retry_count=max_retries,
                            processing_time=time.time() - start_time
                        ))
                    
                    return fallback(*args, **kwargs)
                except Exception as fallback_error:
                    if error_tracker:
                        error_tracker.record_error(ErrorMetric(
                            timestamp=time.time(),
                            error_type=func.__name__,
                            severity=ErrorSeverity.CRITICAL,
                            message=str(fallback_error),
                            traceback=traceback.format_exc(),
                            recovered=False,
                            recovery_strategy=None,
                            retry_count=max_retries,
                            processing_time=time.time() - start_time
                        ))
                    
                    raise
            
            # Record failure
            if error_tracker:
                error_tracker.record_error(ErrorMetric(
                    timestamp=time.time(),
                    error_type=func.__name__,
                    severity=ErrorSeverity.HIGH,
                    message=str(last_exception),
                    traceback=traceback.format_exc(),
                    recovered=False,
                    recovery_strategy=None,
                    retry_count=max_retries,
                    processing_time=time.time() - start_time
                ))
            
            raise last_exception
        
        return wrapper
    
    return decorator

class HealthChecker:
    """System health monitoring and recovery."""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.health_checks: Dict[str, Callable] = {}
        self.last_check_results: Dict[str, bool] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def register_check(self, name: str, check_func: Callable):
        """Register health check."""
        self.health_checks[name] = check_func
    
    def run_checks(self) -> Dict[str, bool]:
        """Run all health checks."""
        with self.lock:
            results = {}
            
            for name, check_func in self.health_checks.items():
                try:
                    results[name] = check_func()
                except Exception as e:
                    self.logger.error(f"Health check '{name}' failed: {e}")
                    results[name] = False
            
            self.last_check_results = results
            return results
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        with self.lock:
            return all(self.last_check_results.values()) if self.last_check_results else True
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        with self.lock:
            healthy_count = sum(1 for v in self.last_check_results.values() if v)
            total_checks = len(self.last_check_results)
            
            return {
                'is_healthy': self.is_healthy(),
                'healthy_checks': healthy_count,
                'total_checks': total_checks,
                'health_status': self.last_check_results,
                'timestamp': datetime.now().isoformat()
            }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize error tracking
    error_tracker = ErrorTracker()
    
    # Example with retry and error tracking
    @robust_handler(max_retries=3, error_tracker=error_tracker)
    def unstable_operation():
        import random
        if random.random() < 0.7:
            raise Exception("Random failure")
        return "Success"
    
    # Test
    try:
        result = unstable_operation()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Print error summary
    summary = error_tracker.get_error_summary()
    print(f"Error Summary: {json.dumps(summary, indent=2, default=str)}")