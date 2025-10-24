# Rate Limiting & Throttling System

import threading
import time
from typing import Optional, Dict, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque
import math

class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    ADAPTIVE = "adaptive"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_tokens: float
    refill_rate: float  # tokens per second
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET

class TokenBucketLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.RLock()
    
    def allow_request(self, tokens_required: float = 1) -> bool:
        """Check if request is allowed."""
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens_required:
                self.tokens -= tokens_required
                return True
            
            return False
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def get_tokens(self) -> float:
        """Get current token count."""
        with self.lock:
            self._refill()
            return self.tokens
    
    def get_time_until_available(self, tokens_required: float = 1) -> float:
        """Get time until tokens are available."""
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens_required:
                return 0
            
            needed = tokens_required - self.tokens
            return needed / self.refill_rate

class SlidingWindowLimiter:
    """Sliding window rate limiter."""
    
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self.lock = threading.RLock()
    
    def allow_request(self) -> bool:
        """Check if request is allowed."""
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove old requests outside window
            while self.requests and self.requests[0] < window_start:
                self.requests.popleft()
            
            # Check limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def get_request_count(self) -> int:
        """Get request count in window."""
        with self.lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove old requests
            while self.requests and self.requests[0] < window_start:
                self.requests.popleft()
            
            return len(self.requests)
    
    def get_time_until_available(self) -> float:
        """Get time until next request is allowed."""
        with self.lock:
            if len(self.requests) < self.max_requests:
                return 0
            
            oldest = self.requests[0]
            return max(0, self.window_seconds - (time.time() - oldest))

class AdaptiveThrottler:
    """Adaptive throttling based on system load."""
    
    def __init__(self, base_rate: float, min_rate: float = 0.1, 
                 max_rate: float = None):
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate or base_rate * 10
        self.current_rate = base_rate
        self.success_count = 0
        self.failure_count = 0
        self.lock = threading.RLock()
    
    def record_success(self) -> None:
        """Record successful request."""
        with self.lock:
            self.success_count += 1
            self._adjust_rate()
    
    def record_failure(self) -> None:
        """Record failed request."""
        with self.lock:
            self.failure_count += 1
            self._adjust_rate()
    
    def _adjust_rate(self) -> None:
        """Adjust rate based on success/failure ratio."""
        total = self.success_count + self.failure_count
        
        if total == 0:
            return
        
        success_rate = self.success_count / total
        
        if success_rate > 0.9:
            # Increase rate
            self.current_rate = min(self.max_rate, self.current_rate * 1.1)
        elif success_rate < 0.5:
            # Decrease rate
            self.current_rate = max(self.min_rate, self.current_rate * 0.9)
        
        # Reset counters periodically
        if total > 1000:
            self.success_count = 0
            self.failure_count = 0
    
    def get_rate(self) -> float:
        """Get current rate."""
        with self.lock:
            return self.current_rate

class LeakyBucketLimiter:
    """Leaky bucket rate limiter."""
    
    def __init__(self, capacity: int, leak_rate: float):
        self.capacity = capacity
        self.leak_rate = leak_rate  # items per second
        self.queue_size = 0
        self.last_leak = time.time()
        self.lock = threading.RLock()
    
    def allow_request(self) -> bool:
        """Check if request can be queued."""
        with self.lock:
            self._leak()
            
            if self.queue_size < self.capacity:
                self.queue_size += 1
                return True
            
            return False
    
    def _leak(self) -> None:
        """Leak items from bucket."""
        now = time.time()
        elapsed = now - self.last_leak
        
        leaked = elapsed * self.leak_rate
        self.queue_size = max(0, self.queue_size - leaked)
        self.last_leak = now
    
    def get_queue_size(self) -> float:
        """Get current queue size."""
        with self.lock:
            self._leak()
            return self.queue_size

class RateLimitCluster:
    """Rate limiting for multiple endpoints."""
    
    def __init__(self):
        self.limiters: Dict[str, TokenBucketLimiter] = {}
        self.lock = threading.RLock()
    
    def add_endpoint(self, endpoint: str, capacity: float, 
                     refill_rate: float) -> None:
        """Add rate-limited endpoint."""
        with self.lock:
            self.limiters[endpoint] = TokenBucketLimiter(capacity, refill_rate)
    
    def allow_request(self, endpoint: str, tokens: float = 1) -> bool:
        """Check if request is allowed."""
        with self.lock:
            if endpoint not in self.limiters:
                return True
            
            return self.limiters[endpoint].allow_request(tokens)
    
    def get_status(self) -> Dict:
        """Get cluster status."""
        with self.lock:
            return {
                endpoint: {
                    'tokens': limiter.get_tokens(),
                    'capacity': limiter.capacity,
                    'rate': limiter.refill_rate
                }
                for endpoint, limiter in self.limiters.items()
            }

class ThrottleDecorator:
    """Decorator for rate limiting."""
    
    def __init__(self, limiter, tokens_required: float = 1):
        self.limiter = limiter
        self.tokens_required = tokens_required
    
    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Wait until allowed
            while not self.limiter.allow_request(self.tokens_required):
                wait_time = self.limiter.get_time_until_available(self.tokens_required)
                time.sleep(min(wait_time, 0.1))
            
            return func(*args, **kwargs)
        
        return wrapper

class ConcurrencyLimiter:
    """Limit concurrent requests."""
    
    def __init__(self, max_concurrent: int):
        self.max_concurrent = max_concurrent
        self.current_count = 0
        self.semaphore = threading.Semaphore(max_concurrent)
        self.lock = threading.RLock()
    
    def acquire(self) -> bool:
        """Try to acquire a slot."""
        return self.semaphore.acquire(blocking=False)
    
    def release(self) -> None:
        """Release a slot."""
        self.semaphore.release()
    
    def get_available_slots(self) -> int:
        """Get available concurrent slots."""
        with self.lock:
            return self.max_concurrent - self.current_count
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

# Example usage
if __name__ == "__main__":
    # Token bucket
    token_limiter = TokenBucketLimiter(capacity=10, refill_rate=2)
    
    # Sliding window
    window_limiter = SlidingWindowLimiter(max_requests=5, window_seconds=1.0)
    
    # Adaptive throttler
    adaptive = AdaptiveThrottler(base_rate=1.0)
    
    # Test token bucket
    print("Token Bucket Test:")
    for i in range(3):
        allowed = token_limiter.allow_request()
        print(f"Request {i}: {allowed}")
    
    print(f"\nAvailable tokens: {token_limiter.get_tokens():.2f}")
    print(f"Wait time: {token_limiter.get_time_until_available():.2f}s")
    
    # Test sliding window
    print(f"\nSliding Window Test:")
    for i in range(6):
        allowed = window_limiter.allow_request()
        print(f"Request {i}: {allowed}")
    
    # Test concurrency limiter
    print(f"\nConcurrency Limiter Test:")
    limiter = ConcurrencyLimiter(max_concurrent=2)
    print(f"Available slots: {limiter.get_available_slots()}")
