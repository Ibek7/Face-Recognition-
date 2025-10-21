# API Rate Limiting and Quota Management System

import time
import threading
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"

@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests_per_second: float
    burst_size: int
    time_window_seconds: int

@dataclass
class QuotaLimit:
    """Quota configuration."""
    requests_per_day: int
    requests_per_month: int
    max_concurrent_requests: int

class TokenBucket:
    """Token bucket algorithm for rate limiting."""
    
    def __init__(self, capacity: float, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill_time = time.time()
        self.lock = threading.RLock()
    
    def allow_request(self, tokens_required: int = 1) -> bool:
        """Check if request is allowed."""
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens_required:
                self.tokens -= tokens_required
                return True
            
            return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_time
        
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill_time = now
    
    def get_tokens_available(self) -> float:
        """Get available tokens."""
        with self.lock:
            self._refill()
            return self.tokens

class SlidingWindowCounter:
    """Sliding window counter for rate limiting."""
    
    def __init__(self, window_size_seconds: int, max_requests: int):
        self.window_size_seconds = window_size_seconds
        self.max_requests = max_requests
        self.requests: deque = deque()
        self.lock = threading.RLock()
    
    def allow_request(self) -> bool:
        """Check if request is allowed."""
        with self.lock:
            now = time.time()
            window_start = now - self.window_size_seconds
            
            # Remove old requests outside window
            while self.requests and self.requests[0] < window_start:
                self.requests.popleft()
            
            # Check if within limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def get_request_count(self) -> int:
        """Get current request count in window."""
        with self.lock:
            now = time.time()
            window_start = now - self.window_size_seconds
            
            # Remove old requests
            while self.requests and self.requests[0] < window_start:
                self.requests.popleft()
            
            return len(self.requests)
    
    def get_retry_after_seconds(self) -> float:
        """Get seconds until next request is allowed."""
        with self.lock:
            if not self.requests:
                return 0.0
            
            oldest_request = self.requests[0]
            return max(0, self.window_size_seconds - (time.time() - oldest_request))

class ClientRateLimiter:
    """Rate limiter for individual clients."""
    
    def __init__(self, client_id: str, strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
                 rate_limit: RateLimit = None):
        self.client_id = client_id
        self.strategy = strategy
        self.rate_limit = rate_limit or RateLimit(requests_per_second=10, burst_size=20, time_window_seconds=60)
        
        if strategy == RateLimitStrategy.TOKEN_BUCKET:
            self.limiter = TokenBucket(
                capacity=self.rate_limit.burst_size,
                refill_rate=self.rate_limit.requests_per_second
            )
        elif strategy == RateLimitStrategy.SLIDING_WINDOW:
            self.limiter = SlidingWindowCounter(
                window_size_seconds=self.rate_limit.time_window_seconds,
                max_requests=int(self.rate_limit.requests_per_second * self.rate_limit.time_window_seconds)
            )
    
    def is_allowed(self) -> Tuple[bool, Optional[float]]:
        """Check if request is allowed, return (allowed, retry_after_seconds)."""
        
        if isinstance(self.limiter, TokenBucket):
            allowed = self.limiter.allow_request()
            retry_after = 1.0 / self.rate_limit.requests_per_second if not allowed else None
        else:
            allowed = self.limiter.allow_request()
            retry_after = self.limiter.get_retry_after_seconds() if not allowed else None
        
        return allowed, retry_after
    
    def get_status(self) -> Dict:
        """Get rate limiter status."""
        if isinstance(self.limiter, TokenBucket):
            return {
                'strategy': self.strategy.value,
                'tokens_available': self.limiter.get_tokens_available(),
                'capacity': self.limiter.capacity
            }
        else:
            return {
                'strategy': self.strategy.value,
                'requests_in_window': self.limiter.get_request_count(),
                'max_requests': self.limiter.max_requests
            }

class QuotaManager:
    """Manage API quotas for clients."""
    
    def __init__(self):
        self.client_quotas: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            'daily_requests': 0,
            'monthly_requests': 0,
            'current_concurrent': 0,
            'last_reset_daily': datetime.now(),
            'last_reset_monthly': datetime.now()
        })
        self.lock = threading.RLock()
    
    def increment_request(self, client_id: str) -> bool:
        """Increment request counter for client."""
        with self.lock:
            quota = self.client_quotas[client_id]
            quota['daily_requests'] += 1
            quota['monthly_requests'] += 1
            return True
    
    def increment_concurrent(self, client_id: str, quota_limit: QuotaLimit) -> bool:
        """Increment concurrent request counter."""
        with self.lock:
            quota = self.client_quotas[client_id]
            
            if quota['current_concurrent'] >= quota_limit.max_concurrent_requests:
                return False
            
            quota['current_concurrent'] += 1
            return True
    
    def decrement_concurrent(self, client_id: str):
        """Decrement concurrent request counter."""
        with self.lock:
            quota = self.client_quotas[client_id]
            quota['current_concurrent'] = max(0, quota['current_concurrent'] - 1)
    
    def check_daily_quota(self, client_id: str, quota_limit: QuotaLimit) -> Tuple[bool, int]:
        """Check if daily quota exceeded."""
        with self.lock:
            quota = self.client_quotas[client_id]
            
            # Reset if day changed
            if (datetime.now() - quota['last_reset_daily']).days > 0:
                quota['daily_requests'] = 0
                quota['last_reset_daily'] = datetime.now()
            
            remaining = quota_limit.requests_per_day - quota['daily_requests']
            return remaining > 0, remaining
    
    def check_monthly_quota(self, client_id: str, quota_limit: QuotaLimit) -> Tuple[bool, int]:
        """Check if monthly quota exceeded."""
        with self.lock:
            quota = self.client_quotas[client_id]
            
            # Reset if month changed
            if (datetime.now() - quota['last_reset_monthly']).days > 30:
                quota['monthly_requests'] = 0
                quota['last_reset_monthly'] = datetime.now()
            
            remaining = quota_limit.requests_per_month - quota['monthly_requests']
            return remaining > 0, remaining
    
    def get_quota_status(self, client_id: str, quota_limit: QuotaLimit) -> Dict:
        """Get quota status for client."""
        with self.lock:
            quota = self.client_quotas[client_id]
            
            daily_allowed, daily_remaining = self.check_daily_quota(client_id, quota_limit)
            monthly_allowed, monthly_remaining = self.check_monthly_quota(client_id, quota_limit)
            
            return {
                'client_id': client_id,
                'daily': {
                    'used': quota['daily_requests'],
                    'limit': quota_limit.requests_per_day,
                    'remaining': daily_remaining
                },
                'monthly': {
                    'used': quota['monthly_requests'],
                    'limit': quota_limit.requests_per_month,
                    'remaining': monthly_remaining
                },
                'concurrent': {
                    'current': quota['current_concurrent'],
                    'limit': quota_limit.max_concurrent_requests
                }
            }

class RateLimitingMiddleware:
    """Middleware for rate limiting and quota enforcement."""
    
    def __init__(self, default_rate_limit: RateLimit = None,
                 default_quota_limit: QuotaLimit = None):
        self.default_rate_limit = default_rate_limit or RateLimit(
            requests_per_second=10,
            burst_size=20,
            time_window_seconds=60
        )
        self.default_quota_limit = default_quota_limit or QuotaLimit(
            requests_per_day=10000,
            requests_per_month=100000,
            max_concurrent_requests=100
        )
        
        self.client_limiters: Dict[str, ClientRateLimiter] = {}
        self.quota_manager = QuotaManager()
        self.lock = threading.RLock()
    
    def get_or_create_limiter(self, client_id: str) -> ClientRateLimiter:
        """Get or create rate limiter for client."""
        with self.lock:
            if client_id not in self.client_limiters:
                self.client_limiters[client_id] = ClientRateLimiter(
                    client_id,
                    rate_limit=self.default_rate_limit
                )
            
            return self.client_limiters[client_id]
    
    def check_limits(self, client_id: str) -> Tuple[bool, Optional[str]]:
        """Check rate limits and quotas for client."""
        
        # Check rate limit
        limiter = self.get_or_create_limiter(client_id)
        allowed, retry_after = limiter.is_allowed()
        
        if not allowed:
            return False, f"Rate limit exceeded. Retry after {retry_after:.1f}s"
        
        # Check daily quota
        daily_allowed, daily_remaining = self.quota_manager.check_daily_quota(
            client_id, self.default_quota_limit
        )
        
        if not daily_allowed:
            return False, f"Daily quota exceeded"
        
        # Check monthly quota
        monthly_allowed, monthly_remaining = self.quota_manager.check_monthly_quota(
            client_id, self.default_quota_limit
        )
        
        if not monthly_allowed:
            return False, f"Monthly quota exceeded"
        
        # Check concurrent requests
        if not self.quota_manager.increment_concurrent(client_id, self.default_quota_limit):
            return False, "Maximum concurrent requests reached"
        
        # Increment request counters
        self.quota_manager.increment_request(client_id)
        
        return True, None
    
    def release_concurrent(self, client_id: str):
        """Release concurrent request."""
        self.quota_manager.decrement_concurrent(client_id)
    
    def get_client_status(self, client_id: str) -> Dict:
        """Get detailed status for client."""
        limiter = self.get_or_create_limiter(client_id)
        
        return {
            'rate_limiter': limiter.get_status(),
            'quota': self.quota_manager.get_quota_status(client_id, self.default_quota_limit)
        }

# Example usage
if __name__ == "__main__":
    # Create middleware
    middleware = RateLimitingMiddleware()
    
    # Simulate requests
    client_id = "client_123"
    
    for i in range(15):
        allowed, error = middleware.check_limits(client_id)
        
        if allowed:
            print(f"Request {i+1}: ALLOWED")
            middleware.release_concurrent(client_id)
        else:
            print(f"Request {i+1}: BLOCKED - {error}")
    
    # Get status
    status = middleware.get_client_status(client_id)
    print(f"\nClient Status:")
    print(json.dumps(status, indent=2, default=str))
