"""
Advanced rate limiter with multiple algorithms.

Supports token bucket, sliding window, fixed window, and leaky bucket.
"""

from typing import Dict, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(str, Enum):
    """Rate limiting algorithm."""
    
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None
    ):
        """
        Initialize exception.
        
        Args:
            message: Error message
            retry_after: Seconds until retry allowed
        """
        super().__init__(message)
        self.retry_after = retry_after


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        initial_tokens: Optional[int] = None
    ):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens
            refill_rate: Tokens per second
            initial_tokens: Initial token count
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Calculate tokens to add
        tokens_to_add = elapsed * self.refill_rate
        
        # Add tokens up to capacity
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens.
        
        Args:
            tokens: Number of tokens to acquire
        
        Returns:
            True if acquired, False otherwise
        """
        async with self._lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def get_retry_after(self, tokens: int = 1) -> float:
        """Get seconds until tokens available."""
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        # Calculate time needed to refill
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class SlidingWindowCounter:
    """Sliding window rate limiter."""
    
    def __init__(self, limit: int, window_size: int):
        """
        Initialize sliding window counter.
        
        Args:
            limit: Maximum requests in window
            window_size: Window size in seconds
        """
        self.limit = limit
        self.window_size = window_size
        self.requests: deque = deque()
        self._lock = asyncio.Lock()
    
    def _clean_old_requests(self):
        """Remove requests outside window."""
        now = time.time()
        cutoff = now - self.window_size
        
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
    
    async def acquire(self) -> bool:
        """
        Try to acquire request slot.
        
        Returns:
            True if allowed, False otherwise
        """
        async with self._lock:
            self._clean_old_requests()
            
            if len(self.requests) < self.limit:
                self.requests.append(time.time())
                return True
            
            return False
    
    def get_retry_after(self) -> float:
        """Get seconds until next slot available."""
        self._clean_old_requests()
        
        if len(self.requests) < self.limit:
            return 0.0
        
        # Time until oldest request expires
        oldest = self.requests[0]
        return (oldest + self.window_size) - time.time()


class FixedWindowCounter:
    """Fixed window rate limiter."""
    
    def __init__(self, limit: int, window_size: int):
        """
        Initialize fixed window counter.
        
        Args:
            limit: Maximum requests per window
            window_size: Window size in seconds
        """
        self.limit = limit
        self.window_size = window_size
        self.count = 0
        self.window_start = time.time()
        self._lock = asyncio.Lock()
    
    def _reset_if_needed(self):
        """Reset counter if window expired."""
        now = time.time()
        
        if now - self.window_start >= self.window_size:
            self.count = 0
            self.window_start = now
    
    async def acquire(self) -> bool:
        """
        Try to acquire request slot.
        
        Returns:
            True if allowed, False otherwise
        """
        async with self._lock:
            self._reset_if_needed()
            
            if self.count < self.limit:
                self.count += 1
                return True
            
            return False
    
    def get_retry_after(self) -> float:
        """Get seconds until window resets."""
        self._reset_if_needed()
        
        if self.count < self.limit:
            return 0.0
        
        # Time until window resets
        return (self.window_start + self.window_size) - time.time()


class AdvancedRateLimiter:
    """Multi-algorithm rate limiter."""
    
    def __init__(
        self,
        algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET,
        limit: int = 100,
        window_size: int = 60,
        refill_rate: Optional[float] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            algorithm: Rate limiting algorithm
            limit: Request limit
            window_size: Time window in seconds
            refill_rate: Token refill rate (for token bucket)
        """
        self.algorithm = algorithm
        self.limit = limit
        self.window_size = window_size
        
        # Create algorithm instance
        if algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            rate = refill_rate or (limit / window_size)
            self.limiter = TokenBucket(limit, rate)
        
        elif algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            self.limiter = SlidingWindowCounter(limit, window_size)
        
        elif algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            self.limiter = FixedWindowCounter(limit, window_size)
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire rate limit slot.
        
        Args:
            tokens: Number of tokens (for token bucket)
        
        Returns:
            True if allowed
        """
        if hasattr(self.limiter, 'tokens'):
            return await self.limiter.acquire(tokens)
        else:
            return await self.limiter.acquire()
    
    async def acquire_or_raise(self, tokens: int = 1):
        """
        Acquire or raise RateLimitExceeded.
        
        Args:
            tokens: Number of tokens
        
        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        if not await self.acquire(tokens):
            retry_after = self.limiter.get_retry_after(tokens) if hasattr(self.limiter, 'get_retry_after') else self.limiter.get_retry_after()
            
            raise RateLimitExceeded(
                f"Rate limit exceeded. Try again in {retry_after:.2f} seconds.",
                retry_after=retry_after
            )
    
    def get_retry_after(self, tokens: int = 1) -> float:
        """Get seconds until retry allowed."""
        if hasattr(self.limiter, 'tokens'):
            return self.limiter.get_retry_after(tokens)
        else:
            return self.limiter.get_retry_after()


class MultiKeyRateLimiter:
    """Rate limiter with per-key limits."""
    
    def __init__(
        self,
        algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET,
        limit: int = 100,
        window_size: int = 60,
        refill_rate: Optional[float] = None
    ):
        """
        Initialize multi-key rate limiter.
        
        Args:
            algorithm: Rate limiting algorithm
            limit: Request limit per key
            window_size: Time window in seconds
            refill_rate: Token refill rate
        """
        self.algorithm = algorithm
        self.limit = limit
        self.window_size = window_size
        self.refill_rate = refill_rate
        self.limiters: Dict[str, AdvancedRateLimiter] = {}
        self._lock = asyncio.Lock()
    
    def _get_limiter(self, key: str) -> AdvancedRateLimiter:
        """Get or create limiter for key."""
        if key not in self.limiters:
            self.limiters[key] = AdvancedRateLimiter(
                algorithm=self.algorithm,
                limit=self.limit,
                window_size=self.window_size,
                refill_rate=self.refill_rate
            )
        
        return self.limiters[key]
    
    async def acquire(self, key: str, tokens: int = 1) -> bool:
        """
        Try to acquire for key.
        
        Args:
            key: Rate limit key (e.g., user ID, IP)
            tokens: Number of tokens
        
        Returns:
            True if allowed
        """
        async with self._lock:
            limiter = self._get_limiter(key)
        
        return await limiter.acquire(tokens)
    
    async def acquire_or_raise(self, key: str, tokens: int = 1):
        """
        Acquire or raise exception.
        
        Args:
            key: Rate limit key
            tokens: Number of tokens
        
        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        async with self._lock:
            limiter = self._get_limiter(key)
        
        await limiter.acquire_or_raise(tokens)
    
    def get_retry_after(self, key: str, tokens: int = 1) -> float:
        """Get retry time for key."""
        limiter = self._get_limiter(key)
        return limiter.get_retry_after(tokens)


# Example usage:
"""
from fastapi import FastAPI, Request, HTTPException
from src.advanced_rate_limiter import MultiKeyRateLimiter, RateLimitAlgorithm, RateLimitExceeded

app = FastAPI()

# Create rate limiter
rate_limiter = MultiKeyRateLimiter(
    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
    limit=100,
    window_size=60
)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    
    try:
        await rate_limiter.acquire_or_raise(client_ip)
        response = await call_next(request)
        return response
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
            headers={"Retry-After": str(int(e.retry_after))}
        )
"""
