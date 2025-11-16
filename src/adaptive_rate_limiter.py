"""
Adaptive Rate Limiter

Smart rate limiting with:
- Token bucket algorithm
- Sliding window counters
- Adaptive rate adjustment based on load
- Per-user, per-IP, and per-endpoint limits
- Redis-backed distributed rate limiting
- Rate limit headers (X-RateLimit-*)
"""

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import redis.asyncio as aioredis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitStrategy(str, Enum):
    """Rate limiting strategy"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests: int  # Number of requests allowed
    window: int  # Time window in seconds
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    burst: Optional[int] = None  # Burst capacity (for token bucket)
    
    def __post_init__(self):
        if self.burst is None:
            self.burst = self.requests * 2


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    limit: int
    remaining: int
    reset: int  # Unix timestamp
    retry_after: Optional[int] = None  # Seconds to wait
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers"""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(self.reset)
        }
        
        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)
        
        return headers


class TokenBucketLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis = redis_client
        self.local_buckets: Dict[str, Dict[str, Any]] = {}
    
    async def check_limit(
        self,
        key: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Check if request is allowed"""
        if self.redis:
            return await self._check_redis(key, config)
        else:
            return await self._check_local(key, config)
    
    async def _check_local(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        """Check rate limit using local storage"""
        now = time.time()
        
        # Initialize bucket if not exists
        if key not in self.local_buckets:
            self.local_buckets[key] = {
                "tokens": config.burst,
                "last_update": now
            }
        
        bucket = self.local_buckets[key]
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = now - bucket["last_update"]
        tokens_to_add = time_elapsed * (config.requests / config.window)
        
        # Update bucket
        bucket["tokens"] = min(
            config.burst,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_update"] = now
        
        # Check if request allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            allowed = True
            remaining = int(bucket["tokens"])
        else:
            allowed = False
            remaining = 0
        
        # Calculate reset time
        if bucket["tokens"] < config.burst:
            time_to_full = (config.burst - bucket["tokens"]) * (config.window / config.requests)
            reset = int(now + time_to_full)
        else:
            reset = int(now + config.window)
        
        # Calculate retry after
        retry_after = None
        if not allowed:
            retry_after = int((1 - bucket["tokens"]) * (config.window / config.requests))
        
        return RateLimitResult(
            allowed=allowed,
            limit=config.requests,
            remaining=remaining,
            reset=reset,
            retry_after=retry_after
        )
    
    async def _check_redis(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        """Check rate limit using Redis"""
        now = time.time()
        redis_key = f"ratelimit:token_bucket:{key}"
        
        # Get current bucket state
        bucket_data = await self.redis.get(redis_key)
        
        if bucket_data:
            bucket = eval(bucket_data)
        else:
            bucket = {
                "tokens": config.burst,
                "last_update": now
            }
        
        # Calculate tokens to add
        time_elapsed = now - bucket["last_update"]
        tokens_to_add = time_elapsed * (config.requests / config.window)
        
        # Update bucket
        bucket["tokens"] = min(
            config.burst,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_update"] = now
        
        # Check if allowed
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            allowed = True
            remaining = int(bucket["tokens"])
        else:
            allowed = False
            remaining = 0
        
        # Save bucket state
        await self.redis.setex(
            redis_key,
            config.window * 2,
            str(bucket)
        )
        
        # Calculate reset
        reset = int(now + config.window)
        
        # Calculate retry after
        retry_after = None
        if not allowed:
            retry_after = int((1 - bucket["tokens"]) * (config.window / config.requests))
        
        return RateLimitResult(
            allowed=allowed,
            limit=config.requests,
            remaining=remaining,
            reset=reset,
            retry_after=retry_after
        )


class SlidingWindowLimiter:
    """Sliding window rate limiter"""
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis = redis_client
        self.local_windows: Dict[str, list] = {}
    
    async def check_limit(
        self,
        key: str,
        config: RateLimitConfig
    ) -> RateLimitResult:
        """Check if request is allowed"""
        if self.redis:
            return await self._check_redis(key, config)
        else:
            return await self._check_local(key, config)
    
    async def _check_local(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        """Check rate limit using local storage"""
        now = time.time()
        window_start = now - config.window
        
        # Initialize window if not exists
        if key not in self.local_windows:
            self.local_windows[key] = []
        
        # Remove old requests
        self.local_windows[key] = [
            timestamp for timestamp in self.local_windows[key]
            if timestamp > window_start
        ]
        
        # Count requests in window
        request_count = len(self.local_windows[key])
        
        # Check if allowed
        if request_count < config.requests:
            self.local_windows[key].append(now)
            allowed = True
            remaining = config.requests - request_count - 1
        else:
            allowed = False
            remaining = 0
        
        # Calculate reset (when oldest request expires)
        if self.local_windows[key]:
            reset = int(self.local_windows[key][0] + config.window)
        else:
            reset = int(now + config.window)
        
        # Calculate retry after
        retry_after = None
        if not allowed and self.local_windows[key]:
            retry_after = int(self.local_windows[key][0] + config.window - now)
        
        return RateLimitResult(
            allowed=allowed,
            limit=config.requests,
            remaining=max(0, remaining),
            reset=reset,
            retry_after=retry_after
        )
    
    async def _check_redis(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        """Check rate limit using Redis"""
        now = time.time()
        redis_key = f"ratelimit:sliding_window:{key}"
        
        # Remove old requests
        await self.redis.zremrangebyscore(redis_key, 0, now - config.window)
        
        # Count requests in window
        request_count = await self.redis.zcard(redis_key)
        
        # Check if allowed
        if request_count < config.requests:
            await self.redis.zadd(redis_key, {str(now): now})
            await self.redis.expire(redis_key, config.window)
            allowed = True
            remaining = config.requests - request_count - 1
        else:
            allowed = False
            remaining = 0
        
        # Calculate reset
        oldest = await self.redis.zrange(redis_key, 0, 0, withscores=True)
        if oldest:
            reset = int(oldest[0][1] + config.window)
        else:
            reset = int(now + config.window)
        
        # Calculate retry after
        retry_after = None
        if not allowed and oldest:
            retry_after = int(oldest[0][1] + config.window - now)
        
        return RateLimitResult(
            allowed=allowed,
            limit=config.requests,
            remaining=max(0, remaining),
            reset=reset,
            retry_after=retry_after
        )


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter with multiple strategies
    and automatic adjustment based on system load
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    ):
        self.default_strategy = default_strategy
        self.redis: Optional[aioredis.Redis] = None
        self.redis_url = redis_url
        
        # Rate limit configs per key pattern
        self.configs: Dict[str, RateLimitConfig] = {}
        
        # Limiters
        self.token_bucket = None
        self.sliding_window = None
        
        # System load factor (1.0 = normal, <1.0 = reduce limits, >1.0 = increase)
        self.load_factor = 1.0
    
    async def initialize(self):
        """Initialize Redis connection"""
        if self.redis_url:
            self.redis = await aioredis.from_url(self.redis_url)
        
        self.token_bucket = TokenBucketLimiter(self.redis)
        self.sliding_window = SlidingWindowLimiter(self.redis)
        
        logger.info("Adaptive rate limiter initialized")
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
    
    def configure(
        self,
        pattern: str,
        requests: int,
        window: int,
        strategy: Optional[RateLimitStrategy] = None,
        burst: Optional[int] = None
    ):
        """
        Configure rate limit for a key pattern
        
        Args:
            pattern: Key pattern (e.g., "user:*", "ip:*", "endpoint:/api/detect")
            requests: Number of requests allowed
            window: Time window in seconds
            strategy: Rate limiting strategy
            burst: Burst capacity
        """
        config = RateLimitConfig(
            requests=requests,
            window=window,
            strategy=strategy or self.default_strategy,
            burst=burst
        )
        
        self.configs[pattern] = config
        logger.info(f"Configured rate limit for '{pattern}': {requests} req/{window}s")
    
    def _get_config(self, key: str) -> Optional[RateLimitConfig]:
        """Get config for key"""
        # Direct match
        if key in self.configs:
            return self.configs[key]
        
        # Pattern match
        for pattern, config in self.configs.items():
            if "*" in pattern:
                prefix = pattern.split("*")[0]
                if key.startswith(prefix):
                    return config
        
        return None
    
    async def check_limit(
        self,
        key: str,
        requests: Optional[int] = None,
        window: Optional[int] = None,
        strategy: Optional[RateLimitStrategy] = None
    ) -> RateLimitResult:
        """
        Check rate limit for key
        
        Args:
            key: Rate limit key (e.g., "user:123", "ip:1.2.3.4")
            requests: Override request limit
            window: Override time window
            strategy: Override strategy
        
        Returns:
            RateLimitResult
        """
        # Get config
        config = self._get_config(key)
        
        if not config:
            if requests is None or window is None:
                raise ValueError(f"No config found for key '{key}' and no override provided")
            
            config = RateLimitConfig(
                requests=requests,
                window=window,
                strategy=strategy or self.default_strategy
            )
        
        # Apply load factor
        adjusted_config = RateLimitConfig(
            requests=int(config.requests * self.load_factor),
            window=config.window,
            strategy=config.strategy,
            burst=int(config.burst * self.load_factor) if config.burst else None
        )
        
        # Select limiter based on strategy
        if adjusted_config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            limiter = self.token_bucket
        elif adjusted_config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            limiter = self.sliding_window
        else:
            limiter = self.token_bucket
        
        # Check limit
        result = await limiter.check_limit(key, adjusted_config)
        
        return result
    
    def set_load_factor(self, factor: float):
        """
        Adjust rate limits based on system load
        
        Args:
            factor: Load factor (1.0 = normal, 0.5 = half capacity, 2.0 = double)
        """
        self.load_factor = max(0.1, min(2.0, factor))
        logger.info(f"Rate limit load factor set to {self.load_factor}")
    
    async def get_stats(self, key: str) -> Dict[str, Any]:
        """Get rate limit statistics for key"""
        if not self.redis:
            return {}
        
        stats = {}
        
        # Token bucket stats
        tb_key = f"ratelimit:token_bucket:{key}"
        tb_data = await self.redis.get(tb_key)
        if tb_data:
            stats["token_bucket"] = eval(tb_data)
        
        # Sliding window stats
        sw_key = f"ratelimit:sliding_window:{key}"
        sw_count = await self.redis.zcard(sw_key)
        stats["sliding_window_count"] = sw_count
        
        return stats


# FastAPI middleware example
class RateLimitMiddleware:
    """Rate limiting middleware for FastAPI"""
    
    def __init__(self, limiter: AdaptiveRateLimiter):
        self.limiter = limiter
    
    async def __call__(self, request, call_next):
        """Process request with rate limiting"""
        # Extract rate limit key (e.g., from IP or user ID)
        client_ip = request.client.host
        key = f"ip:{client_ip}"
        
        # Check rate limit
        result = await self.limiter.check_limit(key)
        
        # Add rate limit headers
        headers = result.to_headers()
        
        if not result.allowed:
            # Rate limit exceeded
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"},
                headers=headers
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value
        
        return response


# Example usage
async def main():
    """Example usage"""
    # Initialize limiter
    limiter = AdaptiveRateLimiter()
    await limiter.initialize()
    
    # Configure rate limits
    limiter.configure("user:*", requests=100, window=60)  # 100 req/min per user
    limiter.configure("ip:*", requests=50, window=60)  # 50 req/min per IP
    limiter.configure("endpoint:/api/detect", requests=10, window=1)  # 10 req/sec for endpoint
    
    # Simulate requests
    print("Testing rate limiter...\n")
    
    user_key = "user:123"
    
    for i in range(15):
        result = await limiter.check_limit(user_key)
        
        print(f"Request {i+1}:")
        print(f"  Allowed: {result.allowed}")
        print(f"  Remaining: {result.remaining}/{result.limit}")
        print(f"  Reset: {datetime.fromtimestamp(result.reset).isoformat()}")
        if result.retry_after:
            print(f"  Retry after: {result.retry_after}s")
        print()
        
        if not result.allowed:
            print(f"Rate limit exceeded! Waiting {result.retry_after}s...")
            await asyncio.sleep(result.retry_after)
    
    # Adjust for high load
    print("Reducing capacity due to high load...")
    limiter.set_load_factor(0.5)
    
    result = await limiter.check_limit(user_key)
    print(f"With 50% capacity: Remaining = {result.remaining}/{result.limit}\n")
    
    await limiter.close()


if __name__ == "__main__":
    asyncio.run(main())
