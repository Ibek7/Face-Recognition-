"""
Simple in-memory rate limiter middleware for FastAPI/Starlette.

This implements a token-bucket style limiter keyed by client IP.
It's intentionally lightweight and suitable for single-process development.
For production use, replace with a distributed rate limiter (Redis, etc.).
"""
from typing import Callable, Dict
import time
import logging
import os

from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        # capacity: maximum tokens in bucket
        # refill_rate: tokens per second
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.timestamp = time.monotonic()

    def consume(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self.timestamp
        # refill
        refill = elapsed * self.refill_rate
        if refill > 0:
            self.tokens = min(self.capacity, self.tokens + refill)
            self.timestamp = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware enforcing a token-bucket per-client-IP.

    Configuration via environment variables:
      RATE_LIMIT_CAPACITY (int) default 60
      RATE_LIMIT_REFILL_PER_SEC (float) default 1.0
    """

    def __init__(self, app, get_identifier: Callable[[Request], str] = None):
        super().__init__(app)
        cap = int(os.getenv("RATE_LIMIT_CAPACITY", "60"))
        refill = float(os.getenv("RATE_LIMIT_REFILL_PER_SEC", "1.0"))
        self.capacity = cap
        self.refill = refill
        self.buckets: Dict[str, TokenBucket] = {}
        self.get_identifier = get_identifier or (lambda req: req.client.host if req.client else "unknown")
        logger.info(f"RateLimiter configured: capacity={cap}, refill_per_sec={refill}")

    async def dispatch(self, request: Request, call_next):
        try:
            ident = self.get_identifier(request)
            bucket = self.buckets.get(ident)
            if bucket is None:
                bucket = TokenBucket(self.capacity, self.refill)
                self.buckets[ident] = bucket

            allowed = bucket.consume(1)
            if not allowed:
                # Too many requests
                return JSONResponse({"detail": "Too Many Requests"}, status_code=429)

        except Exception as e:
            logger.exception("Error in rate limiter middleware: %s", e)

        response = await call_next(request)
        return response
