"""
Response caching layer for FastAPI applications.

Provides HTTP response caching with TTL and cache invalidation.
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any, Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with metadata."""
    
    def __init__(
        self,
        content: bytes,
        status_code: int,
        headers: dict,
        ttl: int
    ):
        """
        Initialize cache entry.
        
        Args:
            content: Response content
            status_code: HTTP status code
            headers: Response headers
            ttl: Time to live in seconds
        """
        self.content = content
        self.status_code = status_code
        self.headers = headers
        self.created_at = time.time()
        self.ttl = ttl
        self.hits = 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.created_at > self.ttl
    
    def get_age(self) -> int:
        """Get entry age in seconds."""
        return int(time.time() - self.created_at)


class ResponseCache:
    """HTTP response cache manager."""
    
    def __init__(
        self,
        default_ttl: int = 300,
        max_size: int = 1000
    ):
        """
        Initialize response cache.
        
        Args:
            default_ttl: Default TTL in seconds
            max_size: Maximum cache entries
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def _generate_key(
        self,
        method: str,
        path: str,
        query_params: str,
        headers: Optional[dict] = None
    ) -> str:
        """
        Generate cache key.
        
        Args:
            method: HTTP method
            path: Request path
            query_params: Query parameters
            headers: Request headers to include
        
        Returns:
            Cache key
        """
        key_parts = [method, path, query_params]
        
        # Include specific headers if needed
        if headers:
            vary_headers = ["Accept", "Accept-Encoding"]
            for header in vary_headers:
                if header in headers:
                    key_parts.append(f"{header}:{headers[header]}")
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(
        self,
        method: str,
        path: str,
        query_params: str
    ) -> Optional[CacheEntry]:
        """
        Get cached response.
        
        Args:
            method: HTTP method
            path: Request path
            query_params: Query parameters
        
        Returns:
            Cache entry or None
        """
        key = self._generate_key(method, path, query_params)
        entry = self.cache.get(key)
        
        if not entry:
            self.stats["misses"] += 1
            return None
        
        # Check expiration
        if entry.is_expired():
            del self.cache[key]
            self.stats["misses"] += 1
            return None
        
        # Update stats
        entry.hits += 1
        self.stats["hits"] += 1
        
        return entry
    
    def set(
        self,
        method: str,
        path: str,
        query_params: str,
        content: bytes,
        status_code: int,
        headers: dict,
        ttl: Optional[int] = None
    ):
        """
        Cache response.
        
        Args:
            method: HTTP method
            path: Request path
            query_params: Query parameters
            content: Response content
            status_code: HTTP status code
            headers: Response headers
            ttl: Time to live
        """
        # Check cache size
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        key = self._generate_key(method, path, query_params)
        ttl = ttl or self.default_ttl
        
        entry = CacheEntry(content, status_code, headers, ttl)
        self.cache[key] = entry
        
        logger.debug(f"Cached response: {method} {path} (TTL: {ttl}s)")
    
    def invalidate(self, pattern: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            pattern: Path pattern to match (None = clear all)
        """
        if pattern is None:
            # Clear all
            count = len(self.cache)
            self.cache.clear()
            logger.info(f"Invalidated all cache entries ({count})")
        else:
            # Clear matching pattern
            keys_to_delete = [
                key for key in self.cache
                if pattern in key
            ]
            
            for key in keys_to_delete:
                del self.cache[key]
            
            logger.info(
                f"Invalidated {len(keys_to_delete)} cache entries "
                f"matching '{pattern}'"
            )
    
    def _evict_oldest(self):
        """Evict oldest cache entry."""
        if not self.cache:
            return
        
        oldest_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].created_at
        )
        
        del self.cache[oldest_key]
        self.stats["evictions"] += 1
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests * 100
            if total_requests > 0
            else 0
        )
        
        return {
            "entries": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "hit_rate": round(hit_rate, 2)
        }


class CacheMiddleware(BaseHTTPMiddleware):
    """Cache middleware for FastAPI."""
    
    def __init__(
        self,
        app,
        cache: ResponseCache,
        cache_methods: Optional[list] = None,
        exclude_paths: Optional[list] = None
    ):
        """
        Initialize cache middleware.
        
        Args:
            app: FastAPI application
            cache: Response cache instance
            cache_methods: HTTP methods to cache
            exclude_paths: Paths to exclude from caching
        """
        super().__init__(app)
        self.cache = cache
        self.cache_methods = cache_methods or ["GET"]
        self.exclude_paths = exclude_paths or []
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request with caching."""
        # Check if should cache
        if not self._should_cache(request):
            return await call_next(request)
        
        # Try to get from cache
        query_params = str(request.query_params)
        cached = self.cache.get(
            request.method,
            request.url.path,
            query_params
        )
        
        if cached:
            # Return cached response
            response = Response(
                content=cached.content,
                status_code=cached.status_code,
                headers=dict(cached.headers)
            )
            response.headers["X-Cache"] = "HIT"
            response.headers["Age"] = str(cached.get_age())
            
            return response
        
        # Get fresh response
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            self.cache.set(
                request.method,
                request.url.path,
                query_params,
                body,
                response.status_code,
                dict(response.headers)
            )
            
            # Create new response with cached body
            response = Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            response.headers["X-Cache"] = "MISS"
        
        return response
    
    def _should_cache(self, request: Request) -> bool:
        """Check if request should be cached."""
        # Check method
        if request.method not in self.cache_methods:
            return False
        
        # Check excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return False
        
        # Check cache-control header
        cache_control = request.headers.get("Cache-Control", "")
        if "no-cache" in cache_control or "no-store" in cache_control:
            return False
        
        return True


# Global cache instance
response_cache = ResponseCache(default_ttl=300, max_size=1000)


# Example usage:
"""
from fastapi import FastAPI
from src.response_cache import CacheMiddleware, response_cache

app = FastAPI()

# Add cache middleware
app.add_middleware(
    CacheMiddleware,
    cache=response_cache,
    cache_methods=["GET"],
    exclude_paths=["/health", "/admin"]
)

@app.get("/api/data")
async def get_data():
    # This will be cached
    return {"data": "expensive_operation"}

@app.post("/api/invalidate")
async def invalidate_cache(pattern: str = None):
    # Invalidate cache
    response_cache.invalidate(pattern)
    return {"message": "Cache invalidated"}

@app.get("/api/cache/stats")
async def cache_stats():
    return response_cache.get_stats()
"""
