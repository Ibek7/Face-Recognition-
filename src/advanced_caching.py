# Advanced Caching Strategy System

import time
import threading
from typing import Any, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import hashlib
import json

class CacheTier(Enum):
    """Cache tier levels."""
    L1 = 1  # In-memory local
    L2 = 2  # Shared memory/Redis
    L3 = 3  # Persistent storage
    CLOUD = 4  # Cloud storage

class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live

@dataclass
class CacheEntry:
    """Cache entry metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[int] = None  # seconds
    tier: CacheTier = CacheTier.L1
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def update_access(self):
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'key': self.key,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'ttl': self.ttl,
            'tier': self.tier.value,
            'expired': self.is_expired()
        }

class L1Cache:
    """L1 in-memory cache."""
    
    def __init__(self, max_size: int = 1000, policy: EvictionPolicy = EvictionPolicy.LRU):
        self.max_size = max_size
        self.policy = policy
        self.cache: Dict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None
            
            entry.update_access()
            self.hits += 1
            
            # Move to end (for LRU)
            if self.policy == EvictionPolicy.LRU:
                self.cache.move_to_end(key)
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: int = None):
        """Put value in cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            
            entry = CacheEntry(key=key, value=value, ttl=ttl, tier=CacheTier.L1)
            self.cache[key] = entry
            
            # Evict if necessary
            if len(self.cache) > self.max_size:
                self._evict()
    
    def _evict(self):
        """Evict entry based on policy."""
        if not self.cache:
            return
        
        if self.policy == EvictionPolicy.LRU:
            # Remove first (oldest)
            key = next(iter(self.cache))
        elif self.policy == EvictionPolicy.LFU:
            # Remove least frequently used
            key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].access_count)
        elif self.policy == EvictionPolicy.FIFO:
            # Remove oldest by creation time
            key = min(self.cache.keys(),
                     key=lambda k: self.cache[k].created_at)
        else:
            key = next(iter(self.cache))
        
        del self.cache[key]
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'policy': self.policy.value
            }

class L2Cache:
    """L2 distributed cache (Redis-like)."""
    
    def __init__(self, persistence_enabled: bool = True):
        self.cache: Dict[str, CacheEntry] = {}
        self.persistence_enabled = persistence_enabled
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            if entry.is_expired():
                del self.cache[key]
                self.misses += 1
                return None
            
            entry.update_access()
            self.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl: int = None):
        """Put value in cache."""
        with self.lock:
            entry = CacheEntry(key=key, value=value, ttl=ttl, tier=CacheTier.L2)
            self.cache[key] = entry
    
    def invalidate(self, pattern: str = None):
        """Invalidate entries."""
        with self.lock:
            if pattern is None:
                self.cache.clear()
            else:
                keys_to_delete = [k for k in self.cache if pattern in k]
                for k in keys_to_delete:
                    del self.cache[k]

class CacheKeyBuilder:
    """Build cache keys with namespacing."""
    
    def __init__(self, namespace: str = "cache"):
        self.namespace = namespace
    
    def build_key(self, *parts, **params) -> str:
        """Build cache key."""
        key_parts = [self.namespace] + list(parts)
        key = ":".join(str(p) for p in key_parts)
        
        # Add parameters to key if present
        if params:
            params_str = json.dumps(params, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            key = f"{key}::{params_hash}"
        
        return key

class MultiTierCache:
    """Multi-tier caching system."""
    
    def __init__(self):
        self.l1 = L1Cache(max_size=1000)
        self.l2 = L2Cache()
        self.lock = threading.RLock()
        self.key_builder = CacheKeyBuilder()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy."""
        
        # Try L1
        value = self.l1.get(key)
        if value is not None:
            return value
        
        # Try L2
        value = self.l2.get(key)
        if value is not None:
            # Populate L1
            self.l1.put(key, value)
            return value
        
        return None
    
    def put(self, key: str, value: Any, ttl: int = None, 
           tiers: list = None):
        """Put value in specified tiers."""
        if tiers is None:
            tiers = [CacheTier.L1, CacheTier.L2]
        
        if CacheTier.L1 in tiers:
            self.l1.put(key, value, ttl)
        
        if CacheTier.L2 in tiers:
            self.l2.put(key, value, ttl)
    
    def invalidate(self, pattern: str = None, tiers: list = None):
        """Invalidate entries."""
        if tiers is None:
            tiers = [CacheTier.L1, CacheTier.L2]
        
        if CacheTier.L1 in tiers:
            # For L1, need to iterate and check
            pass
        
        if CacheTier.L2 in tiers:
            self.l2.invalidate(pattern)

class CachedFunction:
    """Decorator for caching function results."""
    
    def __init__(self, cache: MultiTierCache, ttl: int = 3600, 
                key_builder: CacheKeyBuilder = None):
        self.cache = cache
        self.ttl = ttl
        self.key_builder = key_builder or CacheKeyBuilder()
        self.call_count = 0
        self.cache_hits = 0
    
    def __call__(self, func: Callable) -> Callable:
        """Apply caching decorator."""
        
        def wrapper(*args, **kwargs):
            # Build cache key
            key = self.key_builder.build_key(
                func.__module__,
                func.__name__,
                *args,
                **kwargs
            )
            
            # Try to get from cache
            cached_value = self.cache.get(key)
            if cached_value is not None:
                self.cache_hits += 1
                return cached_value
            
            # Call function
            self.call_count += 1
            result = func(*args, **kwargs)
            
            # Cache result
            self.cache.put(key, result, self.ttl)
            
            return result
        
        wrapper.cache_hits = lambda: self.cache_hits
        wrapper.call_count = lambda: self.call_count
        wrapper.hit_rate = lambda: (self.cache_hits / self.call_count * 100) if self.call_count > 0 else 0
        
        return wrapper

class CacheWarmer:
    """Preload cache with data."""
    
    def __init__(self, cache: MultiTierCache):
        self.cache = cache
    
    def warm_cache(self, data_loader: Callable, keys: list, 
                  ttl: int = None):
        """Warm cache with data."""
        
        for key in keys:
            try:
                value = data_loader(key)
                self.cache.put(key, value, ttl)
            except Exception as e:
                print(f"Error warming cache for key {key}: {e}")

# Example usage
if __name__ == "__main__":
    # Create multi-tier cache
    cache = MultiTierCache()
    
    # Add data
    cache.put("user:1:profile", {"name": "John", "age": 30}, ttl=3600)
    cache.put("user:2:profile", {"name": "Jane", "age": 28}, ttl=3600)
    
    # Retrieve data
    user1 = cache.get("user:1:profile")
    print(f"User 1: {user1}")
    
    # Check stats
    l1_stats = cache.l1.get_stats()
    print(f"\nL1 Cache Stats:")
    print(f"  Hit Rate: {l1_stats['hit_rate']:.1f}%")
    print(f"  Size: {l1_stats['size']}/{l1_stats['max_size']}")
    
    # Test cached function
    cache_mgr = MultiTierCache()
    key_builder = CacheKeyBuilder("functions")
    decorator = CachedFunction(cache_mgr, ttl=300, key_builder=key_builder)
    
    @decorator
    def expensive_operation(x, y):
        time.sleep(0.1)
        return x + y
    
    result1 = expensive_operation(5, 3)
    result2 = expensive_operation(5, 3)  # From cache
    
    print(f"\nCached Function:")
    print(f"  Calls: {expensive_operation.call_count()}")
    print(f"  Cache Hits: {expensive_operation.cache_hits()}")
    print(f"  Hit Rate: {expensive_operation.hit_rate():.1f}%")
