# Advanced Multi-Level Caching System

import threading
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict
import json

class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    ARC = "arc"  # Adaptive Replacement Cache

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'key': self.key,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'ttl': self.ttl,
            'expired': self.is_expired()
        }

class L1Cache:
    """Level 1 Cache - Fast, small capacity."""
    
    def __init__(self, capacity: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.capacity = capacity
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                return None
            
            # Update access info
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self.lock:
            if len(self.cache) >= self.capacity:
                self._evict()
            
            now = time.time()
            self.cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                ttl=ttl
            )
    
    def _evict(self) -> None:
        """Evict entry based on strategy."""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].last_accessed)
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            key = min(self.cache.keys(),
                     key=lambda k: self.cache[k].access_count)
        elif self.strategy == CacheStrategy.FIFO:
            # Evict oldest entry
            key = min(self.cache.keys(),
                     key=lambda k: self.cache[k].created_at)
        else:
            # Default to LRU
            key = min(self.cache.keys(),
                     key=lambda k: self.cache[k].last_accessed)
        
        del self.cache[key]
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'capacity': self.capacity,
                'strategy': self.strategy.value
            }

class L2Cache:
    """Level 2 Cache - Slower, larger capacity."""
    
    def __init__(self, capacity: int = 10000, strategy: CacheStrategy = CacheStrategy.LFU):
        self.capacity = capacity
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                return None
            
            # Move to end for LRU-like behavior
            self.cache.move_to_end(key)
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache."""
        with self.lock:
            if len(self.cache) >= self.capacity:
                self._evict()
            
            now = time.time()
            self.cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                ttl=ttl
            )
            self.cache.move_to_end(key)
    
    def _evict(self) -> None:
        """Evict entry."""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove oldest (first)
            self.cache.popitem(last=False)
        elif self.strategy == CacheStrategy.LFU:
            # Find least frequently used
            key = min(self.cache.keys(),
                     key=lambda k: self.cache[k].access_count)
            del self.cache[key]
        else:
            # Default to LRU
            self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()

class MultiLevelCache:
    """Multi-level cache with warming and invalidation."""
    
    def __init__(self, 
                 l1_capacity: int = 1000,
                 l2_capacity: int = 10000,
                 l1_strategy: CacheStrategy = CacheStrategy.LRU,
                 l2_strategy: CacheStrategy = CacheStrategy.LFU):
        self.l1_cache = L1Cache(l1_capacity, l1_strategy)
        self.l2_cache = L2Cache(l2_capacity, l2_strategy)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            # Try L1
            value = self.l1_cache.get(key)
            if value is not None:
                self.hit_count += 1
                return value
            
            # Try L2
            value = self.l2_cache.get(key)
            if value is not None:
                self.hit_count += 1
                # Promote to L1
                self.l1_cache.put(key, value)
                return value
            
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, 
            level: int = 1) -> None:
        """Put value in cache."""
        with self.lock:
            if level == 1:
                self.l1_cache.put(key, value, ttl)
            elif level == 2:
                self.l2_cache.put(key, value, ttl)
            else:
                # Put in both
                self.l1_cache.put(key, value, ttl)
                self.l2_cache.put(key, value, ttl)
    
    def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        with self.lock:
            if key in self.l1_cache.cache:
                del self.l1_cache.cache[key]
            if key in self.l2_cache.cache:
                del self.l2_cache.cache[key]
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate entries matching pattern."""
        count = 0
        
        with self.lock:
            # L1
            keys_to_delete = [k for k in self.l1_cache.cache.keys() 
                             if pattern in k]
            for key in keys_to_delete:
                del self.l1_cache.cache[key]
                count += 1
            
            # L2
            keys_to_delete = [k for k in self.l2_cache.cache.keys()
                             if pattern in k]
            for key in keys_to_delete:
                del self.l2_cache.cache[key]
                count += 1
        
        return count
    
    def warm_cache(self, data: Dict[str, Any], level: int = 2) -> None:
        """Warm cache with initial data."""
        with self.lock:
            for key, value in data.items():
                self.put(key, value, level=level)
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        with self.lock:
            total = self.hit_count + self.miss_count
            if total == 0:
                return 0.0
            return self.hit_count / total
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            total = self.hit_count + self.miss_count
            return {
                'hits': self.hit_count,
                'misses': self.miss_count,
                'hit_rate': self.get_hit_rate(),
                'l1': self.l1_cache.get_stats(),
                'l2': self.l2_cache.get_stats()
            }
    
    def clear(self) -> None:
        """Clear all caches."""
        with self.lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.hit_count = 0
            self.miss_count = 0

class CacheWarmingStrategy:
    """Cache warming strategies."""
    
    @staticmethod
    def lazy_loading(cache: MultiLevelCache, 
                     loader: Callable) -> None:
        """Lazy load cache on first miss."""
        # Implemented via loader function
        pass
    
    @staticmethod
    def eager_loading(cache: MultiLevelCache,
                      initial_data: Dict[str, Any]) -> None:
        """Eagerly load cache."""
        cache.warm_cache(initial_data)
    
    @staticmethod
    def time_based_refresh(cache: MultiLevelCache,
                          loader: Callable,
                          refresh_interval: float) -> None:
        """Refresh cache at intervals."""
        def refresh():
            while True:
                time.sleep(refresh_interval)
                data = loader()
                cache.warm_cache(data)
        
        thread = threading.Thread(target=refresh, daemon=True)
        thread.start()

class InvalidationStrategy:
    """Cache invalidation strategies."""
    
    @staticmethod
    def time_based(cache: MultiLevelCache, 
                   ttl: float) -> None:
        """Invalidate after TTL."""
        # Handled by CacheEntry.is_expired()
        pass
    
    @staticmethod
    def pattern_based(cache: MultiLevelCache,
                      pattern: str) -> int:
        """Invalidate by pattern."""
        return cache.invalidate_pattern(pattern)
    
    @staticmethod
    def event_based(cache: MultiLevelCache,
                    event_handler: Callable) -> None:
        """Invalidate on event."""
        event_handler(cache)

# Example usage
if __name__ == "__main__":
    cache = MultiLevelCache(l1_capacity=100, l2_capacity=1000)
    
    # Add data
    cache.put("user:1", {"id": 1, "name": "John"})
    cache.put("user:2", {"id": 2, "name": "Jane"})
    
    # Get data
    user1 = cache.get("user:1")
    print(f"User 1: {user1}")
    
    # Warm cache
    initial_data = {
        "user:3": {"id": 3, "name": "Bob"},
        "user:4": {"id": 4, "name": "Alice"}
    }
    cache.warm_cache(initial_data)
    
    # Invalidate pattern
    count = cache.invalidate_pattern("user:3")
    print(f"Invalidated {count} entries")
    
    # Get stats
    stats = cache.get_stats()
    print(f"\nCache Stats:")
    print(json.dumps(stats, indent=2))
