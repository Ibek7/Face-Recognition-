# Advanced Caching System for Face Recognition

import hashlib
import json
import time
import pickle
import threading
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from collections import OrderedDict
import logging
import psutil
import heapq

@dataclass
class CacheEntry:
    """Represents a cached item."""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    ttl: Optional[int]  # Time to live in seconds
    size_bytes: int
    hit_count: int = 0
    metadata: Dict[str, Any] = None

class LRUCache:
    """Least Recently Used (LRU) cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 500):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self.current_memory_bytes = 0
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check if TTL expired
            if entry.ttl and (time.time() - entry.created_at) > entry.ttl:
                self._remove_entry(key)
                return None
            
            # Update access info for LRU
            entry.accessed_at = time.time()
            entry.hit_count += 1
            self.cache.move_to_end(key)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            metadata: Optional[Dict] = None):
        """Set value in cache."""
        with self.lock:
            size = self._estimate_size(value)
            
            # Check if we need to evict entries
            self._ensure_space(size, key in self.cache)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl=ttl,
                size_bytes=size,
                metadata=metadata or {}
            )
            
            if key in self.cache:
                self.current_memory_bytes -= self.cache[key].size_bytes
            
            self.cache[key] = entry
            self.current_memory_bytes += size
            self.cache.move_to_end(key)
    
    def _ensure_space(self, required_size: int, is_update: bool = False):
        """Ensure enough space for new entry."""
        memory_limit_bytes = self.max_memory_mb * 1024 * 1024
        
        # Evict expired entries first
        self._evict_expired()
        
        # Evict LRU entries if needed
        while (self.current_memory_bytes + required_size > memory_limit_bytes and 
               len(self.cache) > 0):
            first_key = next(iter(self.cache))
            self._remove_entry(first_key)
        
        # Check size limit
        while len(self.cache) >= self.max_size and not is_update:
            first_key = next(iter(self.cache))
            self._remove_entry(first_key)
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            self.current_memory_bytes -= entry.size_bytes
            del self.cache[key]
    
    def _evict_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        keys_to_remove = []
        
        for key, entry in self.cache.items():
            if entry.ttl and (current_time - entry.created_at) > entry.ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._remove_entry(key)
    
    def clear(self):
        """Clear all cache."""
        with self.lock:
            self.cache.clear()
            self.current_memory_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_count = sum(entry.hit_count for entry in self.cache.values())
            total_requests = hit_count + len(self.cache)
            hit_rate = (hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'memory_usage_mb': self.current_memory_bytes / (1024 * 1024),
                'hit_count': hit_count,
                'hit_rate': hit_rate,
                'entries': [{
                    'key': entry.key,
                    'size_bytes': entry.size_bytes,
                    'hit_count': entry.hit_count,
                    'age_seconds': time.time() - entry.created_at
                } for entry in self.cache.values()]
            }

class MultiLevelCache:
    """Multi-level caching system with L1 (memory) and L2 (disk)."""
    
    def __init__(self, cache_dir: str = "./cache", max_memory_mb: float = 500,
                 max_disk_mb: float = 5000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.l1_cache = LRUCache(max_size=1000, max_memory_mb=max_memory_mb)
        self.max_disk_mb = max_disk_mb
        self.current_disk_bytes = 0
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from L1 or L2 cache."""
        # Check L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Check L2 cache (disk)
        disk_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        if disk_file.exists():
            try:
                with open(disk_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Move to L1 cache for faster access
                self.l1_cache.set(key, cached_data['value'], ttl=cached_data.get('ttl'))
                return cached_data['value']
            except Exception as e:
                self.logger.warning(f"Failed to load from L2 cache: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            level: str = 'auto'):
        """Set value in L1 or L2 cache."""
        size = len(pickle.dumps(value))
        
        if level == 'auto':
            # Small objects go to L1, large to L2
            level = 'l1' if size < 1024 * 100 else 'l2'
        
        if level == 'l1':
            self.l1_cache.set(key, value, ttl=ttl)
        elif level == 'l2':
            self._save_to_disk(key, value, ttl)
    
    def _save_to_disk(self, key: str, value: Any, ttl: Optional[int] = None):
        """Save value to disk cache."""
        disk_file = self.cache_dir / f"{self._hash_key(key)}.cache"
        
        try:
            cache_data = {
                'key': key,
                'value': value,
                'ttl': ttl,
                'created_at': time.time()
            }
            
            with open(disk_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.current_disk_bytes += disk_file.stat().st_size
        except Exception as e:
            self.logger.error(f"Failed to save to disk cache: {e}")
    
    def _hash_key(self, key: str) -> str:
        """Hash key for file naming."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def clear(self):
        """Clear both cache levels."""
        self.l1_cache.clear()
        
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except:
                pass
        
        self.current_disk_bytes = 0

class CacheableFunction:
    """Decorator for caching function results."""
    
    def __init__(self, cache: MultiLevelCache, ttl: Optional[int] = 3600):
        self.cache = cache
        self.ttl = ttl
    
    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            self.cache.set(cache_key, result, ttl=self.ttl)
            
            return result
        
        return wrapper
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function arguments."""
        key_parts = [func_name]
        
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        
        return ":".join(key_parts)

class FaceEmbeddingCache:
    """Specialized cache for face embeddings."""
    
    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)
        self.embedding_stats = {
            'total_cached': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def get_embedding(self, image_id: str) -> Optional[dict]:
        """Get cached face embedding."""
        cache_key = f"face_embedding:{image_id}"
        result = self.cache.get(cache_key)
        
        if result:
            self.embedding_stats['cache_hits'] += 1
        else:
            self.embedding_stats['cache_misses'] += 1
        
        return result
    
    def cache_embedding(self, image_id: str, embedding: dict, ttl: int = 86400):
        """Cache face embedding."""
        cache_key = f"face_embedding:{image_id}"
        self.cache.set(cache_key, embedding, ttl=ttl, level='auto')
        self.embedding_stats['total_cached'] += 1
    
    def get_batch_embeddings(self, image_ids: List[str]) -> Dict[str, dict]:
        """Get multiple embeddings from cache."""
        embeddings = {}
        
        for image_id in image_ids:
            embedding = self.get_embedding(image_id)
            if embedding:
                embeddings[image_id] = embedding
        
        return embeddings
    
    def cache_batch_embeddings(self, embeddings: Dict[str, dict], ttl: int = 86400):
        """Cache multiple embeddings."""
        for image_id, embedding in embeddings.items():
            self.cache_embedding(image_id, embedding, ttl)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = (self.embedding_stats['cache_hits'] + 
                         self.embedding_stats['cache_misses'])
        hit_rate = (self.embedding_stats['cache_hits'] / total_requests * 100 
                   if total_requests > 0 else 0)
        
        return {
            **self.embedding_stats,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'cache_stats': self.cache.l1_cache.get_stats()
        }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create multi-level cache
    cache = MultiLevelCache(cache_dir="./face_cache", max_memory_mb=500)
    
    # Create embedding cache
    embedding_cache = FaceEmbeddingCache(cache)
    
    # Cache some embeddings
    test_embeddings = {
        'img_001': {'vector': [0.1] * 512, 'confidence': 0.95},
        'img_002': {'vector': [0.2] * 512, 'confidence': 0.92},
    }
    
    embedding_cache.cache_batch_embeddings(test_embeddings)
    
    # Retrieve embeddings
    retrieved = embedding_cache.get_embedding('img_001')
    print(f"Retrieved embedding: {retrieved}")
    
    # Get statistics
    stats = embedding_cache.get_stats()
    print(f"Cache stats: {json.dumps(stats, indent=2, default=str)}")