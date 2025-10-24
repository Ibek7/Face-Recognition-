# Request Deduplication & Smart Caching

import hashlib
import threading
import time
import json
from typing import Dict, Any, Optional, Callable, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

class DeduplicationStrategy(Enum):
    """Request deduplication strategies."""
    EXACT_MATCH = "exact_match"
    SEMANTIC = "semantic"
    FUZZY = "fuzzy"
    TIME_WINDOW = "time_window"

class CacheInvalidationPolicy(Enum):
    """Cache invalidation policies."""
    TTL = "ttl"
    LRU = "lru"
    LFU = "lfu"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"

@dataclass
class RequestSignature:
    """Request signature for deduplication."""
    request_id: str
    signature_hash: str
    timestamp: float = field(default_factory=time.time)
    request_data: Dict[str, Any] = field(default_factory=dict)
    response_cache: Optional[Any] = None
    hit_count: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'signature_hash': self.signature_hash,
            'timestamp': self.timestamp,
            'hit_count': self.hit_count
        }

class RequestDeduplicator:
    """Deduplicate incoming requests."""
    
    def __init__(self, strategy: DeduplicationStrategy = DeduplicationStrategy.EXACT_MATCH,
                 window_size: int = 1000, retention_sec: int = 300):
        self.strategy = strategy
        self.window_size = window_size
        self.retention_sec = retention_sec
        
        self.request_signatures: Dict[str, RequestSignature] = {}
        self.pending_requests: Dict[str, Tuple[float, Any]] = {}
        self.lock = threading.RLock()
    
    def generate_signature(self, request_data: Dict) -> str:
        """Generate request signature."""
        if self.strategy == DeduplicationStrategy.EXACT_MATCH:
            return self._exact_match_signature(request_data)
        elif self.strategy == DeduplicationStrategy.SEMANTIC:
            return self._semantic_signature(request_data)
        elif self.strategy == DeduplicationStrategy.FUZZY:
            return self._fuzzy_signature(request_data)
        elif self.strategy == DeduplicationStrategy.TIME_WINDOW:
            return self._time_window_signature(request_data)
        
        return self._exact_match_signature(request_data)
    
    def _exact_match_signature(self, request_data: Dict) -> str:
        """Exact match signature."""
        data_str = json.dumps(request_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _semantic_signature(self, request_data: Dict) -> str:
        """Semantic signature (ignoring order)."""
        # Normalize keys
        normalized = {}
        for key in sorted(request_data.keys()):
            if isinstance(request_data[key], (list, set)):
                normalized[key] = tuple(sorted(request_data[key]))
            else:
                normalized[key] = request_data[key]
        
        data_str = json.dumps(normalized, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _fuzzy_signature(self, request_data: Dict) -> str:
        """Fuzzy matching signature."""
        # Include only key fields, ignore minor variations
        key_fields = {k: v for k, v in request_data.items() 
                     if not k.startswith('_')}
        
        data_str = json.dumps(key_fields, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _time_window_signature(self, request_data: Dict) -> str:
        """Time-window based signature."""
        # Group by time window (5 sec intervals)
        current_window = int(time.time() / 5)
        base_sig = self._exact_match_signature(request_data)
        window_sig = f"{base_sig}_{current_window}"
        return hashlib.sha256(window_sig.encode()).hexdigest()
    
    def check_duplicate(self, request_data: Dict) -> Tuple[bool, Optional[Any]]:
        """Check if request is duplicate."""
        signature = self.generate_signature(request_data)
        
        with self.lock:
            if signature in self.request_signatures:
                req_sig = self.request_signatures[signature]
                
                # Check if response is cached
                if req_sig.response_cache is not None:
                    req_sig.hit_count += 1
                    return True, req_sig.response_cache
                
                # Check if request is pending
                if signature in self.pending_requests:
                    return True, None  # Duplicate, waiting for response
            
            # Cleanup old signatures
            self._cleanup_old_signatures()
            
            return False, None
    
    def register_request(self, request_id: str, request_data: Dict) -> str:
        """Register new request."""
        signature = self.generate_signature(request_data)
        
        with self.lock:
            req_sig = RequestSignature(
                request_id=request_id,
                signature_hash=signature,
                request_data=request_data
            )
            
            self.request_signatures[signature] = req_sig
            self.pending_requests[signature] = (time.time(), None)
        
        return signature
    
    def cache_response(self, signature: str, response: Any) -> None:
        """Cache response for signature."""
        with self.lock:
            if signature in self.request_signatures:
                self.request_signatures[signature].response_cache = response
                
                # Remove from pending
                if signature in self.pending_requests:
                    del self.pending_requests[signature]
    
    def _cleanup_old_signatures(self) -> None:
        """Remove old signatures."""
        cutoff_time = time.time() - self.retention_sec
        
        to_delete = [
            sig for sig, req_sig in self.request_signatures.items()
            if req_sig.timestamp < cutoff_time
        ]
        
        for sig in to_delete:
            del self.request_signatures[sig]
    
    def get_stats(self) -> Dict:
        """Get deduplication statistics."""
        with self.lock:
            total_cached = len(self.request_signatures)
            total_pending = len(self.pending_requests)
            total_hits = sum(r.hit_count for r in self.request_signatures.values())
            
            return {
                'strategy': self.strategy.value,
                'total_signatures': total_cached,
                'pending_requests': total_pending,
                'total_cache_hits': total_hits,
                'avg_hits_per_request': total_hits / total_cached if total_cached > 0 else 0
            }

class SmartCacheManager:
    """Intelligent cache management."""
    
    def __init__(self, max_size: int = 10000, 
                 invalidation_policy: CacheInvalidationPolicy = CacheInvalidationPolicy.LRU):
        self.max_size = max_size
        self.policy = invalidation_policy
        
        self.cache: Dict[str, Tuple[Any, float, int]] = {}  # value, timestamp, access_count
        self.access_history: deque = deque(maxlen=max_size)
        self.lock = threading.RLock()
    
    def get(self, key: str, ttl_sec: int = None) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            value, timestamp, access_count = self.cache[key]
            
            # Check TTL
            if ttl_sec and (time.time() - timestamp) > ttl_sec:
                del self.cache[key]
                return None
            
            # Update access info
            self.cache[key] = (value, timestamp, access_count + 1)
            self.access_history.append(key)
            
            return value
    
    def set(self, key: str, value: Any, ttl_sec: int = None) -> None:
        """Set value in cache."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_entry()
            
            self.cache[key] = (value, time.time(), 0)
    
    def _evict_entry(self) -> None:
        """Evict entry based on policy."""
        if not self.cache:
            return
        
        if self.policy == CacheInvalidationPolicy.LRU:
            # Evict least recently used
            key_to_evict = min(
                self.cache.keys(),
                key=lambda k: self.cache[k][1]
            )
        elif self.policy == CacheInvalidationPolicy.LFU:
            # Evict least frequently used
            key_to_evict = min(
                self.cache.keys(),
                key=lambda k: self.cache[k][2]
            )
        else:
            # Default to FIFO
            key_to_evict = next(iter(self.cache.keys()))
        
        del self.cache[key_to_evict]
    
    def invalidate(self, key: str = None) -> None:
        """Invalidate cache entry or entire cache."""
        with self.lock:
            if key:
                self.cache.pop(key, None)
            else:
                self.cache.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            if not self.cache:
                return {
                    'size': 0,
                    'max_size': self.max_size,
                    'policy': self.policy.value,
                    'utilization': 0.0
                }
            
            total_accesses = sum(self.cache[k][2] for k in self.cache)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'policy': self.policy.value,
                'utilization': len(self.cache) / self.max_size,
                'total_accesses': total_accesses,
                'avg_accesses': total_accesses / len(self.cache) if self.cache else 0
            }

class RequestCache:
    """Integrated request caching system."""
    
    def __init__(self):
        self.deduplicator = RequestDeduplicator()
        self.cache_manager = SmartCacheManager()
        self.request_handlers: Dict[str, Callable] = {}
        self.lock = threading.RLock()
    
    def register_handler(self, request_type: str, handler: Callable) -> None:
        """Register request handler."""
        with self.lock:
            self.request_handlers[request_type] = handler
    
    def process_request(self, request_id: str, request_type: str, 
                       request_data: Dict) -> Any:
        """Process request with deduplication."""
        # Check for duplicate
        is_duplicate, cached_response = self.deduplicator.check_duplicate(request_data)
        
        if is_duplicate and cached_response is not None:
            return cached_response
        
        # Register request
        signature = self.deduplicator.register_request(request_id, request_data)
        
        # Get handler
        handler = self.request_handlers.get(request_type)
        if not handler:
            raise ValueError(f"No handler for request type: {request_type}")
        
        # Execute
        response = handler(request_data)
        
        # Cache response
        self.deduplicator.cache_response(signature, response)
        self.cache_manager.set(signature, response)
        
        return response
    
    def get_metrics(self) -> Dict:
        """Get caching metrics."""
        return {
            'deduplication': self.deduplicator.get_stats(),
            'cache_manager': self.cache_manager.get_stats()
        }

# Example usage
if __name__ == "__main__":
    # Create cache system
    cache = RequestCache()
    
    # Register handler
    def face_detection_handler(request_data: Dict) -> Dict:
        """Handle face detection request."""
        return {'faces': [{'id': 1, 'confidence': 0.95}]}
    
    cache.register_handler('face_detection', face_detection_handler)
    
    # Process requests
    req1 = cache.process_request(
        'req1', 'face_detection',
        {'image': 'img.jpg', 'threshold': 0.8}
    )
    print(f"Request 1: {req1}")
    
    # Duplicate request (should hit cache)
    req2 = cache.process_request(
        'req2', 'face_detection',
        {'image': 'img.jpg', 'threshold': 0.8}
    )
    print(f"Request 2 (cached): {req2}")
    
    # Get metrics
    metrics = cache.get_metrics()
    print(f"\nMetrics:")
    print(json.dumps(metrics, indent=2))
