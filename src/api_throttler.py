"""
Advanced request throttling with priority queues and quotas.

Provides sophisticated throttling with user quotas and priority handling.
"""

from typing import Dict, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import asyncio
from collections import deque
import heapq
import logging

logger = logging.getLogger(__name__)


class RequestPriority(str, Enum):
    """Request priority level."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class QuotaType(str, Enum):
    """Quota type."""
    
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    CONCURRENT_REQUESTS = "concurrent_requests"


class ThrottledRequest:
    """Throttled request."""
    
    def __init__(
        self,
        request_id: str,
        user_id: str,
        priority: RequestPriority = RequestPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize throttled request.
        
        Args:
            request_id: Unique request ID
            user_id: User identifier
            priority: Request priority
            metadata: Request metadata
        """
        self.request_id = request_id
        self.user_id = user_id
        self.priority = priority
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.wait_time: Optional[float] = None
    
    def __lt__(self, other):
        """Compare for priority queue (higher priority first)."""
        priority_order = {
            RequestPriority.CRITICAL: 0,
            RequestPriority.HIGH: 1,
            RequestPriority.NORMAL: 2,
            RequestPriority.LOW: 3
        }
        
        # First by priority, then by creation time
        if self.priority != other.priority:
            return priority_order[self.priority] < priority_order[other.priority]
        
        return self.created_at < other.created_at


class UserQuota:
    """User quota tracker."""
    
    def __init__(
        self,
        user_id: str,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        concurrent_limit: int = 10
    ):
        """
        Initialize user quota.
        
        Args:
            user_id: User identifier
            requests_per_minute: RPM limit
            requests_per_hour: RPH limit
            requests_per_day: RPD limit
            concurrent_limit: Max concurrent requests
        """
        self.user_id = user_id
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.concurrent_limit = concurrent_limit
        
        # Request history
        self.minute_requests: deque = deque()
        self.hour_requests: deque = deque()
        self.day_requests: deque = deque()
        
        # Concurrent tracking
        self.concurrent_count = 0
    
    def _clean_old_requests(self):
        """Remove expired requests from history."""
        now = datetime.utcnow()
        
        # Clean minute requests
        one_minute_ago = now - timedelta(minutes=1)
        while self.minute_requests and self.minute_requests[0] < one_minute_ago:
            self.minute_requests.popleft()
        
        # Clean hour requests
        one_hour_ago = now - timedelta(hours=1)
        while self.hour_requests and self.hour_requests[0] < one_hour_ago:
            self.hour_requests.popleft()
        
        # Clean day requests
        one_day_ago = now - timedelta(days=1)
        while self.day_requests and self.day_requests[0] < one_day_ago:
            self.day_requests.popleft()
    
    def can_make_request(self) -> bool:
        """Check if user can make request."""
        self._clean_old_requests()
        
        # Check all quota types
        if len(self.minute_requests) >= self.requests_per_minute:
            return False
        
        if len(self.hour_requests) >= self.requests_per_hour:
            return False
        
        if len(self.day_requests) >= self.requests_per_day:
            return False
        
        if self.concurrent_count >= self.concurrent_limit:
            return False
        
        return True
    
    def record_request_start(self):
        """Record request start."""
        now = datetime.utcnow()
        
        self.minute_requests.append(now)
        self.hour_requests.append(now)
        self.day_requests.append(now)
        self.concurrent_count += 1
    
    def record_request_end(self):
        """Record request completion."""
        self.concurrent_count = max(0, self.concurrent_count - 1)
    
    def get_quota_status(self) -> dict:
        """Get current quota status."""
        self._clean_old_requests()
        
        return {
            "user_id": self.user_id,
            "requests_per_minute": {
                "used": len(self.minute_requests),
                "limit": self.requests_per_minute,
                "remaining": self.requests_per_minute - len(self.minute_requests)
            },
            "requests_per_hour": {
                "used": len(self.hour_requests),
                "limit": self.requests_per_hour,
                "remaining": self.requests_per_hour - len(self.hour_requests)
            },
            "requests_per_day": {
                "used": len(self.day_requests),
                "limit": self.requests_per_day,
                "remaining": self.requests_per_day - len(self.day_requests)
            },
            "concurrent": {
                "active": self.concurrent_count,
                "limit": self.concurrent_limit,
                "available": self.concurrent_limit - self.concurrent_count
            }
        }


class ApiThrottler:
    """Advanced API throttler with priority queues."""
    
    def __init__(
        self,
        global_rate_limit: int = 100,
        default_user_quota: Optional[Dict[str, int]] = None
    ):
        """
        Initialize API throttler.
        
        Args:
            global_rate_limit: Global requests per second
            default_user_quota: Default quota settings
        """
        self.global_rate_limit = global_rate_limit
        self.default_user_quota = default_user_quota or {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "requests_per_day": 10000,
            "concurrent_limit": 10
        }
        
        # Priority queue (heap)
        self._queue: list = []
        self._queue_lock = asyncio.Lock()
        
        # User quotas
        self.quotas: Dict[str, UserQuota] = {}
        
        # Processing
        self._processing = False
        self._processor_task: Optional[asyncio.Task] = None
        
        # Global rate limiting
        self._global_tokens = global_rate_limit
        self._last_refill = asyncio.get_event_loop().time()
        
        # Metrics
        self.total_requests = 0
        self.throttled_requests = 0
        self.completed_requests = 0
    
    def _get_quota(self, user_id: str) -> UserQuota:
        """Get or create user quota."""
        if user_id not in self.quotas:
            self.quotas[user_id] = UserQuota(
                user_id=user_id,
                **self.default_user_quota
            )
        
        return self.quotas[user_id]
    
    def _refill_global_tokens(self):
        """Refill global rate limit tokens."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.global_rate_limit
        self._global_tokens = min(
            self.global_rate_limit,
            self._global_tokens + tokens_to_add
        )
        
        self._last_refill = now
    
    async def enqueue_request(
        self,
        request: ThrottledRequest
    ) -> asyncio.Event:
        """
        Enqueue request for processing.
        
        Args:
            request: Throttled request
        
        Returns:
            Event that will be set when request can proceed
        """
        self.total_requests += 1
        
        # Create event for request
        event = asyncio.Event()
        
        async with self._queue_lock:
            heapq.heappush(self._queue, (request, event))
        
        logger.debug(
            f"Enqueued request {request.request_id} "
            f"(priority: {request.priority.value})"
        )
        
        return event
    
    async def _process_queue(self):
        """Process queued requests."""
        while self._processing:
            try:
                async with self._queue_lock:
                    if not self._queue:
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Get highest priority request
                    request, event = heapq.heappop(self._queue)
                
                # Check quotas
                quota = self._get_quota(request.user_id)
                
                if not quota.can_make_request():
                    # Re-queue request
                    async with self._queue_lock:
                        heapq.heappush(self._queue, (request, event))
                    
                    self.throttled_requests += 1
                    await asyncio.sleep(0.1)
                    continue
                
                # Check global rate limit
                self._refill_global_tokens()
                
                if self._global_tokens < 1:
                    # Re-queue request
                    async with self._queue_lock:
                        heapq.heappush(self._queue, (request, event))
                    
                    await asyncio.sleep(0.1)
                    continue
                
                # Consume global token
                self._global_tokens -= 1
                
                # Record request start
                quota.record_request_start()
                request.started_at = datetime.utcnow()
                request.wait_time = (
                    request.started_at - request.created_at
                ).total_seconds()
                
                # Signal request can proceed
                event.set()
                
                logger.debug(
                    f"Processing request {request.request_id} "
                    f"(waited: {request.wait_time:.2f}s)"
                )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(1)
    
    def start_processing(self):
        """Start request processor."""
        if self._processing:
            logger.warning("Processor already running")
            return
        
        self._processing = True
        self._processor_task = asyncio.create_task(self._process_queue())
        
        logger.info("Started API throttler processor")
    
    async def stop_processing(self):
        """Stop request processor."""
        if not self._processing:
            return
        
        self._processing = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped API throttler processor")
    
    def complete_request(self, request: ThrottledRequest):
        """Mark request as completed."""
        quota = self._get_quota(request.user_id)
        quota.record_request_end()
        
        request.completed_at = datetime.utcnow()
        self.completed_requests += 1
    
    def get_user_quota_status(self, user_id: str) -> dict:
        """Get quota status for user."""
        quota = self._get_quota(user_id)
        return quota.get_quota_status()
    
    def get_stats(self) -> dict:
        """Get throttler statistics."""
        return {
            "total_requests": self.total_requests,
            "throttled_requests": self.throttled_requests,
            "completed_requests": self.completed_requests,
            "queue_size": len(self._queue),
            "active_users": len(self.quotas),
            "global_rate_limit": self.global_rate_limit,
            "global_tokens_available": int(self._global_tokens)
        }


# Example usage:
"""
from fastapi import FastAPI, Request, HTTPException
from src.api_throttler import ApiThrottler, ThrottledRequest, RequestPriority

app = FastAPI()

# Create throttler
throttler = ApiThrottler(
    global_rate_limit=100,
    default_user_quota={
        "requests_per_minute": 60,
        "requests_per_hour": 1000,
        "concurrent_limit": 10
    }
)

@app.on_event("startup")
async def startup():
    throttler.start_processing()

@app.middleware("http")
async def throttle_middleware(request: Request, call_next):
    user_id = request.headers.get("X-User-ID", "anonymous")
    
    # Create throttled request
    throttled_request = ThrottledRequest(
        request_id=str(id(request)),
        user_id=user_id,
        priority=RequestPriority.NORMAL
    )
    
    # Enqueue and wait
    event = await throttler.enqueue_request(throttled_request)
    await event.wait()
    
    # Process request
    try:
        response = await call_next(request)
        return response
    finally:
        throttler.complete_request(throttled_request)

@app.get("/quota")
async def get_quota(user_id: str):
    return throttler.get_user_quota_status(user_id)

@app.get("/stats")
async def get_stats():
    return throttler.get_stats()
"""
