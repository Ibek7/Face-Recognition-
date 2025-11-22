"""
Webhook dispatcher for outbound event notifications.

Provides reliable webhook delivery with retry logic and failure handling.
"""

import asyncio
import hashlib
import hmac
import json
import time
from typing import Optional, Dict, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import httpx
import logging

logger = logging.getLogger(__name__)


class WebhookStatus(str, Enum):
    """Webhook delivery status."""
    
    PENDING = "pending"
    SENDING = "sending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"


class WebhookEvent:
    """Webhook event data."""
    
    def __init__(
        self,
        event_type: str,
        payload: dict,
        url: str,
        headers: Optional[dict] = None,
        secret: Optional[str] = None
    ):
        """
        Initialize webhook event.
        
        Args:
            event_type: Type of event
            payload: Event payload
            url: Webhook URL
            headers: Custom headers
            secret: Secret for signature
        """
        self.id = self._generate_id()
        self.event_type = event_type
        self.payload = payload
        self.url = url
        self.headers = headers or {}
        self.secret = secret
        self.status = WebhookStatus.PENDING
        self.attempts = 0
        self.max_attempts = 3
        self.created_at = datetime.utcnow()
        self.last_attempt_at: Optional[datetime] = None
        self.next_retry_at: Optional[datetime] = None
        self.error: Optional[str] = None
    
    def _generate_id(self) -> str:
        """Generate unique event ID."""
        timestamp = str(time.time()).encode()
        random_bytes = str(time.time_ns()).encode()
        return hashlib.sha256(timestamp + random_bytes).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "payload": self.payload,
            "status": self.status.value,
            "attempts": self.attempts,
            "created_at": self.created_at.isoformat(),
            "last_attempt_at": self.last_attempt_at.isoformat() if self.last_attempt_at else None,
            "error": self.error
        }


class WebhookDispatcher:
    """Dispatch webhooks with retry logic."""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delays: Optional[List[int]] = None,
        timeout: int = 30
    ):
        """
        Initialize webhook dispatcher.
        
        Args:
            max_retries: Maximum retry attempts
            retry_delays: Delay between retries (seconds)
            timeout: Request timeout (seconds)
        """
        self.max_retries = max_retries
        self.retry_delays = retry_delays or [5, 30, 300]  # 5s, 30s, 5min
        self.timeout = timeout
        self.queue: asyncio.Queue = asyncio.Queue()
        self.subscribers: Dict[str, List[str]] = {}
        self._running = False
        self._workers: List[asyncio.Task] = []
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "pending": 0
        }
    
    def subscribe(self, event_type: str, url: str, secret: Optional[str] = None):
        """
        Subscribe to webhook events.
        
        Args:
            event_type: Type of event to subscribe to
            url: Webhook URL
            secret: Secret for signature
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        subscriber_data = {"url": url}
        if secret:
            subscriber_data["secret"] = secret
        
        self.subscribers[event_type].append(subscriber_data)
        
        logger.info(f"Subscribed {url} to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, url: str):
        """
        Unsubscribe from webhook events.
        
        Args:
            event_type: Event type
            url: Webhook URL
        """
        if event_type in self.subscribers:
            self.subscribers[event_type] = [
                sub for sub in self.subscribers[event_type]
                if sub["url"] != url
            ]
            
            logger.info(f"Unsubscribed {url} from event type: {event_type}")
    
    async def dispatch(
        self,
        event_type: str,
        payload: dict,
        headers: Optional[dict] = None
    ):
        """
        Dispatch webhook to all subscribers.
        
        Args:
            event_type: Type of event
            payload: Event payload
            headers: Custom headers
        """
        if event_type not in self.subscribers:
            logger.debug(f"No subscribers for event type: {event_type}")
            return
        
        for subscriber in self.subscribers[event_type]:
            event = WebhookEvent(
                event_type=event_type,
                payload=payload,
                url=subscriber["url"],
                headers=headers,
                secret=subscriber.get("secret")
            )
            
            await self.queue.put(event)
            self.stats["total"] += 1
            self.stats["pending"] += 1
            
            logger.debug(f"Queued webhook: {event.id} to {event.url}")
    
    async def _send_webhook(self, event: WebhookEvent) -> bool:
        """
        Send webhook HTTP request.
        
        Args:
            event: Webhook event
        
        Returns:
            True if successful
        """
        event.status = WebhookStatus.SENDING
        event.attempts += 1
        event.last_attempt_at = datetime.utcnow()
        
        # Prepare payload
        payload = {
            "id": event.id,
            "event_type": event.event_type,
            "timestamp": event.created_at.isoformat(),
            "data": event.payload
        }
        
        payload_json = json.dumps(payload)
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "WebhookDispatcher/1.0",
            "X-Webhook-ID": event.id,
            "X-Event-Type": event.event_type,
            **event.headers
        }
        
        # Add signature if secret provided
        if event.secret:
            signature = self._generate_signature(payload_json, event.secret)
            headers["X-Webhook-Signature"] = signature
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    event.url,
                    content=payload_json,
                    headers=headers
                )
                
                # Check response
                if 200 <= response.status_code < 300:
                    event.status = WebhookStatus.SUCCESS
                    self.stats["success"] += 1
                    self.stats["pending"] -= 1
                    
                    logger.info(
                        f"Webhook delivered: {event.id} to {event.url} "
                        f"(status: {response.status_code})"
                    )
                    return True
                
                else:
                    event.error = f"HTTP {response.status_code}"
                    logger.warning(
                        f"Webhook failed: {event.id} to {event.url} "
                        f"(status: {response.status_code})"
                    )
                    return False
        
        except Exception as e:
            event.error = str(e)
            logger.error(f"Webhook error: {event.id} to {event.url}: {e}")
            return False
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """
        Generate HMAC signature.
        
        Args:
            payload: JSON payload
            secret: Secret key
        
        Returns:
            HMAC signature
        """
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
    
    async def _worker(self):
        """Process webhook queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                
                success = await self._send_webhook(event)
                
                # Retry logic
                if not success and event.attempts < self.max_retries:
                    event.status = WebhookStatus.RETRY
                    
                    # Calculate retry delay
                    delay_index = min(event.attempts - 1, len(self.retry_delays) - 1)
                    delay = self.retry_delays[delay_index]
                    
                    event.next_retry_at = datetime.utcnow() + timedelta(seconds=delay)
                    
                    logger.info(
                        f"Retrying webhook {event.id} in {delay}s "
                        f"(attempt {event.attempts}/{self.max_retries})"
                    )
                    
                    # Re-queue after delay
                    await asyncio.sleep(delay)
                    await self.queue.put(event)
                
                elif not success:
                    event.status = WebhookStatus.FAILED
                    self.stats["failed"] += 1
                    self.stats["pending"] -= 1
                    
                    logger.error(
                        f"Webhook permanently failed: {event.id} "
                        f"after {event.attempts} attempts"
                    )
            
            except asyncio.TimeoutError:
                continue
            
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    async def start(self, num_workers: int = 3):
        """
        Start webhook dispatcher.
        
        Args:
            num_workers: Number of worker tasks
        """
        if self._running:
            logger.warning("Webhook dispatcher already running")
            return
        
        self._running = True
        
        # Start workers
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker())
            self._workers.append(worker)
        
        logger.info(f"Started webhook dispatcher with {num_workers} workers")
    
    async def stop(self):
        """Stop webhook dispatcher."""
        if not self._running:
            return
        
        self._running = False
        
        # Wait for workers
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("Stopped webhook dispatcher")
    
    def get_stats(self) -> dict:
        """Get dispatcher statistics."""
        return {
            **self.stats,
            "queue_size": self.queue.qsize(),
            "subscribers": sum(len(subs) for subs in self.subscribers.values())
        }


# Global webhook dispatcher
webhook_dispatcher = WebhookDispatcher(max_retries=3)


# Example usage:
"""
from fastapi import FastAPI
from src.webhook_dispatcher import webhook_dispatcher

app = FastAPI()

@app.on_event("startup")
async def startup():
    # Subscribe to events
    webhook_dispatcher.subscribe(
        "user.created",
        "https://example.com/webhooks",
        secret="my_secret_key"
    )
    
    # Start dispatcher
    await webhook_dispatcher.start(num_workers=5)

@app.on_event("shutdown")
async def shutdown():
    await webhook_dispatcher.stop()

@app.post("/api/users")
async def create_user(user_data: dict):
    # Create user...
    
    # Dispatch webhook
    await webhook_dispatcher.dispatch(
        "user.created",
        payload={"user_id": "123", "email": "user@example.com"}
    )
    
    return {"message": "User created"}

@app.get("/webhooks/stats")
async def webhook_stats():
    return webhook_dispatcher.get_stats()
"""
