"""
Webhook receiver with signature verification and retry.

Handles incoming webhooks with security validation and event processing.
"""

from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import asyncio
import hmac
import hashlib
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class WebhookStatus(str, Enum):
    """Webhook processing status."""
    
    RECEIVED = "received"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class SignatureAlgorithm(str, Enum):
    """Signature algorithms."""
    
    SHA256 = "sha256"
    SHA1 = "sha1"
    MD5 = "md5"


class WebhookEvent:
    """Webhook event."""
    
    def __init__(
        self,
        event_id: str,
        event_type: str,
        payload: dict,
        signature: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize webhook event.
        
        Args:
            event_id: Unique event identifier
            event_type: Event type
            payload: Event payload
            signature: Webhook signature
            headers: Request headers
        """
        self.event_id = event_id
        self.event_type = event_type
        self.payload = payload
        self.signature = signature
        self.headers = headers or {}
        self.received_at = datetime.utcnow()
        self.status = WebhookStatus.RECEIVED
        self.processed_at: Optional[datetime] = None
        self.retry_count = 0
        self.error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "status": self.status.value,
            "received_at": self.received_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "retry_count": self.retry_count,
            "error_message": self.error_message
        }


class SignatureVerifier:
    """Webhook signature verifier."""
    
    def __init__(
        self,
        secret: str,
        algorithm: SignatureAlgorithm = SignatureAlgorithm.SHA256
    ):
        """
        Initialize signature verifier.
        
        Args:
            secret: Signing secret
            algorithm: Hash algorithm
        """
        self.secret = secret.encode() if isinstance(secret, str) else secret
        self.algorithm = algorithm
    
    def verify(
        self,
        payload: bytes,
        signature: str,
        prefix: str = "sha256="
    ) -> bool:
        """
        Verify webhook signature.
        
        Args:
            payload: Request payload
            signature: Provided signature
            prefix: Signature prefix (e.g., "sha256=")
        
        Returns:
            True if signature valid
        """
        # Get hash function
        if self.algorithm == SignatureAlgorithm.SHA256:
            hash_func = hashlib.sha256
        elif self.algorithm == SignatureAlgorithm.SHA1:
            hash_func = hashlib.sha1
        else:
            hash_func = hashlib.md5
        
        # Calculate expected signature
        expected = hmac.new(
            self.secret,
            payload,
            hash_func
        ).hexdigest()
        
        # Remove prefix if present
        if signature.startswith(prefix):
            signature = signature[len(prefix):]
        
        # Constant-time comparison
        return hmac.compare_digest(expected, signature)
    
    def generate(self, payload: bytes, prefix: str = "sha256=") -> str:
        """Generate signature for payload."""
        if self.algorithm == SignatureAlgorithm.SHA256:
            hash_func = hashlib.sha256
        elif self.algorithm == SignatureAlgorithm.SHA1:
            hash_func = hashlib.sha1
        else:
            hash_func = hashlib.md5
        
        signature = hmac.new(
            self.secret,
            payload,
            hash_func
        ).hexdigest()
        
        return f"{prefix}{signature}"


class WebhookHandler:
    """Webhook event handler."""
    
    def __init__(
        self,
        secret: Optional[str] = None,
        verify_signatures: bool = True,
        max_retries: int = 3,
        retry_delay: int = 60
    ):
        """
        Initialize webhook handler.
        
        Args:
            secret: Signing secret
            verify_signatures: Enable signature verification
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
        """
        self.verify_signatures = verify_signatures
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.verifier = SignatureVerifier(secret) if secret else None
        self.handlers: Dict[str, List[Callable]] = {}
        self.events: Dict[str, WebhookEvent] = {}
        
        # Retry queue
        self._retry_queue: asyncio.Queue = asyncio.Queue()
        self._retry_task: Optional[asyncio.Task] = None
    
    def register_handler(self, event_type: str, handler: Callable):
        """
        Register event handler.
        
        Args:
            event_type: Event type to handle
            handler: Async handler function
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        
        self.handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")
    
    async def handle_webhook(
        self,
        payload: dict,
        signature: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> WebhookEvent:
        """
        Handle incoming webhook.
        
        Args:
            payload: Webhook payload
            signature: Webhook signature
            headers: Request headers
        
        Returns:
            Webhook event
        
        Raises:
            ValueError: If signature verification fails
        """
        # Verify signature
        if self.verify_signatures:
            if not signature or not self.verifier:
                raise ValueError("Signature verification required but not provided")
            
            payload_bytes = json.dumps(payload, separators=(',', ':')).encode()
            
            if not self.verifier.verify(payload_bytes, signature):
                raise ValueError("Invalid webhook signature")
        
        # Create event
        event = WebhookEvent(
            event_id=payload.get("id", str(datetime.utcnow().timestamp())),
            event_type=payload.get("type", "unknown"),
            payload=payload,
            signature=signature,
            headers=headers
        )
        
        self.events[event.event_id] = event
        
        # Process event
        await self._process_event(event)
        
        return event
    
    async def _process_event(self, event: WebhookEvent):
        """Process webhook event."""
        event.status = WebhookStatus.PROCESSING
        
        try:
            # Get handlers for event type
            handlers = self.handlers.get(event.event_type, [])
            
            if not handlers:
                logger.warning(f"No handlers registered for event type: {event.event_type}")
                event.status = WebhookStatus.COMPLETED
                event.processed_at = datetime.utcnow()
                return
            
            # Execute handlers
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Handler error for {event.event_type}: {e}")
                    raise
            
            # Mark as completed
            event.status = WebhookStatus.COMPLETED
            event.processed_at = datetime.utcnow()
            
            logger.info(f"Webhook event processed: {event.event_id}")
        
        except Exception as e:
            event.status = WebhookStatus.FAILED
            event.error_message = str(e)
            
            # Queue for retry
            if event.retry_count < self.max_retries:
                event.status = WebhookStatus.RETRYING
                await self._retry_queue.put(event)
            
            logger.error(f"Webhook processing failed: {e}")
    
    async def _retry_loop(self):
        """Retry failed webhooks."""
        while True:
            try:
                event = await self._retry_queue.get()
                
                # Wait before retry
                await asyncio.sleep(self.retry_delay)
                
                # Increment retry count
                event.retry_count += 1
                
                logger.info(
                    f"Retrying webhook {event.event_id} "
                    f"(attempt {event.retry_count}/{self.max_retries})"
                )
                
                # Retry processing
                await self._process_event(event)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retry loop error: {e}")
    
    def start_retry_worker(self):
        """Start retry worker."""
        if self._retry_task:
            logger.warning("Retry worker already running")
            return
        
        self._retry_task = asyncio.create_task(self._retry_loop())
        logger.info("Started webhook retry worker")
    
    async def stop_retry_worker(self):
        """Stop retry worker."""
        if not self._retry_task:
            return
        
        self._retry_task.cancel()
        
        try:
            await self._retry_task
        except asyncio.CancelledError:
            pass
        
        self._retry_task = None
        logger.info("Stopped webhook retry worker")
    
    def get_event(self, event_id: str) -> Optional[WebhookEvent]:
        """Get webhook event by ID."""
        return self.events.get(event_id)
    
    def list_events(
        self,
        event_type: Optional[str] = None,
        status: Optional[WebhookStatus] = None,
        limit: int = 100
    ) -> List[WebhookEvent]:
        """
        List webhook events.
        
        Args:
            event_type: Filter by event type
            status: Filter by status
            limit: Maximum events to return
        
        Returns:
            List of webhook events
        """
        events = list(self.events.values())
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if status:
            events = [e for e in events if e.status == status]
        
        # Sort by received time (newest first)
        events.sort(key=lambda e: e.received_at, reverse=True)
        
        return events[:limit]
    
    def get_stats(self) -> dict:
        """Get webhook statistics."""
        total = len(self.events)
        
        return {
            "total_events": total,
            "completed": sum(
                1 for e in self.events.values()
                if e.status == WebhookStatus.COMPLETED
            ),
            "failed": sum(
                1 for e in self.events.values()
                if e.status == WebhookStatus.FAILED
            ),
            "retrying": sum(
                1 for e in self.events.values()
                if e.status == WebhookStatus.RETRYING
            ),
            "registered_handlers": len(self.handlers)
        }


# Decorator for webhook handlers
def webhook_handler(handler_instance: WebhookHandler, event_type: str):
    """
    Decorator for webhook handlers.
    
    Args:
        handler_instance: WebhookHandler instance
        event_type: Event type to handle
    """
    def decorator(func: Callable) -> Callable:
        handler_instance.register_handler(event_type, func)
        return func
    
    return decorator


# Example usage:
"""
from fastapi import FastAPI, Request, Header, HTTPException
from src.webhook_handler import WebhookHandler, webhook_handler

app = FastAPI()

# Create webhook handler
webhook_handler_instance = WebhookHandler(
    secret="your-webhook-secret",
    verify_signatures=True
)

# Register event handlers
@webhook_handler(webhook_handler_instance, "user.created")
async def handle_user_created(event):
    user_data = event.payload.get("data")
    print(f"New user created: {user_data}")

@webhook_handler(webhook_handler_instance, "payment.completed")
async def handle_payment(event):
    payment_data = event.payload.get("data")
    print(f"Payment completed: {payment_data}")

@app.on_event("startup")
async def startup():
    webhook_handler_instance.start_retry_worker()

@app.post("/webhooks")
async def receive_webhook(
    request: Request,
    signature: str = Header(None, alias="X-Webhook-Signature")
):
    payload = await request.json()
    
    try:
        event = await webhook_handler_instance.handle_webhook(
            payload=payload,
            signature=signature
        )
        return {"event_id": event.event_id, "status": event.status.value}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/webhooks/events")
async def list_events():
    events = webhook_handler_instance.list_events()
    return [e.to_dict() for e in events]
"""
