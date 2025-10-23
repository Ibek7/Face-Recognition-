# Webhook System for Event Notifications

import threading
import time
import requests
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import hmac

class WebhookEventType(Enum):
    """Types of webhook events."""
    FACE_DETECTED = "face.detected"
    FACE_RECOGNIZED = "face.recognized"
    MODEL_TRAINED = "model.trained"
    QUOTA_EXCEEDED = "quota.exceeded"
    PIPELINE_COMPLETED = "pipeline.completed"
    ERROR_OCCURRED = "error.occurred"
    HEALTH_CHECK = "health.check"

class WebhookStatus(Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    EXPIRED = "expired"

@dataclass
class WebhookPayload:
    """Webhook event payload."""
    event_type: WebhookEventType
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    attempt: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'data': self.data,
            'source': self.source,
            'attempt': self.attempt
        }

@dataclass
class WebhookDelivery:
    """Track webhook delivery attempt."""
    webhook_id: str
    payload: WebhookPayload
    status: WebhookStatus
    timestamp: float = field(default_factory=time.time)
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'webhook_id': self.webhook_id,
            'status': self.status.value,
            'timestamp': self.timestamp,
            'response_code': self.response_code,
            'error': self.error,
            'duration_ms': self.duration_ms
        }

@dataclass
class Webhook:
    """Webhook definition."""
    webhook_id: str
    url: str
    event_types: List[WebhookEventType]
    is_active: bool = True
    secret: Optional[str] = None
    max_retries: int = 3
    timeout_seconds: int = 10
    created_at: float = field(default_factory=time.time)
    headers: Dict[str, str] = field(default_factory=dict)
    
    def should_receive_event(self, event_type: WebhookEventType) -> bool:
        """Check if webhook should receive event."""
        return self.is_active and event_type in self.event_types
    
    def sign_payload(self, payload: str) -> str:
        """Create HMAC signature for payload."""
        if not self.secret:
            return ""
        
        return hmac.new(
            self.secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

class WebhookManager:
    """Manage webhooks."""
    
    def __init__(self):
        self.webhooks: Dict[str, Webhook] = {}
        self.deliveries: List[WebhookDelivery] = []
        self.lock = threading.RLock()
    
    def register_webhook(self, webhook: Webhook):
        """Register webhook."""
        with self.lock:
            self.webhooks[webhook.webhook_id] = webhook
    
    def unregister_webhook(self, webhook_id: str):
        """Unregister webhook."""
        with self.lock:
            if webhook_id in self.webhooks:
                del self.webhooks[webhook_id]
    
    def update_webhook(self, webhook_id: str, **kwargs):
        """Update webhook properties."""
        with self.lock:
            webhook = self.webhooks.get(webhook_id)
            if webhook:
                for key, value in kwargs.items():
                    if hasattr(webhook, key):
                        setattr(webhook, key, value)
    
    def get_webhooks_for_event(self, event_type: WebhookEventType) -> List[Webhook]:
        """Get webhooks interested in event."""
        with self.lock:
            return [w for w in self.webhooks.values() 
                   if w.should_receive_event(event_type)]
    
    def record_delivery(self, delivery: WebhookDelivery):
        """Record delivery attempt."""
        with self.lock:
            self.deliveries.append(delivery)
    
    def get_delivery_history(self, webhook_id: str, limit: int = 100) -> List[WebhookDelivery]:
        """Get delivery history."""
        with self.lock:
            return [d for d in self.deliveries if d.webhook_id == webhook_id][-limit:]

class WebhookDispatcher:
    """Dispatch webhooks."""
    
    def __init__(self, webhook_manager: WebhookManager, 
                max_workers: int = 4):
        self.manager = webhook_manager
        self.max_workers = max_workers
        self.is_running = False
        self.worker_threads: List[threading.Thread] = []
        self.event_queue: List[tuple] = []
        self.lock = threading.RLock()
    
    def start(self):
        """Start dispatcher."""
        if self.is_running:
            return
        
        self.is_running = True
        
        for _ in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.worker_threads.append(worker)
    
    def stop(self):
        """Stop dispatcher."""
        self.is_running = False
        
        for worker in self.worker_threads:
            worker.join(timeout=5)
        
        self.worker_threads.clear()
    
    def dispatch(self, event_type: WebhookEventType, data: Dict[str, Any],
                source: str = ""):
        """Dispatch event to webhooks."""
        
        payload = WebhookPayload(
            event_type=event_type,
            data=data,
            source=source
        )
        
        webhooks = self.manager.get_webhooks_for_event(event_type)
        
        for webhook in webhooks:
            with self.lock:
                self.event_queue.append((webhook, payload))
    
    def _worker_loop(self):
        """Worker loop for dispatching webhooks."""
        while self.is_running:
            webhook, payload = None, None
            
            with self.lock:
                if self.event_queue:
                    webhook, payload = self.event_queue.pop(0)
            
            if webhook and payload:
                self._send_webhook(webhook, payload)
            else:
                time.sleep(0.1)
    
    def _send_webhook(self, webhook: Webhook, payload: WebhookPayload):
        """Send webhook with retries."""
        
        for attempt in range(webhook.max_retries):
            payload.attempt = attempt + 1
            
            start_time = time.time()
            
            try:
                response = self._make_request(webhook, payload)
                duration = (time.time() - start_time) * 1000
                
                delivery = WebhookDelivery(
                    webhook_id=webhook.webhook_id,
                    payload=payload,
                    status=WebhookStatus.DELIVERED if response.status_code < 400 
                           else WebhookStatus.RETRYING,
                    response_code=response.status_code,
                    response_body=response.text[:500],
                    duration_ms=duration
                )
                
                self.manager.record_delivery(delivery)
                
                if response.status_code < 400:
                    return
            
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                
                delivery = WebhookDelivery(
                    webhook_id=webhook.webhook_id,
                    payload=payload,
                    status=WebhookStatus.FAILED if attempt == webhook.max_retries - 1 
                           else WebhookStatus.RETRYING,
                    error=str(e),
                    duration_ms=duration
                )
                
                self.manager.record_delivery(delivery)
            
            if attempt < webhook.max_retries - 1:
                # Exponential backoff
                time.sleep(2 ** attempt)
    
    def _make_request(self, webhook: Webhook, payload: WebhookPayload) -> requests.Response:
        """Make HTTP request to webhook."""
        
        payload_json = json.dumps(payload.to_dict())
        headers = webhook.headers.copy()
        headers['Content-Type'] = 'application/json'
        
        # Add signature if secret exists
        if webhook.secret:
            signature = webhook.sign_payload(payload_json)
            headers['X-Webhook-Signature'] = signature
        
        return requests.post(
            webhook.url,
            data=payload_json,
            headers=headers,
            timeout=webhook.timeout_seconds
        )

class WebhookValidator:
    """Validate incoming webhook signatures."""
    
    @staticmethod
    def validate_signature(payload: str, signature: str, secret: str) -> bool:
        """Validate webhook signature."""
        
        expected = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected)

class WebhookFilter:
    """Filter webhook events."""
    
    def __init__(self):
        self.filters: List[Callable] = []
    
    def add_filter(self, predicate: Callable) -> 'WebhookFilter':
        """Add filter."""
        self.filters.append(predicate)
        return self
    
    def should_dispatch(self, event_type: WebhookEventType, data: Dict) -> bool:
        """Check if should dispatch."""
        return all(f(event_type, data) for f in self.filters)

# Example usage
if __name__ == "__main__":
    # Create manager and dispatcher
    manager = WebhookManager()
    dispatcher = WebhookDispatcher(manager)
    
    # Register webhook
    webhook = Webhook(
        webhook_id="wh_1",
        url="https://example.com/webhooks",
        event_types=[WebhookEventType.FACE_DETECTED, WebhookEventType.FACE_RECOGNIZED],
        secret="my_secret_key"
    )
    manager.register_webhook(webhook)
    
    # Start dispatcher
    dispatcher.start()
    
    # Dispatch event
    dispatcher.dispatch(
        WebhookEventType.FACE_DETECTED,
        {"face_id": "face_123", "confidence": 0.95},
        source="camera_1"
    )
    
    print("Webhook dispatched!")
    time.sleep(1)
    
    # Get delivery history
    history = manager.get_delivery_history("wh_1")
    print(f"Deliveries: {len(history)}")
    
    dispatcher.stop()
