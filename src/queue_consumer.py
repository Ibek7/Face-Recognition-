"""
Message queue consumer with dead letter handling.

Processes messages from queues with retry and error handling.
"""

from typing import Dict, Optional, Callable, Any, List
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class MessageStatus(str, Enum):
    """Message processing status."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class QueueMessage:
    """Queue message."""
    
    def __init__(
        self,
        message_id: str,
        queue_name: str,
        payload: dict,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize queue message.
        
        Args:
            message_id: Unique message ID
            queue_name: Queue name
            payload: Message payload
            metadata: Message metadata
        """
        self.message_id = message_id
        self.queue_name = queue_name
        self.payload = payload
        self.metadata = metadata or {}
        self.status = MessageStatus.PENDING
        self.received_at = datetime.utcnow()
        self.processed_at: Optional[datetime] = None
        self.retry_count = 0
        self.max_retries = 3
        self.error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "queue_name": self.queue_name,
            "payload": self.payload,
            "metadata": self.metadata,
            "status": self.status.value,
            "received_at": self.received_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "retry_count": self.retry_count,
            "error_message": self.error_message
        }


class DeadLetterQueue:
    """Dead letter queue for failed messages."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize dead letter queue.
        
        Args:
            max_size: Maximum messages to store
        """
        self.max_size = max_size
        self.messages: Dict[str, QueueMessage] = {}
    
    def add(self, message: QueueMessage):
        """Add message to dead letter queue."""
        message.status = MessageStatus.DEAD_LETTER
        
        # Enforce max size
        if len(self.messages) >= self.max_size:
            # Remove oldest message
            oldest_id = min(
                self.messages.keys(),
                key=lambda k: self.messages[k].received_at
            )
            del self.messages[oldest_id]
        
        self.messages[message.message_id] = message
        
        logger.warning(
            f"Message {message.message_id} moved to dead letter queue: "
            f"{message.error_message}"
        )
    
    def get(self, message_id: str) -> Optional[QueueMessage]:
        """Get message from dead letter queue."""
        return self.messages.get(message_id)
    
    def list(self, limit: int = 100) -> List[QueueMessage]:
        """List dead letter messages."""
        messages = list(self.messages.values())
        messages.sort(key=lambda m: m.received_at, reverse=True)
        return messages[:limit]
    
    def retry_message(self, message_id: str) -> Optional[QueueMessage]:
        """Remove message from DLQ for retry."""
        message = self.messages.pop(message_id, None)
        
        if message:
            message.status = MessageStatus.PENDING
            message.retry_count = 0
            logger.info(f"Message {message_id} removed from DLQ for retry")
        
        return message
    
    def clear(self):
        """Clear all dead letter messages."""
        count = len(self.messages)
        self.messages.clear()
        logger.info(f"Cleared {count} messages from dead letter queue")


class QueueConsumer:
    """Message queue consumer."""
    
    def __init__(
        self,
        queue_name: str,
        handler: Callable,
        max_retries: int = 3,
        retry_delay: int = 60,
        concurrent_workers: int = 5
    ):
        """
        Initialize queue consumer.
        
        Args:
            queue_name: Queue to consume from
            handler: Async message handler
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            concurrent_workers: Number of concurrent workers
        """
        self.queue_name = queue_name
        self.handler = handler
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.concurrent_workers = concurrent_workers
        
        # Message queue
        self._queue: asyncio.Queue = asyncio.Queue()
        
        # Dead letter queue
        self.dead_letter_queue = DeadLetterQueue()
        
        # Worker tasks
        self._workers: List[asyncio.Task] = []
        self._running = False
        
        # Metrics
        self.processed_count = 0
        self.failed_count = 0
        self.dlq_count = 0
    
    async def enqueue(self, message: QueueMessage):
        """Add message to queue."""
        message.max_retries = self.max_retries
        await self._queue.put(message)
    
    async def _worker(self, worker_id: int):
        """Worker to process messages."""
        logger.info(f"Queue worker {worker_id} started for {self.queue_name}")
        
        while self._running:
            try:
                # Get message with timeout
                try:
                    message = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process message
                await self._process_message(message)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Queue worker {worker_id} stopped")
    
    async def _process_message(self, message: QueueMessage):
        """Process single message."""
        message.status = MessageStatus.PROCESSING
        
        logger.info(
            f"Processing message {message.message_id} "
            f"(attempt {message.retry_count + 1}/{message.max_retries + 1})"
        )
        
        try:
            # Call handler
            await self.handler(message)
            
            # Mark as completed
            message.status = MessageStatus.COMPLETED
            message.processed_at = datetime.utcnow()
            self.processed_count += 1
            
            logger.info(f"Message {message.message_id} processed successfully")
        
        except Exception as e:
            message.error_message = str(e)
            message.retry_count += 1
            
            logger.error(
                f"Message {message.message_id} processing failed: {e}"
            )
            
            # Check if should retry
            if message.retry_count <= message.max_retries:
                message.status = MessageStatus.PENDING
                
                # Re-queue with delay
                logger.info(
                    f"Re-queuing message {message.message_id} "
                    f"(retry {message.retry_count}/{message.max_retries})"
                )
                
                await asyncio.sleep(self.retry_delay)
                await self._queue.put(message)
            else:
                # Move to dead letter queue
                message.status = MessageStatus.FAILED
                self.failed_count += 1
                self.dlq_count += 1
                self.dead_letter_queue.add(message)
    
    def start(self):
        """Start consuming messages."""
        if self._running:
            logger.warning("Queue consumer already running")
            return
        
        self._running = True
        
        # Start workers
        for i in range(self.concurrent_workers):
            task = asyncio.create_task(self._worker(i))
            self._workers.append(task)
        
        logger.info(
            f"Started queue consumer for {self.queue_name} "
            f"with {self.concurrent_workers} workers"
        )
    
    async def stop(self):
        """Stop consuming messages."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        
        self._workers.clear()
        
        logger.info(f"Stopped queue consumer for {self.queue_name}")
    
    def get_stats(self) -> dict:
        """Get consumer statistics."""
        return {
            "queue_name": self.queue_name,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "dlq_count": self.dlq_count,
            "queue_size": self._queue.qsize(),
            "workers": len(self._workers),
            "running": self._running
        }


class MultiQueueConsumer:
    """Manage multiple queue consumers."""
    
    def __init__(self):
        """Initialize multi-queue consumer."""
        self.consumers: Dict[str, QueueConsumer] = {}
    
    def register_consumer(
        self,
        queue_name: str,
        handler: Callable,
        **kwargs
    ):
        """
        Register queue consumer.
        
        Args:
            queue_name: Queue name
            handler: Message handler
            **kwargs: Additional consumer parameters
        """
        consumer = QueueConsumer(
            queue_name=queue_name,
            handler=handler,
            **kwargs
        )
        
        self.consumers[queue_name] = consumer
        
        logger.info(f"Registered consumer for queue: {queue_name}")
    
    async def enqueue(self, queue_name: str, message: QueueMessage):
        """Enqueue message to specific queue."""
        if queue_name not in self.consumers:
            raise ValueError(f"Consumer not found for queue: {queue_name}")
        
        await self.consumers[queue_name].enqueue(message)
    
    def start_all(self):
        """Start all consumers."""
        for consumer in self.consumers.values():
            consumer.start()
        
        logger.info(f"Started {len(self.consumers)} queue consumers")
    
    async def stop_all(self):
        """Stop all consumers."""
        for consumer in self.consumers.values():
            await consumer.stop()
        
        logger.info("Stopped all queue consumers")
    
    def get_consumer(self, queue_name: str) -> Optional[QueueConsumer]:
        """Get consumer for queue."""
        return self.consumers.get(queue_name)
    
    def get_all_stats(self) -> Dict[str, dict]:
        """Get statistics for all consumers."""
        return {
            name: consumer.get_stats()
            for name, consumer in self.consumers.items()
        }


# Decorator for queue handlers
def queue_handler(
    consumer_manager: MultiQueueConsumer,
    queue_name: str,
    **kwargs
):
    """
    Decorator for queue handlers.
    
    Args:
        consumer_manager: MultiQueueConsumer instance
        queue_name: Queue to handle
        **kwargs: Consumer parameters
    """
    def decorator(func: Callable) -> Callable:
        consumer_manager.register_consumer(queue_name, func, **kwargs)
        return func
    
    return decorator


# Example usage:
"""
from src.queue_consumer import MultiQueueConsumer, QueueMessage, queue_handler

# Create consumer manager
consumer_manager = MultiQueueConsumer()

# Register handlers
@queue_handler(consumer_manager, "emails", concurrent_workers=3)
async def handle_email(message: QueueMessage):
    email_data = message.payload
    print(f"Sending email to: {email_data['to']}")
    # Send email logic

@queue_handler(consumer_manager, "notifications", concurrent_workers=5)
async def handle_notification(message: QueueMessage):
    notification_data = message.payload
    print(f"Sending notification: {notification_data}")
    # Send notification logic

# Start consumers
consumer_manager.start_all()

# Enqueue messages
await consumer_manager.enqueue(
    "emails",
    QueueMessage(
        message_id="msg-123",
        queue_name="emails",
        payload={"to": "user@example.com", "subject": "Hello"}
    )
)

# Get stats
stats = consumer_manager.get_all_stats()
print(stats)

# Cleanup
await consumer_manager.stop_all()
"""
