"""
Multi-channel notification router.

Routes notifications through email, SMS, push, and other channels.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


class NotificationChannel(str, Enum):
    """Notification delivery channel."""
    
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"


class NotificationPriority(str, Enum):
    """Notification priority."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationStatus(str, Enum):
    """Notification delivery status."""
    
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


class Notification:
    """Notification message."""
    
    def __init__(
        self,
        notification_id: str,
        recipient: str,
        channel: NotificationChannel,
        subject: Optional[str] = None,
        body: str = "",
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize notification.
        
        Args:
            notification_id: Unique notification ID
            recipient: Recipient identifier
            channel: Delivery channel
            subject: Notification subject
            body: Notification body
            priority: Notification priority
            metadata: Additional metadata
        """
        self.notification_id = notification_id
        self.recipient = recipient
        self.channel = channel
        self.subject = subject
        self.body = body
        self.priority = priority
        self.metadata = metadata or {}
        
        self.status = NotificationStatus.PENDING
        self.created_at = datetime.utcnow()
        self.sent_at: Optional[datetime] = None
        self.delivered_at: Optional[datetime] = None
        self.retry_count = 0
        self.error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "notification_id": self.notification_id,
            "recipient": self.recipient,
            "channel": self.channel.value,
            "subject": self.subject,
            "body": self.body,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


class NotificationProvider:
    """Base notification provider."""
    
    def __init__(self, channel: NotificationChannel):
        """
        Initialize provider.
        
        Args:
            channel: Provider channel
        """
        self.channel = channel
    
    async def send(self, notification: Notification) -> bool:
        """
        Send notification.
        
        Args:
            notification: Notification to send
        
        Returns:
            True if sent successfully
        """
        raise NotImplementedError


class EmailProvider(NotificationProvider):
    """Email notification provider."""
    
    def __init__(self, smtp_config: dict):
        """
        Initialize email provider.
        
        Args:
            smtp_config: SMTP configuration
        """
        super().__init__(NotificationChannel.EMAIL)
        self.smtp_config = smtp_config
    
    async def send(self, notification: Notification) -> bool:
        """Send email notification."""
        logger.info(
            f"Sending email to {notification.recipient}: {notification.subject}"
        )
        
        # In production, use actual email library
        # For now, simulate sending
        await asyncio.sleep(0.1)
        
        return True


class SmsProvider(NotificationProvider):
    """SMS notification provider."""
    
    def __init__(self, api_key: str, sender_id: str):
        """
        Initialize SMS provider.
        
        Args:
            api_key: SMS service API key
            sender_id: Sender identifier
        """
        super().__init__(NotificationChannel.SMS)
        self.api_key = api_key
        self.sender_id = sender_id
    
    async def send(self, notification: Notification) -> bool:
        """Send SMS notification."""
        logger.info(
            f"Sending SMS to {notification.recipient}: {notification.body[:50]}"
        )
        
        # In production, use SMS service API
        await asyncio.sleep(0.1)
        
        return True


class PushProvider(NotificationProvider):
    """Push notification provider."""
    
    def __init__(self, fcm_key: Optional[str] = None, apns_key: Optional[str] = None):
        """
        Initialize push provider.
        
        Args:
            fcm_key: Firebase Cloud Messaging key
            apns_key: Apple Push Notification Service key
        """
        super().__init__(NotificationChannel.PUSH)
        self.fcm_key = fcm_key
        self.apns_key = apns_key
    
    async def send(self, notification: Notification) -> bool:
        """Send push notification."""
        logger.info(
            f"Sending push to {notification.recipient}: {notification.subject}"
        )
        
        # In production, use FCM/APNS
        await asyncio.sleep(0.1)
        
        return True


class WebhookProvider(NotificationProvider):
    """Webhook notification provider."""
    
    def __init__(self):
        """Initialize webhook provider."""
        super().__init__(NotificationChannel.WEBHOOK)
    
    async def send(self, notification: Notification) -> bool:
        """Send webhook notification."""
        import httpx
        
        webhook_url = notification.recipient
        
        logger.info(f"Sending webhook to {webhook_url}")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook_url,
                json={
                    "subject": notification.subject,
                    "body": notification.body,
                    "metadata": notification.metadata
                }
            )
            
            return response.status_code == 200


class NotificationRouter:
    """Route notifications to appropriate channels."""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: int = 60
    ):
        """
        Initialize notification router.
        
        Args:
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.providers: Dict[NotificationChannel, NotificationProvider] = {}
        self.notifications: Dict[str, Notification] = {}
        
        # Retry queue
        self._retry_queue: asyncio.Queue = asyncio.Queue()
        self._retry_task: Optional[asyncio.Task] = None
    
    def register_provider(
        self,
        channel: NotificationChannel,
        provider: NotificationProvider
    ):
        """
        Register notification provider.
        
        Args:
            channel: Notification channel
            provider: Provider instance
        """
        self.providers[channel] = provider
        logger.info(f"Registered provider for channel: {channel.value}")
    
    async def send(self, notification: Notification):
        """
        Send notification.
        
        Args:
            notification: Notification to send
        """
        self.notifications[notification.notification_id] = notification
        
        # Get provider
        provider = self.providers.get(notification.channel)
        
        if not provider:
            logger.error(
                f"No provider registered for channel: {notification.channel.value}"
            )
            notification.status = NotificationStatus.FAILED
            notification.error_message = "Provider not found"
            return
        
        # Send notification
        await self._send_with_retry(notification, provider)
    
    async def _send_with_retry(
        self,
        notification: Notification,
        provider: NotificationProvider
    ):
        """Send notification with retry logic."""
        try:
            # Send notification
            success = await provider.send(notification)
            
            if success:
                notification.status = NotificationStatus.SENT
                notification.sent_at = datetime.utcnow()
                
                logger.info(
                    f"Notification {notification.notification_id} sent successfully"
                )
            else:
                raise Exception("Provider returned failure")
        
        except Exception as e:
            notification.error_message = str(e)
            notification.retry_count += 1
            
            logger.error(
                f"Notification {notification.notification_id} failed: {e}"
            )
            
            # Queue for retry
            if notification.retry_count <= self.max_retries:
                notification.status = NotificationStatus.RETRYING
                await self._retry_queue.put(notification)
            else:
                notification.status = NotificationStatus.FAILED
    
    async def _retry_loop(self):
        """Retry failed notifications."""
        while True:
            try:
                notification = await self._retry_queue.get()
                
                # Wait before retry
                await asyncio.sleep(self.retry_delay)
                
                logger.info(
                    f"Retrying notification {notification.notification_id} "
                    f"(attempt {notification.retry_count}/{self.max_retries})"
                )
                
                # Get provider and retry
                provider = self.providers.get(notification.channel)
                if provider:
                    await self._send_with_retry(notification, provider)
            
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
        logger.info("Started notification retry worker")
    
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
        logger.info("Stopped notification retry worker")
    
    async def send_multi_channel(
        self,
        recipient: str,
        subject: str,
        body: str,
        channels: List[NotificationChannel],
        **kwargs
    ):
        """
        Send notification to multiple channels.
        
        Args:
            recipient: Recipient identifier
            subject: Notification subject
            body: Notification body
            channels: List of channels
            **kwargs: Additional notification parameters
        """
        timestamp = datetime.utcnow().timestamp()
        
        tasks = []
        for channel in channels:
            notification = Notification(
                notification_id=f"{channel.value}_{timestamp}_{recipient}",
                recipient=recipient,
                channel=channel,
                subject=subject,
                body=body,
                **kwargs
            )
            
            tasks.append(self.send(notification))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_notification(
        self,
        notification_id: str
    ) -> Optional[Notification]:
        """Get notification by ID."""
        return self.notifications.get(notification_id)
    
    def list_notifications(
        self,
        channel: Optional[NotificationChannel] = None,
        status: Optional[NotificationStatus] = None,
        limit: int = 100
    ) -> List[Notification]:
        """
        List notifications.
        
        Args:
            channel: Filter by channel
            status: Filter by status
            limit: Maximum notifications to return
        
        Returns:
            List of notifications
        """
        notifications = list(self.notifications.values())
        
        if channel:
            notifications = [n for n in notifications if n.channel == channel]
        
        if status:
            notifications = [n for n in notifications if n.status == status]
        
        # Sort by created time (newest first)
        notifications.sort(key=lambda n: n.created_at, reverse=True)
        
        return notifications[:limit]
    
    def get_stats(self) -> dict:
        """Get notification statistics."""
        total = len(self.notifications)
        
        return {
            "total_notifications": total,
            "sent": sum(
                1 for n in self.notifications.values()
                if n.status == NotificationStatus.SENT
            ),
            "failed": sum(
                1 for n in self.notifications.values()
                if n.status == NotificationStatus.FAILED
            ),
            "retrying": sum(
                1 for n in self.notifications.values()
                if n.status == NotificationStatus.RETRYING
            ),
            "by_channel": {
                channel.value: sum(
                    1 for n in self.notifications.values()
                    if n.channel == channel
                )
                for channel in NotificationChannel
            }
        }


# Example usage:
"""
from src.notification_router import (
    NotificationRouter, Notification,
    EmailProvider, SmsProvider, PushProvider,
    NotificationChannel, NotificationPriority
)

# Create router
router = NotificationRouter(max_retries=3)

# Register providers
router.register_provider(
    NotificationChannel.EMAIL,
    EmailProvider(smtp_config={})
)

router.register_provider(
    NotificationChannel.SMS,
    SmsProvider(api_key="...", sender_id="MyApp")
)

router.register_provider(
    NotificationChannel.PUSH,
    PushProvider(fcm_key="...")
)

# Start retry worker
router.start_retry_worker()

# Send notification
notification = Notification(
    notification_id="notif-123",
    recipient="user@example.com",
    channel=NotificationChannel.EMAIL,
    subject="Welcome!",
    body="Thanks for signing up",
    priority=NotificationPriority.HIGH
)

await router.send(notification)

# Send to multiple channels
await router.send_multi_channel(
    recipient="user123",
    subject="Important Update",
    body="Your account needs attention",
    channels=[
        NotificationChannel.EMAIL,
        NotificationChannel.SMS,
        NotificationChannel.PUSH
    ]
)

# Get stats
stats = router.get_stats()
print(stats)
"""
