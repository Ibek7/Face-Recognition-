"""
Notification Service

Multi-channel notification system for:
- Email notifications
- SMS notifications
- Webhook notifications
- Slack/Discord integration
- Push notifications

Features:
- Template-based messages
- Priority queuing
- Retry logic
- Rate limiting
- Notification history
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import asyncio
import aiohttp
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Notification channel types"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"
    PUSH = "push"


class Priority(int, Enum):
    """Notification priority"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class NotificationTemplate:
    """Notification message template"""
    subject: str
    body: str
    html_body: Optional[str] = None
    
    def render(self, **kwargs) -> Dict[str, str]:
        """Render template with variables"""
        return {
            "subject": self.subject.format(**kwargs),
            "body": self.body.format(**kwargs),
            "html_body": self.html_body.format(**kwargs) if self.html_body else None
        }


@dataclass
class Notification:
    """Notification message"""
    id: str
    type: NotificationType
    recipient: str
    subject: str
    body: str
    html_body: Optional[str] = None
    priority: Priority = Priority.NORMAL
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    sent_at: Optional[datetime] = None
    status: str = "pending"
    error: Optional[str] = None
    retry_count: int = 0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()


class EmailNotifier:
    """Email notification handler"""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        from_email: Optional[str] = None,
        use_tls: bool = True
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email or username
        self.use_tls = use_tls
    
    async def send(self, notification: Notification) -> bool:
        """Send email notification"""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = notification.subject
            msg['From'] = self.from_email
            msg['To'] = notification.recipient
            
            # Add text body
            text_part = MIMEText(notification.body, 'plain')
            msg.attach(text_part)
            
            # Add HTML body if provided
            if notification.html_body:
                html_part = MIMEText(notification.html_body, 'html')
                msg.attach(html_part)
            
            # Send email
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_smtp,
                msg,
                notification.recipient
            )
            
            logger.info(f"Email sent to {notification.recipient}: {notification.subject}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            notification.error = str(e)
            return False
    
    def _send_smtp(self, msg: MIMEMultipart, recipient: str):
        """Send email via SMTP (sync)"""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()
            
            if self.username and self.password:
                server.login(self.username, self.password)
            
            server.send_message(msg)


class WebhookNotifier:
    """Webhook notification handler"""
    
    def __init__(self, default_timeout: int = 30):
        self.default_timeout = default_timeout
    
    async def send(self, notification: Notification) -> bool:
        """Send webhook notification"""
        try:
            webhook_url = notification.recipient
            
            payload = {
                "id": notification.id,
                "subject": notification.subject,
                "body": notification.body,
                "priority": notification.priority.name,
                "timestamp": notification.created_at.isoformat(),
                "metadata": notification.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.default_timeout)
                ) as response:
                    response.raise_for_status()
                    
                    logger.info(f"Webhook sent to {webhook_url}: {notification.subject}")
                    return True
        
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            notification.error = str(e)
            return False


class SlackNotifier:
    """Slack notification handler"""
    
    def __init__(self, webhook_url: Optional[str] = None, token: Optional[str] = None):
        self.webhook_url = webhook_url
        self.token = token
    
    async def send(self, notification: Notification) -> bool:
        """Send Slack notification"""
        try:
            # Build Slack message
            message = {
                "text": notification.subject,
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": notification.subject
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": notification.body
                        }
                    }
                ]
            }
            
            # Add priority indicator
            priority_emoji = {
                Priority.LOW: "🔵",
                Priority.NORMAL: "🟢",
                Priority.HIGH: "🟠",
                Priority.URGENT: "🔴"
            }
            
            message["blocks"].append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"{priority_emoji[notification.priority]} Priority: *{notification.priority.name}*"
                    }
                ]
            })
            
            # Send via webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url or notification.recipient,
                    json=message
                ) as response:
                    response.raise_for_status()
                    
                    logger.info(f"Slack notification sent: {notification.subject}")
                    return True
        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            notification.error = str(e)
            return False


class NotificationService:
    """Central notification service"""
    
    def __init__(
        self,
        email_config: Optional[Dict[str, Any]] = None,
        slack_webhook: Optional[str] = None,
        max_retries: int = 3
    ):
        self.max_retries = max_retries
        
        # Initialize handlers
        self.handlers = {}
        
        if email_config:
            self.handlers[NotificationType.EMAIL] = EmailNotifier(**email_config)
        
        self.handlers[NotificationType.WEBHOOK] = WebhookNotifier()
        
        if slack_webhook:
            self.handlers[NotificationType.SLACK] = SlackNotifier(webhook_url=slack_webhook)
        
        # Notification queue
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
        # Notification history
        self.history: List[Notification] = []
        
        # Templates
        self.templates: Dict[str, NotificationTemplate] = {}
        
        # Worker task
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
    
    def register_template(self, name: str, template: NotificationTemplate):
        """Register a notification template"""
        self.templates[name] = template
        logger.info(f"Registered template: {name}")
    
    async def send_notification(
        self,
        notification_type: NotificationType,
        recipient: str,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        priority: Priority = Priority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        notification_id: Optional[str] = None
    ) -> str:
        """
        Send a notification
        
        Args:
            notification_type: Type of notification
            recipient: Recipient (email, phone, webhook URL)
            subject: Notification subject
            body: Notification body
            html_body: HTML body (for email)
            priority: Notification priority
            metadata: Additional metadata
            notification_id: Optional custom ID
        
        Returns:
            Notification ID
        """
        if notification_id is None:
            notification_id = f"{notification_type.value}_{datetime.now().timestamp()}"
        
        notification = Notification(
            id=notification_id,
            type=notification_type,
            recipient=recipient,
            subject=subject,
            body=body,
            html_body=html_body,
            priority=priority,
            metadata=metadata or {}
        )
        
        # Add to queue (priority queue - higher priority first)
        await self.queue.put((-priority.value, notification))
        
        logger.info(f"Queued notification {notification_id} with priority {priority.name}")
        
        return notification_id
    
    async def send_from_template(
        self,
        template_name: str,
        notification_type: NotificationType,
        recipient: str,
        priority: Priority = Priority.NORMAL,
        **template_vars
    ) -> str:
        """Send notification using a template"""
        if template_name not in self.templates:
            raise ValueError(f"Template not found: {template_name}")
        
        template = self.templates[template_name]
        rendered = template.render(**template_vars)
        
        return await self.send_notification(
            notification_type=notification_type,
            recipient=recipient,
            subject=rendered["subject"],
            body=rendered["body"],
            html_body=rendered["html_body"],
            priority=priority,
            metadata={"template": template_name, "vars": template_vars}
        )
    
    async def start_worker(self):
        """Start notification worker"""
        if self._running:
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._process_notifications())
        logger.info("Notification worker started")
    
    async def stop_worker(self):
        """Stop notification worker"""
        if not self._running:
            return
        
        self._running = False
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Notification worker stopped")
    
    async def _process_notifications(self):
        """Process notification queue"""
        while self._running:
            try:
                # Get next notification
                _, notification = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                
                # Send notification
                await self._send_with_retry(notification)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing notification: {e}")
    
    async def _send_with_retry(self, notification: Notification):
        """Send notification with retry logic"""
        handler = self.handlers.get(notification.type)
        
        if not handler:
            logger.error(f"No handler for notification type: {notification.type}")
            notification.status = "failed"
            notification.error = f"No handler for type {notification.type}"
            self.history.append(notification)
            return
        
        while notification.retry_count <= self.max_retries:
            try:
                success = await handler.send(notification)
                
                if success:
                    notification.status = "sent"
                    notification.sent_at = datetime.now()
                    self.history.append(notification)
                    logger.info(f"Notification {notification.id} sent successfully")
                    return
                
                notification.retry_count += 1
                
                if notification.retry_count <= self.max_retries:
                    # Exponential backoff
                    wait_time = min(2 ** notification.retry_count, 60)
                    logger.info(f"Retrying notification {notification.id} in {wait_time}s...")
                    await asyncio.sleep(wait_time)
            
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
                notification.retry_count += 1
                notification.error = str(e)
        
        notification.status = "failed"
        self.history.append(notification)
        logger.error(f"Notification {notification.id} failed after {self.max_retries} retries")
    
    def get_history(
        self,
        limit: Optional[int] = None,
        status: Optional[str] = None,
        notification_type: Optional[NotificationType] = None
    ) -> List[Notification]:
        """Get notification history"""
        history = self.history
        
        if status:
            history = [n for n in history if n.status == status]
        
        if notification_type:
            history = [n for n in history if n.type == notification_type]
        
        if limit:
            history = history[-limit:]
        
        return history


# Pre-defined templates
DEFAULT_TEMPLATES = {
    "face_detected": NotificationTemplate(
        subject="Face Detected: {name}",
        body="A face was detected for {name} at {timestamp}.\n\nConfidence: {confidence}%\nLocation: {location}",
        html_body="<h2>Face Detected</h2><p>A face was detected for <strong>{name}</strong></p><p>Time: {timestamp}<br>Confidence: {confidence}%<br>Location: {location}</p>"
    ),
    "new_person_enrolled": NotificationTemplate(
        subject="New Person Enrolled: {name}",
        body="New person '{name}' has been enrolled in the system.\n\nEmail: {email}\nEnrolled at: {timestamp}",
        html_body="<h2>New Person Enrolled</h2><p><strong>{name}</strong> has been added to the system.</p><p>Email: {email}<br>Time: {timestamp}</p>"
    ),
    "system_alert": NotificationTemplate(
        subject="System Alert: {alert_type}",
        body="System alert: {message}\n\nSeverity: {severity}\nTimestamp: {timestamp}",
        html_body="<h2>System Alert</h2><p><strong>{alert_type}</strong></p><p>{message}</p><p>Severity: {severity}<br>Time: {timestamp}</p>"
    )
}


# Example usage
async def main():
    """Example usage"""
    # Initialize service
    service = NotificationService(
        email_config={
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "your-email@gmail.com",
            "password": "your-password",
            "use_tls": True
        },
        slack_webhook="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    )
    
    # Register default templates
    for name, template in DEFAULT_TEMPLATES.items():
        service.register_template(name, template)
    
    # Start worker
    await service.start_worker()
    
    # Send webhook notification
    await service.send_notification(
        notification_type=NotificationType.WEBHOOK,
        recipient="https://example.com/webhook",
        subject="Test Webhook",
        body="This is a test webhook notification",
        priority=Priority.HIGH
    )
    
    # Send from template
    await service.send_from_template(
        template_name="face_detected",
        notification_type=NotificationType.WEBHOOK,
        recipient="https://example.com/webhook",
        priority=Priority.NORMAL,
        name="John Doe",
        timestamp=datetime.now().isoformat(),
        confidence=95.5,
        location="Main Entrance"
    )
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Get history
    history = service.get_history(limit=10)
    print(f"\nNotification History ({len(history)} items):")
    for notification in history:
        print(f"  {notification.id}: {notification.status} - {notification.subject}")
    
    # Stop worker
    await service.stop_worker()


if __name__ == "__main__":
    asyncio.run(main())
