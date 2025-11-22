"""
Email notification service with template support.

Provides email sending with HTML/text templates and attachments.
"""

import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any
from pathlib import Path
from jinja2 import Template
import aiosmtplib
import logging

logger = logging.getLogger(__name__)


class EmailConfig:
    """Email service configuration."""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None
    ):
        """
        Initialize email config.
        
        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            use_tls: Use TLS encryption
            from_email: Default from email
            from_name: Default from name
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.from_email = from_email or username
        self.from_name = from_name


class EmailMessage:
    """Email message."""
    
    def __init__(
        self,
        to: List[str],
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        from_email: Optional[str] = None,
        from_name: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize email message.
        
        Args:
            to: Recipient email addresses
            subject: Email subject
            body: Plain text body
            html_body: HTML body
            from_email: Sender email
            from_name: Sender name
            cc: CC recipients
            bcc: BCC recipients
            attachments: File paths to attach
            headers: Custom headers
        """
        self.to = to
        self.subject = subject
        self.body = body
        self.html_body = html_body
        self.from_email = from_email
        self.from_name = from_name
        self.cc = cc or []
        self.bcc = bcc or []
        self.attachments = attachments or []
        self.headers = headers or {}


class EmailTemplate:
    """Email template with Jinja2."""
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize email template.
        
        Args:
            template_dir: Directory containing templates
        """
        self.template_dir = Path(template_dir) if template_dir else None
        self.templates: Dict[str, Template] = {}
    
    def register_template(self, name: str, content: str):
        """
        Register template.
        
        Args:
            name: Template name
            content: Template content (Jinja2)
        """
        self.templates[name] = Template(content)
    
    def load_template(self, name: str, filename: str):
        """
        Load template from file.
        
        Args:
            name: Template name
            filename: Template filename
        """
        if not self.template_dir:
            raise ValueError("Template directory not set")
        
        template_path = self.template_dir / filename
        
        with open(template_path, 'r') as f:
            content = f.read()
        
        self.register_template(name, content)
    
    def render(self, name: str, context: Dict[str, Any]) -> str:
        """
        Render template.
        
        Args:
            name: Template name
            context: Template context
        
        Returns:
            Rendered content
        """
        if name not in self.templates:
            raise ValueError(f"Template not found: {name}")
        
        return self.templates[name].render(**context)


class EmailNotifier:
    """Email notification service."""
    
    def __init__(self, config: EmailConfig):
        """
        Initialize email notifier.
        
        Args:
            config: Email configuration
        """
        self.config = config
        self.template = EmailTemplate()
        self.sent_count = 0
        self.failed_count = 0
    
    async def send(self, message: EmailMessage) -> bool:
        """
        Send email asynchronously.
        
        Args:
            message: Email message
        
        Returns:
            True if sent successfully
        """
        try:
            # Create MIME message
            mime_msg = self._create_mime_message(message)
            
            # Send via SMTP
            async with aiosmtplib.SMTP(
                hostname=self.config.smtp_host,
                port=self.config.smtp_port,
                use_tls=self.config.use_tls
            ) as smtp:
                # Login if credentials provided
                if self.config.username and self.config.password:
                    await smtp.login(
                        self.config.username,
                        self.config.password
                    )
                
                # Send email
                await smtp.send_message(mime_msg)
            
            self.sent_count += 1
            logger.info(f"Email sent to {', '.join(message.to)}")
            
            return True
        
        except Exception as e:
            self.failed_count += 1
            logger.error(f"Failed to send email: {e}")
            return False
    
    def send_sync(self, message: EmailMessage) -> bool:
        """
        Send email synchronously.
        
        Args:
            message: Email message
        
        Returns:
            True if sent successfully
        """
        try:
            # Create MIME message
            mime_msg = self._create_mime_message(message)
            
            # Connect to SMTP server
            if self.config.use_tls:
                smtp = smtplib.SMTP(
                    self.config.smtp_host,
                    self.config.smtp_port
                )
                smtp.starttls()
            else:
                smtp = smtplib.SMTP(
                    self.config.smtp_host,
                    self.config.smtp_port
                )
            
            # Login
            if self.config.username and self.config.password:
                smtp.login(self.config.username, self.config.password)
            
            # Send
            smtp.send_message(mime_msg)
            smtp.quit()
            
            self.sent_count += 1
            logger.info(f"Email sent to {', '.join(message.to)}")
            
            return True
        
        except Exception as e:
            self.failed_count += 1
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _create_mime_message(self, message: EmailMessage) -> MIMEMultipart:
        """Create MIME message."""
        # Create message
        mime_msg = MIMEMultipart('alternative')
        
        # Set headers
        from_email = message.from_email or self.config.from_email
        from_name = message.from_name or self.config.from_name
        
        if from_name:
            mime_msg['From'] = f"{from_name} <{from_email}>"
        else:
            mime_msg['From'] = from_email
        
        mime_msg['To'] = ', '.join(message.to)
        mime_msg['Subject'] = message.subject
        
        if message.cc:
            mime_msg['Cc'] = ', '.join(message.cc)
        
        # Custom headers
        for key, value in message.headers.items():
            mime_msg[key] = value
        
        # Add plain text body
        text_part = MIMEText(message.body, 'plain')
        mime_msg.attach(text_part)
        
        # Add HTML body if provided
        if message.html_body:
            html_part = MIMEText(message.html_body, 'html')
            mime_msg.attach(html_part)
        
        # Add attachments
        for filepath in message.attachments:
            self._add_attachment(mime_msg, filepath)
        
        return mime_msg
    
    def _add_attachment(self, mime_msg: MIMEMultipart, filepath: str):
        """Add attachment to message."""
        try:
            path = Path(filepath)
            
            with open(path, 'rb') as f:
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(f.read())
            
            encoders.encode_base64(attachment)
            attachment.add_header(
                'Content-Disposition',
                f'attachment; filename={path.name}'
            )
            
            mime_msg.attach(attachment)
        
        except Exception as e:
            logger.error(f"Failed to attach file {filepath}: {e}")
    
    async def send_template(
        self,
        to: List[str],
        subject: str,
        template_name: str,
        context: Dict[str, Any],
        **kwargs
    ) -> bool:
        """
        Send email using template.
        
        Args:
            to: Recipients
            subject: Subject
            template_name: Template name
            context: Template context
            **kwargs: Additional message parameters
        
        Returns:
            True if sent successfully
        """
        # Render template
        html_body = self.template.render(template_name, context)
        
        # Create message
        message = EmailMessage(
            to=to,
            subject=subject,
            body="",  # Plain text fallback
            html_body=html_body,
            **kwargs
        )
        
        return await self.send(message)
    
    def get_stats(self) -> dict:
        """Get email statistics."""
        return {
            "sent": self.sent_count,
            "failed": self.failed_count,
            "success_rate": (
                self.sent_count / (self.sent_count + self.failed_count) * 100
                if self.sent_count + self.failed_count > 0
                else 0
            )
        }


# Example usage:
"""
from src.email_notifier import EmailNotifier, EmailConfig, EmailMessage

# Configure email service
config = EmailConfig(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    username="your-email@gmail.com",
    password="your-password",
    use_tls=True,
    from_email="noreply@example.com",
    from_name="Face Recognition System"
)

notifier = EmailNotifier(config)

# Register template
notifier.template.register_template(
    "welcome",
    '''
    <h1>Welcome {{name}}!</h1>
    <p>Thank you for signing up.</p>
    '''
)

# Send simple email
message = EmailMessage(
    to=["user@example.com"],
    subject="Test Email",
    body="This is a test email",
    html_body="<p>This is a <b>test</b> email</p>"
)

await notifier.send(message)

# Send templated email
await notifier.send_template(
    to=["user@example.com"],
    subject="Welcome!",
    template_name="welcome",
    context={"name": "John"}
)

# Get statistics
stats = notifier.get_stats()
print(f"Sent: {stats['sent']}, Failed: {stats['failed']}")
"""
