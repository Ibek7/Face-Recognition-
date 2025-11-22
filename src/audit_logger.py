"""
Audit logging system for security and compliance.

Provides structured logging of security-relevant events.
"""

import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from fastapi import Request
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    
    # Authentication events
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    TOKEN_REFRESH = "auth.token.refresh"
    
    # Authorization events
    ACCESS_GRANTED = "authz.access.granted"
    ACCESS_DENIED = "authz.access.denied"
    PERMISSION_CHANGE = "authz.permission.change"
    
    # Data events
    DATA_CREATE = "data.create"
    DATA_READ = "data.read"
    DATA_UPDATE = "data.update"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    
    # System events
    CONFIG_CHANGE = "system.config.change"
    ADMIN_ACTION = "system.admin.action"
    ERROR = "system.error"
    SECURITY_ALERT = "system.security.alert"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEvent:
    """Audit event data structure."""
    
    def __init__(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Initialize audit event.
        
        Args:
            event_type: Type of event
            severity: Event severity
            user_id: User identifier
            resource: Resource affected
            action: Action performed
            result: Action result
            metadata: Additional metadata
            ip_address: Client IP address
            user_agent: Client user agent
        """
        self.timestamp = datetime.utcnow()
        self.event_type = event_type
        self.severity = severity
        self.user_id = user_id
        self.resource = resource
        self.action = action
        self.result = result
        self.metadata = metadata or {}
        self.ip_address = ip_address
        self.user_agent = user_agent
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "user_id": self.user_id,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "metadata": self.metadata,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class AuditLogger:
    """Audit logger with multiple backends."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        console_output: bool = True
    ):
        """
        Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
            console_output: Enable console output
        """
        self.log_file = log_file
        self.console_output = console_output
        self.event_queue = asyncio.Queue()
        self._running = False
        
        # Setup file logger
        if log_file:
            self._setup_file_logger(log_file)
    
    def _setup_file_logger(self, log_file: str):
        """Setup file-based audit logger."""
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(message)s')
        )
        
        audit_logger = logging.getLogger('audit')
        audit_logger.setLevel(logging.INFO)
        audit_logger.addHandler(file_handler)
        audit_logger.propagate = False
    
    async def log(self, event: AuditEvent):
        """
        Log audit event.
        
        Args:
            event: Audit event to log
        """
        # Add to queue for async processing
        await self.event_queue.put(event)
    
    async def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity = AuditSeverity.INFO,
        **kwargs
    ):
        """
        Log audit event (convenience method).
        
        Args:
            event_type: Type of event
            severity: Event severity
            **kwargs: Additional event parameters
        """
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            **kwargs
        )
        await self.log(event)
    
    async def log_from_request(
        self,
        request: Request,
        event_type: AuditEventType,
        severity: AuditSeverity = AuditSeverity.INFO,
        **kwargs
    ):
        """
        Log audit event from FastAPI request.
        
        Args:
            request: FastAPI request
            event_type: Type of event
            severity: Event severity
            **kwargs: Additional event parameters
        """
        # Extract client info
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("User-Agent")
        
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            ip_address=ip_address,
            user_agent=user_agent,
            **kwargs
        )
        
        await self.log(event)
    
    async def start_processor(self):
        """Start async event processor."""
        if self._running:
            return
        
        self._running = True
        logger.info("Started audit log processor")
        
        while self._running:
            try:
                event = await self.event_queue.get()
                self._write_event(event)
            except Exception as e:
                logger.error(f"Error processing audit event: {e}")
    
    async def stop_processor(self):
        """Stop event processor."""
        self._running = False
        logger.info("Stopped audit log processor")
    
    def _write_event(self, event: AuditEvent):
        """Write event to backends."""
        event_json = event.to_json()
        
        # Write to file
        if self.log_file:
            audit_logger = logging.getLogger('audit')
            audit_logger.info(event_json)
        
        # Write to console
        if self.console_output:
            severity_color = {
                AuditSeverity.INFO: "",
                AuditSeverity.WARNING: "\033[93m",
                AuditSeverity.ERROR: "\033[91m",
                AuditSeverity.CRITICAL: "\033[91m\033[1m"
            }
            reset = "\033[0m"
            
            color = severity_color.get(event.severity, "")
            logger.info(f"{color}AUDIT: {event_json}{reset}")
    
    async def search(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> list:
        """
        Search audit logs.
        
        Args:
            user_id: Filter by user ID
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
        
        Returns:
            List of matching events
        """
        # This is a simple implementation
        # In production, use a proper database/search engine
        
        if not self.log_file:
            return []
        
        events = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    event_dict = json.loads(line.strip())
                    
                    # Apply filters
                    if user_id and event_dict.get("user_id") != user_id:
                        continue
                    
                    if event_type and event_dict.get("event_type") != event_type.value:
                        continue
                    
                    if start_time:
                        event_time = datetime.fromisoformat(event_dict["timestamp"])
                        if event_time < start_time:
                            continue
                    
                    if end_time:
                        event_time = datetime.fromisoformat(event_dict["timestamp"])
                        if event_time > end_time:
                            continue
                    
                    events.append(event_dict)
                
                except Exception as e:
                    logger.error(f"Error parsing audit log line: {e}")
        
        return events


# Global audit logger
audit_logger = AuditLogger(
    log_file="logs/audit.log",
    console_output=True
)


# Example usage:
"""
from fastapi import FastAPI, Request
from src.audit_logger import audit_logger, AuditEventType, AuditSeverity

app = FastAPI()

@app.on_event("startup")
async def startup():
    # Start audit processor
    asyncio.create_task(audit_logger.start_processor())

@app.on_event("shutdown")
async def shutdown():
    await audit_logger.stop_processor()

@app.post("/api/login")
async def login(request: Request, username: str):
    # Log login attempt
    await audit_logger.log_from_request(
        request,
        event_type=AuditEventType.LOGIN_SUCCESS,
        severity=AuditSeverity.INFO,
        user_id=username,
        action="login"
    )
    
    return {"message": "Logged in"}

@app.delete("/api/data/{item_id}")
async def delete_item(request: Request, item_id: str):
    # Log data deletion
    await audit_logger.log_from_request(
        request,
        event_type=AuditEventType.DATA_DELETE,
        severity=AuditSeverity.WARNING,
        user_id="user123",
        resource=f"item/{item_id}",
        action="delete",
        result="success"
    )
    
    return {"message": "Deleted"}

# Search audit logs
events = await audit_logger.search(
    user_id="user123",
    event_type=AuditEventType.DATA_DELETE
)
"""
