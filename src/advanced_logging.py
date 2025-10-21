# Advanced Logging System

import logging
import logging.handlers
import json
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import traceback

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: float
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    context: Dict[str, Any] = None
    exception: Optional[str] = None

class StructuredFormatter(logging.Formatter):
    """Structured JSON logging formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        
        log_entry = LogEntry(
            timestamp=record.created,
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno
        )
        
        if record.exc_info:
            log_entry.exception = traceback.format_exception(*record.exc_info)[0]
        
        return json.dumps(asdict(log_entry), default=str)

class ContextualLogger:
    """Logger with context management."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._context = threading.local()
    
    def set_context(self, **kwargs):
        """Set context variables."""
        if not hasattr(self._context, 'data'):
            self._context.data = {}
        self._context.data.update(kwargs)
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context."""
        if hasattr(self._context, 'data'):
            return self._context.data.copy()
        return {}
    
    def clear_context(self):
        """Clear context."""
        if hasattr(self._context, 'data'):
            self._context.data.clear()
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log with context."""
        context = self.get_context()
        
        if context:
            msg = f"{msg} | context: {json.dumps(context)}"
        
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)

class RotatingFileLogger:
    """Rotating file logger with compression."""
    
    def __init__(self, log_dir: str = "logs", max_bytes: int = 10485760, backup_count: int = 5):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("app")
        self.logger.setLevel(logging.DEBUG)
        
        # Rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "app.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        
        formatter = StructuredFormatter()
        handler.setFormatter(formatter)
        
        self.logger.addHandler(handler)
    
    def get_logger(self) -> logging.Logger:
        """Get logger instance."""
        return self.logger

class PerformanceLogger:
    """Log performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger("performance")
        self.metrics: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
    
    def log_metric(self, metric_name: str, value: float, unit: str = "", tags: Dict = None):
        """Log performance metric."""
        
        metric = {
            'timestamp': datetime.now().isoformat(),
            'metric': metric_name,
            'value': value,
            'unit': unit,
            'tags': tags or {}
        }
        
        with self.lock:
            self.metrics.append(metric)
            self.logger.info(f"Metric: {metric_name}={value}{unit}", extra={'metric': metric})
    
    def log_function_call(self, func_name: str, duration_ms: float, success: bool = True):
        """Log function call performance."""
        status = "success" if success else "failure"
        self.log_metric(
            f"{func_name}_duration",
            duration_ms,
            unit="ms",
            tags={'status': status}
        )
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get recorded metrics."""
        with self.lock:
            return self.metrics.copy()

class AuditLogger:
    """Audit logging for security and compliance."""
    
    def __init__(self):
        self.logger = logging.getLogger("audit")
        self.audit_trail: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
    
    def log_access(self, user_id: str, resource: str, action: str, success: bool = True):
        """Log resource access."""
        
        audit_log = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'ACCESS',
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'success': success
        }
        
        with self.lock:
            self.audit_trail.append(audit_log)
            self.logger.info(
                f"Access: {user_id} -> {action} {resource}",
                extra={'audit': audit_log}
            )
    
    def log_modification(self, user_id: str, resource: str, old_value: Any, new_value: Any):
        """Log resource modification."""
        
        audit_log = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'MODIFICATION',
            'user_id': user_id,
            'resource': resource,
            'old_value': old_value,
            'new_value': new_value
        }
        
        with self.lock:
            self.audit_trail.append(audit_log)
            self.logger.warning(
                f"Modified: {user_id} changed {resource}",
                extra={'audit': audit_log}
            )
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event."""
        
        audit_log = {
            'timestamp': datetime.now().isoformat(),
            'event_type': f"SECURITY_{event_type}",
            'details': details
        }
        
        with self.lock:
            self.audit_trail.append(audit_log)
            self.logger.critical(
                f"Security Event: {event_type}",
                extra={'audit': audit_log}
            )
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail."""
        with self.lock:
            return self.audit_trail.copy()

class LogAnalyzer:
    """Analyze logs for patterns and anomalies."""
    
    def __init__(self):
        self.logger = logging.getLogger("analyzer")
    
    def analyze_error_patterns(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """Analyze error patterns."""
        
        error_logs = [log for log in logs if log.level == "ERROR"]
        
        if not error_logs:
            return {}
        
        # Group by module
        errors_by_module = {}
        for log in error_logs:
            if log.module not in errors_by_module:
                errors_by_module[log.module] = []
            errors_by_module[log.module].append(log)
        
        # Calculate statistics
        stats = {
            'total_errors': len(error_logs),
            'errors_by_module': {
                module: len(logs)
                for module, logs in errors_by_module.items()
            },
            'most_common_module': max(errors_by_module.keys(), 
                                      key=lambda k: len(errors_by_module[k])) if errors_by_module else None
        }
        
        return stats
    
    def find_anomalies(self, logs: List[LogEntry]) -> List[LogEntry]:
        """Find anomalous logs."""
        
        # Simple heuristic: logs with exceptions or critical level
        anomalies = [
            log for log in logs 
            if log.level in ["CRITICAL", "ERROR"] or log.exception
        ]
        
        return anomalies

# Global logger instances
_rotating_logger = RotatingFileLogger()
_performance_logger = PerformanceLogger()
_audit_logger = AuditLogger()

def get_logger(name: str) -> ContextualLogger:
    """Get contextual logger."""
    return ContextualLogger(name)

def get_performance_logger() -> PerformanceLogger:
    """Get performance logger."""
    return _performance_logger

def get_audit_logger() -> AuditLogger:
    """Get audit logger."""
    return _audit_logger

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test contextual logger
    logger = get_logger("test")
    logger.set_context(user_id="user123", request_id="req456")
    
    logger.info("Processing request")
    logger.warning("High memory usage detected")
    
    # Test performance logger
    perf_logger = get_performance_logger()
    perf_logger.log_metric("inference_time", 45.2, unit="ms")
    perf_logger.log_function_call("face_detection", 123.5, success=True)
    
    # Test audit logger
    audit_logger = get_audit_logger()
    audit_logger.log_access("user123", "/api/faces", "GET", success=True)
    audit_logger.log_modification("admin", "model_weights", "v1.0", "v1.1")
    
    print("Logging system initialized successfully!")
