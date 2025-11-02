"""
Centralized logging configuration for the Face Recognition System.

This module provides a comprehensive logging setup with:
- Multiple log levels and handlers
- Automatic log rotation
- Structured JSON logging option
- Performance tracking
- Context-aware logging
"""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import traceback


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname:8}"
                f"{self.RESET}"
            )
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Formatter that outputs logs in JSON format."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted string
        """
        log_data: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that adds context information to log records."""
    
    def process(self, msg, kwargs):
        """
        Process log message with context.
        
        Args:
            msg: Log message
            kwargs: Additional keyword arguments
            
        Returns:
            Tuple of (message, kwargs)
        """
        # Add context from extra dict
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Merge adapter context with extra
        kwargs['extra'].update(self.extra)
        
        return msg, kwargs


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: str = "face_recognition.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    enable_json: bool = False,
    enable_console: bool = True,
    enable_colors: bool = True,
) -> logging.Logger:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None for no file logging)
        log_file: Name of the log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        enable_json: Whether to use JSON formatting for file logs
        enable_console: Whether to enable console logging
        enable_colors: Whether to use colored console output
        
    Returns:
        Configured logger instance
    """
    # Get root logger
    logger = logging.getLogger()
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create formatters
    console_format = (
        '%(levelname)s | %(asctime)s | %(name)s | '
        '%(funcName)s:%(lineno)d | %(message)s'
    )
    file_format = (
        '%(asctime)s | %(levelname)-8s | %(name)s | '
        '%(module)s:%(funcName)s:%(lineno)d | %(message)s'
    )
    
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if enable_colors and sys.stdout.isatty():
            console_formatter = ColoredFormatter(
                console_format,
                datefmt=date_format
            )
        else:
            console_formatter = logging.Formatter(
                console_format,
                datefmt=date_format
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_dir:
        # Create log directory if it doesn't exist
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        log_file_path = log_path / log_file
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        
        if enable_json:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                file_format,
                datefmt=date_format
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file_path}")
    
    # Add error file handler for ERROR and above
    if log_dir:
        error_file_path = log_path / f"error_{log_file}"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    logger.info(f"Logging configured: level={log_level}, console={enable_console}, file={log_dir is not None}")
    
    return logger


def get_logger(
    name: str,
    context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Get a logger instance with optional context.
    
    Args:
        name: Logger name (usually __name__)
        context: Optional context dictionary to add to all log messages
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if context:
        return ContextLogger(logger, context)
    
    return logger


class PerformanceLogger:
    """Context manager for logging function performance."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        """
        Initialize performance logger.
        
        Args:
            logger: Logger instance
            operation: Name of the operation being timed
        """
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        self.logger.debug(f"Starting: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration."""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            if exc_type:
                self.logger.error(
                    f"Failed: {self.operation} after {duration:.3f}s",
                    exc_info=(exc_type, exc_val, exc_tb)
                )
            else:
                self.logger.info(f"Completed: {self.operation} in {duration:.3f}s")


def configure_from_env() -> logging.Logger:
    """
    Configure logging from environment variables.
    
    Environment Variables:
        LOG_LEVEL: Logging level (default: INFO)
        LOG_DIR: Directory for log files (default: logs)
        LOG_FILE: Log file name (default: face_recognition.log)
        LOG_JSON: Enable JSON logging (default: false)
        LOG_CONSOLE: Enable console logging (default: true)
        LOG_COLORS: Enable colored output (default: true)
        
    Returns:
        Configured logger instance
    """
    return setup_logging(
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        log_dir=os.getenv('LOG_DIR', 'logs'),
        log_file=os.getenv('LOG_FILE', 'face_recognition.log'),
        enable_json=os.getenv('LOG_JSON', 'false').lower() == 'true',
        enable_console=os.getenv('LOG_CONSOLE', 'true').lower() == 'true',
        enable_colors=os.getenv('LOG_COLORS', 'true').lower() == 'true',
    )


# Example usage
if __name__ == "__main__":
    # Configure logging
    logger = setup_logging(
        log_level="DEBUG",
        log_dir="logs",
        enable_json=False,
        enable_colors=True
    )
    
    # Get module logger
    module_logger = get_logger(__name__, context={'module': 'logging_config'})
    
    # Test different log levels
    module_logger.debug("Debug message")
    module_logger.info("Info message")
    module_logger.warning("Warning message")
    module_logger.error("Error message")
    
    # Test performance logging
    with PerformanceLogger(module_logger, "Example Operation"):
        import time
        time.sleep(0.5)
    
    # Test exception logging
    try:
        raise ValueError("Example exception")
    except Exception as e:
        module_logger.exception("An error occurred")
    
    print("\nLogging configuration test complete!")
