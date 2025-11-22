"""
Retry mechanism with exponential backoff utility.

Provides decorators and utilities for retrying failed operations.
"""

import asyncio
import functools
import logging
import random
import time
from typing import Callable, Type, Tuple, Optional

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delay
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"Failed after {max_attempts} attempts: {func.__name__}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        initial_delay * (exponential_base ** (attempt - 1)),
                        max_delay
                    )
                    
                    # Add jitter
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}. "
                        f"Retrying in {delay:.2f}s. Error: {e}"
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"Failed after {max_attempts} attempts: {func.__name__}"
                        )
                        raise
                    
                    delay = min(
                        initial_delay * (exponential_base ** (attempt - 1)),
                        max_delay
                    )
                    
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}. "
                        f"Retrying in {delay:.2f}s. Error: {e}"
                    )
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class RetryContext:
    """Context manager for retry operations."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        """
        Initialize retry context.
        
        Args:
            max_attempts: Maximum retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            exponential_base: Exponential backoff base
            exceptions: Exceptions to catch
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.exceptions = exceptions
        self.attempt = 0
    
    def __enter__(self):
        """Enter context."""
        self.attempt += 1
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        if exc_type is None:
            return True
        
        if not issubclass(exc_type, self.exceptions):
            return False
        
        if self.attempt >= self.max_attempts:
            logger.error(f"Failed after {self.max_attempts} attempts")
            return False
        
        delay = min(
            self.initial_delay * (self.exponential_base ** (self.attempt - 1)),
            self.max_delay
        )
        delay = delay * (0.5 + random.random())
        
        logger.warning(
            f"Attempt {self.attempt}/{self.max_attempts} failed. "
            f"Retrying in {delay:.2f}s"
        )
        
        time.sleep(delay)
        return True


def retry_on_exception(
    operation: Callable,
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> any:
    """
    Retry operation with simple linear backoff.
    
    Args:
        operation: Function to retry
        max_attempts: Maximum attempts
        delay: Delay between retries
        exceptions: Exceptions to catch
        
    Returns:
        Operation result
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except exceptions as e:
            if attempt == max_attempts:
                raise
            
            logger.warning(f"Attempt {attempt} failed, retrying in {delay}s")
            time.sleep(delay)


# Example usage:
"""
# Decorator usage
@retry(max_attempts=5, initial_delay=2.0)
def fetch_data():
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()

# Async decorator
@retry(max_attempts=3, exceptions=(ConnectionError,))
async def async_fetch():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com") as resp:
            return await resp.json()

# Context manager
retry_ctx = RetryContext(max_attempts=3)
while retry_ctx.attempt < retry_ctx.max_attempts:
    with retry_ctx:
        # Your operation here
        risky_operation()

# Function call
result = retry_on_exception(
    lambda: requests.get("https://api.example.com"),
    max_attempts=3,
    delay=2.0
)
"""
