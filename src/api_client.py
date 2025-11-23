"""
Generic API client with retry logic and circuit breaker.

Provides resilient HTTP client with automatic retries and failure handling.
"""

from typing import Dict, Optional, Any, List
from enum import Enum
import asyncio
import httpx
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker state."""
    
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


class RetryStrategy(str, Enum):
    """Retry strategy."""
    
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening
            recovery_timeout: Seconds before trying again
            success_threshold: Successes needed to close
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time: Optional[datetime] = None
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                
                if elapsed >= self.recovery_timeout:
                    logger.info("Circuit breaker entering half-open state")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    return False
            
            return True
        
        return False
    
    def record_success(self):
        """Record successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.success_threshold:
                logger.info("Circuit breaker closing after recovery")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker reopening after failed recovery attempt")
            self.state = CircuitState.OPEN
            self.success_count = 0
        
        elif self.failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker opening after {self.failure_count} failures"
            )
            self.state = CircuitState.OPEN


class ApiClient:
    """Generic API client with resilience features."""
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        retry_delay: float = 1.0,
        use_circuit_breaker: bool = True,
        default_headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL for API
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_strategy: Retry backoff strategy
            retry_delay: Initial retry delay
            use_circuit_breaker: Enable circuit breaker
            default_headers: Default headers for all requests
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_strategy = retry_strategy
        self.retry_delay = retry_delay
        self.default_headers = default_headers or {}
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker() if use_circuit_breaker else None
        
        # HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True
        )
        
        # Metrics
        self.request_count = 0
        self.error_count = 0
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy."""
        if self.retry_strategy == RetryStrategy.FIXED:
            return self.retry_delay
        
        elif self.retry_strategy == RetryStrategy.LINEAR:
            return self.retry_delay * attempt
        
        elif self.retry_strategy == RetryStrategy.EXPONENTIAL:
            return self.retry_delay * (2 ** (attempt - 1))
        
        return self.retry_delay
    
    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> httpx.Response:
        """Make request with retry logic."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Merge headers
        headers = {**self.default_headers, **kwargs.pop("headers", {})}
        
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    **kwargs
                )
                
                # Check for retryable status codes
                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Server error: {response.status_code}",
                        request=response.request,
                        response=response
                    )
                
                return response
            
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt}/{self.max_retries}), "
                        f"retrying in {delay}s: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Request failed after {self.max_retries} attempts: {str(e)}"
                    )
        
        # All retries exhausted
        raise last_exception
    
    async def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with circuit breaker and retry.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters
        
        Returns:
            HTTP response
        
        Raises:
            Exception: If circuit is open or request fails
        """
        # Check circuit breaker
        if self.circuit_breaker and self.circuit_breaker.is_open():
            raise Exception("Circuit breaker is open, request rejected")
        
        self.request_count += 1
        
        try:
            response = await self._request_with_retry(method, endpoint, **kwargs)
            
            # Record success
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
            
            return response
        
        except Exception as e:
            self.error_count += 1
            
            # Record failure
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            raise
    
    async def get(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make GET request."""
        return await self.request("GET", endpoint, **kwargs)
    
    async def post(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make POST request."""
        return await self.request("POST", endpoint, **kwargs)
    
    async def put(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make PUT request."""
        return await self.request("PUT", endpoint, **kwargs)
    
    async def delete(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make DELETE request."""
        return await self.request("DELETE", endpoint, **kwargs)
    
    async def patch(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make PATCH request."""
        return await self.request("PATCH", endpoint, **kwargs)
    
    def get_metrics(self) -> dict:
        """Get client metrics."""
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "circuit_state": self.circuit_breaker.state.value if self.circuit_breaker else None
        }


class AuthenticatedApiClient(ApiClient):
    """API client with authentication support."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        bearer_token: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize authenticated API client.
        
        Args:
            base_url: Base URL
            api_key: API key for X-API-Key header
            bearer_token: Bearer token for Authorization
            **kwargs: Additional ApiClient parameters
        """
        headers = kwargs.pop("default_headers", {})
        
        if api_key:
            headers["X-API-Key"] = api_key
        
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        
        super().__init__(base_url, default_headers=headers, **kwargs)


class RateLimitedApiClient(ApiClient):
    """API client with built-in rate limiting."""
    
    def __init__(
        self,
        base_url: str,
        requests_per_second: float = 10.0,
        **kwargs
    ):
        """
        Initialize rate-limited API client.
        
        Args:
            base_url: Base URL
            requests_per_second: Max requests per second
            **kwargs: Additional ApiClient parameters
        """
        super().__init__(base_url, **kwargs)
        
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time: Optional[float] = None
        self._rate_limit_lock = asyncio.Lock()
    
    async def request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make rate-limited request."""
        async with self._rate_limit_lock:
            # Wait if needed to respect rate limit
            if self.last_request_time:
                elapsed = asyncio.get_event_loop().time() - self.last_request_time
                
                if elapsed < self.min_interval:
                    await asyncio.sleep(self.min_interval - elapsed)
            
            self.last_request_time = asyncio.get_event_loop().time()
        
        return await super().request(method, endpoint, **kwargs)


# Example usage:
"""
from src.api_client import ApiClient, AuthenticatedApiClient, RetryStrategy

# Basic client
client = ApiClient(
    base_url="https://api.example.com",
    max_retries=3,
    retry_strategy=RetryStrategy.EXPONENTIAL
)

# Make requests
response = await client.get("/users/123")
data = response.json()

# Authenticated client
auth_client = AuthenticatedApiClient(
    base_url="https://api.example.com",
    bearer_token="your-token-here"
)

# Create resource
response = await auth_client.post(
    "/users",
    json={"name": "John Doe"}
)

# Check metrics
metrics = client.get_metrics()
print(f"Error rate: {metrics['error_rate']:.2%}")

# Cleanup
await client.close()
"""
