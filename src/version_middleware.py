"""
API version management middleware for FastAPI.

Supports versioning via headers, path, or query parameters.
"""

from typing import Callable, Optional, List
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import re


class VersionMiddleware(BaseHTTPMiddleware):
    """Handle API versioning in requests."""
    
    def __init__(
        self,
        app,
        default_version: str = "v1",
        supported_versions: Optional[List[str]] = None,
    ):
        """
        Initialize version middleware.
        
        Args:
            app: FastAPI app
            default_version: Default API version
            supported_versions: List of supported versions
        """
        super().__init__(app)
        self.default_version = default_version
        self.supported_versions = supported_versions or ["v1"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with versioning."""
        # Extract version
        version = self._get_version(request)
        
        # Validate
        if version not in self.supported_versions:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": "Unsupported API version",
                    "version": version,
                    "supported": self.supported_versions,
                }
            )
        
        # Store in request state
        request.state.api_version = version
        
        # Process
        response = await call_next(request)
        response.headers["X-API-Version"] = version
        
        return response
    
    def _get_version(self, request: Request) -> str:
        """Extract version from request."""
        # Check header
        if "X-API-Version" in request.headers:
            return request.headers["X-API-Version"]
        
        # Check path
        match = re.match(r"^/(v\d+)/", request.url.path)
        if match:
            return match.group(1)
        
        # Check query
        if "version" in request.query_params:
            return request.query_params["version"]
        
        return self.default_version


def get_version(request: Request) -> str:
    """Get API version from request."""
    return getattr(request.state, "api_version", "v1")
