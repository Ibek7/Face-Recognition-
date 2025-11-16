#!/usr/bin/env python3
"""
API Versioning Middleware

Implements API versioning strategies:
- URL path versioning (/v1/, /v2/)
- Header versioning (Accept: application/vnd.api+json;version=1)
- Query parameter versioning (?version=1)
- Custom header versioning (X-API-Version: 1)

Features:
- Multiple versioning strategies
- Version deprecation warnings
- Version sunset dates
- Default version handling
- Version-specific routing
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, List
from enum import Enum

from fastapi import Request, Response, HTTPException, FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import Headers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VersioningStrategy(str, Enum):
    """API versioning strategies"""
    URL_PATH = "url_path"  # /v1/endpoint
    HEADER = "header"  # X-API-Version: 1
    ACCEPT_HEADER = "accept_header"  # Accept: application/vnd.api.v1+json
    QUERY_PARAM = "query_param"  # ?version=1


class APIVersion:
    """API version metadata"""
    
    def __init__(
        self,
        version: str,
        release_date: datetime,
        deprecated: bool = False,
        deprecation_date: Optional[datetime] = None,
        sunset_date: Optional[datetime] = None,
        changelog: Optional[str] = None
    ):
        self.version = version
        self.release_date = release_date
        self.deprecated = deprecated
        self.deprecation_date = deprecation_date
        self.sunset_date = sunset_date
        self.changelog = changelog
    
    def is_sunset(self) -> bool:
        """Check if version has reached sunset date"""
        if self.sunset_date is None:
            return False
        return datetime.utcnow() >= self.sunset_date
    
    def days_until_sunset(self) -> Optional[int]:
        """Get days until sunset"""
        if self.sunset_date is None:
            return None
        delta = self.sunset_date - datetime.utcnow()
        return max(0, delta.days)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "release_date": self.release_date.isoformat(),
            "deprecated": self.deprecated,
            "deprecation_date": self.deprecation_date.isoformat() if self.deprecation_date else None,
            "sunset_date": self.sunset_date.isoformat() if self.sunset_date else None,
            "days_until_sunset": self.days_until_sunset(),
            "changelog": self.changelog
        }


class VersionRegistry:
    """Registry of API versions"""
    
    def __init__(self, default_version: str = "1"):
        self.versions: Dict[str, APIVersion] = {}
        self.default_version = default_version
    
    def register(self, version: APIVersion):
        """Register an API version"""
        self.versions[version.version] = version
        logger.info(f"Registered API version {version.version}")
    
    def get_version(self, version: str) -> Optional[APIVersion]:
        """Get version metadata"""
        return self.versions.get(version)
    
    def get_latest_version(self) -> str:
        """Get latest non-deprecated version"""
        active_versions = [
            v for v in self.versions.values()
            if not v.deprecated and not v.is_sunset()
        ]
        
        if not active_versions:
            return self.default_version
        
        # Sort by release date, return latest
        latest = max(active_versions, key=lambda v: v.release_date)
        return latest.version
    
    def get_all_versions(self) -> List[Dict[str, Any]]:
        """Get all versions info"""
        return [v.to_dict() for v in self.versions.values()]
    
    def validate_version(self, version: str) -> bool:
        """Check if version is valid and not sunset"""
        api_version = self.get_version(version)
        
        if api_version is None:
            return False
        
        if api_version.is_sunset():
            return False
        
        return True


class VersionExtractor:
    """Extract version from request using different strategies"""
    
    def __init__(self, strategy: VersioningStrategy):
        self.strategy = strategy
    
    def extract(self, request: Request) -> Optional[str]:
        """Extract version from request"""
        if self.strategy == VersioningStrategy.URL_PATH:
            return self._extract_from_url(request)
        
        elif self.strategy == VersioningStrategy.HEADER:
            return self._extract_from_header(request)
        
        elif self.strategy == VersioningStrategy.ACCEPT_HEADER:
            return self._extract_from_accept_header(request)
        
        elif self.strategy == VersioningStrategy.QUERY_PARAM:
            return self._extract_from_query(request)
        
        return None
    
    def _extract_from_url(self, request: Request) -> Optional[str]:
        """Extract version from URL path (/v1/endpoint)"""
        path = request.url.path
        match = re.search(r'/v(\d+)/', path)
        
        if match:
            return match.group(1)
        
        return None
    
    def _extract_from_header(self, request: Request) -> Optional[str]:
        """Extract version from X-API-Version header"""
        return request.headers.get("X-API-Version")
    
    def _extract_from_accept_header(self, request: Request) -> Optional[str]:
        """Extract version from Accept header"""
        accept = request.headers.get("Accept", "")
        
        # Pattern: application/vnd.api.v1+json
        match = re.search(r'application/vnd\.api\.v(\d+)\+json', accept)
        
        if match:
            return match.group(1)
        
        # Pattern: application/vnd.api+json;version=1
        match = re.search(r'version=(\d+)', accept)
        
        if match:
            return match.group(1)
        
        return None
    
    def _extract_from_query(self, request: Request) -> Optional[str]:
        """Extract version from query parameter"""
        return request.query_params.get("version")


class APIVersioningMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for API versioning"""
    
    def __init__(
        self,
        app: FastAPI,
        strategy: VersioningStrategy = VersioningStrategy.URL_PATH,
        default_version: str = "1",
        strict_mode: bool = False
    ):
        super().__init__(app)
        self.strategy = strategy
        self.registry = VersionRegistry(default_version)
        self.extractor = VersionExtractor(strategy)
        self.strict_mode = strict_mode
        
        # Register default versions
        self._register_default_versions()
    
    def _register_default_versions(self):
        """Register default API versions"""
        # Version 1 (current)
        self.registry.register(
            APIVersion(
                version="1",
                release_date=datetime(2024, 1, 1),
                deprecated=False,
                changelog="Initial API release"
            )
        )
        
        # Version 2 (upcoming)
        self.registry.register(
            APIVersion(
                version="2",
                release_date=datetime(2025, 1, 1),
                deprecated=False,
                changelog="Added WebSocket support, improved error handling"
            )
        )
    
    async def dispatch(self, request: Request, call_next):
        """Process request with version handling"""
        # Extract version from request
        version = self.extractor.extract(request)
        
        # Use default if not specified
        if version is None:
            if self.strict_mode:
                return Response(
                    content='{"error": "API version not specified"}',
                    status_code=400,
                    media_type="application/json"
                )
            
            version = self.registry.default_version
            logger.debug(f"Using default API version: {version}")
        
        # Validate version
        if not self.registry.validate_version(version):
            api_version = self.registry.get_version(version)
            
            if api_version and api_version.is_sunset():
                # Version is sunset
                return Response(
                    content=f'{{"error": "API version {version} has been sunset", "latest_version": "{self.registry.get_latest_version()}"}}',
                    status_code=410,  # Gone
                    media_type="application/json"
                )
            else:
                # Invalid version
                return Response(
                    content=f'{{"error": "Invalid API version", "supported_versions": {list(self.registry.versions.keys())}}}',
                    status_code=400,
                    media_type="application/json"
                )
        
        # Add version to request state
        request.state.api_version = version
        
        # Process request
        response = await call_next(request)
        
        # Add version headers to response
        response.headers["X-API-Version"] = version
        
        # Add deprecation warning if applicable
        api_version = self.registry.get_version(version)
        if api_version and api_version.deprecated:
            response.headers["Deprecation"] = "true"
            
            if api_version.sunset_date:
                response.headers["Sunset"] = api_version.sunset_date.isoformat()
                
                days_left = api_version.days_until_sunset()
                if days_left is not None:
                    response.headers["X-Days-Until-Sunset"] = str(days_left)
            
            # Add Link header to latest version
            latest = self.registry.get_latest_version()
            if latest != version:
                response.headers["Link"] = f'</v{latest}/>; rel="latest-version"'
        
        return response


class VersionedAPIRoute(APIRoute):
    """Custom API route with version support"""
    
    def __init__(self, *args, versions: Optional[List[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.versions = versions or []


def versioned_endpoint(versions: List[str]):
    """
    Decorator to mark endpoint as available in specific versions
    
    Usage:
        @app.get("/users")
        @versioned_endpoint(versions=["1", "2"])
        async def get_users():
            return {"users": []}
    """
    def decorator(func: Callable):
        func._api_versions = versions
        return func
    return decorator


def get_api_version(request: Request) -> str:
    """Get API version from request state"""
    return getattr(request.state, "api_version", "1")


def setup_versioning_routes(app: FastAPI, middleware: APIVersioningMiddleware):
    """Setup versioning information endpoints"""
    
    @app.get("/versions")
    async def list_versions():
        """List all API versions"""
        return {
            "versions": middleware.registry.get_all_versions(),
            "default": middleware.registry.default_version,
            "latest": middleware.registry.get_latest_version()
        }
    
    @app.get("/version")
    async def current_version(request: Request):
        """Get current API version"""
        version = get_api_version(request)
        api_version = middleware.registry.get_version(version)
        
        if api_version:
            return api_version.to_dict()
        
        return {"version": version}


# Example usage with FastAPI
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI(title="Versioned Face Recognition API")
    
    # Add versioning middleware
    versioning = APIVersioningMiddleware(
        app,
        strategy=VersioningStrategy.URL_PATH,
        default_version="1",
        strict_mode=False
    )
    
    app.add_middleware(APIVersioningMiddleware, strategy=VersioningStrategy.URL_PATH)
    
    # Setup versioning routes
    setup_versioning_routes(app, versioning)
    
    # Version 1 endpoints
    @app.get("/v1/detect")
    @versioned_endpoint(versions=["1"])
    async def detect_v1(request: Request):
        """Face detection V1"""
        version = get_api_version(request)
        return {
            "version": version,
            "endpoint": "detect",
            "message": "Detection V1 - Basic detection"
        }
    
    # Version 2 endpoints (improved)
    @app.get("/v2/detect")
    @versioned_endpoint(versions=["2"])
    async def detect_v2(request: Request):
        """Face detection V2 (improved)"""
        version = get_api_version(request)
        return {
            "version": version,
            "endpoint": "detect",
            "message": "Detection V2 - Enhanced with landmarks",
            "features": ["faster", "more_accurate", "landmarks"]
        }
    
    # Endpoint available in multiple versions
    @app.get("/v{version:int}/health")
    async def health(request: Request, version: int):
        """Health check (all versions)"""
        return {
            "status": "healthy",
            "version": str(version)
        }
    
    # Default version (no version in path)
    @app.get("/")
    async def root(request: Request):
        """Root endpoint"""
        return {
            "message": "Face Recognition API",
            "version": get_api_version(request),
            "documentation": "/docs"
        }
    
    # Run server
    logger.info("Starting versioned API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
