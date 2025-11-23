"""
API version manager for backward compatibility.

Handles version detection, routing, and deprecation warnings.
"""

from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
from functools import wraps
import re
import logging

logger = logging.getLogger(__name__)


class VersionStatus(str, Enum):
    """API version status."""
    
    CURRENT = "current"
    SUPPORTED = "supported"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"


class ApiVersion:
    """API version definition."""
    
    def __init__(
        self,
        version: str,
        status: VersionStatus = VersionStatus.CURRENT,
        release_date: Optional[datetime] = None,
        deprecation_date: Optional[datetime] = None,
        sunset_date: Optional[datetime] = None
    ):
        """
        Initialize API version.
        
        Args:
            version: Version string (e.g., "1.0", "2.1")
            status: Version status
            release_date: Release date
            deprecation_date: Deprecation date
            sunset_date: Sunset (end-of-life) date
        """
        self.version = version
        self.status = status
        self.release_date = release_date or datetime.utcnow()
        self.deprecation_date = deprecation_date
        self.sunset_date = sunset_date
    
    def is_active(self) -> bool:
        """Check if version is active."""
        return self.status in [VersionStatus.CURRENT, VersionStatus.SUPPORTED]
    
    def is_deprecated(self) -> bool:
        """Check if version is deprecated."""
        return self.status == VersionStatus.DEPRECATED
    
    def days_until_sunset(self) -> Optional[int]:
        """Get days until sunset."""
        if not self.sunset_date:
            return None
        
        delta = self.sunset_date - datetime.utcnow()
        return max(0, delta.days)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "status": self.status.value,
            "release_date": self.release_date.isoformat(),
            "deprecation_date": self.deprecation_date.isoformat() if self.deprecation_date else None,
            "sunset_date": self.sunset_date.isoformat() if self.sunset_date else None,
            "days_until_sunset": self.days_until_sunset()
        }


class VersionExtractor:
    """Extract API version from requests."""
    
    # Version patterns
    HEADER_PATTERN = re.compile(r"application/vnd\.api\+json;version=(\d+\.\d+)")
    PATH_PATTERN = re.compile(r"/v(\d+\.\d+)/")
    QUERY_PATTERN = re.compile(r"version=(\d+\.\d+)")
    
    @classmethod
    def from_header(cls, accept_header: str) -> Optional[str]:
        """Extract version from Accept header."""
        if not accept_header:
            return None
        
        match = cls.HEADER_PATTERN.search(accept_header)
        return match.group(1) if match else None
    
    @classmethod
    def from_path(cls, path: str) -> Optional[str]:
        """Extract version from URL path."""
        match = cls.PATH_PATTERN.search(path)
        return match.group(1) if match else None
    
    @classmethod
    def from_query(cls, query_string: str) -> Optional[str]:
        """Extract version from query string."""
        if not query_string:
            return None
        
        match = cls.QUERY_PATTERN.search(query_string)
        return match.group(1) if match else None
    
    @classmethod
    def from_custom_header(cls, header_value: str) -> Optional[str]:
        """Extract version from custom header."""
        if not header_value:
            return None
        
        # Simple version string like "1.0" or "2.1"
        if re.match(r"^\d+\.\d+$", header_value):
            return header_value
        
        return None


class ApiVersionManager:
    """Manage API versions."""
    
    def __init__(
        self,
        default_version: str = "1.0",
        version_header: str = "X-API-Version"
    ):
        """
        Initialize API version manager.
        
        Args:
            default_version: Default version to use
            version_header: Header name for version
        """
        self.default_version = default_version
        self.version_header = version_header
        self.versions: Dict[str, ApiVersion] = {}
        self.route_handlers: Dict[str, Dict[str, Callable]] = {}
    
    def register_version(
        self,
        version: str,
        status: VersionStatus = VersionStatus.CURRENT,
        **kwargs
    ):
        """
        Register API version.
        
        Args:
            version: Version string
            status: Version status
            **kwargs: Additional version parameters
        """
        api_version = ApiVersion(version=version, status=status, **kwargs)
        self.versions[version] = api_version
        
        logger.info(f"Registered API version: {version} ({status.value})")
    
    def deprecate_version(
        self,
        version: str,
        sunset_in_days: int = 180
    ):
        """
        Deprecate version with sunset date.
        
        Args:
            version: Version to deprecate
            sunset_in_days: Days until sunset
        """
        if version not in self.versions:
            raise ValueError(f"Version not found: {version}")
        
        api_version = self.versions[version]
        api_version.status = VersionStatus.DEPRECATED
        api_version.deprecation_date = datetime.utcnow()
        api_version.sunset_date = datetime.utcnow() + timedelta(days=sunset_in_days)
        
        logger.warning(
            f"Deprecated version {version}, sunset in {sunset_in_days} days"
        )
    
    def sunset_version(self, version: str):
        """Mark version as sunset (end-of-life)."""
        if version not in self.versions:
            raise ValueError(f"Version not found: {version}")
        
        api_version = self.versions[version]
        api_version.status = VersionStatus.SUNSET
        
        logger.warning(f"Version {version} has reached end-of-life")
    
    def get_version(
        self,
        accept_header: Optional[str] = None,
        path: Optional[str] = None,
        query_string: Optional[str] = None,
        custom_header: Optional[str] = None
    ) -> str:
        """
        Detect API version from request.
        
        Args:
            accept_header: Accept header
            path: Request path
            query_string: Query string
            custom_header: Custom version header
        
        Returns:
            Version string
        """
        # Priority: custom header > path > query > accept header > default
        version = None
        
        if custom_header:
            version = VersionExtractor.from_custom_header(custom_header)
        
        if not version and path:
            version = VersionExtractor.from_path(path)
        
        if not version and query_string:
            version = VersionExtractor.from_query(query_string)
        
        if not version and accept_header:
            version = VersionExtractor.from_header(accept_header)
        
        if not version:
            version = self.default_version
        
        # Validate version exists
        if version not in self.versions:
            logger.warning(f"Unknown version {version}, using default {self.default_version}")
            version = self.default_version
        
        return version
    
    def check_version_status(self, version: str) -> dict:
        """
        Check version status and get warnings.
        
        Args:
            version: Version to check
        
        Returns:
            Status information
        """
        if version not in self.versions:
            return {
                "valid": False,
                "error": f"Unknown version: {version}"
            }
        
        api_version = self.versions[version]
        result = {"valid": True, "version": api_version.to_dict()}
        
        if api_version.status == VersionStatus.SUNSET:
            result["warning"] = f"Version {version} has reached end-of-life"
        elif api_version.status == VersionStatus.DEPRECATED:
            days_left = api_version.days_until_sunset()
            result["warning"] = (
                f"Version {version} is deprecated. "
                f"Sunset in {days_left} days."
            )
        
        return result
    
    def list_versions(self, include_sunset: bool = False) -> List[dict]:
        """
        List all API versions.
        
        Args:
            include_sunset: Include sunset versions
        
        Returns:
            List of version info
        """
        versions = []
        
        for version in self.versions.values():
            if not include_sunset and version.status == VersionStatus.SUNSET:
                continue
            
            versions.append(version.to_dict())
        
        # Sort by version number (newest first)
        versions.sort(key=lambda v: tuple(map(int, v["version"].split("."))), reverse=True)
        
        return versions
    
    def register_route_handler(
        self,
        version: str,
        route: str,
        handler: Callable
    ):
        """
        Register version-specific route handler.
        
        Args:
            version: API version
            route: Route path
            handler: Handler function
        """
        if version not in self.route_handlers:
            self.route_handlers[version] = {}
        
        self.route_handlers[version][route] = handler
        
        logger.info(f"Registered handler for {route} (v{version})")
    
    def get_route_handler(
        self,
        version: str,
        route: str
    ) -> Optional[Callable]:
        """Get version-specific route handler."""
        if version in self.route_handlers:
            return self.route_handlers[version].get(route)
        
        return None


# Decorator for version-specific endpoints
def versioned(
    manager: ApiVersionManager,
    min_version: Optional[str] = None,
    max_version: Optional[str] = None
):
    """
    Decorator for version-specific endpoints.
    
    Args:
        manager: API version manager
        min_version: Minimum supported version
        max_version: Maximum supported version
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract version from request context
            # This is framework-specific
            version = kwargs.get("version") or manager.default_version
            
            # Check version bounds
            if min_version:
                if tuple(map(int, version.split("."))) < tuple(map(int, min_version.split("."))):
                    raise ValueError(f"Version {version} not supported. Minimum: {min_version}")
            
            if max_version:
                if tuple(map(int, version.split("."))) > tuple(map(int, max_version.split("."))):
                    raise ValueError(f"Version {version} not supported. Maximum: {max_version}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Example usage:
"""
from fastapi import FastAPI, Request, Header
from src.api_version_manager import ApiVersionManager, VersionStatus

app = FastAPI()
version_manager = ApiVersionManager(default_version="1.0")

@app.on_event("startup")
async def startup():
    # Register versions
    version_manager.register_version("1.0", VersionStatus.SUPPORTED)
    version_manager.register_version("1.1", VersionStatus.CURRENT)
    version_manager.register_version("2.0", VersionStatus.CURRENT)
    
    # Deprecate old version
    version_manager.deprecate_version("1.0", sunset_in_days=90)

@app.middleware("http")
async def version_middleware(request: Request, call_next):
    # Detect version
    version = version_manager.get_version(
        accept_header=request.headers.get("accept"),
        path=request.url.path,
        custom_header=request.headers.get("x-api-version")
    )
    
    # Check status
    status = version_manager.check_version_status(version)
    
    # Add headers
    response = await call_next(request)
    response.headers["X-API-Version"] = version
    
    if "warning" in status:
        response.headers["X-API-Warning"] = status["warning"]
    
    return response

@app.get("/api/versions")
async def list_versions():
    return version_manager.list_versions()
"""
