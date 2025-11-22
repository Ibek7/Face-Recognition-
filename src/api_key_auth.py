"""
API Key authentication middleware for FastAPI.

Provides header-based authentication using API keys.
"""

from typing import Optional, Dict, Set
from datetime import datetime, timedelta
from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader
import hashlib
import secrets
import logging

logger = logging.getLogger(__name__)


# API Key storage (use database in production)
API_KEYS: Dict[str, dict] = {}


class APIKeyAuth:
    """API Key authentication manager."""
    
    def __init__(
        self,
        header_name: str = "X-API-Key",
        auto_error: bool = True
    ):
        """
        Initialize API key authentication.
        
        Args:
            header_name: HTTP header name for API key
            auto_error: Raise error if key is invalid
        """
        self.header_name = header_name
        self.auto_error = auto_error
        self.api_key_header = APIKeyHeader(
            name=header_name,
            auto_error=auto_error
        )
    
    def generate_key(
        self,
        user_id: str,
        permissions: Optional[Set[str]] = None,
        expires_days: Optional[int] = None
    ) -> str:
        """
        Generate new API key.
        
        Args:
            user_id: User identifier
            permissions: Set of permissions
            expires_days: Days until expiration
        
        Returns:
            Generated API key
        """
        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)
        
        # Hash for storage
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Calculate expiration
        expiration = None
        if expires_days:
            expiration = datetime.utcnow() + timedelta(days=expires_days)
        
        # Store key metadata
        API_KEYS[key_hash] = {
            "user_id": user_id,
            "permissions": permissions or set(),
            "created_at": datetime.utcnow(),
            "expires_at": expiration,
            "last_used": None,
            "usage_count": 0
        }
        
        logger.info(f"Generated API key for user: {user_id}")
        return raw_key
    
    def revoke_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
        
        Returns:
            True if revoked
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash in API_KEYS:
            del API_KEYS[key_hash]
            logger.info(f"Revoked API key: {key_hash[:8]}...")
            return True
        
        return False
    
    async def validate(
        self,
        request: Request,
        required_permissions: Optional[Set[str]] = None
    ) -> dict:
        """
        Validate API key from request.
        
        Args:
            request: FastAPI request
            required_permissions: Required permissions
        
        Returns:
            Key metadata
        
        Raises:
            HTTPException: If key is invalid
        """
        # Extract API key from header
        api_key = request.headers.get(self.header_name)
        
        if not api_key:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required"
                )
            return None
        
        # Hash and lookup
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_data = API_KEYS.get(key_hash)
        
        if not key_data:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )
            return None
        
        # Check expiration
        if key_data["expires_at"]:
            if datetime.utcnow() > key_data["expires_at"]:
                if self.auto_error:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="API key expired"
                    )
                return None
        
        # Check permissions
        if required_permissions:
            user_permissions = key_data["permissions"]
            
            if not required_permissions.issubset(user_permissions):
                if self.auto_error:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions"
                    )
                return None
        
        # Update usage statistics
        key_data["last_used"] = datetime.utcnow()
        key_data["usage_count"] += 1
        
        return key_data
    
    def get_key_info(self, api_key: str) -> Optional[dict]:
        """
        Get API key information.
        
        Args:
            api_key: API key
        
        Returns:
            Key metadata or None
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return API_KEYS.get(key_hash)


# Global instance
api_key_auth = APIKeyAuth()


# Example usage in api_server.py:
"""
from fastapi import Depends, FastAPI
from src.api_key_auth import api_key_auth

app = FastAPI()

# Require API key for all routes
@app.middleware("http")
async def authenticate(request: Request, call_next):
    # Skip auth for health check
    if request.url.path == "/health":
        return await call_next(request)
    
    # Validate API key
    await api_key_auth.validate(request)
    
    response = await call_next(request)
    return response

# Or per-route authentication
@app.get("/protected")
async def protected_route(
    key_data: dict = Depends(api_key_auth.validate)
):
    return {"user_id": key_data["user_id"]}

# Generate key for new user
api_key = api_key_auth.generate_key(
    user_id="user123",
    permissions={"read", "write"},
    expires_days=30
)
print(f"Your API key: {api_key}")

# Revoke key
api_key_auth.revoke_key(api_key)
"""
