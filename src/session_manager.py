"""
Session management with Redis backend.

Provides session storage, expiration, and user session tracking.
"""

import json
import hashlib
import secrets
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
import redis.asyncio as aioredis
import logging

logger = logging.getLogger(__name__)


class SessionConfig:
    """Session configuration."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        session_timeout: int = 3600,
        cookie_name: str = "session_id",
        cookie_secure: bool = True,
        cookie_httponly: bool = True,
        cookie_samesite: str = "lax"
    ):
        """
        Initialize session config.
        
        Args:
            redis_url: Redis connection URL
            session_timeout: Session timeout in seconds
            cookie_name: Session cookie name
            cookie_secure: Use secure cookie
            cookie_httponly: Use httponly cookie
            cookie_samesite: Cookie samesite policy
        """
        self.redis_url = redis_url
        self.session_timeout = session_timeout
        self.cookie_name = cookie_name
        self.cookie_secure = cookie_secure
        self.cookie_httponly = cookie_httponly
        self.cookie_samesite = cookie_samesite


class Session:
    """User session."""
    
    def __init__(self, session_id: str, data: Dict[str, Any]):
        """
        Initialize session.
        
        Args:
            session_id: Unique session ID
            data: Session data
        """
        self.session_id = session_id
        self.data = data
        self.modified = False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get session value."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set session value."""
        self.data[key] = value
        self.modified = True
    
    def delete(self, key: str):
        """Delete session key."""
        if key in self.data:
            del self.data[key]
            self.modified = True
    
    def clear(self):
        """Clear all session data."""
        self.data.clear()
        self.modified = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.data.copy()


class SessionManager:
    """Redis-backed session manager."""
    
    def __init__(self, config: SessionConfig):
        """
        Initialize session manager.
        
        Args:
            config: Session configuration
        """
        self.config = config
        self.redis: Optional[aioredis.Redis] = None
        self.session_prefix = "session:"
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis = await aioredis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await self.redis.ping()
            
            logger.info("Connected to Redis for session management")
        
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis")
    
    def generate_session_id(self) -> str:
        """
        Generate secure session ID.
        
        Returns:
            Session ID
        """
        random_bytes = secrets.token_bytes(32)
        timestamp = str(datetime.utcnow().timestamp()).encode()
        
        return hashlib.sha256(random_bytes + timestamp).hexdigest()
    
    async def create_session(
        self,
        data: Optional[Dict[str, Any]] = None
    ) -> Session:
        """
        Create new session.
        
        Args:
            data: Initial session data
        
        Returns:
            Session object
        """
        session_id = self.generate_session_id()
        session_data = data or {}
        
        # Add metadata
        session_data["_created_at"] = datetime.utcnow().isoformat()
        session_data["_last_accessed"] = datetime.utcnow().isoformat()
        
        # Store in Redis
        await self._save_session(session_id, session_data)
        
        logger.info(f"Created session: {session_id[:8]}...")
        
        return Session(session_id, session_data)
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID.
        
        Args:
            session_id: Session ID
        
        Returns:
            Session object or None
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        key = self._get_key(session_id)
        
        # Get from Redis
        data_json = await self.redis.get(key)
        
        if not data_json:
            return None
        
        # Parse data
        data = json.loads(data_json)
        
        # Update last accessed
        data["_last_accessed"] = datetime.utcnow().isoformat()
        await self._save_session(session_id, data)
        
        return Session(session_id, data)
    
    async def save_session(self, session: Session):
        """
        Save session to Redis.
        
        Args:
            session: Session to save
        """
        if session.modified:
            await self._save_session(session.session_id, session.data)
            session.modified = False
    
    async def _save_session(self, session_id: str, data: Dict[str, Any]):
        """Save session data to Redis."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        key = self._get_key(session_id)
        data_json = json.dumps(data)
        
        # Set with expiration
        await self.redis.setex(
            key,
            self.config.session_timeout,
            data_json
        )
    
    async def delete_session(self, session_id: str):
        """
        Delete session.
        
        Args:
            session_id: Session ID
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        key = self._get_key(session_id)
        await self.redis.delete(key)
        
        logger.info(f"Deleted session: {session_id[:8]}...")
    
    async def refresh_session(self, session_id: str):
        """
        Refresh session timeout.
        
        Args:
            session_id: Session ID
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        key = self._get_key(session_id)
        await self.redis.expire(key, self.config.session_timeout)
    
    async def get_all_sessions(self, user_id: Optional[str] = None) -> list:
        """
        Get all sessions, optionally filtered by user.
        
        Args:
            user_id: Optional user ID filter
        
        Returns:
            List of session IDs
        """
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        pattern = f"{self.session_prefix}*"
        session_ids = []
        
        async for key in self.redis.scan_iter(match=pattern):
            session_id = key.replace(self.session_prefix, "")
            
            if user_id:
                # Filter by user_id
                session = await self.get_session(session_id)
                if session and session.get("user_id") == user_id:
                    session_ids.append(session_id)
            else:
                session_ids.append(session_id)
        
        return session_ids
    
    async def delete_user_sessions(self, user_id: str):
        """
        Delete all sessions for a user.
        
        Args:
            user_id: User ID
        """
        sessions = await self.get_all_sessions(user_id=user_id)
        
        for session_id in sessions:
            await self.delete_session(session_id)
        
        logger.info(f"Deleted {len(sessions)} sessions for user: {user_id}")
    
    def _get_key(self, session_id: str) -> str:
        """Get Redis key for session."""
        return f"{self.session_prefix}{session_id}"
    
    async def get_stats(self) -> dict:
        """Get session statistics."""
        if not self.redis:
            raise RuntimeError("Redis not connected")
        
        pattern = f"{self.session_prefix}*"
        total_sessions = 0
        
        async for _ in self.redis.scan_iter(match=pattern):
            total_sessions += 1
        
        return {
            "total_sessions": total_sessions,
            "session_timeout": self.config.session_timeout
        }


# Global session manager
session_config = SessionConfig()
session_manager = SessionManager(session_config)


# Example usage:
"""
from fastapi import FastAPI, Request, Response, Cookie
from src.session_manager import session_manager, SessionConfig

app = FastAPI()

@app.on_event("startup")
async def startup():
    # Configure session
    config = SessionConfig(
        redis_url="redis://localhost:6379",
        session_timeout=3600,
        cookie_name="session_id"
    )
    
    global session_manager
    session_manager = SessionManager(config)
    
    # Connect to Redis
    await session_manager.connect()

@app.on_event("shutdown")
async def shutdown():
    await session_manager.disconnect()

@app.post("/login")
async def login(response: Response, username: str, password: str):
    # Authenticate user...
    
    # Create session
    session = await session_manager.create_session({
        "user_id": "123",
        "username": username
    })
    
    # Set session cookie
    response.set_cookie(
        key="session_id",
        value=session.session_id,
        httponly=True,
        secure=True
    )
    
    return {"message": "Logged in"}

@app.get("/profile")
async def get_profile(session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Get session
    session = await session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    user_id = session.get("user_id")
    username = session.get("username")
    
    return {
        "user_id": user_id,
        "username": username
    }

@app.post("/logout")
async def logout(response: Response, session_id: str = Cookie(None)):
    if session_id:
        await session_manager.delete_session(session_id)
    
    response.delete_cookie("session_id")
    
    return {"message": "Logged out"}

@app.get("/sessions/stats")
async def session_stats():
    return await session_manager.get_stats()
"""
