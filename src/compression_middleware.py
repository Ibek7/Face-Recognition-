"""
Response compression middleware for FastAPI.

Compresses HTTP responses to reduce bandwidth and improve performance.
"""

import gzip
import brotli
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message
import logging

logger = logging.getLogger(__name__)


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware for response compression."""
    
    def __init__(
        self,
        app,
        minimum_size: int = 500,
        compression_level: int = 6,
        exclude_paths: list = None,
        exclude_media_types: list = None,
    ):
        """
        Initialize compression middleware.
        
        Args:
            app: FastAPI application
            minimum_size: Minimum response size for compression (bytes)
            compression_level: Compression level (1-9)
            exclude_paths: Paths to exclude from compression
            exclude_media_types: Media types to exclude
        """
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
        self.exclude_paths = exclude_paths or []
        self.exclude_media_types = exclude_media_types or [
            "image/",
            "video/",
            "audio/",
            "application/zip",
            "application/gzip",
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and compress response.
        
        Args:
            request: Incoming request
            call_next: Next handler
            
        Returns:
            Possibly compressed response
        """
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Get response
        response = await call_next(request)
        
        # Check if compression should be applied
        if not self._should_compress(request, response):
            return response
        
        # Determine compression algorithm
        accept_encoding = request.headers.get("accept-encoding", "")
        
        if "br" in accept_encoding:
            # Brotli compression
            return await self._compress_brotli(response)
        elif "gzip" in accept_encoding:
            # Gzip compression
            return await self._compress_gzip(response)
        
        return response
    
    def _should_compress(self, request: Request, response: Response) -> bool:
        """
        Determine if response should be compressed.
        
        Args:
            request: Request object
            response: Response object
            
        Returns:
            True if should compress
        """
        # Check if already compressed
        if "content-encoding" in response.headers:
            return False
        
        # Check media type
        content_type = response.headers.get("content-type", "")
        for excluded in self.exclude_media_types:
            if content_type.startswith(excluded):
                return False
        
        # Check size
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) < self.minimum_size:
            return False
        
        return True
    
    async def _compress_gzip(self, response: Response) -> Response:
        """
        Compress response with gzip.
        
        Args:
            response: Original response
            
        Returns:
            Compressed response
        """
        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Check minimum size
        if len(body) < self.minimum_size:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
        
        # Compress
        compressed = gzip.compress(body, compresslevel=self.compression_level)
        
        # Create new response
        headers = dict(response.headers)
        headers["content-encoding"] = "gzip"
        headers["content-length"] = str(len(compressed))
        
        logger.debug(
            f"Compressed response: {len(body)} -> {len(compressed)} bytes "
            f"({(1 - len(compressed)/len(body))*100:.1f}% reduction)"
        )
        
        return Response(
            content=compressed,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )
    
    async def _compress_brotli(self, response: Response) -> Response:
        """
        Compress response with Brotli.
        
        Args:
            response: Original response
            
        Returns:
            Compressed response
        """
        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        
        # Check minimum size
        if len(body) < self.minimum_size:
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
        
        # Compress with Brotli
        compressed = brotli.compress(body, quality=self.compression_level)
        
        # Create new response
        headers = dict(response.headers)
        headers["content-encoding"] = "br"
        headers["content-length"] = str(len(compressed))
        
        logger.debug(
            f"Brotli compressed: {len(body)} -> {len(compressed)} bytes "
            f"({(1 - len(compressed)/len(body))*100:.1f}% reduction)"
        )
        
        return Response(
            content=compressed,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )


# Example usage:
"""
from fastapi import FastAPI
from src.compression_middleware import CompressionMiddleware

app = FastAPI()

app.add_middleware(
    CompressionMiddleware,
    minimum_size=500,
    compression_level=6,
    exclude_paths=["/metrics", "/health"]
)
"""
