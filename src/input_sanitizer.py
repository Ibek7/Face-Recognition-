"""
Input sanitization utilities for security.

Prevents XSS, SQL injection, and other input-based attacks.
"""

import re
import html
from typing import Any, Dict, List, Optional, Union
import bleach
import logging

logger = logging.getLogger(__name__)


class InputSanitizer:
    """Sanitize user input for security."""
    
    # Dangerous patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|\#|\/\*|\*\/)",
        r"(\bOR\b.*=.*)",
        r"(\bAND\b.*=.*)",
        r"(;.*\b(SELECT|INSERT|UPDATE|DELETE)\b)"
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
        r"<object",
        r"<embed"
    ]
    
    # Allowed HTML tags for rich text
    ALLOWED_TAGS = [
        'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'blockquote', 'code', 'pre', 'ul', 'ol', 'li', 'a'
    ]
    
    ALLOWED_ATTRIBUTES = {
        'a': ['href', 'title'],
        'img': ['src', 'alt']
    }
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize input sanitizer.
        
        Args:
            strict_mode: Enable strict sanitization
        """
        self.strict_mode = strict_mode
    
    def sanitize_string(
        self,
        text: str,
        max_length: Optional[int] = None,
        allow_html: bool = False
    ) -> str:
        """
        Sanitize string input.
        
        Args:
            text: Input text
            max_length: Maximum length
            allow_html: Allow safe HTML tags
        
        Returns:
            Sanitized string
        """
        if not isinstance(text, str):
            return ""
        
        # Trim whitespace
        text = text.strip()
        
        # Enforce length limit
        if max_length:
            text = text[:max_length]
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        if allow_html:
            # Allow safe HTML tags
            text = self._sanitize_html(text)
        else:
            # Escape all HTML
            text = html.escape(text)
        
        # Check for dangerous patterns
        if self.strict_mode:
            if self._contains_sql_injection(text):
                logger.warning(f"Blocked SQL injection attempt: {text[:50]}...")
                raise ValueError("Invalid input: potential SQL injection")
            
            if self._contains_xss(text):
                logger.warning(f"Blocked XSS attempt: {text[:50]}...")
                raise ValueError("Invalid input: potential XSS attack")
        
        return text
    
    def sanitize_email(self, email: str) -> str:
        """
        Sanitize email address.
        
        Args:
            email: Email address
        
        Returns:
            Sanitized email
        
        Raises:
            ValueError: If email is invalid
        """
        email = self.sanitize_string(email, max_length=254)
        
        # Basic email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if not re.match(email_pattern, email):
            raise ValueError("Invalid email format")
        
        return email.lower()
    
    def sanitize_url(
        self,
        url: str,
        allowed_schemes: Optional[List[str]] = None
    ) -> str:
        """
        Sanitize URL.
        
        Args:
            url: URL to sanitize
            allowed_schemes: Allowed URL schemes
        
        Returns:
            Sanitized URL
        
        Raises:
            ValueError: If URL is invalid
        """
        url = self.sanitize_string(url, max_length=2048)
        
        # Default allowed schemes
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
        
        # Extract scheme
        scheme_match = re.match(r'^([a-zA-Z][a-zA-Z0-9+.-]*?):', url)
        
        if scheme_match:
            scheme = scheme_match.group(1).lower()
            
            if scheme not in allowed_schemes:
                raise ValueError(f"URL scheme not allowed: {scheme}")
        else:
            raise ValueError("Invalid URL: missing scheme")
        
        # Block javascript: and data: URLs
        if url.lower().startswith(('javascript:', 'data:', 'vbscript:')):
            raise ValueError("Dangerous URL scheme")
        
        return url
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename.
        
        Args:
            filename: Filename to sanitize
        
        Returns:
            Safe filename
        """
        # Remove path separators
        filename = filename.replace('/', '').replace('\\', '')
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*\x00-\x1f]', '', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:255 - len(ext) - 1] + '.' + ext if ext else name[:255]
        
        # Ensure not empty
        if not filename:
            filename = "unnamed"
        
        return filename
    
    def sanitize_dict(
        self,
        data: Dict[str, Any],
        allowed_keys: Optional[List[str]] = None,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """
        Sanitize dictionary recursively.
        
        Args:
            data: Dictionary to sanitize
            allowed_keys: Allowed keys (None = all)
            max_depth: Maximum recursion depth
        
        Returns:
            Sanitized dictionary
        """
        if max_depth <= 0:
            raise ValueError("Maximum recursion depth exceeded")
        
        sanitized = {}
        
        for key, value in data.items():
            # Filter keys
            if allowed_keys and key not in allowed_keys:
                continue
            
            # Sanitize key
            safe_key = self.sanitize_string(str(key), max_length=100)
            
            # Sanitize value based on type
            if isinstance(value, str):
                safe_value = self.sanitize_string(value)
            elif isinstance(value, dict):
                safe_value = self.sanitize_dict(value, max_depth=max_depth - 1)
            elif isinstance(value, list):
                safe_value = self.sanitize_list(value, max_depth=max_depth - 1)
            elif isinstance(value, (int, float, bool)):
                safe_value = value
            elif value is None:
                safe_value = None
            else:
                # Convert unknown types to string
                safe_value = self.sanitize_string(str(value))
            
            sanitized[safe_key] = safe_value
        
        return sanitized
    
    def sanitize_list(
        self,
        data: List[Any],
        max_depth: int = 10
    ) -> List[Any]:
        """
        Sanitize list recursively.
        
        Args:
            data: List to sanitize
            max_depth: Maximum recursion depth
        
        Returns:
            Sanitized list
        """
        if max_depth <= 0:
            raise ValueError("Maximum recursion depth exceeded")
        
        sanitized = []
        
        for item in data:
            if isinstance(item, str):
                safe_item = self.sanitize_string(item)
            elif isinstance(item, dict):
                safe_item = self.sanitize_dict(item, max_depth=max_depth - 1)
            elif isinstance(item, list):
                safe_item = self.sanitize_list(item, max_depth=max_depth - 1)
            elif isinstance(item, (int, float, bool)):
                safe_item = item
            elif item is None:
                safe_item = None
            else:
                safe_item = self.sanitize_string(str(item))
            
            sanitized.append(safe_item)
        
        return sanitized
    
    def _sanitize_html(self, text: str) -> str:
        """Sanitize HTML using bleach."""
        return bleach.clean(
            text,
            tags=self.ALLOWED_TAGS,
            attributes=self.ALLOWED_ATTRIBUTES,
            strip=True
        )
    
    def _contains_sql_injection(self, text: str) -> bool:
        """Check for SQL injection patterns."""
        text_upper = text.upper()
        
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_upper, re.IGNORECASE):
                return True
        
        return False
    
    def _contains_xss(self, text: str) -> bool:
        """Check for XSS patterns."""
        for pattern in self.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False


# Global sanitizer
sanitizer = InputSanitizer(strict_mode=True)


# Convenience functions
def sanitize(text: str, **kwargs) -> str:
    """Sanitize string input."""
    return sanitizer.sanitize_string(text, **kwargs)


def sanitize_email(email: str) -> str:
    """Sanitize email address."""
    return sanitizer.sanitize_email(email)


def sanitize_url(url: str, **kwargs) -> str:
    """Sanitize URL."""
    return sanitizer.sanitize_url(url, **kwargs)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename."""
    return sanitizer.sanitize_filename(filename)


# Example usage:
"""
from fastapi import FastAPI, HTTPException
from src.input_sanitizer import sanitize, sanitize_email, sanitize_url

app = FastAPI()

@app.post("/api/users")
async def create_user(username: str, email: str, bio: str):
    try:
        # Sanitize inputs
        safe_username = sanitize(username, max_length=50)
        safe_email = sanitize_email(email)
        safe_bio = sanitize(bio, max_length=500, allow_html=True)
        
        # Create user...
        return {
            "username": safe_username,
            "email": safe_email,
            "bio": safe_bio
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/upload")
async def upload_file(filename: str):
    safe_filename = sanitize_filename(filename)
    return {"filename": safe_filename}
"""
