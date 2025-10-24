# Advanced API Gateway

import threading
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib

class AuthenticationMethod(Enum):
    """Authentication methods."""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    MTLS = "mtls"

@dataclass
class Route:
    """API route."""
    path: str
    method: str
    target_service: str
    version: str = "v1"
    authenticated: bool = True
    rate_limit: Optional[int] = None
    
    def matches(self, incoming_path: str, incoming_method: str) -> bool:
        """Check if route matches request."""
        return self.path == incoming_path and self.method == incoming_method

@dataclass
class APIKey:
    """API key."""
    key: str
    client_id: str
    scopes: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used: float = 0
    active: bool = True

class AuthenticationManager:
    """Manages API authentication."""
    
    def __init__(self):
        self.api_keys: Dict[str, APIKey] = {}
        self.jwt_secret = "secret-key"
        self.lock = threading.RLock()
    
    def register_api_key(self, client_id: str, scopes: List[str] = None) -> str:
        """Register new API key."""
        import uuid
        key = str(uuid.uuid4())
        
        with self.lock:
            self.api_keys[key] = APIKey(
                key=key,
                client_id=client_id,
                scopes=scopes or []
            )
        
        return key
    
    def validate_api_key(self, key: str) -> Tuple[bool, str]:
        """Validate API key."""
        with self.lock:
            if key not in self.api_keys:
                return False, "Invalid API key"
            
            api_key = self.api_keys[key]
            if not api_key.active:
                return False, "API key inactive"
            
            api_key.last_used = time.time()
            return True, api_key.client_id
    
    def revoke_api_key(self, key: str) -> None:
        """Revoke API key."""
        with self.lock:
            if key in self.api_keys:
                self.api_keys[key].active = False
    
    def validate_jwt(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """Validate JWT token."""
        try:
            import jwt
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return True, payload
        except Exception:
            return False, None
    
    def create_jwt(self, client_id: str, claims: Dict = None) -> str:
        """Create JWT token."""
        try:
            import jwt
            payload = {'client_id': client_id, 'exp': time.time() + 3600}
            if claims:
                payload.update(claims)
            return jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        except Exception:
            return ""

class RequestTransformer:
    """Transform API requests."""
    
    def __init__(self):
        self.transformers: Dict[str, Callable] = {}
    
    def register_transformer(self, name: str, transformer: Callable) -> None:
        """Register transformer."""
        self.transformers[name] = transformer
    
    def transform_request(self, request: Dict, transformers: List[str]) -> Dict:
        """Apply transformers to request."""
        for transformer_name in transformers:
            if transformer_name in self.transformers:
                request = self.transformers[transformer_name](request)
        
        return request
    
    def add_headers(self, request: Dict, headers: Dict) -> Dict:
        """Add headers to request."""
        if 'headers' not in request:
            request['headers'] = {}
        request['headers'].update(headers)
        return request
    
    def add_auth_header(self, request: Dict, token: str) -> Dict:
        """Add authorization header."""
        return self.add_headers(request, {'Authorization': f'Bearer {token}'})
    
    def add_version_header(self, request: Dict, version: str) -> Dict:
        """Add API version header."""
        return self.add_headers(request, {'X-API-Version': version})

class ResponseTransformer:
    """Transform API responses."""
    
    def __init__(self):
        self.transformers: Dict[str, Callable] = {}
    
    def register_transformer(self, name: str, transformer: Callable) -> None:
        """Register transformer."""
        self.transformers[name] = transformer
    
    def transform_response(self, response: Dict, transformers: List[str]) -> Dict:
        """Apply transformers to response."""
        for transformer_name in transformers:
            if transformer_name in self.transformers:
                response = self.transformers[transformer_name](response)
        
        return response
    
    def wrap_response(self, data: Any, status: str = "success") -> Dict:
        """Wrap response data."""
        return {
            'status': status,
            'data': data,
            'timestamp': time.time()
        }
    
    def add_cache_headers(self, response: Dict, ttl: int = 300) -> Dict:
        """Add cache headers."""
        if 'headers' not in response:
            response['headers'] = {}
        response['headers']['Cache-Control'] = f'max-age={ttl}'
        return response

class RateLimitBucket:
    """Rate limit bucket."""
    
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self.requests: List[float] = []
        self.lock = threading.RLock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed."""
        now = time.time()
        
        with self.lock:
            # Remove old requests outside window
            self.requests = [r for r in self.requests if now - r < self.window]
            
            if len(self.requests) < self.limit:
                self.requests.append(now)
                return True
            
            return False

class AdvancedAPIGateway:
    """Advanced API Gateway."""
    
    def __init__(self):
        self.routes: Dict[str, Route] = {}
        self.auth_manager = AuthenticationManager()
        self.request_transformer = RequestTransformer()
        self.response_transformer = ResponseTransformer()
        self.rate_limiters: Dict[str, RateLimitBucket] = {}
        self.middlewares: List[Callable] = []
        self.lock = threading.RLock()
    
    def register_route(self, route: Route) -> None:
        """Register API route."""
        with self.lock:
            key = f"{route.method}:{route.path}"
            self.routes[key] = route
            
            if route.rate_limit:
                self.rate_limiters[key] = RateLimitBucket(route.rate_limit, 60)
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add gateway middleware."""
        self.middlewares.append(middleware)
    
    def handle_request(self, path: str, method: str, headers: Dict = None,
                      body: Dict = None, client_id: str = None) -> Dict:
        """Handle incoming request."""
        headers = headers or {}
        body = body or {}
        
        # Find matching route
        route_key = f"{method}:{path}"
        if route_key not in self.routes:
            return {'error': 'Route not found', 'status': 404}
        
        route = self.routes[route_key]
        
        # Apply middlewares
        for middleware in self.middlewares:
            result = middleware({'path': path, 'method': method, 'headers': headers})
            if result and result.get('error'):
                return result
        
        # Check authentication
        if route.authenticated:
            auth_header = headers.get('Authorization', '')
            if not auth_header:
                return {'error': 'Missing authorization', 'status': 401}
            
            token = auth_header.replace('Bearer ', '')
            valid, client = self.auth_manager.validate_api_key(token)
            if not valid:
                return {'error': 'Invalid token', 'status': 401}
            
            client_id = client
        
        # Check rate limit
        if route_key in self.rate_limiters:
            if not self.rate_limiters[route_key].is_allowed():
                return {'error': 'Rate limit exceeded', 'status': 429}
        
        # Transform request
        request = {'path': path, 'method': method, 'headers': headers, 'body': body}
        request = self.request_transformer.transform_request(request, ['add_version'])
        
        # Route to service (simulated)
        response = {
            'status': 200,
            'data': {'service': route.target_service, 'version': route.version}
        }
        
        # Transform response
        response = self.response_transformer.transform_response(response, [])
        response = self.response_transformer.add_cache_headers(response)
        
        return response
    
    def get_gateway_stats(self) -> Dict:
        """Get gateway statistics."""
        with self.lock:
            return {
                'routes': len(self.routes),
                'rate_limiters': len(self.rate_limiters),
                'middlewares': len(self.middlewares)
            }

class APIVersionManager:
    """Manage API versions."""
    
    def __init__(self):
        self.versions: Dict[str, Dict] = {}
        self.lock = threading.RLock()
    
    def register_version(self, version: str, description: str,
                        deprecated: bool = False) -> None:
        """Register API version."""
        with self.lock:
            self.versions[version] = {
                'description': description,
                'deprecated': deprecated,
                'created_at': time.time()
            }
    
    def get_version_info(self, version: str) -> Optional[Dict]:
        """Get version information."""
        with self.lock:
            return self.versions.get(version)
    
    def deprecate_version(self, version: str, sunset_date: float) -> None:
        """Deprecate API version."""
        with self.lock:
            if version in self.versions:
                self.versions[version]['deprecated'] = True
                self.versions[version]['sunset_date'] = sunset_date

class APIAggregator:
    """Aggregate multiple API calls."""
    
    def __init__(self, gateway: AdvancedAPIGateway):
        self.gateway = gateway
    
    def aggregate_requests(self, requests: List[Tuple[str, str, Dict]]) -> Dict:
        """Aggregate multiple requests."""
        results = {}
        
        for path, method, headers in requests:
            response = self.gateway.handle_request(path, method, headers)
            results[f"{method}:{path}"] = response
        
        return {'aggregated': results, 'count': len(results)}

# Example usage
if __name__ == "__main__":
    gateway = AdvancedAPIGateway()
    
    # Register routes
    routes = [
        Route(path="/api/faces", method="GET", target_service="face-service"),
        Route(path="/api/faces", method="POST", target_service="face-service"),
        Route(path="/api/recognition", method="POST", target_service="recognition-service"),
    ]
    
    for route in routes:
        gateway.register_route(route)
    
    # Register API key
    key = gateway.auth_manager.register_api_key("client-1", ["faces:read", "faces:write"])
    print(f"API Key: {key}")
    
    # Handle request
    response = gateway.handle_request(
        path="/api/faces",
        method="GET",
        headers={'Authorization': f'Bearer {key}'}
    )
    print(f"\nResponse: {json.dumps(response, indent=2)}")
    
    # Get gateway stats
    stats = gateway.get_gateway_stats()
    print(f"\nGateway Stats: {json.dumps(stats, indent=2)}")
