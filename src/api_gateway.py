# API Gateway with Request/Response Middleware

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json

class RequestMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

class RouteMatchType(Enum):
    """Route matching types."""
    EXACT = "exact"
    PREFIX = "prefix"
    REGEX = "regex"

@dataclass
class Request:
    """HTTP request."""
    method: RequestMethod
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None
    query_params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'method': self.method.value,
            'path': self.path,
            'headers': self.headers,
            'body': self.body,
            'query_params': self.query_params,
            'timestamp': self.timestamp,
            'request_id': self.request_id
        }

@dataclass
class Response:
    """HTTP response."""
    status_code: int
    body: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'status_code': self.status_code,
            'body': self.body,
            'headers': self.headers,
            'timestamp': self.timestamp,
            'duration_ms': self.duration_ms
        }

class Middleware(ABC):
    """Base class for middleware."""
    
    @abstractmethod
    def process_request(self, request: Request) -> Optional[Response]:
        """Process request. Return None to continue, Response to short-circuit."""
        pass
    
    @abstractmethod
    def process_response(self, request: Request, response: Response) -> Response:
        """Process response."""
        pass

class AuthenticationMiddleware(Middleware):
    """Authenticate requests."""
    
    def __init__(self, token_validator: Callable):
        self.token_validator = token_validator
    
    def process_request(self, request: Request) -> Optional[Response]:
        """Validate token."""
        token = request.headers.get('Authorization', '')
        
        if not token.startswith('Bearer '):
            return Response(status_code=401, body="Missing authorization token")
        
        token_value = token[7:]
        
        if not self.token_validator(token_value):
            return Response(status_code=401, body="Invalid token")
        
        return None
    
    def process_response(self, request: Request, response: Response) -> Response:
        """Pass through response."""
        return response

class RateLimitMiddleware(Middleware):
    """Rate limiting middleware."""
    
    def __init__(self, requests_per_second: float = 100):
        self.requests_per_second = requests_per_second
        self.request_times: Dict[str, List[float]] = {}
        self.lock = threading.RLock()
    
    def process_request(self, request: Request) -> Optional[Response]:
        """Check rate limit."""
        client_id = request.headers.get('X-Client-ID', 'unknown')
        
        with self.lock:
            if client_id not in self.request_times:
                self.request_times[client_id] = []
            
            # Remove old requests
            cutoff = time.time() - 1
            self.request_times[client_id] = [
                t for t in self.request_times[client_id] if t > cutoff
            ]
            
            # Check limit
            if len(self.request_times[client_id]) >= self.requests_per_second:
                return Response(status_code=429, body="Rate limit exceeded")
            
            self.request_times[client_id].append(time.time())
        
        return None
    
    def process_response(self, request: Request, response: Response) -> Response:
        """Pass through response."""
        return response

class LoggingMiddleware(Middleware):
    """Log requests and responses."""
    
    def __init__(self):
        self.logs: List[Dict] = []
        self.lock = threading.RLock()
    
    def process_request(self, request: Request) -> Optional[Response]:
        """Log request."""
        with self.lock:
            self.logs.append({
                'type': 'request',
                'timestamp': time.time(),
                'method': request.method.value,
                'path': request.path,
                'request_id': request.request_id
            })
        
        return None
    
    def process_response(self, request: Request, response: Response) -> Response:
        """Log response."""
        with self.lock:
            self.logs.append({
                'type': 'response',
                'timestamp': time.time(),
                'status_code': response.status_code,
                'request_id': request.request_id,
                'duration_ms': response.duration_ms
            })
        
        return response

class Route:
    """API route."""
    
    def __init__(self, path: str, method: RequestMethod, 
                handler: Callable, match_type: RouteMatchType = RouteMatchType.EXACT):
        self.path = path
        self.method = method
        self.handler = handler
        self.match_type = match_type
    
    def matches(self, request: Request) -> bool:
        """Check if route matches request."""
        if request.method != self.method:
            return False
        
        if self.match_type == RouteMatchType.EXACT:
            return request.path == self.path
        elif self.match_type == RouteMatchType.PREFIX:
            return request.path.startswith(self.path)
        
        return False

class APIGateway:
    """API Gateway."""
    
    def __init__(self, name: str):
        self.name = name
        self.routes: List[Route] = []
        self.middlewares: List[Middleware] = []
        self.lock = threading.RLock()
        self.request_count = 0
        self.error_count = 0
    
    def register_route(self, path: str, method: RequestMethod, 
                      handler: Callable):
        """Register route."""
        route = Route(path, method, handler)
        with self.lock:
            self.routes.append(route)
    
    def add_middleware(self, middleware: Middleware):
        """Add middleware."""
        with self.lock:
            self.middlewares.append(middleware)
    
    def handle_request(self, request: Request) -> Response:
        """Handle request."""
        request.request_id = f"req_{int(time.time() * 1000)}"
        start_time = time.time()
        
        with self.lock:
            self.request_count += 1
        
        try:
            # Process through middlewares
            for middleware in self.middlewares:
                response = middleware.process_request(request)
                if response:
                    return response
            
            # Find and execute route handler
            with self.lock:
                routes_copy = self.routes.copy()
            
            for route in routes_copy:
                if route.matches(request):
                    response = Response(
                        status_code=200,
                        body=route.handler(request)
                    )
                    break
            else:
                response = Response(
                    status_code=404,
                    body="Route not found"
                )
            
            # Process response through middlewares
            for middleware in self.middlewares:
                response = middleware.process_response(request, response)
            
            response.duration_ms = (time.time() - start_time) * 1000
            
            return response
        
        except Exception as e:
            with self.lock:
                self.error_count += 1
            
            return Response(
                status_code=500,
                body=f"Internal server error: {str(e)}"
            )
    
    def get_stats(self) -> Dict:
        """Get gateway statistics."""
        with self.lock:
            error_rate = (self.error_count / self.request_count * 100) \
                        if self.request_count > 0 else 0
            
            return {
                'name': self.name,
                'total_requests': self.request_count,
                'errors': self.error_count,
                'error_rate': error_rate,
                'routes': len(self.routes),
                'middlewares': len(self.middlewares)
            }

class GatewayRoute:
    """Gateway route builder."""
    
    def __init__(self, gateway: APIGateway):
        self.gateway = gateway
    
    def get(self, path: str, handler: Callable) -> 'GatewayRoute':
        """Register GET route."""
        self.gateway.register_route(path, RequestMethod.GET, handler)
        return self
    
    def post(self, path: str, handler: Callable) -> 'GatewayRoute':
        """Register POST route."""
        self.gateway.register_route(path, RequestMethod.POST, handler)
        return self
    
    def put(self, path: str, handler: Callable) -> 'GatewayRoute':
        """Register PUT route."""
        self.gateway.register_route(path, RequestMethod.PUT, handler)
        return self
    
    def delete(self, path: str, handler: Callable) -> 'GatewayRoute':
        """Register DELETE route."""
        self.gateway.register_route(path, RequestMethod.DELETE, handler)
        return self

# Example usage
if __name__ == "__main__":
    # Create gateway
    gateway = APIGateway("Face Recognition API Gateway")
    
    # Add middlewares
    def validate_token(token):
        return token == "valid_token"
    
    gateway.add_middleware(AuthenticationMiddleware(validate_token))
    gateway.add_middleware(RateLimitMiddleware(requests_per_second=100))
    gateway.add_middleware(LoggingMiddleware())
    
    # Register routes
    def get_faces(request):
        return json.dumps({"faces": []})
    
    def create_face(request):
        return json.dumps({"face_id": "face_123"})
    
    routes = GatewayRoute(gateway)
    routes.get("/api/faces", get_faces)
    routes.post("/api/faces", create_face)
    
    # Handle request
    request = Request(
        method=RequestMethod.GET,
        path="/api/faces",
        headers={"Authorization": "Bearer valid_token"}
    )
    
    response = gateway.handle_request(request)
    print(f"Status: {response.status_code}")
    print(f"Body: {response.body}")
    print(f"Duration: {response.duration_ms:.2f}ms")
    
    # Get stats
    stats = gateway.get_stats()
    print(f"\nGateway Stats:")
    print(json.dumps(stats, indent=2))
