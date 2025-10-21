# API Documentation & OpenAPI/Swagger Integration

import json
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
from enum import Enum
import inspect

class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "get"
    POST = "post"
    PUT = "put"
    DELETE = "delete"
    PATCH = "patch"
    HEAD = "head"
    OPTIONS = "options"

@dataclass
class Parameter:
    """API parameter definition."""
    name: str
    param_type: str
    required: bool = True
    description: str = ""
    example: Any = None
    default: Any = None
    schema: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        param_dict = {
            'name': self.name,
            'in': 'query',  # or 'path', 'header', 'body'
            'required': self.required,
            'description': self.description
        }
        
        if self.schema:
            param_dict['schema'] = self.schema
        else:
            param_dict['type'] = self.param_type
        
        if self.example is not None:
            param_dict['example'] = self.example
        
        if self.default is not None:
            param_dict['default'] = self.default
        
        return param_dict

@dataclass
class Response:
    """API response definition."""
    status_code: int
    description: str
    content_type: str = "application/json"
    schema: Optional[Dict] = None
    example: Any = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        response_dict = {
            'description': self.description,
            'content': {
                self.content_type: {}
            }
        }
        
        if self.schema:
            response_dict['content'][self.content_type]['schema'] = self.schema
        
        if self.example:
            response_dict['content'][self.content_type]['example'] = self.example
        
        return response_dict

@dataclass
class APIEndpoint:
    """API endpoint definition."""
    path: str
    method: HTTPMethod
    summary: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: List[Parameter] = field(default_factory=list)
    request_body: Optional[Dict] = None
    responses: List[Response] = field(default_factory=list)
    deprecated: bool = False
    security: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to OpenAPI operation object."""
        operation = {
            'summary': self.summary,
            'description': self.description,
            'tags': self.tags,
            'deprecated': self.deprecated
        }
        
        # Parameters
        if self.parameters:
            operation['parameters'] = [p.to_dict() for p in self.parameters]
        
        # Request body
        if self.request_body:
            operation['requestBody'] = self.request_body
        
        # Responses
        responses_dict = {}
        if self.responses:
            for response in self.responses:
                responses_dict[str(response.status_code)] = response.to_dict()
        else:
            responses_dict['200'] = {
                'description': 'Successful response'
            }
        
        operation['responses'] = responses_dict
        
        # Security
        if self.security:
            operation['security'] = self.security
        
        return operation

class EndpointRegistry:
    """Registry for API endpoints."""
    
    def __init__(self):
        self.endpoints: Dict[str, Dict[str, APIEndpoint]] = {}
    
    def register(self, endpoint: APIEndpoint):
        """Register endpoint."""
        if endpoint.path not in self.endpoints:
            self.endpoints[endpoint.path] = {}
        
        self.endpoints[endpoint.path][endpoint.method.value] = endpoint
    
    def get_endpoint(self, path: str, method: HTTPMethod) -> Optional[APIEndpoint]:
        """Get endpoint."""
        if path in self.endpoints:
            return self.endpoints[path].get(method.value)
        return None
    
    def get_all_endpoints(self) -> List[APIEndpoint]:
        """Get all endpoints."""
        endpoints = []
        for path_endpoints in self.endpoints.values():
            endpoints.extend(path_endpoints.values())
        return endpoints

class OpenAPISpec:
    """OpenAPI specification generator."""
    
    def __init__(self, title: str, version: str, description: str = ""):
        self.title = title
        self.version = version
        self.description = description
        self.registry = EndpointRegistry()
        self.servers: List[Dict] = []
        self.tags: List[Dict] = []
        self.security_schemes: Dict[str, Dict] = {}
    
    def add_endpoint(self, endpoint: APIEndpoint):
        """Add endpoint."""
        self.registry.register(endpoint)
    
    def add_server(self, url: str, description: str = ""):
        """Add server."""
        self.servers.append({
            'url': url,
            'description': description
        })
    
    def add_tag(self, name: str, description: str = ""):
        """Add tag."""
        self.tags.append({
            'name': name,
            'description': description
        })
    
    def add_security_scheme(self, name: str, scheme_type: str, 
                          **kwargs):
        """Add security scheme."""
        self.security_schemes[name] = {
            'type': scheme_type,
            **kwargs
        }
    
    def to_dict(self) -> Dict:
        """Convert to OpenAPI dictionary."""
        spec = {
            'openapi': '3.0.0',
            'info': {
                'title': self.title,
                'version': self.version,
                'description': self.description
            },
            'paths': self._build_paths(),
            'components': {
                'securitySchemes': self.security_schemes
            }
        }
        
        if self.servers:
            spec['servers'] = self.servers
        
        if self.tags:
            spec['tags'] = self.tags
        
        return spec
    
    def _build_paths(self) -> Dict:
        """Build OpenAPI paths object."""
        paths = {}
        
        for path, methods in self.registry.endpoints.items():
            path_item = {}
            
            for method, endpoint in methods.items():
                path_item[method] = endpoint.to_dict()
            
            paths[path] = path_item
        
        return paths
    
    def to_json(self) -> str:
        """Convert to JSON."""
        return json.dumps(self.to_dict(), indent=2)

class APIDocumentation:
    """API documentation generator."""
    
    def __init__(self, service_name: str, version: str):
        self.service_name = service_name
        self.version = version
        self.endpoints: List[APIEndpoint] = []
        self.models: Dict[str, Dict] = {}
        self.examples: Dict[str, Any] = {}
    
    def document_endpoint(self, path: str, method: HTTPMethod,
                         summary: str, description: str = "",
                         parameters: List[Parameter] = None,
                         request_body: Dict = None,
                         responses: List[Response] = None,
                         tags: List[str] = None) -> APIEndpoint:
        """Document an endpoint."""
        
        endpoint = APIEndpoint(
            path=path,
            method=method,
            summary=summary,
            description=description,
            parameters=parameters or [],
            request_body=request_body,
            responses=responses or [],
            tags=tags or []
        )
        
        self.endpoints.append(endpoint)
        return endpoint
    
    def add_model(self, name: str, schema: Dict):
        """Add data model."""
        self.models[name] = schema
    
    def add_example(self, name: str, example: Any):
        """Add usage example."""
        self.examples[name] = example
    
    def generate_markdown(self) -> str:
        """Generate Markdown documentation."""
        
        md = f"# {self.service_name} API Documentation\n\n"
        md += f"**Version:** {self.version}\n\n"
        
        # Table of contents
        md += "## Table of Contents\n"
        for endpoint in self.endpoints:
            md += f"- [{endpoint.method.value.upper()} {endpoint.path}](#{endpoint.method.value}-{endpoint.path.replace('/', '-')})\n"
        md += "\n"
        
        # Endpoints
        md += "## Endpoints\n\n"
        for endpoint in self.endpoints:
            md += self._endpoint_to_markdown(endpoint)
            md += "\n"
        
        # Models
        if self.models:
            md += "## Data Models\n\n"
            for name, schema in self.models.items():
                md += f"### {name}\n\n"
                md += f"```json\n{json.dumps(schema, indent=2)}\n```\n\n"
        
        # Examples
        if self.examples:
            md += "## Examples\n\n"
            for name, example in self.examples.items():
                md += f"### {name}\n\n"
                md += f"```json\n{json.dumps(example, indent=2)}\n```\n\n"
        
        return md
    
    def _endpoint_to_markdown(self, endpoint: APIEndpoint) -> str:
        """Convert endpoint to Markdown."""
        
        md = f"### {endpoint.method.value.upper()} {endpoint.path}\n\n"
        
        if endpoint.summary:
            md += f"**Summary:** {endpoint.summary}\n\n"
        
        if endpoint.description:
            md += f"**Description:** {endpoint.description}\n\n"
        
        if endpoint.tags:
            md += f"**Tags:** {', '.join(endpoint.tags)}\n\n"
        
        # Parameters
        if endpoint.parameters:
            md += "**Parameters:**\n\n"
            for param in endpoint.parameters:
                required = "required" if param.required else "optional"
                md += f"- `{param.name}` ({param.param_type}, {required}): {param.description}\n"
            md += "\n"
        
        # Request body
        if endpoint.request_body:
            md += "**Request Body:**\n\n"
            md += "```json\n"
            md += json.dumps(endpoint.request_body, indent=2)
            md += "\n```\n\n"
        
        # Responses
        if endpoint.responses:
            md += "**Responses:**\n\n"
            for response in endpoint.responses:
                md += f"- **{response.status_code}:** {response.description}\n"
            md += "\n"
        
        return md

class APIBuilder:
    """Builder for API documentation."""
    
    def __init__(self, service_name: str, version: str):
        self.doc = APIDocumentation(service_name, version)
        self.spec = OpenAPISpec(service_name, version)
    
    def endpoint(self, path: str, method: HTTPMethod, 
                summary: str, description: str = "",
                tags: List[str] = None) -> 'EndpointBuilder':
        """Start building endpoint."""
        return EndpointBuilder(self, path, method, summary, description, tags)
    
    def generate_openapi(self) -> Dict:
        """Generate OpenAPI spec."""
        for endpoint in self.doc.endpoints:
            self.spec.add_endpoint(endpoint)
        
        return self.spec.to_dict()
    
    def generate_markdown(self) -> str:
        """Generate Markdown docs."""
        return self.doc.generate_markdown()

class EndpointBuilder:
    """Builder for endpoints."""
    
    def __init__(self, api_builder: APIBuilder, path: str, method: HTTPMethod,
                summary: str, description: str, tags: List[str]):
        self.api_builder = api_builder
        self.path = path
        self.method = method
        self.summary = summary
        self.description = description
        self.tags = tags or []
        self.parameters: List[Parameter] = []
        self.request_body = None
        self.responses: List[Response] = []
    
    def add_parameter(self, name: str, param_type: str, 
                     required: bool = True, description: str = "") -> 'EndpointBuilder':
        """Add parameter."""
        self.parameters.append(Parameter(name, param_type, required, description))
        return self
    
    def set_request_body(self, request_body: Dict) -> 'EndpointBuilder':
        """Set request body."""
        self.request_body = request_body
        return self
    
    def add_response(self, status_code: int, description: str,
                    schema: Dict = None) -> 'EndpointBuilder':
        """Add response."""
        self.responses.append(Response(status_code, description, schema=schema))
        return self
    
    def build(self) -> APIEndpoint:
        """Build endpoint."""
        endpoint = self.api_builder.doc.document_endpoint(
            path=self.path,
            method=self.method,
            summary=self.summary,
            description=self.description,
            parameters=self.parameters,
            request_body=self.request_body,
            responses=self.responses,
            tags=self.tags
        )
        
        return endpoint

# Example usage
if __name__ == "__main__":
    # Create API builder
    builder = APIBuilder("Face Recognition API", "1.0.0")
    
    # Document endpoints
    (builder
        .endpoint("/api/faces", HTTPMethod.POST, 
                 "Register a face", "Register a new face for recognition",
                 tags=["faces"])
        .add_parameter("image_path", "string", True, "Path to face image")
        .add_response(201, "Face registered successfully")
        .add_response(400, "Invalid image")
        .build())
    
    (builder
        .endpoint("/api/faces/{id}", HTTPMethod.GET,
                 "Get face", "Retrieve face information",
                 tags=["faces"])
        .add_parameter("id", "string", True, "Face ID")
        .add_response(200, "Face found")
        .add_response(404, "Face not found")
        .build())
    
    # Generate documentation
    print("=== OpenAPI Specification ===\n")
    openapi = builder.generate_openapi()
    print(json.dumps(openapi, indent=2))
    
    print("\n=== Markdown Documentation ===\n")
    markdown = builder.generate_markdown()
    print(markdown)
