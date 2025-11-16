#!/usr/bin/env python3
"""
API Documentation Generator

Automatically generates comprehensive API documentation from FastAPI applications
with support for OpenAPI/Swagger, Markdown, and ReDoc formats.

Features:
- OpenAPI specification generation
- Interactive Swagger UI
- ReDoc documentation
- Markdown export
- Endpoint discovery and analysis
- Schema extraction
- Example generation
- Versioning support

Usage:
    python scripts/generate_docs.py --app src.api_server:app --output docs/api
    python scripts/generate_docs.py --format markdown --output API.md
    python scripts/generate_docs.py --serve --port 8080
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

try:
    from fastapi import FastAPI
    from fastapi.openapi.utils import get_openapi
    import yaml
except ImportError:
    print("Error: FastAPI and PyYAML required. Install with: pip install fastapi pyyaml")
    sys.exit(1)


class DocumentationGenerator:
    """Generate comprehensive API documentation."""
    
    def __init__(self, app: FastAPI, output_dir: Path):
        """
        Initialize documentation generator.
        
        Args:
            app: FastAPI application instance
            output_dir: Directory for generated documentation
        """
        self.app = app
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_openapi(self) -> Dict[str, Any]:
        """
        Generate OpenAPI specification.
        
        Returns:
            OpenAPI specification dictionary
        """
        return get_openapi(
            title=self.app.title,
            version=self.app.version,
            openapi_version=self.app.openapi_version,
            description=self.app.description,
            routes=self.app.routes,
        )
    
    def save_openapi_json(self, filename: str = "openapi.json") -> Path:
        """
        Save OpenAPI spec as JSON.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        openapi_spec = self.generate_openapi()
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(openapi_spec, f, indent=2)
        
        print(f"✓ Generated OpenAPI JSON: {output_path}")
        return output_path
    
    def save_openapi_yaml(self, filename: str = "openapi.yaml") -> Path:
        """
        Save OpenAPI spec as YAML.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        openapi_spec = self.generate_openapi()
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            yaml.dump(openapi_spec, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Generated OpenAPI YAML: {output_path}")
        return output_path
    
    def generate_markdown(self) -> str:
        """
        Generate Markdown documentation.
        
        Returns:
            Markdown content
        """
        openapi_spec = self.generate_openapi()
        
        # Build markdown content
        lines = []
        lines.append(f"# {openapi_spec['info']['title']}")
        lines.append(f"\nVersion: {openapi_spec['info']['version']}")
        lines.append(f"\n{openapi_spec['info'].get('description', '')}\n")
        
        # Add table of contents
        lines.append("## Table of Contents\n")
        for path, methods in openapi_spec['paths'].items():
            for method in methods.keys():
                if method not in ['parameters', 'servers']:
                    endpoint_name = methods[method].get('summary', path)
                    anchor = self._create_anchor(f"{method.upper()} {path}")
                    lines.append(f"- [{method.upper()} {path}](#{anchor})")
        
        lines.append("\n## Endpoints\n")
        
        # Document each endpoint
        for path, methods in openapi_spec['paths'].items():
            for method, details in methods.items():
                if method in ['parameters', 'servers']:
                    continue
                
                lines.append(f"### {method.upper()} {path}\n")
                
                # Summary and description
                if 'summary' in details:
                    lines.append(f"**{details['summary']}**\n")
                if 'description' in details:
                    lines.append(f"{details['description']}\n")
                
                # Tags
                if 'tags' in details:
                    lines.append(f"**Tags:** {', '.join(details['tags'])}\n")
                
                # Parameters
                if 'parameters' in details:
                    lines.append("#### Parameters\n")
                    lines.append("| Name | In | Type | Required | Description |")
                    lines.append("|------|-----|------|----------|-------------|")
                    for param in details['parameters']:
                        name = param['name']
                        location = param['in']
                        schema = param.get('schema', {})
                        param_type = schema.get('type', 'any')
                        required = '✓' if param.get('required', False) else ''
                        description = param.get('description', '')
                        lines.append(f"| {name} | {location} | {param_type} | {required} | {description} |")
                    lines.append("")
                
                # Request body
                if 'requestBody' in details:
                    lines.append("#### Request Body\n")
                    content = details['requestBody'].get('content', {})
                    for content_type, schema_info in content.items():
                        lines.append(f"**Content-Type:** `{content_type}`\n")
                        if 'schema' in schema_info:
                            lines.append("```json")
                            lines.append(json.dumps(
                                self._generate_example(schema_info['schema'], openapi_spec),
                                indent=2
                            ))
                            lines.append("```\n")
                
                # Responses
                if 'responses' in details:
                    lines.append("#### Responses\n")
                    for status_code, response in details['responses'].items():
                        description = response.get('description', '')
                        lines.append(f"**{status_code}** - {description}\n")
                        
                        if 'content' in response:
                            for content_type, schema_info in response['content'].items():
                                lines.append(f"Content-Type: `{content_type}`\n")
                                if 'schema' in schema_info:
                                    lines.append("```json")
                                    lines.append(json.dumps(
                                        self._generate_example(schema_info['schema'], openapi_spec),
                                        indent=2
                                    ))
                                    lines.append("```\n")
                
                lines.append("---\n")
        
        # Add schemas section
        if 'components' in openapi_spec and 'schemas' in openapi_spec['components']:
            lines.append("## Schemas\n")
            for schema_name, schema_def in openapi_spec['components']['schemas'].items():
                lines.append(f"### {schema_name}\n")
                if 'description' in schema_def:
                    lines.append(f"{schema_def['description']}\n")
                
                lines.append("```json")
                lines.append(json.dumps(
                    self._generate_example({'$ref': f'#/components/schemas/{schema_name}'}, openapi_spec),
                    indent=2
                ))
                lines.append("```\n")
        
        return '\n'.join(lines)
    
    def save_markdown(self, filename: str = "API.md") -> Path:
        """
        Save Markdown documentation.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        markdown_content = self.generate_markdown()
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write(markdown_content)
        
        print(f"✓ Generated Markdown documentation: {output_path}")
        return output_path
    
    def generate_postman_collection(self) -> Dict[str, Any]:
        """
        Generate Postman collection.
        
        Returns:
            Postman collection dictionary
        """
        openapi_spec = self.generate_openapi()
        
        collection = {
            "info": {
                "name": openapi_spec['info']['title'],
                "description": openapi_spec['info'].get('description', ''),
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "item": []
        }
        
        # Convert endpoints to Postman requests
        for path, methods in openapi_spec['paths'].items():
            for method, details in methods.items():
                if method in ['parameters', 'servers']:
                    continue
                
                request = {
                    "name": details.get('summary', f"{method.upper()} {path}"),
                    "request": {
                        "method": method.upper(),
                        "header": [],
                        "url": {
                            "raw": "{{base_url}}" + path,
                            "host": ["{{base_url}}"],
                            "path": path.strip('/').split('/')
                        }
                    },
                    "response": []
                }
                
                # Add description
                if 'description' in details:
                    request['request']['description'] = details['description']
                
                # Add request body if present
                if 'requestBody' in details:
                    content = details['requestBody'].get('content', {})
                    for content_type, schema_info in content.items():
                        request['request']['header'].append({
                            "key": "Content-Type",
                            "value": content_type
                        })
                        if 'schema' in schema_info:
                            request['request']['body'] = {
                                "mode": "raw",
                                "raw": json.dumps(
                                    self._generate_example(schema_info['schema'], openapi_spec),
                                    indent=2
                                )
                            }
                        break
                
                collection['item'].append(request)
        
        return collection
    
    def save_postman_collection(self, filename: str = "postman_collection.json") -> Path:
        """
        Save Postman collection.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        collection = self.generate_postman_collection()
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(collection, f, indent=2)
        
        print(f"✓ Generated Postman collection: {output_path}")
        return output_path
    
    def generate_all(self):
        """Generate all documentation formats."""
        self.save_openapi_json()
        self.save_openapi_yaml()
        self.save_markdown()
        self.save_postman_collection()
        
        # Generate HTML with Swagger UI
        self._generate_swagger_ui()
        
        print(f"\n✓ All documentation generated in: {self.output_dir}")
    
    def _generate_swagger_ui(self):
        """Generate HTML with embedded Swagger UI."""
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.app.title} - API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
    <style>
        body {{ margin: 0; padding: 0; }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            SwaggerUIBundle({{
                url: './openapi.json',
                dom_id: '#swagger-ui',
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                layout: "BaseLayout"
            }});
        }};
    </script>
</body>
</html>"""
        
        output_path = self.output_dir / "index.html"
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"✓ Generated Swagger UI: {output_path}")
    
    def _generate_example(self, schema: Dict[str, Any], openapi_spec: Dict[str, Any]) -> Any:
        """
        Generate example data from schema.
        
        Args:
            schema: JSON schema
            openapi_spec: Full OpenAPI specification for resolving references
            
        Returns:
            Example data
        """
        # Handle references
        if '$ref' in schema:
            ref_path = schema['$ref'].split('/')
            ref_schema = openapi_spec
            for part in ref_path[1:]:  # Skip '#'
                ref_schema = ref_schema[part]
            return self._generate_example(ref_schema, openapi_spec)
        
        # Use example if provided
        if 'example' in schema:
            return schema['example']
        
        schema_type = schema.get('type', 'object')
        
        if schema_type == 'object':
            result = {}
            properties = schema.get('properties', {})
            for prop_name, prop_schema in properties.items():
                result[prop_name] = self._generate_example(prop_schema, openapi_spec)
            return result
        
        elif schema_type == 'array':
            items_schema = schema.get('items', {})
            return [self._generate_example(items_schema, openapi_spec)]
        
        elif schema_type == 'string':
            return schema.get('default', 'string')
        
        elif schema_type == 'integer':
            return schema.get('default', 0)
        
        elif schema_type == 'number':
            return schema.get('default', 0.0)
        
        elif schema_type == 'boolean':
            return schema.get('default', False)
        
        return None
    
    def _create_anchor(self, text: str) -> str:
        """
        Create markdown anchor from text.
        
        Args:
            text: Text to convert to anchor
            
        Returns:
            Anchor string
        """
        return text.lower().replace(' ', '-').replace('/', '')


def import_app(app_path: str) -> FastAPI:
    """
    Import FastAPI application from module path.
    
    Args:
        app_path: Module path in format 'module:app'
        
    Returns:
        FastAPI application instance
    """
    module_path, app_name = app_path.split(':')
    
    # Add current directory to path
    sys.path.insert(0, str(Path.cwd()))
    
    # Import module
    import importlib
    module = importlib.import_module(module_path)
    
    # Get app instance
    app = getattr(module, app_name)
    
    if not isinstance(app, FastAPI):
        raise ValueError(f"{app_name} is not a FastAPI instance")
    
    return app


async def serve_docs(output_dir: Path, port: int = 8000):
    """
    Serve documentation with a simple HTTP server.
    
    Args:
        output_dir: Documentation directory
        port: Server port
    """
    import http.server
    import socketserver
    import os
    
    os.chdir(output_dir)
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"✓ Serving documentation at http://localhost:{port}")
        print(f"  Press Ctrl+C to stop")
        httpd.serve_forever()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive API documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all documentation formats
  python scripts/generate_docs.py --app src.api_server:app --output docs/api
  
  # Generate only markdown
  python scripts/generate_docs.py --app src.api_server:app --format markdown --output API.md
  
  # Serve documentation
  python scripts/generate_docs.py --app src.api_server:app --serve --port 8080
        """
    )
    
    parser.add_argument(
        '--app',
        required=True,
        help='FastAPI app in format "module:app"'
    )
    
    parser.add_argument(
        '--output',
        default='docs/api',
        help='Output directory or file (default: docs/api)'
    )
    
    parser.add_argument(
        '--format',
        choices=['all', 'openapi', 'markdown', 'postman'],
        default='all',
        help='Documentation format (default: all)'
    )
    
    parser.add_argument(
        '--serve',
        action='store_true',
        help='Serve documentation with HTTP server'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Server port (default: 8000)'
    )
    
    args = parser.parse_args()
    
    try:
        # Import FastAPI app
        print(f"Loading application: {args.app}")
        app = import_app(args.app)
        print(f"✓ Loaded {app.title} v{app.version}")
        
        # Create generator
        output_path = Path(args.output)
        if args.format != 'markdown' and output_path.suffix:
            output_dir = output_path.parent
        else:
            output_dir = output_path
        
        generator = DocumentationGenerator(app, output_dir)
        
        # Generate documentation
        print(f"\nGenerating documentation...")
        
        if args.format == 'all':
            generator.generate_all()
        elif args.format == 'openapi':
            generator.save_openapi_json()
            generator.save_openapi_yaml()
        elif args.format == 'markdown':
            if output_path.suffix:
                generator.save_markdown(output_path.name)
            else:
                generator.save_markdown()
        elif args.format == 'postman':
            generator.save_postman_collection()
        
        # Serve if requested
        if args.serve:
            asyncio.run(serve_docs(output_dir, args.port))
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
