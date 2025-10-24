# Comprehensive Request Validation & Schema Registry

import threading
import json
import re
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, field
from enum import Enum

class FieldType(Enum):
    """Field data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    ENUM = "enum"
    DATE = "date"
    UUID = "uuid"

class ValidationError(Enum):
    """Validation error types."""
    MISSING_FIELD = "missing_field"
    INVALID_TYPE = "invalid_type"
    INVALID_FORMAT = "invalid_format"
    VALUE_OUT_OF_RANGE = "value_out_of_range"
    PATTERN_MISMATCH = "pattern_mismatch"
    ENUM_VIOLATION = "enum_violation"
    CUSTOM_VIOLATION = "custom_violation"

@dataclass
class FieldSchema:
    """Field schema definition."""
    name: str
    field_type: FieldType
    required: bool = True
    default: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    enum_values: Optional[List[Any]] = None
    custom_validators: List[Callable] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.field_type.value,
            'required': self.required,
            'description': self.description
        }

@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    errors: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings
        }

class FieldValidator:
    """Validate individual fields."""
    
    @staticmethod
    def validate_field(value: Any, schema: FieldSchema) -> List[Dict[str, str]]:
        """Validate field against schema."""
        errors = []
        
        # Check required
        if schema.required and value is None:
            errors.append({
                'field': schema.name,
                'error_type': ValidationError.MISSING_FIELD.value,
                'message': f"Field '{schema.name}' is required"
            })
            return errors
        
        if value is None:
            return errors
        
        # Check type
        if not FieldValidator._check_type(value, schema.field_type):
            errors.append({
                'field': schema.name,
                'error_type': ValidationError.INVALID_TYPE.value,
                'message': f"Field '{schema.name}' must be {schema.field_type.value}"
            })
            return errors
        
        # Check enum
        if schema.enum_values and value not in schema.enum_values:
            errors.append({
                'field': schema.name,
                'error_type': ValidationError.ENUM_VIOLATION.value,
                'message': f"Field '{schema.name}' must be one of {schema.enum_values}"
            })
        
        # Check range
        if isinstance(value, (int, float)):
            if schema.min_value is not None and value < schema.min_value:
                errors.append({
                    'field': schema.name,
                    'error_type': ValidationError.VALUE_OUT_OF_RANGE.value,
                    'message': f"Field '{schema.name}' must be >= {schema.min_value}"
                })
            
            if schema.max_value is not None and value > schema.max_value:
                errors.append({
                    'field': schema.name,
                    'error_type': ValidationError.VALUE_OUT_OF_RANGE.value,
                    'message': f"Field '{schema.name}' must be <= {schema.max_value}"
                })
        
        # Check length
        if isinstance(value, str):
            if schema.min_length and len(value) < schema.min_length:
                errors.append({
                    'field': schema.name,
                    'error_type': ValidationError.VALUE_OUT_OF_RANGE.value,
                    'message': f"Field '{schema.name}' min length is {schema.min_length}"
                })
            
            if schema.max_length and len(value) > schema.max_length:
                errors.append({
                    'field': schema.name,
                    'error_type': ValidationError.VALUE_OUT_OF_RANGE.value,
                    'message': f"Field '{schema.name}' max length is {schema.max_length}"
                })
            
            # Check pattern
            if schema.pattern and not re.match(schema.pattern, value):
                errors.append({
                    'field': schema.name,
                    'error_type': ValidationError.PATTERN_MISMATCH.value,
                    'message': f"Field '{schema.name}' doesn't match pattern"
                })
        
        # Custom validators
        for validator in schema.custom_validators:
            try:
                if not validator(value):
                    errors.append({
                        'field': schema.name,
                        'error_type': ValidationError.CUSTOM_VIOLATION.value,
                        'message': f"Custom validation failed for '{schema.name}'"
                    })
            except Exception as e:
                errors.append({
                    'field': schema.name,
                    'error_type': ValidationError.CUSTOM_VIOLATION.value,
                    'message': str(e)
                })
        
        return errors
    
    @staticmethod
    def _check_type(value: Any, field_type: FieldType) -> bool:
        """Check if value matches field type."""
        if field_type == FieldType.STRING:
            return isinstance(value, str)
        elif field_type == FieldType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        elif field_type == FieldType.FLOAT:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif field_type == FieldType.BOOLEAN:
            return isinstance(value, bool)
        elif field_type == FieldType.ARRAY:
            return isinstance(value, list)
        elif field_type == FieldType.OBJECT:
            return isinstance(value, dict)
        elif field_type == FieldType.UUID:
            if not isinstance(value, str):
                return False
            return re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', value) is not None
        elif field_type == FieldType.DATE:
            if not isinstance(value, str):
                return False
            return re.match(r'^\d{4}-\d{2}-\d{2}$', value) is not None
        
        return True

class SchemaDefinition:
    """Define request/response schema."""
    
    def __init__(self, schema_id: str, name: str):
        self.schema_id = schema_id
        self.name = name
        self.fields: Dict[str, FieldSchema] = {}
        self.created_at = time.time() if 'time' in dir() else 0
    
    def add_field(self, field: FieldSchema) -> None:
        """Add field to schema."""
        self.fields[field.name] = field
    
    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data against schema."""
        errors = []
        sanitized_data = {}
        
        for field_name, field_schema in self.fields.items():
            value = data.get(field_name)
            
            # Validate field
            field_errors = FieldValidator.validate_field(value, field_schema)
            errors.extend(field_errors)
            
            # Sanitize data
            if value is not None:
                sanitized_data[field_name] = value
            elif field_schema.default is not None:
                sanitized_data[field_name] = field_schema.default
        
        # Check for extra fields
        for data_key in data.keys():
            if data_key not in self.fields:
                errors.append({
                    'field': data_key,
                    'error_type': 'unknown_field',
                    'message': f"Unknown field '{data_key}'"
                })
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            sanitized_data=sanitized_data
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'schema_id': self.schema_id,
            'name': self.name,
            'fields': {name: field.to_dict() for name, field in self.fields.items()}
        }

class SchemaRegistry:
    """Registry for schemas."""
    
    def __init__(self):
        self.schemas: Dict[str, SchemaDefinition] = {}
        self.schema_versions: Dict[str, List[SchemaDefinition]] = {}
        self.lock = threading.RLock()
    
    def register_schema(self, schema: SchemaDefinition) -> None:
        """Register schema."""
        with self.lock:
            self.schemas[schema.schema_id] = schema
            
            # Track versions
            if schema.name not in self.schema_versions:
                self.schema_versions[schema.name] = []
            
            self.schema_versions[schema.name].append(schema)
    
    def get_schema(self, schema_id: str) -> Optional[SchemaDefinition]:
        """Get schema by ID."""
        with self.lock:
            return self.schemas.get(schema_id)
    
    def get_latest_schema(self, schema_name: str) -> Optional[SchemaDefinition]:
        """Get latest version of schema."""
        with self.lock:
            versions = self.schema_versions.get(schema_name)
            return versions[-1] if versions else None
    
    def list_schemas(self) -> List[Dict]:
        """List all schemas."""
        with self.lock:
            return [schema.to_dict() for schema in self.schemas.values()]

class RequestValidator:
    """Validate incoming requests."""
    
    def __init__(self, registry: SchemaRegistry):
        self.registry = registry
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.lock = threading.RLock()
    
    def validate(self, data: Dict[str, Any], schema_id: str,
                use_cache: bool = False) -> ValidationResult:
        """Validate request against schema."""
        # Check cache
        cache_key = f"{schema_id}:{json.dumps(data, sort_keys=True, default=str)}"
        
        with self.lock:
            if use_cache and cache_key in self.validation_cache:
                return self.validation_cache[cache_key]
        
        # Get schema
        schema = self.registry.get_schema(schema_id)
        if not schema:
            return ValidationResult(
                is_valid=False,
                errors=[{'error': 'Unknown schema', 'schema_id': schema_id}]
            )
        
        # Validate
        result = schema.validate(data)
        
        # Cache result
        with self.lock:
            self.validation_cache[cache_key] = result
            
            # Keep only last 1000 validations
            if len(self.validation_cache) > 1000:
                # Remove oldest
                oldest_key = next(iter(self.validation_cache))
                del self.validation_cache[oldest_key]
        
        return result

# Example usage
if __name__ == "__main__":
    import time
    
    # Create schema registry
    registry = SchemaRegistry()
    
    # Define schema
    schema = SchemaDefinition("schema1", "CreateFaceDetection")
    schema.add_field(FieldSchema(
        name="image",
        field_type=FieldType.STRING,
        required=True,
        min_length=1,
        max_length=1000
    ))
    schema.add_field(FieldSchema(
        name="confidence",
        field_type=FieldType.FLOAT,
        required=False,
        default=0.5,
        min_value=0.0,
        max_value=1.0
    ))
    schema.add_field(FieldSchema(
        name="mode",
        field_type=FieldType.ENUM,
        required=True,
        enum_values=["fast", "accurate", "balanced"]
    ))
    
    # Register schema
    registry.register_schema(schema)
    
    # Create validator
    validator = RequestValidator(registry)
    
    # Validate valid request
    valid_data = {
        "image": "test.jpg",
        "confidence": 0.8,
        "mode": "accurate"
    }
    result = validator.validate(valid_data, "schema1")
    print(f"Valid request: {result.is_valid}")
    
    # Validate invalid request
    invalid_data = {
        "image": "",
        "confidence": 1.5,
        "mode": "invalid"
    }
    result = validator.validate(invalid_data, "schema1")
    print(f"Invalid request: {result.is_valid}")
    print(f"Errors: {result.errors}")
