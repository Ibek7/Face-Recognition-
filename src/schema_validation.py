# Request/Response Schema Validation System

import re
import json
from typing import Any, Dict, List, Optional, Union, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
import threading

class DataType(Enum):
    """Supported data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    DATETIME = "datetime"
    EMAIL = "email"
    URL = "url"

class ValidationErrorType(Enum):
    """Validation error types."""
    TYPE_MISMATCH = "type_mismatch"
    VALUE_OUT_OF_RANGE = "value_out_of_range"
    PATTERN_MISMATCH = "pattern_mismatch"
    REQUIRED_FIELD_MISSING = "required_field_missing"
    INVALID_FORMAT = "invalid_format"
    ARRAY_LENGTH_VIOLATION = "array_length_violation"
    CUSTOM_VALIDATION_FAILED = "custom_validation_failed"

@dataclass
class ValidationError:
    """Validation error details."""
    field: str
    error_type: ValidationErrorType
    message: str
    value: Any = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'field': self.field,
            'error_type': self.error_type.value,
            'message': self.message,
            'value': str(self.value) if self.value is not None else None
        }

@dataclass
class FieldSchema:
    """Field schema definition."""
    name: str
    data_type: DataType
    required: bool = True
    default: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    enum_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None
    items_schema: Optional['FieldSchema'] = None
    nested_schema: Optional['Schema'] = None
    
    def validate(self, value: Any) -> List[ValidationError]:
        """Validate field value."""
        errors = []
        
        # Check required
        if value is None:
            if self.required:
                errors.append(ValidationError(
                    field=self.name,
                    error_type=ValidationErrorType.REQUIRED_FIELD_MISSING,
                    message=f"Required field '{self.name}' is missing"
                ))
            return errors
        
        # Type validation
        if not self._validate_type(value):
            errors.append(ValidationError(
                field=self.name,
                error_type=ValidationErrorType.TYPE_MISMATCH,
                message=f"Field '{self.name}' has invalid type. Expected {self.data_type.value}",
                value=value
            ))
            return errors
        
        # Range validation
        if self.data_type in (DataType.INTEGER, DataType.FLOAT):
            if self.min_value is not None and value < self.min_value:
                errors.append(ValidationError(
                    field=self.name,
                    error_type=ValidationErrorType.VALUE_OUT_OF_RANGE,
                    message=f"Field '{self.name}' value {value} is less than minimum {self.min_value}",
                    value=value
                ))
            
            if self.max_value is not None and value > self.max_value:
                errors.append(ValidationError(
                    field=self.name,
                    error_type=ValidationErrorType.VALUE_OUT_OF_RANGE,
                    message=f"Field '{self.name}' value {value} is greater than maximum {self.max_value}",
                    value=value
                ))
        
        # Length validation
        if self.data_type == DataType.STRING:
            if self.min_length and len(value) < self.min_length:
                errors.append(ValidationError(
                    field=self.name,
                    error_type=ValidationErrorType.ARRAY_LENGTH_VIOLATION,
                    message=f"Field '{self.name}' length {len(value)} is less than minimum {self.min_length}",
                    value=value
                ))
            
            if self.max_length and len(value) > self.max_length:
                errors.append(ValidationError(
                    field=self.name,
                    error_type=ValidationErrorType.ARRAY_LENGTH_VIOLATION,
                    message=f"Field '{self.name}' length {len(value)} exceeds maximum {self.max_length}",
                    value=value
                ))
            
            # Pattern validation
            if self.pattern and not re.match(self.pattern, value):
                errors.append(ValidationError(
                    field=self.name,
                    error_type=ValidationErrorType.PATTERN_MISMATCH,
                    message=f"Field '{self.name}' does not match pattern {self.pattern}",
                    value=value
                ))
        
        # Enum validation
        if self.enum_values and value not in self.enum_values:
            errors.append(ValidationError(
                field=self.name,
                error_type=ValidationErrorType.INVALID_FORMAT,
                message=f"Field '{self.name}' value '{value}' not in allowed values",
                value=value
            ))
        
        # Array validation
        if self.data_type == DataType.ARRAY and isinstance(value, list):
            if self.min_length and len(value) < self.min_length:
                errors.append(ValidationError(
                    field=self.name,
                    error_type=ValidationErrorType.ARRAY_LENGTH_VIOLATION,
                    message=f"Array '{self.name}' has fewer than {self.min_length} items",
                    value=value
                ))
            
            if self.max_length and len(value) > self.max_length:
                errors.append(ValidationError(
                    field=self.name,
                    error_type=ValidationErrorType.ARRAY_LENGTH_VIOLATION,
                    message=f"Array '{self.name}' has more than {self.max_length} items",
                    value=value
                ))
            
            # Validate items
            if self.items_schema:
                for i, item in enumerate(value):
                    item_errors = self.items_schema.validate(item)
                    for error in item_errors:
                        error.field = f"{self.name}[{i}].{error.field}"
                        errors.append(error)
        
        # Nested object validation
        if self.data_type == DataType.OBJECT and isinstance(value, dict):
            if self.nested_schema:
                nested_errors = self.nested_schema.validate(value)
                for error in nested_errors:
                    error.field = f"{self.name}.{error.field}"
                    errors.append(error)
        
        # Custom validation
        if self.custom_validator:
            try:
                if not self.custom_validator(value):
                    errors.append(ValidationError(
                        field=self.name,
                        error_type=ValidationErrorType.CUSTOM_VALIDATION_FAILED,
                        message=f"Custom validation failed for field '{self.name}'",
                        value=value
                    ))
            except Exception as e:
                errors.append(ValidationError(
                    field=self.name,
                    error_type=ValidationErrorType.CUSTOM_VALIDATION_FAILED,
                    message=f"Custom validation error for field '{self.name}': {str(e)}",
                    value=value
                ))
        
        return errors
    
    def _validate_type(self, value: Any) -> bool:
        """Validate value type."""
        if self.data_type == DataType.STRING:
            return isinstance(value, str)
        elif self.data_type == DataType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        elif self.data_type == DataType.FLOAT:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif self.data_type == DataType.BOOLEAN:
            return isinstance(value, bool)
        elif self.data_type == DataType.ARRAY:
            return isinstance(value, list)
        elif self.data_type == DataType.OBJECT:
            return isinstance(value, dict)
        elif self.data_type == DataType.DATETIME:
            return isinstance(value, (str, datetime))
        elif self.data_type == DataType.EMAIL:
            return isinstance(value, str) and re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', value)
        elif self.data_type == DataType.URL:
            return isinstance(value, str) and re.match(r'^https?://', value)
        
        return False

class Schema:
    """Request/response schema."""
    
    def __init__(self, name: str, fields: Dict[str, FieldSchema] = None):
        self.name = name
        self.fields = fields or {}
        self.lock = threading.RLock()
        self.validation_cache: Dict[str, List[ValidationError]] = {}
    
    def add_field(self, field: FieldSchema):
        """Add field to schema."""
        with self.lock:
            self.fields[field.name] = field
    
    def validate(self, data: Dict[str, Any], use_cache: bool = False) -> List[ValidationError]:
        """Validate data against schema."""
        
        if use_cache:
            cache_key = json.dumps(data, sort_keys=True, default=str)
            if cache_key in self.validation_cache:
                return self.validation_cache[cache_key]
        
        errors = []
        
        with self.lock:
            fields_copy = self.fields.copy()
        
        # Validate all fields
        for field_name, field_schema in fields_copy.items():
            value = data.get(field_name)
            field_errors = field_schema.validate(value)
            errors.extend(field_errors)
        
        # Check for extra fields
        for key in data.keys():
            if key not in fields_copy:
                errors.append(ValidationError(
                    field=key,
                    error_type=ValidationErrorType.INVALID_FORMAT,
                    message=f"Unknown field '{key}'",
                    value=data[key]
                ))
        
        if use_cache:
            cache_key = json.dumps(data, sort_keys=True, default=str)
            self.validation_cache[cache_key] = errors
        
        return errors
    
    def is_valid(self, data: Dict[str, Any]) -> bool:
        """Check if data is valid."""
        return len(self.validate(data)) == 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'fields': {
                name: {
                    'type': field.data_type.value,
                    'required': field.required,
                    'min': field.min_value,
                    'max': field.max_value,
                    'pattern': field.pattern,
                    'enum': field.enum_values
                }
                for name, field in self.fields.items()
            }
        }

class SchemaBuilder:
    """Builder for schemas."""
    
    def __init__(self, schema_name: str):
        self.schema = Schema(schema_name)
    
    def add_string(self, name: str, required: bool = True, 
                  min_length: int = None, max_length: int = None,
                  pattern: str = None) -> 'SchemaBuilder':
        """Add string field."""
        self.schema.add_field(FieldSchema(
            name=name,
            data_type=DataType.STRING,
            required=required,
            min_length=min_length,
            max_length=max_length,
            pattern=pattern
        ))
        return self
    
    def add_integer(self, name: str, required: bool = True,
                   min_value: int = None, max_value: int = None) -> 'SchemaBuilder':
        """Add integer field."""
        self.schema.add_field(FieldSchema(
            name=name,
            data_type=DataType.INTEGER,
            required=required,
            min_value=min_value,
            max_value=max_value
        ))
        return self
    
    def add_email(self, name: str, required: bool = True) -> 'SchemaBuilder':
        """Add email field."""
        self.schema.add_field(FieldSchema(
            name=name,
            data_type=DataType.EMAIL,
            required=required
        ))
        return self
    
    def build(self) -> Schema:
        """Build schema."""
        return self.schema

# Example usage
if __name__ == "__main__":
    # Create schema
    schema = (SchemaBuilder("UserRegistration")
        .add_string("username", min_length=3, max_length=50)
        .add_email("email")
        .add_string("password", min_length=8)
        .add_integer("age", min_value=18, max_value=120)
        .build())
    
    # Valid data
    valid_data = {
        "username": "john_doe",
        "email": "john@example.com",
        "password": "securepass123",
        "age": 25
    }
    
    errors = schema.validate(valid_data)
    print(f"Valid data - Errors: {len(errors)}")
    
    # Invalid data
    invalid_data = {
        "username": "ab",
        "email": "not-an-email",
        "password": "short",
        "age": 150,
        "extra_field": "not allowed"
    }
    
    errors = schema.validate(invalid_data)
    print(f"\nInvalid data - Errors: {len(errors)}")
    for error in errors:
        print(f"  - {error.message}")
