"""
Environment configuration validator.

Validates environment variables, types, and required fields.
"""

from typing import Dict, Any, Optional, List, Callable, Type, Union
from enum import Enum
import os
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation severity level."""
    
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationResult:
    """Validation result."""
    
    def __init__(
        self,
        field: str,
        level: ValidationLevel,
        message: str
    ):
        """
        Initialize validation result.
        
        Args:
            field: Field name
            level: Severity level
            message: Error/warning message
        """
        self.field = field
        self.level = level
        self.message = message
    
    def __repr__(self) -> str:
        return f"<ValidationResult {self.level.value.upper()}: {self.field} - {self.message}>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "level": self.level.value,
            "message": self.message
        }


class FieldValidator:
    """Field validator with type checking."""
    
    def __init__(
        self,
        field_name: str,
        field_type: Type,
        required: bool = True,
        default: Optional[Any] = None,
        pattern: Optional[str] = None,
        choices: Optional[List[Any]] = None,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        custom_validator: Optional[Callable[[Any], bool]] = None,
        description: Optional[str] = None
    ):
        """
        Initialize field validator.
        
        Args:
            field_name: Field name
            field_type: Expected type
            required: Is field required
            default: Default value
            pattern: Regex pattern (for strings)
            choices: Allowed values
            min_value: Minimum value (for numbers)
            max_value: Maximum value (for numbers)
            custom_validator: Custom validation function
            description: Field description
        """
        self.field_name = field_name
        self.field_type = field_type
        self.required = required
        self.default = default
        self.pattern = re.compile(pattern) if pattern else None
        self.choices = choices
        self.min_value = min_value
        self.max_value = max_value
        self.custom_validator = custom_validator
        self.description = description
    
    def validate(self, value: Optional[str]) -> List[ValidationResult]:
        """
        Validate field value.
        
        Args:
            value: Value to validate
        
        Returns:
            List of validation results
        """
        results = []
        
        # Check if required
        if value is None:
            if self.required:
                results.append(ValidationResult(
                    field=self.field_name,
                    level=ValidationLevel.ERROR,
                    message=f"Required field '{self.field_name}' is missing"
                ))
            return results
        
        # Convert to type
        try:
            converted_value = self._convert_type(value)
        except (ValueError, TypeError) as e:
            results.append(ValidationResult(
                field=self.field_name,
                level=ValidationLevel.ERROR,
                message=f"Invalid type for '{self.field_name}': {str(e)}"
            ))
            return results
        
        # Validate pattern
        if self.pattern and isinstance(converted_value, str):
            if not self.pattern.match(converted_value):
                results.append(ValidationResult(
                    field=self.field_name,
                    level=ValidationLevel.ERROR,
                    message=f"Field '{self.field_name}' does not match pattern"
                ))
        
        # Validate choices
        if self.choices and converted_value not in self.choices:
            results.append(ValidationResult(
                field=self.field_name,
                level=ValidationLevel.ERROR,
                message=f"Field '{self.field_name}' must be one of {self.choices}"
            ))
        
        # Validate range
        if isinstance(converted_value, (int, float)):
            if self.min_value is not None and converted_value < self.min_value:
                results.append(ValidationResult(
                    field=self.field_name,
                    level=ValidationLevel.ERROR,
                    message=f"Field '{self.field_name}' must be >= {self.min_value}"
                ))
            
            if self.max_value is not None and converted_value > self.max_value:
                results.append(ValidationResult(
                    field=self.field_name,
                    level=ValidationLevel.ERROR,
                    message=f"Field '{self.field_name}' must be <= {self.max_value}"
                ))
        
        # Custom validation
        if self.custom_validator:
            try:
                if not self.custom_validator(converted_value):
                    results.append(ValidationResult(
                        field=self.field_name,
                        level=ValidationLevel.ERROR,
                        message=f"Field '{self.field_name}' failed custom validation"
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    field=self.field_name,
                    level=ValidationLevel.ERROR,
                    message=f"Custom validation error: {str(e)}"
                ))
        
        return results
    
    def _convert_type(self, value: str) -> Any:
        """Convert string value to type."""
        if self.field_type == bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif self.field_type == int:
            return int(value)
        elif self.field_type == float:
            return float(value)
        elif self.field_type == Path:
            return Path(value)
        else:
            return value
    
    def get_value(self, env_value: Optional[str]) -> Any:
        """Get value with default fallback."""
        if env_value is None:
            return self.default
        
        return self._convert_type(env_value)


class ConfigValidator:
    """Environment configuration validator."""
    
    def __init__(self):
        """Initialize config validator."""
        self.validators: Dict[str, FieldValidator] = {}
    
    def add_field(
        self,
        field_name: str,
        field_type: Type = str,
        **kwargs
    ):
        """
        Add field validator.
        
        Args:
            field_name: Field name
            field_type: Field type
            **kwargs: Additional validator parameters
        """
        validator = FieldValidator(field_name, field_type, **kwargs)
        self.validators[field_name] = validator
    
    def validate(
        self,
        config: Optional[Dict[str, str]] = None
    ) -> List[ValidationResult]:
        """
        Validate configuration.
        
        Args:
            config: Configuration dict (defaults to os.environ)
        
        Returns:
            List of validation results
        """
        if config is None:
            config = dict(os.environ)
        
        results = []
        
        for field_name, validator in self.validators.items():
            value = config.get(field_name)
            field_results = validator.validate(value)
            results.extend(field_results)
        
        return results
    
    def validate_and_raise(self, config: Optional[Dict[str, str]] = None):
        """
        Validate and raise exception on errors.
        
        Args:
            config: Configuration dict
        
        Raises:
            ValueError: If validation fails
        """
        results = self.validate(config)
        
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        
        if warnings:
            for warning in warnings:
                logger.warning(f"{warning.field}: {warning.message}")
        
        if errors:
            error_messages = [f"{e.field}: {e.message}" for e in errors]
            raise ValueError(
                f"Configuration validation failed:\n" + "\n".join(error_messages)
            )
    
    def get_config(
        self,
        config: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Get validated configuration with defaults.
        
        Args:
            config: Configuration dict
        
        Returns:
            Validated configuration
        """
        if config is None:
            config = dict(os.environ)
        
        result = {}
        
        for field_name, validator in self.validators.items():
            env_value = config.get(field_name)
            result[field_name] = validator.get_value(env_value)
        
        return result
    
    def list_fields(self) -> List[dict]:
        """List all configured fields."""
        return [
            {
                "name": v.field_name,
                "type": v.field_type.__name__,
                "required": v.required,
                "default": v.default,
                "description": v.description
            }
            for v in self.validators.values()
        ]


# Pre-configured validators
def url_validator(value: str) -> bool:
    """Validate URL format."""
    url_pattern = re.compile(
        r"^https?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain
        r"localhost|"  # localhost
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # or IP
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE
    )
    return bool(url_pattern.match(value))


def email_validator(value: str) -> bool:
    """Validate email format."""
    email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    return bool(email_pattern.match(value))


def path_exists_validator(value: Union[str, Path]) -> bool:
    """Validate path exists."""
    return Path(value).exists()


# Example usage:
"""
from src.config_validator import ConfigValidator, url_validator, email_validator

# Create validator
validator = ConfigValidator()

# Add fields
validator.add_field(
    "DATABASE_URL",
    str,
    required=True,
    custom_validator=url_validator,
    description="Database connection URL"
)

validator.add_field(
    "PORT",
    int,
    required=False,
    default=8000,
    min_value=1024,
    max_value=65535,
    description="Server port"
)

validator.add_field(
    "LOG_LEVEL",
    str,
    required=False,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    description="Logging level"
)

validator.add_field(
    "ENABLE_METRICS",
    bool,
    required=False,
    default=False,
    description="Enable metrics collection"
)

# Validate
validator.validate_and_raise()

# Get config
config = validator.get_config()
print(config["DATABASE_URL"])
print(config["PORT"])
"""
