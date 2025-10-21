# Advanced Configuration Management System

import json
import yaml
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import threading
from datetime import datetime

@dataclass
class ConfigValue:
    """Configuration value with metadata."""
    key: str
    value: Any
    data_type: str
    description: str = ""
    required: bool = True
    default: Optional[Any] = None
    validation_rules: List[str] = field(default_factory=list)

class ConfigValidator(ABC):
    """Base validator for configuration values."""
    
    @abstractmethod
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate value and return (is_valid, error_message)."""
        pass

class RangeValidator(ConfigValidator):
    """Validate value is within range."""
    
    def __init__(self, min_val: float = None, max_val: float = None):
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate range."""
        if self.min_val is not None and value < self.min_val:
            return False, f"Value {value} is below minimum {self.min_val}"
        
        if self.max_val is not None and value > self.max_val:
            return False, f"Value {value} exceeds maximum {self.max_val}"
        
        return True, None

class ChoiceValidator(ConfigValidator):
    """Validate value is from allowed choices."""
    
    def __init__(self, choices: List[Any]):
        self.choices = choices
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate choice."""
        if value not in self.choices:
            return False, f"Value {value} not in allowed choices: {self.choices}"
        
        return True, None

class TypeValidator(ConfigValidator):
    """Validate value type."""
    
    def __init__(self, expected_type: type):
        self.expected_type = expected_type
    
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate type."""
        if not isinstance(value, self.expected_type):
            return False, f"Expected {self.expected_type}, got {type(value)}"
        
        return True, None

class ConfigurationManager:
    """Manage application configuration."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.config: Dict[str, Any] = {}
        self.validators: Dict[str, List[ConfigValidator]] = {}
        self.lock = threading.RLock()
        self.observers: List[callable] = []
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from file."""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if filename.endswith('.json'):
                config = json.load(f)
            elif filename.endswith(('.yaml', '.yml')):
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {filename}")
        
        with self.lock:
            self.config.update(config)
            self._notify_observers('load', filename)
        
        return config
    
    def load_from_env(self, prefix: str = "APP_"):
        """Load configuration from environment variables."""
        
        env_config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                env_config[config_key] = value
        
        with self.lock:
            self.config.update(env_config)
            self._notify_observers('load_env', prefix)
        
        return env_config
    
    def set_value(self, key: str, value: Any) -> bool:
        """Set configuration value with validation."""
        
        # Validate if validators exist
        if key in self.validators:
            for validator in self.validators[key]:
                is_valid, error_msg = validator.validate(value)
                if not is_valid:
                    raise ValueError(f"Validation failed for {key}: {error_msg}")
        
        with self.lock:
            old_value = self.config.get(key)
            self.config[key] = value
            self._notify_observers('set', {'key': key, 'old': old_value, 'new': value})
        
        return True
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        with self.lock:
            return self.config.get(key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section."""
        with self.lock:
            return {k: v for k, v in self.config.items() if k.startswith(f"{section}.")}
    
    def register_validator(self, key: str, validator: ConfigValidator):
        """Register validator for key."""
        if key not in self.validators:
            self.validators[key] = []
        self.validators[key].append(validator)
    
    def add_observer(self, observer: callable):
        """Add observer for configuration changes."""
        self.observers.append(observer)
    
    def _notify_observers(self, event: str, data: Any):
        """Notify observers of configuration changes."""
        for observer in self.observers:
            try:
                observer(event, data)
            except Exception as e:
                print(f"Observer error: {e}")
    
    def export_config(self, filename: str, include_sensitive: bool = False) -> str:
        """Export configuration to file."""
        
        config_path = self.config_dir / filename
        
        export_config = self.config.copy()
        
        # Remove sensitive values if not included
        if not include_sensitive:
            sensitive_keys = ['password', 'token', 'secret', 'api_key']
            for key in list(export_config.keys()):
                if any(sens in key.lower() for sens in sensitive_keys):
                    export_config[key] = "***REDACTED***"
        
        with open(config_path, 'w') as f:
            if filename.endswith('.json'):
                json.dump(export_config, f, indent=2)
            elif filename.endswith(('.yaml', '.yml')):
                yaml.dump(export_config, f)
        
        return str(config_path)
    
    def validate_config(self) -> tuple[bool, List[str]]:
        """Validate entire configuration."""
        errors = []
        
        for key, validators in self.validators.items():
            value = self.config.get(key)
            
            for validator in validators:
                is_valid, error_msg = validator.validate(value)
                if not is_valid:
                    errors.append(f"{key}: {error_msg}")
        
        return len(errors) == 0, errors
    
    def get_config_report(self) -> Dict[str, Any]:
        """Get configuration report."""
        with self.lock:
            is_valid, errors = self.validate_config()
            
            return {
                'is_valid': is_valid,
                'validation_errors': errors,
                'total_settings': len(self.config),
                'registered_validators': len(self.validators),
                'timestamp': datetime.now().isoformat()
            }

class EnvironmentConfig:
    """Environment-specific configuration."""
    
    ENVIRONMENTS = ['development', 'staging', 'production']
    
    def __init__(self, environment: str = 'development'):
        if environment not in self.ENVIRONMENTS:
            raise ValueError(f"Invalid environment: {environment}")
        
        self.environment = environment
        self.config_manager = ConfigurationManager()
    
    def load_environment_config(self):
        """Load environment-specific configuration."""
        
        # Load base config
        self.config_manager.load_config('config.json')
        
        # Load environment-specific config
        env_config_file = f"config.{self.environment}.json"
        try:
            self.config_manager.load_config(env_config_file)
        except FileNotFoundError:
            pass
        
        # Load from environment variables
        self.config_manager.load_from_env(prefix=f"APP_{self.environment.upper()}_")
    
    def get_config_manager(self) -> ConfigurationManager:
        """Get configuration manager."""
        return self.config_manager
    
    def get_environment(self) -> str:
        """Get current environment."""
        return self.environment

class FeatureFlags:
    """Feature flag management."""
    
    def __init__(self):
        self.flags: Dict[str, bool] = {}
        self.lock = threading.RLock()
    
    def enable_feature(self, feature_name: str):
        """Enable feature."""
        with self.lock:
            self.flags[feature_name] = True
    
    def disable_feature(self, feature_name: str):
        """Disable feature."""
        with self.lock:
            self.flags[feature_name] = False
    
    def is_enabled(self, feature_name: str) -> bool:
        """Check if feature is enabled."""
        with self.lock:
            return self.flags.get(feature_name, False)
    
    def get_enabled_features(self) -> List[str]:
        """Get list of enabled features."""
        with self.lock:
            return [name for name, enabled in self.flags.items() if enabled]
    
    def load_from_config(self, config: Dict[str, bool]):
        """Load feature flags from config."""
        with self.lock:
            self.flags.update(config)

# Global instances
_config_manager: Optional[ConfigurationManager] = None
_feature_flags: Optional[FeatureFlags] = None

def get_config_manager() -> ConfigurationManager:
    """Get global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

def get_feature_flags() -> FeatureFlags:
    """Get global feature flags."""
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = FeatureFlags()
    return _feature_flags

# Example usage
if __name__ == "__main__":
    # Create config manager
    config = get_config_manager()
    
    # Register validators
    config.register_validator('batch_size', RangeValidator(min_val=1, max_val=256))
    config.register_validator('model_type', ChoiceValidator(['resnet', 'vgg', 'mobilenet']))
    
    # Set values
    config.set_value('batch_size', 32)
    config.set_value('model_type', 'resnet')
    config.set_value('learning_rate', 0.001)
    
    # Get values
    print(f"Batch size: {config.get_value('batch_size')}")
    print(f"Model type: {config.get_value('model_type')}")
    
    # Get report
    report = config.get_config_report()
    print(f"\nConfiguration Report:")
    import json
    print(json.dumps(report, indent=2))
    
    # Feature flags
    flags = get_feature_flags()
    flags.enable_feature('advanced_caching')
    flags.enable_feature('federated_learning')
    
    print(f"\nEnabled features: {flags.get_enabled_features()}")
