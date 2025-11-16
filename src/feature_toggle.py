#!/usr/bin/env python3
"""
Feature Toggle System

Alternative implementation for feature management with simpler architecture.
Focuses on runtime configuration and easy integration.
"""

import json
import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureToggle:
    """Simple feature toggle manager"""
    
    def __init__(self, config_file: str = "config/features.json"):
        self.config_file = Path(config_file)
        self.features: Dict[str, bool] = {}
        self.load_config()
    
    def load_config(self):
        """Load feature configuration from file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.features = json.load(f)
            logger.info(f"Loaded {len(self.features)} features")
        else:
            logger.warning(f"Config file not found: {self.config_file}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default feature configuration"""
        self.features = {
            "websocket_notifications": True,
            "advanced_analytics": False,
            "model_optimization": True,
            "batch_processing": True,
            "rate_limiting": True
        }
        self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.features, f, indent=2)
        logger.info(f"Saved configuration to {self.config_file}")
    
    def is_enabled(self, feature_name: str, default: bool = False) -> bool:
        """Check if feature is enabled"""
        return self.features.get(feature_name, default)
    
    def enable(self, feature_name: str):
        """Enable a feature"""
        self.features[feature_name] = True
        self.save_config()
        logger.info(f"Enabled feature: {feature_name}")
    
    def disable(self, feature_name: str):
        """Disable a feature"""
        self.features[feature_name] = False
        self.save_config()
        logger.info(f"Disabled feature: {feature_name}")
    
    def toggle(self, feature_name: str):
        """Toggle a feature"""
        current = self.features.get(feature_name, False)
        self.features[feature_name] = not current
        self.save_config()
        logger.info(f"Toggled feature {feature_name}: {not current}")
    
    def get_all(self) -> Dict[str, bool]:
        """Get all features"""
        return self.features.copy()


# Global instance
_toggle = FeatureToggle()


def feature_enabled(feature_name: str, default: bool = False):
    """Decorator to gate functions with feature flags"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _toggle.is_enabled(feature_name, default):
                return func(*args, **kwargs)
            else:
                logger.warning(f"Feature {feature_name} is disabled")
                return None
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test feature toggles
    toggle = FeatureToggle("config/features.json")
    
    print("Current features:")
    for name, enabled in toggle.get_all().items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {name}")
